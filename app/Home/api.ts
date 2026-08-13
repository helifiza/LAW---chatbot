export type ApiDocument = {
  id: string;
  file_name: string;
  mime_type: string;
  size_bytes: number;
  status: "processing" | "ready" | "failed";
  chunk_count: number;
  error_message: string | null;
  created_at: string;
};

export type ApiMessage = {
  id: number;
  role: "user" | "assistant";
  content: string;
  created_at: string;
};

export type HistoryStatus = "active" | "archived";

export type HistoryInfo = {
  history_id: string;
  user_id: string;
  title: string;
  status: HistoryStatus;
  created_at: string;
  updated_at: string;
};

export type HistorySnapshot = HistoryInfo & {
  documents: ApiDocument[];
  messages: ApiMessage[];
};

export type HistorySummary = {
  history_id: string;
  title: string;
  status: HistoryStatus;
  created_at: string;
  updated_at: string;
};


export type UploadResult = {
  documents: ApiDocument[];
  errors: Array<{ file_name: string; message: string }>;
};

export type QuestionSource = {
  document_id: string;
  file_name: string;
  page_number: number;
  page_end_number: number;
  dieu: string | null;
  score: number;
  excerpt: string;
};

export type QuestionResult = {
  question: string;
  answer: string;
  sources: QuestionSource[];
  retrieved_count: number;
};

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  process.env.NEXT_PUBLIC_RAG_API_BASE_URL ??
  process.env.NEXT_PUBLIC_AUTH_API_BASE_URL ??
  "http://localhost:8000/api/v1";

export class ApiError extends Error {
  status: number;
  code?: string;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

async function readPayload(response: Response) {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return { error: { message: text } };
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${url}`, {
      credentials: "include",
      ...init,
    });
  } catch {
    throw new ApiError(
      "Không kết nối được FastAPI. Hãy kiểm tra backend đang chạy ở cổng 8000.",
      0,
      "NETWORK_ERROR",
    );
  }
  const payload = await readPayload(response);
  if (!response.ok) {
    const message =
      payload?.error?.message ||
      payload?.detail ||
      "Yêu cầu tới backend thất bại.";
    throw new ApiError(message, response.status, payload?.error?.code);
  }
  return payload as T;
}

export async function login(
  email: string,
  password: string,
): Promise<{ message: string; user: { id: string; full_name: string; email: string; role: string } }> {
  return request<{ message: string; user: { id: string; full_name: string; email: string; role: string } }>(
    "/auth/login",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    },
  );
}

export function createHistory(question?: string): Promise<HistoryInfo> {
  const body = question ? JSON.stringify({ question }) : undefined;
  return request<HistoryInfo>("/histories", {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body,
  });
}

export function getHistory(historyId: string): Promise<HistorySnapshot> {
  return request<HistorySnapshot>(`/histories/${historyId}`);
}

export async function closeHistory(historyId: string): Promise<void> {
  await request<null>(`/histories/${historyId}`, { method: "DELETE" });
}

export function uploadHistoryDocuments(
  historyId: string,
  files: File[],
): Promise<UploadResult> {
  const body = new FormData();
  files.forEach((file) => body.append("files", file));
  return request<UploadResult>(`/histories/${historyId}/documents`, {
    method: "POST",
    body,
  });
}

export async function deleteHistoryDocument(
  historyId: string,
  documentId: string,
): Promise<void> {
  await request<null>(`/histories/${historyId}/documents/${documentId}`, {
    method: "DELETE",
  });
}

export async function clearHistoryDocuments(historyId: string): Promise<void> {
  await request<null>(`/histories/${historyId}/documents`, { method: "DELETE" });
}

export function askHistoryQuestion(
  historyId: string,
  question: string,
  topK = 5,
): Promise<QuestionResult> {
  return request<QuestionResult>(`/histories/${historyId}/questions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: topK }),
  });
}

export function listHistories(): Promise<HistorySummary[]> {
  return request<HistorySummary[]>("/histories");
}

export function archiveHistory(historyId: string): Promise<HistoryInfo> {
  return request<HistoryInfo>(`/histories/${historyId}/archive`, { method: "PATCH" });
}

export function reopenHistory(historyId: string): Promise<HistoryInfo> {
  return request<HistoryInfo>(`/histories/${historyId}/reopen`, { method: "PATCH" });
}