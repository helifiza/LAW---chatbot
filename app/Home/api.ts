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

export type SessionInfo = {
  session_id: string;
  status: string;
  created_at: string;
  updated_at: string;
  expires_at: string;
};

export type SessionSnapshot = SessionInfo & {
  documents: ApiDocument[];
  messages: ApiMessage[];
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

const API_BASE_URL =
  process.env.NEXT_PUBLIC_RAG_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";

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
    response = await fetch(`${API_BASE_URL}${url}`, init);
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

export function createSession(): Promise<SessionInfo> {
  return request<SessionInfo>("/sessions", { method: "POST" });
}

export function getSession(sessionId: string): Promise<SessionSnapshot> {
  return request<SessionSnapshot>(`/sessions/${sessionId}`);
}

export async function closeSession(sessionId: string): Promise<void> {
  await request<null>(`/sessions/${sessionId}`, { method: "DELETE" });
}

export function uploadSessionDocuments(
  sessionId: string,
  files: File[],
): Promise<UploadResult> {
  const body = new FormData();
  files.forEach((file) => body.append("files", file));
  return request<UploadResult>(`/sessions/${sessionId}/documents`, {
    method: "POST",
    body,
  });
}

export async function deleteSessionDocument(
  sessionId: string,
  documentId: string,
): Promise<void> {
  await request<null>(`/sessions/${sessionId}/documents/${documentId}`, {
    method: "DELETE",
  });
}

export async function clearSessionDocuments(sessionId: string): Promise<void> {
  await request<null>(`/sessions/${sessionId}/documents`, { method: "DELETE" });
}

export function askSessionQuestion(
  sessionId: string,
  question: string,
  topK = 5,
): Promise<QuestionResult> {
  return request<QuestionResult>(`/sessions/${sessionId}/questions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: topK }),
  });
}
