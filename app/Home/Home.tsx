"use client";

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from "react";
import "./Chat.css";
import styles from "./Home.module.css";
import {
  ApiError,
  askSessionQuestion,
  clearSessionDocuments,
  closeSession,
  createSession,
  deleteSessionDocument,
  getSession,
  uploadSessionDocuments,
  type ApiDocument,
  type SessionSnapshot,
} from "./api";

const MAX_FILES = 5;
const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ALLOWED_EXTENSIONS = ["pdf", "docx", "txt", "md", "csv", "json"];
const SESSION_STORAGE_KEY = "slaw.rag.session_id";

type MessageSource = { fileName: string; locator: string; excerpt: string };
type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  attachments?: string[];
  sources?: MessageSource[];
  error?: boolean;
};

const INITIAL_MESSAGES: Message[] = [
  {
    id: "welcome",
    role: "assistant",
    content:
      "Xin chào! Hãy tải tài liệu lên, chờ hệ thống lập chỉ mục, sau đó đặt câu hỏi. Phiên hiện tại sẽ giữ tài liệu và lịch sử chat khi bạn tải lại trang.",
  },
];

const SUGGESTIONS = [
  "Tóm tắt nội dung quan trọng nhất",
  "Liệt kê các khái niệm chính",
  "Tạo 5 câu hỏi ôn tập",
];

const ICONS: Record<string, ReactNode> = {
  plus: (
    <>
      <path d="M12 5v14" />
      <path d="M5 12h14" />
    </>
  ),
  upload: (
    <>
      <path d="M12 16V4" />
      <path d="m7 9 5-5 5 5" />
      <path d="M5 15v4h14v-4" />
    </>
  ),
  file: (
    <>
      <path d="M6 2h8l4 4v16H6z" />
      <path d="M14 2v5h5" />
      <path d="M9 13h6M9 17h6" />
    </>
  ),
  trash: (
    <>
      <path d="M4 7h16M9 7V4h6v3M7 7l1 14h8l1-14" />
      <path d="M10 11v6M14 11v6" />
    </>
  ),
  user: (
    <>
      <circle cx="12" cy="8" r="4" />
      <path d="M4 21c.6-5 3.3-7 8-7s7.4 2 8 7" />
    </>
  ),
  bot: (
    <>
      <rect x="4" y="7" width="16" height="13" rx="4" />
      <path d="M12 3v4M8 12h.01M16 12h.01M8 16h8" />
    </>
  ),
  menu: <path d="M4 7h16M4 12h16M4 17h16" />,
  close: <path d="m6 6 12 12M18 6 6 18" />,
  clip: (
    <path d="m20 11-8.5 8.5a5 5 0 0 1-7-7L14 3a3.5 3.5 0 0 1 5 5l-9.5 9.5a2 2 0 0 1-3-3L15 6" />
  ),
  send: (
    <>
      <path d="m22 2-7 20-4-9-9-4z" />
      <path d="M22 2 11 13" />
    </>
  ),
  sparkle: (
    <>
      <path d="m12 3 1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5z" />
      <path d="m19 15 .7 2.3L22 18l-2.3.7L19 21l-.7-2.3L16 18l2.3-.7z" />
    </>
  ),
  check: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="m8 12 2.5 2.5L16 9" />
    </>
  ),
  alert: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v6M12 17h.01" />
    </>
  ),
};

function Icon({
  name,
  size = 20,
  strokeWidth = 1.8,
}: {
  name: string;
  size?: number;
  strokeWidth?: number;
}) {
  return (
    <svg
      aria-hidden="true"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {ICONS[name]}
    </svg>
  );
}

function makeId() {
  return typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function fileSignature(file: File) {
  return `${file.name}-${file.size}`;
}
function extensionOf(fileName: string) {
  return fileName.split(".").pop()?.toLowerCase() || "";
}
function formatFileSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
function documentStatus(document: ApiDocument) {
  if (document.status === "processing") return "Đang lập chỉ mục";
  if (document.status === "failed") return "Lỗi lập chỉ mục";
  return `${document.chunk_count} đoạn đã index`;
}
function messagesFromSnapshot(snapshot: SessionSnapshot): Message[] {
  return [
    ...INITIAL_MESSAGES,
    ...snapshot.messages.map((message) => ({
      id: `server-${message.id}`,
      role: message.role,
      content: message.content,
    })),
  ];
}
async function restoreOrCreateSession(): Promise<SessionSnapshot> {
  const stored = window.localStorage.getItem(SESSION_STORAGE_KEY);
  if (stored) {
    try {
      return await getSession(stored);
    } catch (error) {
      if (!(error instanceof ApiError) || ![404, 410].includes(error.status))
        throw error;
      window.localStorage.removeItem(SESSION_STORAGE_KEY);
    }
  }
  const created = await createSession();
  window.localStorage.setItem(SESSION_STORAGE_KEY, created.session_id);
  return { ...created, documents: [], messages: [] };
}

export default function Home() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [documents, setDocuments] = useState<ApiDocument[]>([]);
  const [pendingIds, setPendingIds] = useState<string[]>([]);
  const [messages, setMessages] = useState<Message[]>(INITIAL_MESSAGES);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState("");
  const [isSessionLoading, setIsSessionLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const initializationRef = useRef<Promise<SessionSnapshot> | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const readyDocuments = useMemo(
    () => documents.filter((item) => item.status === "ready"),
    [documents],
  );
  const pendingNames = useMemo(
    () =>
      documents
        .filter((item) => pendingIds.includes(item.id))
        .map((item) => item.file_name),
    [documents, pendingIds],
  );

  useEffect(() => {
    let active = true;
    initializationRef.current ??= restoreOrCreateSession();
    initializationRef.current
      .then((snapshot) => {
        if (!active) return;
        setSessionId(snapshot.session_id);
        setDocuments(snapshot.documents);
        setMessages(messagesFromSnapshot(snapshot));
      })
      .catch((requestError: unknown) => {
        if (active)
          setError(
            requestError instanceof Error
              ? requestError.message
              : "Không khởi tạo được phiên làm việc.",
          );
      })
      .finally(() => {
        if (active) setIsSessionLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isLoading]);
  useEffect(() => {
    if (!textareaRef.current) return;
    textareaRef.current.style.height = "0px";
    textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 140)}px`;
  }, [draft]);

  async function addFiles(fileList: FileList | null) {
    if (!sessionId || isUploading || isLoading || isSessionLoading) return;
    setError("");
    const incoming = Array.from(fileList || []);
    const known = new Set(
      documents.map((item) => `${item.file_name}-${item.size_bytes}`),
    );
    const accepted: File[] = [];
    const rejected: string[] = [];
    const activeCount = documents.filter(
      (item) => item.status !== "failed",
    ).length;
    incoming.forEach((file) => {
      const signature = fileSignature(file);
      if (known.has(signature)) {
        rejected.push(`${file.name}: đã có trong phiên`);
        return;
      }
      if (!ALLOWED_EXTENSIONS.includes(extensionOf(file.name))) {
        rejected.push(`${file.name}: định dạng chưa hỗ trợ`);
        return;
      }
      if (file.size > MAX_FILE_SIZE) {
        rejected.push(`${file.name}: lớn hơn 10 MB`);
        return;
      }
      if (activeCount + accepted.length >= MAX_FILES) {
        rejected.push(`Chỉ được dùng tối đa ${MAX_FILES} file trong một phiên`);
        return;
      }
      accepted.push(file);
      known.add(signature);
    });
    if (!accepted.length) {
      if (rejected.length) setError(rejected.join(" · "));
      return;
    }
    const oldIds = new Set(documents.map((item) => item.id));
    setIsUploading(true);
    try {
      const result = await uploadSessionDocuments(sessionId, accepted);
      setDocuments(result.documents);
      setPendingIds((current) => [
        ...current,
        ...result.documents
          .filter((item) => !oldIds.has(item.id) && item.status === "ready")
          .map((item) => item.id),
      ]);
      const allErrors = [
        ...rejected,
        ...result.errors.map((item) => `${item.file_name}: ${item.message}`),
      ];
      if (allErrors.length) setError(allErrors.join(" · "));
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Không thể upload tài liệu.",
      );
    } finally {
      setIsUploading(false);
    }
  }

  async function removeFile(documentId: string) {
    if (!sessionId || isUploading || isLoading || isSessionLoading) return;
    setError("");
    try {
      await deleteSessionDocument(sessionId, documentId);
      setDocuments((current) =>
        current.filter((item) => item.id !== documentId),
      );
      setPendingIds((current) => current.filter((id) => id !== documentId));
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Không thể xóa tài liệu.",
      );
    }
  }

  async function clearAllFiles() {
    if (!sessionId || isUploading || isLoading || isSessionLoading) return;
    setError("");
    try {
      await clearSessionDocuments(sessionId);
      setDocuments([]);
      setPendingIds([]);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Không thể xóa tài liệu.",
      );
    }
  }

  async function newChat() {
    if (isSessionLoading || isUploading || isLoading) return;
    setIsSessionLoading(true);
    setError("");
    try {
      if (sessionId) {
        try {
          await closeSession(sessionId);
        } catch {
          /* phiên có thể đã hết hạn */
        }
      }
      const created = await createSession();
      window.localStorage.setItem(SESSION_STORAGE_KEY, created.session_id);
      setSessionId(created.session_id);
      setDocuments([]);
      setPendingIds([]);
      setMessages(INITIAL_MESSAGES);
      setDraft("");
      setSidebarOpen(false);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Không tạo được phiên mới.",
      );
    } finally {
      setIsSessionLoading(false);
    }
  }

  async function sendMessage() {
    const question = draft.trim();
    if (!question || isLoading || isUploading || isSessionLoading) return;
    if (!sessionId) {
      setError("Phiên làm việc chưa sẵn sàng.");
      return;
    }
    if (!readyDocuments.length) {
      setError("Bạn cần ít nhất một tài liệu đã lập chỉ mục thành công.");
      inputRef.current?.click();
      return;
    }
    setMessages((current) => [
      ...current,
      {
        id: makeId(),
        role: "user",
        content: question,
        attachments: pendingNames,
      },
    ]);
    setDraft("");
    setError("");
    setPendingIds([]);
    setIsLoading(true);
    try {
      const result = await askSessionQuestion(sessionId, question, 5);
      const sources: MessageSource[] = result.sources.map((source) => ({
        fileName: source.file_name,
        locator: [
          source.page_number === source.page_end_number
            ? `Trang ${source.page_number}`
            : `Trang ${source.page_number}-${source.page_end_number}`,
          source.dieu,
          `score=${source.score.toFixed(3)}`,
        ]
          .filter(Boolean)
          .join(" · "),
        excerpt: source.excerpt,
      }));
      setMessages((current) => [
        ...current,
        { id: makeId(), role: "assistant", content: result.answer, sources },
      ]);
    } catch (requestError) {
      const text =
        requestError instanceof Error
          ? requestError.message
          : "Không thể nhận phản hồi";
      setMessages((current) => [
        ...current,
        {
          id: makeId(),
          role: "assistant",
          content: `Không thể nhận phản hồi: ${text}`,
          error: true,
        },
      ]);
    } finally {
      setIsLoading(false);
      requestAnimationFrame(() => textareaRef.current?.focus());
    }
  }

  function onComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  }
  const statusText = isSessionLoading
    ? "Đang tạo phiên"
    : isUploading
      ? "Đang indexing"
      : "Grounded";

  return (
    <main className={styles.page}>
      <section
        className={styles.shell}
        onDragEnter={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => {
          const target = event.relatedTarget;
          if (
            !target ||
            !(target instanceof Node) ||
            !event.currentTarget.contains(target)
          )
            setIsDragging(false);
        }}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          void addFiles(event.dataTransfer.files);
        }}
      >
        <aside
          className={`${styles.sidebar} ${sidebarOpen ? styles.sidebarOpen : ""}`}
        >
          <div className={styles.sidebarHeader}>
            <div className={styles.brandLine}>
              <span className={styles.logo}>SLAW</span>
              <button
                className={styles.mobileClose}
                type="button"
                aria-label="Đóng menu"
                onClick={() => setSidebarOpen(false)}
              >
                <Icon name="close" />
              </button>
            </div>
            <div className={styles.primaryActions}>
              <button
                className={styles.newChat}
                type="button"
                onClick={() => void newChat()}
                disabled={isSessionLoading || isUploading || isLoading}
              >
                <Icon name="plus" size={17} strokeWidth={2.2} />
                <span>New chat</span>
              </button>
            </div>
          </div>
          <div className={styles.fileSection}>
            <div className={styles.sectionTitle}>
              <span>File</span>
              {documents.length > 0 && (
                <button
                  type="button"
                  onClick={() => void clearAllFiles()}
                  disabled={isSessionLoading || isUploading || isLoading}
                >
                  Clear All
                </button>
              )}
            </div>
            <button
              className={styles.uploadButton}
              type="button"
              onClick={() => inputRef.current?.click()}
              disabled={isUploading || isLoading || isSessionLoading}
            >
              <span>
                <Icon name="upload" size={19} />
              </span>
              <strong>
                {isUploading ? "Đang xử lý tài liệu…" : "Thêm tài liệu"}
              </strong>
              <small>PDF, DOCX, TXT, MD, CSV, JSON</small>
            </button>
            <div className={styles.fileList}>
              {documents.map((document) => (
                <div className={styles.fileItem} key={document.id}>
                  <span className={styles.fileType}>
                    <Icon
                      name={document.status === "failed" ? "alert" : "file"}
                      size={17}
                    />
                  </span>
                  <span className={styles.fileInfo}>
                    <strong title={document.file_name}>
                      {document.file_name}
                    </strong>
                    <small title={document.error_message || undefined}>
                      {formatFileSize(document.size_bytes)} ·{" "}
                      {documentStatus(document)}
                    </small>
                  </span>
                  <button
                    type="button"
                    aria-label={`Xóa ${document.file_name}`}
                    onClick={() => void removeFile(document.id)}
                    disabled={isSessionLoading || isUploading || isLoading}
                  >
                    <Icon name="trash" size={15} />
                  </button>
                </div>
              ))}
              {!documents.length && (
                <p className={styles.emptyFileText}>Chưa có tài liệu nào.</p>
              )}
            </div>
          </div>
          <div className={styles.profile}>
            <span className={styles.profileIcon}>
              <Icon name="user" size={19} />
            </span>
            <span>
              <strong>Phiên cục bộ</strong>
              <small>
                {sessionId ? `Phiên ${sessionId.slice(0, 8)}` : "Đang kết nối"}
              </small>
            </span>
          </div>
        </aside>
        {sidebarOpen && (
          <button
            className={styles.backdrop}
            type="button"
            aria-label="Đóng menu"
            onClick={() => setSidebarOpen(false)}
          />
        )}
        <section className={styles.chatArea}>
          <header className={styles.chatHeader}>
            <button
              className={styles.menuButton}
              type="button"
              aria-label="Mở menu"
              onClick={() => setSidebarOpen(true)}
            >
              <Icon name="menu" />
            </button>
            <div className={styles.chatTitle}>
              <h1>Hỏi đáp tài liệu cá nhân</h1>
              <p>
                {documents.length
                  ? `${readyDocuments.length}/${documents.length} tài liệu sẵn sàng`
                  : "Tải tài liệu để bắt đầu"}
              </p>
            </div>
            <span className={styles.statusBadge}>
              <Icon name="check" size={15} />
              {statusText}
            </span>
          </header>
          <div className={styles.messageScroller}>
            <div className={styles.messageList}>
              {messages.map((message) => {
                const isUser = message.role === "user";
                const attachments = message.attachments ?? [];
                const sources = message.sources ?? [];
                return (
                  <article
                    className={`${styles.messageRow} ${isUser ? styles.userRow : styles.assistantRow}`}
                    key={message.id}
                  >
                    {!isUser && (
                      <span className={`${styles.avatar} ${styles.botAvatar}`}>
                        <Icon name="bot" size={18} />
                      </span>
                    )}
                    <div
                      className={`${styles.bubble} ${isUser ? styles.userBubble : styles.assistantBubble}`}
                    >
                      {attachments.length > 0 && (
                        <div className={styles.attachmentsInMessage}>
                          {attachments.map((name) => (
                            <span key={name}>
                              <Icon name="file" size={14} />
                              {name}
                            </span>
                          ))}
                        </div>
                      )}
                      <p className={message.error ? styles.errorAnswer : ""}>
                        {message.content}
                      </p>
                      {!isUser && sources.length > 0 && (
                        <div className={styles.sources}>
                          <div className={styles.sourcesHeading}>
                            <Icon name="sparkle" size={15} />
                            Nguồn tham khảo
                          </div>
                          <div className={styles.sourceList}>
                            {sources.map((source, index) => (
                              <details
                                className={styles.sourceCard}
                                key={`${source.fileName}-${index}`}
                              >
                                <summary>
                                  <span>{index + 1}</span>
                                  <strong>{source.fileName}</strong>
                                  <small>{source.locator}</small>
                                </summary>
                                <p>{source.excerpt}</p>
                              </details>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                    {isUser && (
                      <span className={`${styles.avatar} ${styles.userAvatar}`}>
                        <Icon name="user" size={18} />
                      </span>
                    )}
                  </article>
                );
              })}
              {messages.length === 1 && (
                <div className={styles.suggestions}>
                  <span>Gợi ý câu hỏi</span>
                  <div>
                    {SUGGESTIONS.map((suggestion) => (
                      <button
                        type="button"
                        key={suggestion}
                        onClick={() => {
                          setDraft(suggestion);
                          textareaRef.current?.focus();
                        }}
                      >
                        <Icon name="sparkle" size={16} />
                        {suggestion}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {isLoading && (
                <article
                  className={`${styles.messageRow} ${styles.assistantRow}`}
                >
                  <span className={`${styles.avatar} ${styles.botAvatar}`}>
                    <Icon name="bot" size={18} />
                  </span>
                  <div className={`${styles.bubble} ${styles.assistantBubble}`}>
                    <div className={styles.typing} aria-label="Đang trả lời">
                      <span />
                      <span />
                      <span />
                    </div>
                  </div>
                </article>
              )}
              <div ref={bottomRef} />
            </div>
          </div>
          <div className={styles.composerDock}>
            <div className={styles.composerInner}>
              {pendingNames.length > 0 && (
                <div className={styles.pendingFiles}>
                  {pendingNames.map((name) => (
                    <span key={name}>
                      <Icon name="file" size={14} />
                      {name}
                    </span>
                  ))}
                </div>
              )}
              {error && (
                <div className={styles.errorMessage} role="alert">
                  <Icon name="alert" size={15} />
                  {error}
                </div>
              )}
              <div className={styles.composer}>
                <button
                  type="button"
                  className={styles.attachButton}
                  aria-label="Đính kèm tài liệu"
                  onClick={() => inputRef.current?.click()}
                  disabled={isUploading || isLoading || isSessionLoading}
                >
                  <Icon name="clip" size={20} />
                </button>
                <textarea
                  ref={textareaRef}
                  value={draft}
                  rows={1}
                  placeholder={
                    readyDocuments.length
                      ? "Hỏi bất cứ điều gì về tài liệu…"
                      : isUploading
                        ? "Đang lập chỉ mục tài liệu…"
                        : "Upload file trước khi chat"
                  }
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={onComposerKeyDown}
                  disabled={isSessionLoading}
                />
                <button
                  type="button"
                  className={styles.sendButton}
                  aria-label="Gửi câu hỏi"
                  disabled={
                    !draft.trim() ||
                    isLoading ||
                    isUploading ||
                    isSessionLoading
                  }
                  onClick={() => void sendMessage()}
                >
                  <Icon name="send" size={19} />
                </button>
              </div>
              <p className={styles.composerHint}>
                Enter để gửi · Shift + Enter để xuống dòng
              </p>
            </div>
          </div>
        </section>
        {isDragging && (
          <div className={styles.dropOverlay}>
            <Icon name="upload" size={40} />
            <strong>Thả tài liệu vào đây</strong>
            <span>PDF, DOCX, TXT, MD, CSV, JSON</span>
          </div>
        )}
        <input
          ref={inputRef}
          className={styles.hiddenInput}
          type="file"
          multiple
          accept=".pdf,.docx,.txt,.md,.csv,.json"
          onChange={(event) => {
            void addFiles(event.target.files);
            event.target.value = "";
          }}
        />
      </section>
    </main>
  );
}
