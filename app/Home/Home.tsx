"use client";

import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import "./Chat.css";
import styles from "./Home.module.css";

const MAX_FILES = 5;
const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ALLOWED_EXTENSIONS = ["pdf", "docx", "txt", "md", "csv", "json"];

type MessageRole = "user" | "assistant";

type Message = {
  id: string;
  role: MessageRole;
  content: string;
  attachments?: string[];
  sources?: Array<{ fileName: string; locator: string; excerpt: string }>;
  demo?: boolean;
  error?: boolean;
};

type DocumentItem = {
  id: string;
  file: File;
};

type HistoryItem = {
  role: MessageRole;
  content: string;
};

type RAGAnswer = {
  answer?: string;
  sources?: Array<{ fileName: string; locator: string; excerpt: string }>;
  demo?: boolean;
  error?: string;
};

const INITIAL_MESSAGES: Message[] = [
  {
    id: "welcome",
    role: "assistant",
    content:
      "Xin chào! Hãy tải tài liệu lên, sau đó đặt câu hỏi. Tôi sẽ trả lời dựa trên nội dung tài liệu và hiển thị rõ nguồn tham khảo.",
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
  search: (
    <>
      <circle cx="11" cy="11" r="7" />
      <path d="m20 20-4-4" />
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
  clip: <path d="m20 11-8.5 8.5a5 5 0 0 1-7-7L14 3a3.5 3.5 0 0 1 5 5l-9.5 9.5a2 2 0 0 1-3-3L15 6" />,
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

function Icon({ name, size = 20, strokeWidth = 1.8 }: { name: string; size?: number; strokeWidth?: number }) {
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
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function fileId(file: File) {
  return `${file.name}-${file.size}-${file.lastModified}`;
}

function extensionOf(fileName: string) {
  return fileName.split(".").pop()?.toLowerCase() || "";
}

function formatFileSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Điểm nối backend RAG của bạn.
 *
 * Backend nên nhận multipart/form-data gồm:
 * - question: string
 * - history: JSON string
 * - files: File[]
 *
 * Và trả JSON:
 * {
 *   answer: string,
 *   sources: [{ fileName, locator, excerpt }]
 * }
 */
async function requestRagAnswer({ question, files, history }: { question: string; files: DocumentItem[]; history: HistoryItem[] }) {
  const endpoint = process.env.NEXT_PUBLIC_RAG_API_URL;

  if (!endpoint) {
    await new Promise((resolve) => window.setTimeout(resolve, 650));
    return {
      answer:
        `Đây là phản hồi mẫu cho câu hỏi: “${question}”.\n\n` +
        "Khi bạn nối backend RAG, nội dung trả lời thực tế sẽ xuất hiện tại chính khung này, kèm các nguồn được backend gửi về.",
      sources: files.slice(0, 2).map((item, index) => ({
        fileName: item.file.name,
        locator: `Nguồn mẫu ${index + 1}`,
        excerpt: "Đoạn nội dung trích dẫn từ tài liệu sẽ được hiển thị tại đây.",
      })),
      demo: true,
    };
  }

  const body = new FormData();
  body.append("question", question);
  body.append(
    "history",
    JSON.stringify(history.map(({ role, content }) => ({ role, content }))),
  );
  files.forEach((item) => body.append("files", item.file));

  const response = await fetch(endpoint, { method: "POST", body });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Backend RAG không phản hồi.");
  }
  return payload;
}

export default function Home() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [pendingIds, setPendingIds] = useState<string[]>([]);
  const [messages, setMessages] = useState<Message[]>(INITIAL_MESSAGES);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const pendingNames = useMemo(
    () =>
      documents
        .filter((document) => pendingIds.includes(document.id))
        .map((document) => document.file.name),
    [documents, pendingIds],
  );

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isLoading]);

  useEffect(() => {
    if (!textareaRef.current) return;
    textareaRef.current.style.height = "0px";
    textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 140)}px`;
  }, [draft]);

  function addFiles(fileList: FileList | null) {
    setError("");
    const incoming = Array.from(fileList || []);
    const knownIds = new Set(documents.map((document) => document.id));
    const accepted: DocumentItem[] = [];
    const rejected: string[] = [];

    incoming.forEach((file) => {
      const id = fileId(file);
      if (knownIds.has(id)) return;
      if (!ALLOWED_EXTENSIONS.includes(extensionOf(file.name))) {
        rejected.push(`${file.name}: định dạng chưa hỗ trợ`);
        return;
      }
      if (file.size > MAX_FILE_SIZE) {
        rejected.push(`${file.name}: lớn hơn 10 MB`);
        return;
      }
      if (documents.length + accepted.length >= MAX_FILES) {
        rejected.push(`Chỉ được tải tối đa ${MAX_FILES} file`);
        return;
      }
      accepted.push({ id, file });
      knownIds.add(id);
    });

    if (accepted.length) {
      setDocuments((current) => [...current, ...accepted]);
      setPendingIds((current) => [...current, ...accepted.map((item) => item.id)]);
    }
    if (rejected.length) setError(rejected.join(" · "));
  }

  function removeFile(id: string) {
    setDocuments((current) => current.filter((document) => document.id !== id));
    setPendingIds((current) => current.filter((pendingId) => pendingId !== id));
  }

  function newChat() {
    setMessages(INITIAL_MESSAGES);
    setDraft("");
    setError("");
    setSidebarOpen(false);
  }

  async function sendMessage() {
    const question = draft.trim();
    if (!question || isLoading) return;
    if (!documents.length) {
      setError("Bạn cần upload ít nhất một tài liệu trước khi đặt câu hỏi.");
      inputRef.current?.click();
      return;
    }

    const userMessage: Message = {
      id: makeId(),
      role: "user",
      content: question,
      attachments: pendingNames,
    };
    const currentHistory = [...messages, userMessage];
    setMessages(currentHistory);
    setDraft("");
    setError("");
    setPendingIds([]);
    setIsLoading(true);

    try {
      const result = await requestRagAnswer({
        question,
        files: documents,
        history: currentHistory.slice(-8),
      });
      setMessages((current) => [
        ...current,
        {
          id: makeId(),
          role: "assistant",
          content: result.answer || "Backend chưa trả về nội dung answer.",
          sources: Array.isArray(result.sources) ? result.sources : [],
          demo: Boolean(result.demo),
        },
      ]);
    } catch (requestError: unknown) {
      const messageText = requestError instanceof Error ? requestError.message : "Không thể nhận phản hồi";
      setMessages((current) => [
        ...current,
        {
          id: makeId(),
          role: "assistant",
          content: `Không thể nhận phản hồi: ${messageText}`,
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
      sendMessage();
    }
  }

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
          const relatedTarget = event.relatedTarget;
          if (!relatedTarget || !(relatedTarget instanceof Node) || !event.currentTarget.contains(relatedTarget)) {
            setIsDragging(false);
          }
        }}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          addFiles(event.dataTransfer.files);
        }}
      >
        <aside className={`${styles.sidebar} ${sidebarOpen ? styles.sidebarOpen : ""}`}>
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
              <button className={styles.newChat} type="button" onClick={newChat}>
                <Icon name="plus" size={17} strokeWidth={2.2} />
                <span>New chat</span>
              </button>
              <button className={styles.searchButton} type="button" aria-label="Tìm kiếm">
                <Icon name="search" size={18} />
              </button>
            </div>
          </div>

          <div className={styles.fileSection}>
            <div className={styles.sectionTitle}>
              <span>File</span>
              {documents.length > 0 && (
                <button
                  type="button"
                  onClick={() => {
                    setDocuments([]);
                    setPendingIds([]);
                  }}
                >
                  Clear All
                </button>
              )}
            </div>

            <button
              className={styles.uploadButton}
              type="button"
              onClick={() => inputRef.current?.click()}
            >
              <span><Icon name="upload" size={19} /></span>
              <strong>Thêm tài liệu</strong>
              <small>PDF, DOCX, TXT…</small>
            </button>

            <div className={styles.fileList}>
              {documents.map((document) => (
                <div className={styles.fileItem} key={document.id}>
                  <span className={styles.fileType}><Icon name="file" size={17} /></span>
                  <span className={styles.fileInfo}>
                    <strong title={document.file.name}>{document.file.name}</strong>
                    <small>{formatFileSize(document.file.size)}</small>
                  </span>
                  <button
                    type="button"
                    aria-label={`Xóa ${document.file.name}`}
                    onClick={() => removeFile(document.id)}
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
            <span className={styles.profileIcon}><Icon name="user" size={19} /></span>
            <span>
              <strong>Username</strong>
              <small>Gói cá nhân</small>
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
                  ? `${documents.length} tài liệu đang được sử dụng`
                  : "Tải tài liệu để bắt đầu"}
              </p>
            </div>
            <span className={styles.statusBadge}>
              <Icon name="check" size={15} />
              Grounded
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
                            <span key={name}><Icon name="file" size={14} />{name}</span>
                          ))}
                        </div>
                      )}

                      <p className={message.error ? styles.errorAnswer : ""}>{message.content}</p>

                      {!isUser && sources.length > 0 && (
                        <div className={styles.sources}>
                          <div className={styles.sourcesHeading}>
                            <Icon name="sparkle" size={15} />
                            Nguồn tham khảo
                          </div>
                          <div className={styles.sourceList}>
                            {sources.map((source, index) => (
                              <details className={styles.sourceCard} key={`${source.fileName}-${index}`}>
                                <summary>
                                  <span>{index + 1}</span>
                                  <strong>{source.fileName}</strong>
                                  <small>{source.locator}</small>
                                </summary>
                                <p>{source.excerpt}</p>
                              </details>
                            ))}
                          </div>
                          {message.demo && (
                            <div className={styles.demoNote}>
                              Đây là dữ liệu mẫu. Điền NEXT_PUBLIC_RAG_API_URL để nối backend.
                            </div>
                          )}
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
                <article className={`${styles.messageRow} ${styles.assistantRow}`}>
                  <span className={`${styles.avatar} ${styles.botAvatar}`}>
                    <Icon name="bot" size={18} />
                  </span>
                  <div className={`${styles.bubble} ${styles.assistantBubble}`}>
                    <div className={styles.typing} aria-label="Đang trả lời">
                      <span /><span /><span />
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
                    <span key={name}><Icon name="file" size={14} />{name}</span>
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
                >
                  <Icon name="clip" size={20} />
                </button>
                <textarea
                  ref={textareaRef}
                  value={draft}
                  rows={1}
                  placeholder={documents.length ? "Hỏi bất cứ điều gì về tài liệu…" : "Upload file trước khi chat"}
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={onComposerKeyDown}
                />
                <button
                  type="button"
                  className={styles.sendButton}
                  aria-label="Gửi câu hỏi"
                  disabled={!draft.trim() || isLoading}
                  onClick={sendMessage}
                >
                  <Icon name="send" size={19} />
                </button>
              </div>
              <p className={styles.composerHint}>Enter để gửi · Shift + Enter để xuống dòng</p>
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
            addFiles(event.target.files);
            event.target.value = "";
          }}
        />
      </section>
    </main>
  );
}
