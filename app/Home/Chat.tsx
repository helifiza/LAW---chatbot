import type { ReactNode } from "react";
import "./Chat.css";

export const metadata = {
  title: "SLaw – Hỏi đáp tài liệu",
  description: "Giao diện chatbot hỏi đáp dựa trên tài liệu.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="vi">
      <body>{children}</body>
    </html>
  );
}
