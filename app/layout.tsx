import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SLaw – Hỏi đáp tài liệu",
  description:
    "Chatbot RAG hỏi đáp tiếng Việt dựa trên tài liệu của từng phiên.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="vi" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
