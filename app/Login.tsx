"use client";

import "./Login.css";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";

async function requestLogin({ email, password }: { email: string; password: string }) {
  const endpoint = process.env.NEXT_PUBLIC_AUTH_API_URL;

  if (!endpoint) {
    // Chế độ demo: chưa nối backend thì chỉ cần có email + password là coi như thành công
    await new Promise((resolve) => window.setTimeout(resolve, 650));
    if (!email || !password) {
      throw new Error("Vui lòng nhập đầy đủ email và mật khẩu.");
    }
    return { success: true, demo: true };
  }

  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Đăng nhập thất bại.");
  }
  return payload;
}

export default function LoginPage() {
  const router = useRouter();
  const [showPassword, setShowPassword] = useState(false);
  const [message, setMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (isSubmitting) return;

    const form = event.currentTarget;
    const email = (form.elements.namedItem("email") as HTMLInputElement).value;
    const password = (form.elements.namedItem("password") as HTMLInputElement).value;

    setIsSubmitting(true);
    setMessage("");

    try {
      await requestLogin({ email, password });
      setMessage("Đăng nhập thành công. Đang chuyển đến trang chính…");
      router.push("/Home");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Đăng nhập thất bại.");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="login-page">
      <section className="login-shell" aria-labelledby="login-title">
        <aside className="brand-panel">
          <div className="brand-copy">
            <p className="eyebrow">Nền tảng pháp luật số</p>
            <h1>
              <span className="title-line">Hỏi đáp pháp luật</span>
              <span>Việt Nam</span>
            </h1>
            <p className="brand-description">
              Tra cứu thông tin, đặt câu hỏi và nhận hỗ trợ pháp lý một cách
              dễ dàng.
            </p>
          </div>

          <div className="brand-footer">
            <div className="wordmark" aria-label="SLaw">
              <span>S</span>Law
            </div>
            <p>support@slaw.vn</p>
          </div>
        </aside>

        <div className="form-panel">
          <div className="form-wrap">
            <header className="form-heading">
              <p className="eyebrow">Tài khoản SLaw</p>
              <h2 id="login-title">Đăng nhập</h2>
              <p>Chào mừng bạn quay lại. Vui lòng nhập thông tin bên dưới.</p>
            </header>

            <form className="login-form" onSubmit={handleSubmit}>
              <div className="field">
                <label htmlFor="email">Email</label>
                <input
                  id="email"
                  name="email"
                  type="email"
                  autoComplete="email"
                  placeholder="tenban@example.com"
                  required
                />
              </div>

              <div className="field">
                <label htmlFor="password">Mật khẩu</label>
                <div className="password-field">
                  <input
                    id="password"
                    name="password"
                    type={showPassword ? "text" : "password"}
                    autoComplete="current-password"
                    placeholder="Nhập mật khẩu"
                    minLength={6}
                    required
                  />
                  <button
                    type="button"
                    className="password-toggle"
                    onClick={() => setShowPassword((value) => !value)}
                    aria-label={showPassword ? "Ẩn mật khẩu" : "Hiện mật khẩu"}
                  >
                    {showPassword ? "Ẩn" : "Hiện"}
                  </button>
                </div>
              </div>

              <div className="form-options">
                <label className="remember">
                  <input type="checkbox" name="remember" />
                  <span>Ghi nhớ đăng nhập</span>
                </label>
                <a href="#forgot-password">Quên mật khẩu?</a>
              </div>

              <button className="submit-button" type="submit" disabled={isSubmitting}>
                {isSubmitting ? "Đang đăng nhập…" : "Đăng nhập"}
              </button>

              <p className="signup-copy">
                Chưa có tài khoản? <Link href="/SignUp">Đăng ký ngay</Link>
              </p>
              <p className="form-message" role="status" aria-live="polite">
                {message}
              </p>
            </form>
          </div>

          <p className="legal-copy">
            Khi tiếp tục, bạn đồng ý với Điều khoản sử dụng và Chính sách bảo
            mật của SLaw.
          </p>
        </div>
      </section>
    </main>
  );
}