"use client";

import "./SignUp.css";
import Link from "next/link";
import { useMemo, useState, type FormEvent } from "react";

function AuthBrand() {
  return (
    <aside className="brand-panel">
      <div className="brand-copy">
        <p className="eyebrow">Nền tảng pháp luật số</p>

        <h1>
          <span className="title-line">Hỏi đáp pháp luật</span>
          <span>Việt Nam</span>
        </h1>

        <p className="brand-description">
          Tra cứu thông tin, đặt câu hỏi và nhận hỗ trợ pháp lý một cách dễ dàng.
        </p>
      </div>

      <div className="brand-footer">
        <div className="wordmark" aria-label="SLaw">
          <span>S</span>Law
        </div>

        <p>support@slaw.vn</p>
      </div>
    </aside>
  );
}

export default function SignupPage() {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [acceptedTerms, setAcceptedTerms] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [message, setMessage] = useState("");

  const passwordsMatch =
    confirmPassword.length === 0 || password === confirmPassword;

  const isValid = useMemo(
    () =>
      fullName.trim().length >= 2 &&
      email.trim().includes("@") &&
      password.length >= 8 &&
      password === confirmPassword &&
      acceptedTerms,
    [fullName, email, password, confirmPassword, acceptedTerms],
  );

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!isValid) return;
    setMessage("Form đăng ký đã sẵn sàng để kết nối với API của SLaw.");
  }

  return (
    <main className="auth-page">
      <section className="auth-shell signup-shell" aria-labelledby="signup-title">
        <AuthBrand />

        <div className="form-panel signup-panel">
          <div className="form-wrap">
            <header className="form-heading signup-heading">
              <p className="eyebrow">Bắt đầu với SLaw</p>
              <h2 id="signup-title">Tạo tài khoản</h2>
              <p>Đăng ký để đặt câu hỏi và nhận hỗ trợ pháp lý thuận tiện hơn.</p>
            </header>

            <form className="auth-form signup-form" onSubmit={handleSubmit}>
              <div className="field">
                <label htmlFor="full-name">Họ và tên</label>
                <input
                  id="full-name"
                  type="text"
                  value={fullName}
                  onChange={(event) => setFullName(event.target.value)}
                  autoComplete="name"
                  placeholder="Nguyễn Văn An"
                  required
                />
              </div>

              <div className="field">
                <label htmlFor="signup-email">Email</label>
                <input
                  id="signup-email"
                  type="email"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  autoComplete="email"
                  placeholder="tenban@example.com"
                  required
                />
              </div>

              <div className="field">
                <label htmlFor="signup-password">Mật khẩu</label>
                <div className="password-field">
                  <input
                    id="signup-password"
                    type={showPassword ? "text" : "password"}
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    autoComplete="new-password"
                    placeholder="Tối thiểu 8 ký tự"
                    minLength={8}
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

              <div className="field">
                <label htmlFor="confirm-password">Xác nhận mật khẩu</label>
                <input
                  id="confirm-password"
                  className={passwordsMatch ? "" : "input-error"}
                  type={showPassword ? "text" : "password"}
                  value={confirmPassword}
                  onChange={(event) => setConfirmPassword(event.target.value)}
                  autoComplete="new-password"
                  placeholder="Nhập lại mật khẩu"
                  aria-describedby="password-error"
                  required
                />
                <p id="password-error" className="field-error" aria-live="polite">
                  {passwordsMatch ? "" : "Mật khẩu xác nhận chưa khớp."}
                </p>
              </div>

              <label className="check-option terms-option">
                <input
                  type="checkbox"
                  checked={acceptedTerms}
                  onChange={(event) => setAcceptedTerms(event.target.checked)}
                  required
                />
                <span>
                  Tôi đồng ý với <Link href="/terms">Điều khoản sử dụng</Link> và{" "}
                  <Link href="/privacy">Chính sách bảo mật</Link>.
                </span>
              </label>

              <button className="submit-button" type="submit" disabled={!isValid}>
                Đăng ký
              </button>

              <p className="switch-copy">
                Đã có tài khoản? <Link href="/">Đăng nhập</Link>
              </p>
              <p className="form-message" role="status" aria-live="polite">
                {message}
              </p>
            </form>
          </div>
        </div>
      </section>
    </main>
  );
}