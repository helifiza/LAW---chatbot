"use client";
import "./Login.css";
import { useState, type FormEvent } from "react";


export default function Home() {
  const [showPassword, setShowPassword] = useState(false);
  const [message, setMessage] = useState("");

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setMessage("Giao diện đã sẵn sàng để kết nối với API đăng nhập của SLaw.");
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

              <button className="submit-button" type="submit">
                Đăng nhập
              </button>

              <p className="signup-copy">
                Chưa có tài khoản? <a href="#signup">Đăng ký ngay</a>
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