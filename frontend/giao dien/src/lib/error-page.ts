export function renderErrorPage(): string {
  return `<!doctype html>
<html lang="vi">
  <head>
    <meta charset="utf-8" />
    <title>HCMUTE-SHIPCODE — Không thể tải trang</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      body { font: 15px/1.6 "Be Vietnam Pro", system-ui, sans-serif; background: #f8f5ed; color: #12203a; display: grid; place-items: center; min-height: 100vh; margin: 0; padding: 1.5rem; }
      .card { max-width: 30rem; width: 100%; text-align: center; padding: 2.5rem 2rem; border-top: 3px solid #f0b600; background: #fffdf8; box-shadow: 0 20px 50px -35px #12203a; }
      h1 { font-size: 1.25rem; margin: 0 0 0.5rem; }
      p { color: #4b5563; margin: 0 0 1.5rem; }
      .actions { display: flex; gap: 0.5rem; justify-content: center; flex-wrap: wrap; }
      a, button { padding: 0.5rem 1rem; border-radius: 0.375rem; font: inherit; cursor: pointer; text-decoration: none; border: 1px solid transparent; }
      .primary { background: #12203a; color: #fff; }
      .secondary { background: #fff; color: #12203a; border-color: #d8d2c6; }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>Không thể tải trang</h1>
      <p>Hệ thống đang gặp sự cố tạm thời. Vui lòng thử tải lại hoặc trở về trang tra cứu.</p>
      <div class="actions">
        <button class="primary" onclick="location.reload()">Thử lại</button>
        <a class="secondary" href="/">Về trang tra cứu</a>
      </div>
    </div>
  </body>
</html>`;
}
