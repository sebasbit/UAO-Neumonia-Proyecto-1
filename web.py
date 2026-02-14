from __future__ import annotations

import base64
import csv
import io
import os
import secrets
import tempfile
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw

from src import integrator

HOST = os.environ.get("NEUMONIA_HOST", "127.0.0.1")
PORT = int(os.environ.get("NEUMONIA_PORT", "8000"))

# Resultados en memoria.
RESULTS: dict[str, dict] = {}


def ensure_dirs():
    """Asegurar que la carpeta de modelos subidos existe."""
    os.makedirs("uploaded_models", exist_ok=True)


def _b64_png(pil_img: Image.Image) -> str:
    """Convertir una imagen de PIL a un texto en base64 de un PNG."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def write_history_row(patient_id: str, label: str, proba_str: str, path: str = "historial.csv") -> None:
    """Agregar una fila de resultado al CSV histórico."""
    with open(path, "a", newline="", encoding="utf-8") as csvfile:
        w = csv.writer(csvfile, delimiter="-")
        w.writerow([patient_id, label, proba_str])


def _html_page(title: str, body: str) -> bytes:
    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --bg1: #f0f7ff;
      --bg2: #e6fffb;
      --page: #f8fafc;

      --card: rgba(255,255,255,.92);
      --text: #0f172a;
      --muted: #475569;
      --border: #e2e8f0;

      --accent: #0ea5a4;   /* clinical teal */
      --accent2: #2563eb;  /* medical blue */
      --success: #16a34a;
      --warning: #f59e0b;
      --danger: #dc2626;

      --chip: #f1f5f9;
      --shadow: 0 10px 30px rgba(15, 23, 42, .08);
    }}

    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(1200px 700px at 18% 0%, var(--bg2), transparent 60%),
        radial-gradient(1200px 700px at 82% 18%, var(--bg1), transparent 55%),
        var(--page);
    }}

    .wrap {{
      max-width: 1120px;
      margin: 0 auto;
      padding: clamp(16px, 3vw, 32px);
      min-height: 100vh;
    }}

    .header {{
      display: flex;
      gap: 12px;
      align-items: center;
      margin: 4px 0 14px;
    }}

    .logo {{
      width: 44px;
      height: 44px;
      border-radius: 14px;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      box-shadow: var(--shadow);
      display: grid;
      place-items: center;
      flex: 0 0 auto;
    }}

    .logo svg {{
      width: 26px;
      height: 26px;
      fill: #fff;
      opacity: .96;
    }}

    .title {{
      font-size: clamp(20px, 2.2vw, 28px);
      font-weight: 850;
      letter-spacing: .2px;
      margin: 0;
      line-height: 1.15;
    }}

    .sub {{
      color: var(--muted);
      margin: 6px 0 0;
      font-size: 14px;
      line-height: 1.5;
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}

    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 16px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(6px);
    }}

    .row {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: end;
    }}

    label {{
      font-size: 12px;
      color: var(--muted);
      display: block;
      margin-bottom: 6px;
    }}

    input[type="text"], input[type="file"] {{
      width: 100%;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #fff;
      color: var(--text);
    }}

    input[type="text"]::placeholder {{ color: #94a3b8; }}

    input[type="text"]:focus, input[type="file"]:focus {{
      outline: none;
      border-color: rgba(14,165,164,.70);
      box-shadow: 0 0 0 4px rgba(14,165,164,.16);
    }}

    .field {{ flex: 1; min-width: 220px; }}

    .btn {{
      padding: 10px 14px;
      border-radius: 12px;
      border: 1px solid rgba(37,99,235,.20);
      background: linear-gradient(135deg, rgba(14,165,164,.16), rgba(37,99,235,.10));
      color: var(--text);
      cursor: pointer;
      font-weight: 750;
      transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
      user-select: none;
      white-space: nowrap;
    }}

    .btn:hover {{
      transform: translateY(-1px);
      border-color: rgba(37,99,235,.35);
      box-shadow: 0 10px 24px rgba(37,99,235,.08);
    }}

    .btn:active {{ transform: translateY(0); }}

    .btn.primary {{
      border: none;
      color: #fff;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      box-shadow: 0 12px 26px rgba(14,165,164,.16);
    }}

    .btn.danger {{
      border-color: rgba(220,38,38,.25);
      background: rgba(220,38,38,.08);
    }}

    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--chip);
      font-size: 12px;
      color: var(--muted);
    }}

    .alert {{
      margin: 12px 0 0;
      padding: 10px 12px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: var(--chip);
      color: var(--text);
      font-size: 13px;
      line-height: 1.45;
    }}

    .alert.ok {{
      border-color: rgba(22,163,74,.28);
      background: rgba(22,163,74,.10);
      color: #14532d;
    }}

    .alert.warn {{
      border-color: rgba(245,158,11,.30);
      background: rgba(245,158,11,.12);
      color: #92400e;
    }}

    .alert.bad {{
      border-color: rgba(220,38,38,.28);
      background: rgba(220,38,38,.10);
      color: #7f1d1d;
    }}

    .imgbox {{
      border-radius: 14px;
      border: 1px dashed var(--border);
      padding: 10px;
      background: linear-gradient(180deg, #fff, #f8fafc);
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: clamp(240px, 42vh, 420px);
      overflow: hidden;
    }}

    img {{
      max-width: 100%;
      height: auto;
      border-radius: 10px;
    }}

    .kpi {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 10px;
    }}

    .kpi .item {{
      flex: 1;
      min-width: 160px;
      padding: 12px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,.75);
    }}

    .kpi .v {{
      font-size: 18px;
      font-weight: 900;
      margin-top: 6px;
      letter-spacing: .2px;
    }}

    .tag {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 800;
      border: 1px solid var(--border);
      background: #fff;
    }}

    .tag.ok {{
      border-color: rgba(22,163,74,.26);
      background: rgba(22,163,74,.08);
      color: var(--success);
    }}

    .tag.warn {{
      border-color: rgba(245,158,11,.28);
      background: rgba(245,158,11,.10);
      color: #92400e;
    }}

    .tag.bad {{
      border-color: rgba(220,38,38,.26);
      background: rgba(220,38,38,.10);
      color: var(--danger);
    }}

    .note {{
      margin-top: 10px;
      font-size: 12px;
      color: var(--muted);
    }}

    a {{ color: var(--accent2); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}

    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      background: rgba(15,23,42,.06);
      border: 1px solid rgba(15,23,42,.08);
      padding: 1px 6px;
      border-radius: 999px;
    }}

    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .imgbox {{ min-height: clamp(220px, 38vh, 380px); }}
    }}

    @media (max-width: 640px) {{
      .row {{ flex-direction: column; align-items: stretch; }}
      .field {{ min-width: 0; }}
      .btn {{ width: 100%; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    {body}
</body>
</html>"""
    return html.encode("utf-8")


def _render_index(message: str | None = None) -> bytes:
    model_path = integrator.get_current_model_path()
    model_badge = (
        f"<span class='pill'>Modelo cargado: <b>{os.path.basename(model_path)}</b></span>"
        if model_path
        else "<span class='pill'>Modelo: <b>NO cargado</b> (sube un .h5)</span>"
    )

    msg_html = ""
    if message:
        m = message.strip().lower()
        cls = "ok" if m.startswith("modelo cargado") else ("bad" if m.startswith("error") else "warn")
        msg_html = f"<div class='alert {cls}'>{message}</div>"

    body = f"""
    <div class="header">
      <div class="logo" aria-hidden="true">
        <svg viewBox="0 0 24 24"><path d="M19 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2Zm-6 16h-2v-5H6v-2h5V7h2v5h5v2h-5v5Z"/></svg>
      </div>
      <div>
        <h1 class="title">Detección rápida de neumonía</h1>
        <p class="sub">Sube el modelo <b>.h5</b> y luego una imagen <b>.dcm / .jpg / .png</b> para predecir. {model_badge}</p>
      </div>
    </div>
    {msg_html}

    <div class="card" style="margin-bottom:14px;">
      <div class="row">
        <form class="row" style="flex:1" action="/upload_model" method="post" enctype="multipart/form-data">
          <div class="field">
            <label>Modelo (.h5)</label>
            <input type="file" name="model_file" accept=".h5" required />
          </div>
          <div>
            <button class="btn primary" type="submit">Cargar modelo</button>
          </div>
        </form>
      </div>
      <div class="note">Si cambias de modelo, vuelve a predecir para recalcular el Grad-CAM.</div>
    </div>

    <div class="card">
      <form action="/predict" method="post" enctype="multipart/form-data">
        <div class="row">
          <div class="field">
            <label>Cédula paciente</label>
            <input type="text" name="patient_id" placeholder="Ej: 1234567890" />
          </div>
          <div class="field">
            <label>Imagen (DICOM/JPG/PNG)</label>
            <input type="file" name="image_file" accept=".dcm,.jpg,.jpeg,.png" required />
          </div>
          <div>
            <button class="btn primary" type="submit">Predecir</button>
          </div>
        </div>
      </form>
      <div class="note">El archivo de historial se guarda como <code>historial.csv</code> en el directorio del proyecto.</div>
    </div>
    """
    return _html_page("UAO Neumonía — Web", body)


def _render_result(token: str, message: str | None = None) -> bytes:
    r = RESULTS[token]
    model_path = integrator.get_current_model_path()
    model_badge = (
        f"<span class='pill'>Modelo: <b>{os.path.basename(model_path)}</b></span>"
        if model_path
        else "<span class='pill'>Modelo: <b>NO cargado</b></span>"
    )

    msg_html = ""
    if message:
        m = message.strip().lower()
        cls = "ok" if "guardado" in m else ("bad" if m.startswith("error") else "warn")
        msg_html = f"<div class='alert {cls}'>{message}</div>"

    img1_b64 = r["img1_b64"]
    img2_b64 = r.get("img2_b64")

    heatmap_box = (
        f"<div class='imgbox'><img alt='Heatmap' src='data:image/png;base64,{img2_b64}' /></div>"
        if img2_b64
        else "<div class='imgbox'><div class='sub'>No se pudo generar Grad-CAM.</div></div>"
    )

    label_lower = str(r.get("label", "")).strip().lower()
    if "normal" in label_lower:
        tag_cls = "ok"
    elif "viral" in label_lower:
        tag_cls = "warn"
    else:  # bacteriana o desconocido
        tag_cls = "bad"

    body = f"""
    <div class="header">
      <div class="logo" aria-hidden="true">
        <svg viewBox="0 0 24 24"><path d="M19 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2Zm-6 16h-2v-5H6v-2h5V7h2v5h5v2h-5v5Z"/></svg>
      </div>
      <div>
        <h1 class="title">Resultados</h1>
        <p class="sub">{model_badge} — <a href="/">↩ Volver</a></p>
      </div>
    </div>
    {msg_html}

    <div class="grid">
      <div class="card">
        <div style="font-weight:800; margin-bottom:10px;">Imagen radiográfica</div>
        <div class="imgbox"><img alt="Radiografía" src="data:image/png;base64,{img1_b64}" /></div>
      </div>
      <div class="card">
        <div style="font-weight:800; margin-bottom:10px;">Heatmap (Grad-CAM)</div>
        {heatmap_box}
      </div>
    </div>

    <div class="card" style="margin-top:14px;">
      <div class="kpi">
        <div class="item">
          <div class="pill">Cédula</div>
          <div class="v">{r["patient_id"] or "Desconocido"}</div>
        </div>
        <div class="item">
          <div class="pill">Resultado</div>
          <div class="v"><span class="tag {tag_cls}">{r["label"]}</span></div>
        </div>
        <div class="item">
          <div class="pill">Probabilidad</div>
          <div class="v">{r["proba"]:.2f}%</div>
        </div>
      </div>

      <div class="row" style="margin-top:14px;">
        <form action="/save_csv" method="post">
          <input type="hidden" name="token" value="{token}" />
          <button class="btn" type="submit">Guardar (CSV)</button>
        </form>

        <form action="/download_pdf" method="post">
          <input type="hidden" name="token" value="{token}" />
          <button class="btn" type="submit">Descargar PDF</button>
        </form>

        <a class="btn danger" href="/" style="display:inline-block;">Borrar / Nueva imagen</a>
      </div>
      <div class="note">El CSV usa el mismo formato del UI original (delimitador <code>-</code>).</div>
    </div>
    """
    return _html_page("UAO Neumonía — Resultados", body)


def _make_pdf_bytes(result: dict) -> bytes:
    W, H = 1240, 1754
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    # Titles
    draw.text((50, 40), "SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA", fill="black")
    draw.text((50, 80), f"Paciente: {result['patient_id'] or 'Desconocido'}", fill="black")
    draw.text((50, 110), f"Resultado: {result['label']}   Probabilidad: {result['proba']:.2f}%", fill="black")
    draw.text((50, 140), f"Fecha/Hora: {time.strftime('%Y-%m-%d %H:%M:%S')}", fill="black")

    # Images
    img1 = Image.open(io.BytesIO(base64.b64decode(result["img1_b64"]))).convert("RGB")
    img1 = img1.resize((540, 540))
    canvas.paste(img1, (50, 210))

    if result.get("img2_b64"):
        img2 = Image.open(io.BytesIO(base64.b64decode(result["img2_b64"]))).convert("RGB")
        img2 = img2.resize((540, 540))
        canvas.paste(img2, (650, 210))
        draw.text((650, 760), "Grad-CAM (Heatmap)", fill="black")
    else:
        draw.text((650, 210), "Grad-CAM no disponible.", fill="black")

    draw.text((50, 760), "Radiografía", fill="black")

    out = io.BytesIO()
    canvas.save(out, format="PDF")
    return out.getvalue()


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, content: bytes, content_type: str = "text/html; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if self.path == "/":
            self._send(HTTPStatus.OK, _render_index())
        else:
            self._send(HTTPStatus.NOT_FOUND,
                       _html_page("404", "<h1 class='title'>404</h1><p class='sub'>No encontrado.</p>"))

    def do_POST(self):
        parsed = urlparse(self.path)

        # Use cgi.FieldStorage for multipart/form-data
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" in ctype:
            import cgi
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
        else:
            length = int(self.headers.get("Content-Length", "0") or "0")
            data = self.rfile.read(length) if length else b""
            form = parse_qs(data.decode("utf-8", errors="ignore"))

        if parsed.path == "/upload_model":
            try:
                if not hasattr(form, "getfirst"):
                    self._send(HTTPStatus.BAD_REQUEST,
                               _render_index("Debe subir un archivo .h5 (multipart/form-data)."))
                    return

                field = form["model_file"] if "model_file" in form else None
                if isinstance(field, list):
                    field = field[0] if field else None
                if field is None or not getattr(field, "filename", ""):
                    self._send(HTTPStatus.BAD_REQUEST, _render_index("No se recibió el archivo del modelo."))
                    return

                filename = os.path.basename(field.filename)
                if not filename.lower().endswith(".h5"):
                    self._send(HTTPStatus.BAD_REQUEST, _render_index("El modelo debe tener extensión .h5"))
                    return

                ensure_dirs()
                model_path = os.path.join("uploaded_models", f"model_{int(time.time())}_{filename}")
                with open(model_path, "wb") as f:
                    f.write(field.file.read())

                integrator.set_current_model_path(model_path)

                self._send(HTTPStatus.OK, _render_index(f"Modelo cargado: {filename}"))
            except Exception as e:
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, _render_index(f"Error cargando modelo: {e}"))
            return

        if parsed.path == "/predict":
            if not hasattr(form, "getfirst"):
                self._send(HTTPStatus.BAD_REQUEST, _render_index("Petición inválida."))
                return

            if not integrator.get_current_model_path():
                self._send(HTTPStatus.BAD_REQUEST, _render_index("Primero debe cargar un modelo .h5"))
                return

            patient_id = (form.getfirst("patient_id") or "").strip()

            field = form["image_file"] if "image_file" in form else None
            if isinstance(field, list):
                field = field[0] if field else None
            if field is None or not getattr(field, "filename", ""):
                self._send(HTTPStatus.BAD_REQUEST, _render_index("No se recibió la imagen."))
                return

            filename = os.path.basename(field.filename)
            data = field.file.read()

            temp_file_path = None
            try:
                # Save uploaded data to a temporary file to use with integrator
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as temp_file:
                    temp_file.write(data)
                    temp_file_path = temp_file.name

                # Process image using the integrator pipeline
                img_array, img_pil_to_show = integrator.load_and_prepare_image(temp_file_path)
                label, proba, heatmap_array = integrator.predict_pneumonia(img_array)

                # Convert heatmap numpy array to PIL Image
                heatmap_pil = Image.fromarray(heatmap_array) if heatmap_array is not None else None

                token = secrets.token_urlsafe(16)
                RESULTS[token] = {
                    "patient_id": patient_id,
                    "label": label,
                    "proba": float(proba) * 100,  # integrator returns 0-1, UI expects 0-100
                    "img1_b64": _b64_png(img_pil_to_show),
                    "img2_b64": _b64_png(heatmap_pil) if heatmap_pil is not None else None,
                }
                self._send(HTTPStatus.OK, _render_result(token))
            except Exception as e:
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, _render_index(f"Error procesando '{filename}': {e}"))
            finally:
                # Clean up the temporary file
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
            return

        if parsed.path == "/save_csv":
            try:
                if hasattr(form, "getfirst"):
                    token = (form.getfirst("token") or "").strip()
                else:
                    token = (form.get("token", [""])[0]).strip()

                if not token or token not in RESULTS:
                    self._send(HTTPStatus.BAD_REQUEST, _render_index("Token inválido para guardar CSV."))
                    return

                r = RESULTS[token]
                write_history_row(r["patient_id"], r["label"], f"{r['proba']:.2f}%")
                self._send(HTTPStatus.OK, _render_result(token, "Guardado en historial.csv"))
            except Exception as e:
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, _render_index(f"Error guardando CSV: {e}"))
            return

        if parsed.path == "/download_pdf":
            try:
                if hasattr(form, "getfirst"):
                    token = (form.getfirst("token") or "").strip()
                else:
                    token = (form.get("token", [""])[0]).strip()

                if not token or token not in RESULTS:
                    self._send(HTTPStatus.BAD_REQUEST, _render_index("Token inválido para PDF."))
                    return

                pdf_bytes = _make_pdf_bytes(RESULTS[token])
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/pdf")
                self.send_header("Content-Disposition",
                                 f'attachment; filename="Reporte_{RESULTS[token]["patient_id"] or "paciente"}.pdf"')
                self.send_header("Content-Length", str(len(pdf_bytes)))
                self.end_headers()
                self.wfile.write(pdf_bytes)
            except Exception as e:
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, _render_index(f"Error generando PDF: {e}"))
            return

        self._send(HTTPStatus.NOT_FOUND,
                   _html_page("404", "<h1 class='title'>404</h1><p class='sub'>No encontrado.</p>"))


def main() -> int:
    ensure_dirs()
    httpd = HTTPServer((HOST, PORT), Handler)
    print(f"\n[UAO-Neumonia] Servidor web corriendo en http://{HOST}:{PORT}")
    print("Cierre la consola (Ctrl+C) para detenerlo.")
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
