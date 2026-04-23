# Web UI (FastAPI + React)

## One command (both API + Vite)

From the repo root:

```bash
./start-web.sh
```

The script: creates **`.venv`** in the project root if missing, installs `requirements.txt`, runs `npm install` in `frontend/`, then starts **uvicorn** (port 8000) and **Vite** (port 5173). You need **Python 3** and **Node/npm** on your PATH. Press **Ctrl+C** to stop both servers.

**PaddleOCR** (optional, for best accuracy) is *not* in the main requirements because PyPI often has no `paddlepaddle` wheel on macOS or Python 3.13+. The stack runs with **EasyOCR + Tesseract** without it. On a machine where it installs, run: `pip install -r requirements-optional-paddle.txt`.

`start-web.sh` waits until `GET /api/health` succeeds before starting Vite, so the first launch can take **tens of seconds** while the API imports and warms EasyOCR (this avoids Vite proxy `ETIMEDOUT` to port 8000).

## 1. Python API (manual)

From the repo root, with the same venv you use for Streamlit:

```bash
pip install -r requirements.txt
uvicorn server.main:app --reload --host 127.0.0.1 --port 8000
```

- `GET /api/health` — health check  
- `GET /api/fields` — field ids for the UI  
- `POST /api/extract` — form: `file`, `dpi` (returns `label` JSON + `preview_png_base64` + `file_hash`)  
- `POST /api/merge-validate` — JSON: `{ "label", "corrections" }`  
- `POST /api/export-jsonl` — appends to `finetune_data/` and returns NDJSON  
- `POST /api/pdf-apply` — form: `file`, `dpi`, `edits` (JSON array of `{x,y,w,h,text}`)  

## 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/api` to `http://127.0.0.1:8000` (see `frontend/vite.config.ts`).

The UI matches the Streamlit app: **sidebar** (⚙️ Options: DPI slider with red accent, checkboxes, legend by category) and **main** (title “IKEA Label Validation”, drag-and-drop zone, scorecard when a file is loaded, light “Label” card for the canvas, dark “Extracted elements” panel for human review).

## 3. Streamlit (unchanged)

```bash
streamlit run app.py
```

## Flow

1. Upload a label in the web UI.  
2. Select a field, edit text, **drag** a rectangle on the image (orange = current draft, green = other saved boxes).  
3. **Re-validate with corrections** — runs the same validator as the dashboard.  
4. **Export JSONL** — training rows + file on disk under `finetune_data/`.
