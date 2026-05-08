# Web UI (FastAPI + React)

The web app now hosts **two backends** behind one Vite dev server:

| Port | Service | Code |
|------|---------|------|
| 8000 | Extractor + human corrections | [server/main.py](server/main.py) |
| 8001 | Visual-validation (region pipeline) | [visual-validation/server/main.py](visual-validation/server/main.py) |
| 5173 | React UI (Vite) | [frontend/](frontend/) |

The frontend talks to both: `/api/*` is proxied to 8000 and `/validation-api/*` is proxied to 8001 (rewritten to `/api/*` server-side). See [frontend/vite.config.ts](frontend/vite.config.ts).

## One command (both APIs + Vite)

From the repo root:

```bash
./start-web.sh
```

The script: creates **`.venv`** in the project root if missing, installs `requirements.txt` AND `visual-validation/requirements.txt`, runs `npm install` in `frontend/`, then starts **both uvicorns** (ports 8000 and 8001) and **Vite** (port 5173). You need **Python 3** and **Node/npm** on your PATH. Press **Ctrl+C** to stop everything.

**PaddleOCR** (optional, for best accuracy) is *not* in the main requirements because PyPI often has no `paddlepaddle` wheel on macOS or Python 3.13+. The stack runs with **EasyOCR + Tesseract** without it. On a machine where it installs, run: `pip install -r requirements-optional-paddle.txt`.

`start-web.sh` waits for `GET /api/health` on **both** ports before starting Vite, so the first launch can take **tens of seconds** while the extractor imports and warms EasyOCR (this avoids Vite proxy `ETIMEDOUT`).

## 1. Extractor API (manual)

From the repo root, with the project venv active:

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

## 2. Visual-validation API (manual)

From the **`visual-validation/`** directory so the `src` package and default `config/regions` / `data/templates/...` paths resolve:

```bash
cd visual-validation
pip install -r requirements.txt
PYTHONPATH=. uvicorn server.main:app --reload --host 127.0.0.1 --port 8001
```

- `GET /api/health` — readiness probe (used by `start-web.sh`)  
- `GET /api/products` — list product codes + versions from `ground_truth/templates/...` and flag whether a manual region schema exists  
- `POST /api/validate` — multipart: `file` (PDF), `product_code`, optional `version`, `page` (default 0), `layout_only` (default true), `layout_tolerance_mm` (default 1.5), `allow_auto_schema` (default false). Returns verdict, per-region results, base64 overview PNGs (template + candidate), and the full HTML report string for the in-app iframe.

This wraps the same `run_region_report_pipeline` that powers `scripts/validate_label.py`, so the CLI command:

```bash
PYTHONPATH=. python scripts/validate_label.py "data/labels/label.pdf" \
  --product L10555-MIADM --layout-only --page 0 --out-dir reports/L10555_MIADM_test
```

is equivalent to a single `POST /api/validate` from the browser.

Override paths via env vars if needed:

- `VISUAL_VALIDATION_GROUND_TRUTH_DIR` — default `<repo>/visual-validation/data/templates/ground-truth-ikea-labels/ground_truth`
- `VISUAL_VALIDATION_REGIONS_DIR` — default `<repo>/visual-validation/config/regions`

## 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/api` → 8000 and `/validation-api` → 8001 (see [frontend/vite.config.ts](frontend/vite.config.ts)).

The UI has a tab bar in the main column:

- **Extractor** — drag-and-drop upload, scorecard, label canvas with bbox drawing, human review/correction, JSONL export. (Existing flow.)
- **Validation** — product code dropdown (auto-selected from the extractor's `ai_240` / AI(240) value when it matches a known template), template-version picker, layout-only / tolerance / page index controls, **side-by-side template-vs-candidate overview images**, per-region table with mm shifts, failed-region details, and an embedded HTML report.
- **QA Dashboard** — final clean one-screen summary for testers: extraction confidence + cache hash + verdict + pass/fail region counts + 3-up comparison panel (extractor preview, approved template overview, candidate validation overview).

The sidebar adapts to the active tab (DPI slider + overlay toggles for the extractor; pipeline summary for validation).

## 4. Streamlit (unchanged)

```bash
streamlit run app.py
```

## Upload cache (shared by both backends)

The extractor caches every accepted upload under [`output/uploads/`](output/uploads) keyed by `file_hash`:

| File | What it holds |
|------|---------------|
| `<hash>.bin` | Raw bytes of the uploaded label (PDF / image) |
| `<hash>.bin.meta` | Tiny JSON sidecar with the sniffed file suffix |
| `<hash>.json` | Full extraction payload (label, dpi, image dims, file name) — same shape `/api/extract` returns |
| `<hash>.preview.png` | The high-res preview image rendered for the UI |

- Path: configured via [`config.UPLOAD_CACHE_DIR`](config.py) on the extractor side and the `EXTRACTOR_UPLOAD_CACHE_DIR` environment variable on the validation side. `start-web.sh` exports it so both servers always agree.
- New endpoints on port 8000:
  - `GET /api/cached/{file_hash}` — replay the cached extraction payload (used by the React app on page load).
  - `DELETE /api/cached/{file_hash}` — wipe the cached upload (used by the "Clear cached upload" button).
- New behaviour on port 8001:
  - `POST /api/validate` accepts **either** `file` (multipart) **or** `file_hash` (form field). When `file_hash` is given, the validator reads the bytes straight from the shared cache — no re-upload.

Frontend wiring:
- The React app stores `file_hash` in `localStorage` after a successful extract. On the next load (or after a refresh) it asks `/api/cached/{hash}` to rehydrate state — so the Extractor view and the Validation tab both work without re-uploading.
- "Clear cached upload" in the Extractor sidebar removes the localStorage key and deletes the server-side cache files.

## Picking the template (manual / metadata file / auto-detect)

The Validation tab supports three ways to choose the template:

1. **Auto-detected** — the extractor reads the `ai_240` (DataMatrix product code) from the candidate label and pre-selects it in the dropdown when it matches a known template under `ground_truth/templates/`. A small banner above the dropdown reports this.
2. **Manual** — the **Product code** dropdown lists every template the catalog knows about (`/validation-api/api/products`). Pick anything you want; it overrides the auto-detected value. The **Template version** dropdown lights up after a product is selected (defaults to the latest `vN`, you can switch to any older one).
3. **Metadata sidecar JSON** — production labels usually arrive with a sidecar metadata file. In the "Metadata (optional)" panel you can either **Upload file** (Choose JSON) or **Paste JSON** straight into the textarea and click **Apply pasted JSON**. The fields are pre-filled into the validation controls; a green chip shows what was loaded (`From foo.json → L10555-MIADM · v1` or `From (pasted) → …`). Click `✕` on the chip to clear it. Anything you load is still editable in the dropdowns afterwards.

### Accepted JSON shape

```json
{
  "product_code": "L10555-MIADM",
  "template_version": "v1",
  "page": 0,
  "layout_only": true,
  "layout_tolerance_mm": 1.5,
  "allow_auto_schema": false
}
```

Only `product_code` is required. The parser is forgiving with key names — these aliases all work:

| Field | Aliases |
|-------|---------|
| `product_code` | `productCode`, `template`, `label_template`, `ikea_template`, `ikea_label_template` |
| `template_version` | `templateVersion`, `version` |
| `page` | `page_index`, `pageIndex` |
| `layout_only` | `layoutOnly` |
| `layout_tolerance_mm` | `tolerance_mm`, `toleranceMm` |
| `allow_auto_schema` | `allowAutoSchema` |

If you want, you can also call the validation HTTP endpoint directly with these as form fields — the same names the JSON uses. See [visual-validation/server/main.py](visual-validation/server/main.py).

## End-to-end flow

1. Drop a label on the Extractor tab — it extracts fields, renders the label canvas, and writes the bytes + extraction to the upload cache.
2. The system reads `ai_240` from the DataMatrix; on the Validation tab the dropdown is pre-selected for that product code.
3. Click **Run validation** — the validation service is told the `file_hash`, reads the same PDF straight from the cache, and runs the region pipeline.
4. The Validation tab shows VALID/INVALID, score vs threshold, the **template overview** and **candidate overview** side-by-side, a per-region table with mm shifts, failed-region cards, and the full HTML report (toggle / open in new tab).
5. Refresh the page — the file is restored automatically. Validation still works without re-uploading.

## QA test directory

Dedicated QA flow tests live under [`tests/qa/`](tests/qa):

- [`tests/qa/test_upload_cache_contract.py`](tests/qa/test_upload_cache_contract.py)  
  Verifies cache write/read/delete contract (`/api/extract` output parity and `/api/cached/{file_hash}` hydration shape).
- [`tests/qa/test_validation_hash_flow.py`](tests/qa/test_validation_hash_flow.py)  
  Verifies validation executes from `file_hash` without re-uploading bytes.
- [`tests/qa/run_qa_tests.sh`](tests/qa/run_qa_tests.sh)  
  Runner script for QA suite.

Run:

```bash
./tests/qa/run_qa_tests.sh
```
