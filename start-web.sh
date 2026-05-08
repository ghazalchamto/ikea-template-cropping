#!/usr/bin/env bash
# Create .venv if needed, install Python + frontend deps, then start API + Vite.
# Usage: ./start-web.sh   (from repo root)
# Press Ctrl+C to stop both servers.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is required but not found in PATH." >&2
  exit 1
fi

VENV="$ROOT/.venv"
if [[ ! -d "$VENV" ]]; then
  echo "Creating virtual environment at .venv …" >&2
  python3 -m venv "$VENV"
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"

echo "Installing Python dependencies (requirements.txt) …" >&2
python -m pip install --upgrade pip -q
pip install -r "$ROOT/requirements.txt" -q

if [[ -f "$ROOT/visual-validation/requirements.txt" ]]; then
  echo "Installing visual-validation dependencies …" >&2
  pip install -r "$ROOT/visual-validation/requirements.txt" -q
fi

if ! command -v uvicorn >/dev/null 2>&1; then
  echo "Error: uvicorn not available after pip install." >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "Error: npm is required for the frontend. Install Node.js: https://nodejs.org/" >&2
  exit 1
fi

echo "Installing frontend dependencies (npm) …" >&2
( cd "$ROOT/frontend" && npm install )

PIDS=()
cleanup() {
  echo ""
  echo "Stopping…" >&2
  for p in "${PIDS[@]:-}"; do
    kill "$p" 2>/dev/null || true
  done
  pkill -P "$$" 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

API="http://127.0.0.1:8000"
VAL_API="http://127.0.0.1:8001"

# Both servers share an upload cache (config.UPLOAD_CACHE_DIR for the
# extractor; EXTRACTOR_UPLOAD_CACHE_DIR for the validation microservice).
# Setting it explicitly here keeps them aligned regardless of cwd.
export EXTRACTOR_UPLOAD_CACHE_DIR="${EXTRACTOR_UPLOAD_CACHE_DIR:-$ROOT/output/uploads}"
mkdir -p "$EXTRACTOR_UPLOAD_CACHE_DIR"

uvicorn server.main:app --reload --host 127.0.0.1 --port 8000 &
UVICORN_PID=$!
PIDS+=("$UVICORN_PID")

# Visual-validation microservice (separate process, separate working dir so its
# `src` and `config/regions` paths resolve cleanly). Inherits the cache dir
# via the exported environment variable above.
( cd "$ROOT/visual-validation" \
  && PYTHONPATH=. uvicorn server.main:app --reload --host 127.0.0.1 --port 8001 ) &
VAL_PID=$!
PIDS+=("$VAL_PID")

# First request can be slow: importing the app warms EasyOCR/Paddle, etc. Vite must
# not proxy until the API is actually accepting connections.
echo "Starting APIs (first load can take 30–90s while OCR warms up)…" >&2

wait_for_health() {
  local url="$1"
  local pid="$2"
  local label="$3"
  if ! command -v curl >/dev/null 2>&1; then
    echo "Warning: curl not found; sleeping 5s for $label." >&2
    sleep 5
    return 0
  fi
  for _ in $(seq 1 240); do
    if curl -sf --connect-timeout 2 --max-time 5 "$url/api/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Error: $label exited before it became ready." >&2
      return 1
    fi
    sleep 0.5
  done
  echo "Error: $label did not respond at $url/api/health in time." >&2
  return 1
}

if ! wait_for_health "$API" "$UVICORN_PID" "extractor API (port 8000)"; then
  exit 1
fi
if ! wait_for_health "$VAL_API" "$VAL_PID" "validation API (port 8001)"; then
  exit 1
fi

( cd "$ROOT/frontend" && npm run dev ) &
PIDS+=($!)

echo "" >&2
echo "  Extractor API:  http://127.0.0.1:8000  (OpenAPI: /docs)" >&2
echo "  Validation API: http://127.0.0.1:8001  (OpenAPI: /docs)" >&2
echo "  UI:             http://127.0.0.1:5173" >&2
echo "                  proxies /api → 8000, /validation-api → 8001" >&2
echo "  venv:           $VENV" >&2
echo "  Press Ctrl+C to stop all." >&2
echo "" >&2

wait
