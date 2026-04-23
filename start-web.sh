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
uvicorn server.main:app --reload --host 127.0.0.1 --port 8000 &
UVICORN_PID=$!
PIDS+=("$UVICORN_PID")

# First request can be slow: importing the app warms EasyOCR/Paddle, etc. Vite must
# not proxy until the API is actually accepting connections.
echo "Starting API (first load can take 30–90s while OCR warms up)…" >&2
READY=0
if command -v curl >/dev/null 2>&1; then
  for _ in $(seq 1 240); do
    if curl -sf --connect-timeout 2 --max-time 5 "$API/api/health" >/dev/null 2>&1; then
      READY=1
      break
    fi
    if ! kill -0 "$UVICORN_PID" 2>/dev/null; then
      echo "Error: uvicorn exited before the API became ready." >&2
      exit 1
    fi
    sleep 0.5
  done
else
  echo "Warning: curl not found; waiting 5s then starting the UI (may see proxy errors until the API is up)." >&2
  sleep 5
  READY=1
fi
if [[ "$READY" -ne 1 ]]; then
  echo "Error: API did not respond at $API/api/health in time. Is something else using port 8000?" >&2
  exit 1
fi

( cd "$ROOT/frontend" && npm run dev ) &
PIDS+=($!)

echo "" >&2
echo "  API:    http://127.0.0.1:8000  (OpenAPI: /docs)" >&2
echo "  UI:     http://127.0.0.1:5173  (proxies /api → API)" >&2
echo "  venv:   $VENV" >&2
echo "  Press Ctrl+C to stop both." >&2
echo "" >&2

wait
