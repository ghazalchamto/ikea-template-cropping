#!/usr/bin/env bash
# Launch the IKEA Label Validation dashboard.
#
# Usage:
#   ./run.sh            — uses whatever Python is active in your shell
#   ./run.sh .venv312   — uses a specific virtual environment
#
# The PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK flag suppresses PaddleOCR's
# online model-source verification, which is not needed for local use.

set -e

VENV="${1:-}"

if [ -n "$VENV" ] && [ -d "$VENV" ]; then
    PYTHON="$VENV/bin/python"
    STREAMLIT="$VENV/bin/streamlit"
else
    PYTHON="${PYTHON:-python3}"
    STREAMLIT="${PYTHON%python*}streamlit"
    # Fall back to module invocation if streamlit is not on PATH
    if ! command -v streamlit &>/dev/null; then
        STREAMLIT="$PYTHON -m streamlit"
    fi
fi

export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

echo "Starting IKEA Label Validation dashboard…"
$STREAMLIT run app.py "$@"
