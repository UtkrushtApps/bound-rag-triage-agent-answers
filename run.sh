#!/usr/bin/env bash
set -euo pipefail

echo "[run.sh] installing dependencies..."
pip install -q -r requirements.txt

# Load .env if present (harmless if absent)
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

echo "[run.sh] running selfcheck (LLM-free readiness probe)..."
python -m agent --selfcheck

echo "ready"
