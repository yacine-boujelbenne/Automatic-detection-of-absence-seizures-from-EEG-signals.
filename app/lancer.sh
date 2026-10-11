#!/usr/bin/env sh
# Lance le lecteur d'absences sur http://127.0.0.1:8765 (Linux / macOS).
cd "$(dirname "$0")/.." || exit 1
PYTHON=".venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="venv/bin/python"
[ -f app/model/recordings.json ] || "$PYTHON" app/prepare_model.py
PYTHONDONTWRITEBYTECODE=1 exec "$PYTHON" app/server.py "$@"
