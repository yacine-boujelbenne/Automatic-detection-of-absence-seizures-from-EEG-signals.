#!/usr/bin/env sh
# Lance l'interface Streamlit sur http://127.0.0.1:8501 (Linux / macOS).
cd "$(dirname "$0")/.." || exit 1
PYTHON=".venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="venv/bin/python"
[ -f app/model/test_scores.csv ] || "$PYTHON" app/prepare_model.py || exit 1
PYTHONDONTWRITEBYTECODE=1 exec "$PYTHON" -m streamlit run app/streamlit_app.py "$@"
