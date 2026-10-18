@echo off
setlocal
cd /d "%~dp0.."
set PYTHONDONTWRITEBYTECODE=1
set PY=.venv\Scripts\python.exe
if not exist %PY% set PY=venv\Scripts\python.exe
if not exist app\model\test_scores.csv %PY% app\prepare_model.py
echo Ouvrez http://127.0.0.1:8501 dans votre navigateur.
%PY% -m streamlit run app\streamlit_app.py
pause
