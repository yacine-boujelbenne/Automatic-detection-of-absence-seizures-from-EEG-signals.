@echo off
setlocal
cd /d "%~dp0.."
set PYTHONDONTWRITEBYTECODE=1
set PY=.venv\Scripts\python.exe
if not exist %PY% set PY=venv\Scripts\python.exe
if not exist app\model\recordings.json %PY% app\prepare_model.py
echo Ouvrez http://127.0.0.1:8765 dans votre navigateur.
%PY% app\server.py
pause
