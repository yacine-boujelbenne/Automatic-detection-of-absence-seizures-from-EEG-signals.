@echo off
setlocal
cd /d "%~dp0.."
set PYTHONDONTWRITEBYTECODE=1
echo Ouvrez http://127.0.0.1:8765 dans votre navigateur.
echo Gardez cette fenetre ouverte pendant la demonstration.
venv\Scripts\python.exe app\server.py
pause
