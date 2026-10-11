@echo off
REM Extraction des 22 fichiers .mat MCOS (timetable MATLAB) vers .npz
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
if not exist %PY% set PY=venv\Scripts\python.exe
%PY% tools\mcos_extract.py --export "data\CRISES ABSENCES" "data\exports"
echo.
pause
