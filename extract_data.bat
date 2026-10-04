@echo off
REM Extraction des 22 fichiers .mat MCOS (timetable MATLAB) vers .npz
REM Outil verifie : tools\mcos_extract.py  -- 22/22 enregistrements
cd /d "%~dp0"
venv\Scripts\python.exe tools\mcos_extract.py --export "data\CRISES ABSENCES" "data\exports"
echo.
pause
