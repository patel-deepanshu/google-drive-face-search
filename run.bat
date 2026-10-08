@echo off
rem Double-click to start Face Search. Creates the venv and installs packages on first run.
cd /d "%~dp0"
if exist venv\Scripts\python.exe goto run
echo Setting up (first run only)...
python -m venv venv || goto fail
venv\Scripts\python.exe -m pip install -r requirements.txt || goto fail
:run
venv\Scripts\python.exe run.py
goto end
:fail
echo Setup failed. Is Python 3.11+ installed and on PATH?
pause
:end
