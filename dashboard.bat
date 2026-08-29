@echo off
REM  Opens the dashboard in your browser and serves it to your phone.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo  [X] Not set up yet. Double-click setup.bat first.
  pause
  exit /b 1
)
echo.
echo  Starting the dashboard. Your browser will open in a moment.
echo  To read it on your phone, use one of the addresses printed below
echo  while the phone is on the same Wi-Fi.
echo.
start "" http://localhost:8000/dashboard
".venv\Scripts\python.exe" -m signalforge.cli dashboard --host 0.0.0.0 --port 8000
pause
