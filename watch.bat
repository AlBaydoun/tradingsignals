@echo off
REM  Runs continuously, checking every 5 minutes. Close the window to stop.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo  [X] Not set up yet. Double-click setup.bat first.
  pause
  exit /b 1
)
echo.
echo  Watching the markets. Close this window to stop.
echo.
".venv\Scripts\python.exe" -m signalforge.cli watch --interval 300
pause
