@echo off
REM  Re-checks data providers, symbol names and affordability.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo  [X] Not set up yet. Double-click setup.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m signalforge.cli doctor
echo.
pause
