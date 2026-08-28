@echo off
REM  Surveys every market the engine knows and ranks what is worth trading.
REM  Needs no trained model and makes no claim about direction.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo  [X] Not set up yet. Double-click setup.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m signalforge.cli hunt --timeframe H1
echo.
pause
