@echo off
REM  Fits a model for every symbol and timeframe in config.yaml.
REM  Takes roughly 30 minutes. Run it once a week.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo  [X] Not set up yet. Double-click setup.bat first.
  pause
  exit /b 1
)
echo.
echo  Training models. This takes about 30 minutes - leave it running.
echo.
".venv\Scripts\python.exe" -m signalforge.cli train --timeframes H1 H4
echo.
pause
