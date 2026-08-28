@echo off
REM ===================================================================
REM  SignalForge - one-time setup for Windows
REM  Double-click this file. It does everything.
REM ===================================================================
setlocal
cd /d "%~dp0"

echo.
echo  ================================================
echo   SignalForge setup
echo  ================================================
echo.

REM ---- 1. Find a usable Python -------------------------------------
set PY=
where py >nul 2>&1 && set PY=py
if "%PY%"=="" (
  where python >nul 2>&1 && set PY=python
)
if "%PY%"=="" (
  echo  [X] Python is not installed, or not on your PATH.
  echo.
  echo      Install it from https://www.python.org/downloads/
  echo      IMPORTANT: tick "Add python.exe to PATH" on the first screen.
  echo.
  pause
  exit /b 1
)

echo  [1/4] Using Python:
%PY% --version
echo.

REM ---- 2. Create the virtual environment ---------------------------
if exist ".venv\Scripts\python.exe" (
  echo  [2/4] Virtual environment already exists - reusing it.
) else (
  echo  [2/4] Creating a private Python environment in .venv ...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo  [X] Could not create the environment. See the error above.
    pause
    exit /b 1
  )
)
echo.

REM ---- 3. Install the dependencies ---------------------------------
echo  [3/4] Installing packages. This takes 2-5 minutes on first run.
echo.
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo  [X] Installation failed. See the error above.
  pause
  exit /b 1
)
echo.

REM ---- 4. Verify ----------------------------------------------------
echo  [4/4] Checking that it can reach the markets ...
echo.
".venv\Scripts\python.exe" -m signalforge.cli doctor

echo.
echo  ================================================
echo   Setup finished.
echo.
echo   Check the MT5 symbol list above against the
echo   Market Watch window in MetaTrader 5.
echo.
echo   Next: double-click  train.bat   (about 30 min)
echo   Then: double-click  signals.bat (any time)
echo  ================================================
echo.
pause
