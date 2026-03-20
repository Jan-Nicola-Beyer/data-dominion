@echo off
title Data Dominion
color 0F

echo.
echo  ============================================
echo       Data Dominion - Starting...
echo  ============================================
echo.

:: ── Check Python ────────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python is not found.
    echo.
    echo  Please run STEP_1_Install_Python.bat first,
    echo  then close and reopen this file.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  Python %PYVER% found.

:: ── First-time setup ────────────────────────────────────────────────────────
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo.
    echo  First-time setup — installing packages...
    echo  This takes 2-3 minutes. Please wait.
    echo.

    echo  [1/3] Creating environment...
    python -m venv "%~dp0.venv"
    if errorlevel 1 (
        echo  [ERROR] Failed to create environment.
        echo  [ERROR] Failed to create environment. >> "%~dp0crash_log.txt"
        pause
        exit /b 1
    )

    echo  [2/3] Installing packages (needs internet)...
    "%~dp0.venv\Scripts\pip.exe" install -r "%~dp0requirements.txt" --disable-pip-version-check 2>> "%~dp0crash_log.txt"
    if errorlevel 1 (
        echo  [ERROR] Package installation failed.
        echo  Check crash_log.txt for details.
        pause
        exit /b 1
    )

    echo  [3/3] Done!
)

:: ── Launch with crash logging ───────────────────────────────────────────────
echo.
echo  Launching Data Dominion...
echo.

:: Write timestamp to crash log
echo ============================================ >> "%~dp0crash_log.txt"
echo  Launch attempt: %date% %time% >> "%~dp0crash_log.txt"
echo  Python: %PYVER% >> "%~dp0crash_log.txt"
echo ============================================ >> "%~dp0crash_log.txt"

:: Use python.exe (not pythonw) so errors are captured
"%~dp0.venv\Scripts\python.exe" "%~dp0app\datadominion.py" 2>> "%~dp0crash_log.txt"

:: If we get here, the app closed — check if it was a crash
if errorlevel 1 (
    echo.
    echo  ============================================
    echo   The app crashed. Error details saved to:
    echo   crash_log.txt
    echo  ============================================
    echo.
    echo  Last error:
    echo  ---
    type "%~dp0crash_log.txt" | more
    echo.
    pause
) else (
    echo  App closed normally.
)
