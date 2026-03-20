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
        pause
        exit /b 1
    )

    echo  [2/3] Installing packages (needs internet)...
    "%~dp0.venv\Scripts\pip.exe" install -r "%~dp0requirements.txt" --quiet --disable-pip-version-check
    if errorlevel 1 (
        echo  [ERROR] Package installation failed.
        echo  Check your internet connection and try again.
        pause
        exit /b 1
    )

    echo  [3/3] Done!
)

:: ── Launch ──────────────────────────────────────────────────────────────────
echo.
echo  Launching Data Dominion...
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0app\datadominion.py"
echo  App started. You can close this window.
timeout /t 3 >nul
