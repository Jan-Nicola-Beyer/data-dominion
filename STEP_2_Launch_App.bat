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
if not exist "%~dp0.lib\customtkinter" (
    echo.
    echo  First-time setup - installing packages...
    echo  This takes 2-3 minutes. Please wait.
    echo.

    python -m pip install -r "%~dp0requirements.txt" --target "%~dp0.lib" --disable-pip-version-check 2>> "%~dp0crash_log.txt"
    if errorlevel 1 (
        echo  [ERROR] Package installation failed.
        echo  Check crash_log.txt for details.
        pause
        exit /b 1
    )

    echo  Packages installed!
)

:: ── Launch with crash logging ───────────────────────────────────────────────
echo.
echo  Launching Data Dominion...
echo.

echo ============================================ >> "%~dp0crash_log.txt"
echo  Launch: %date% %time% / Python %PYVER% >> "%~dp0crash_log.txt"
echo ============================================ >> "%~dp0crash_log.txt"

set PYTHONPATH=%~dp0.lib
python "%~dp0app\datadominion.py" 2>> "%~dp0crash_log.txt"

if errorlevel 1 (
    echo.
    echo  ============================================
    echo   The app crashed. Error details:
    echo  ============================================
    echo.
    type "%~dp0crash_log.txt"
    echo.
    pause
)
