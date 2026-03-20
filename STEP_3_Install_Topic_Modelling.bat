@echo off
title Data Dominion - Install Topic Modelling
color 0F

echo.
echo  ============================================
echo    Install Topic Modelling (Optional)
echo  ============================================
echo.
echo  This downloads additional packages (~500 MB).
echo  Takes 5-10 minutes. Needs internet.
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Run STEP_1 first.
    pause
    exit /b 1
)

echo  Installing...
python -m pip install sentence-transformers bertopic umap-learn hdbscan scikit-learn --target "%~dp0.lib" --disable-pip-version-check
if errorlevel 1 (
    echo.
    echo  [ERROR] Installation failed.
    pause
    exit /b 1
)

echo.
echo  ============================================
echo    Topic Modelling installed!
echo    Restart Data Dominion to use it.
echo  ============================================
echo.
pause
