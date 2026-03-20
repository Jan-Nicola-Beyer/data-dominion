@echo off
title Data Dominion - Install Topic Modelling
color 0F

echo.
echo  ============================================
echo    Install Topic Modelling (Optional)
echo  ============================================
echo.
echo  This downloads additional packages for the
echo  Topic Modelling feature (~500 MB).
echo  Takes 5-10 minutes. Needs internet.
echo.

if not exist "%~dp0.venv\Scripts\pip.exe" (
    echo  [ERROR] Run STEP_2_Launch_App.bat first.
    pause
    exit /b 1
)

echo  Installing...
"%~dp0.venv\Scripts\pip.exe" install sentence-transformers bertopic umap-learn hdbscan scikit-learn --disable-pip-version-check
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
