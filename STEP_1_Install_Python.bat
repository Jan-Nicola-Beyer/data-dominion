@echo off
title Data Dominion - Step 1: Install Python
color 0F

echo.
echo  ============================================
echo    Data Dominion - Step 1: Install Python
echo  ============================================
echo.
echo  This will install Python 3.11 on your computer.
echo  Python is free, safe, and widely used.
echo.
echo  IMPORTANT: In the installer window that opens:
echo    [x] Make sure "Add Python to PATH" is CHECKED
echo    Then click "Install Now"
echo.
echo  Press any key to start the Python installer...
pause >nul

:: Launch the Python installer (per-user, add to PATH)
start "" "%~dp0Install_Python_3.11.9.exe" PrependPath=1 InstallAllUsers=0

echo.
echo  The Python installer should now be open.
echo  Follow the instructions in the installer window.
echo.
echo  After Python is installed, close this window
echo  and double-click STEP_2_Launch_App.bat
echo.
pause
