@echo off
REM Batch file to run the MIU Processor automation on Windows

echo.
echo ============================================================
echo  MIU Batch Processor - Windows Setup
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

REM Check current directory
echo Current directory: %cd%
echo.

REM Install required packages
echo Installing required packages...
python -m pip install --upgrade pip >nul 2>&1
python -m pip install playwright >nul 2>&1
python -m playwright install chromium >nul 2>&1

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

REM Run the automation script
echo.
echo Starting MIU Processor...
echo.

python automation_miu_processor.py

if errorlevel 1 (
    echo.
    echo ERROR: Script failed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Processing Complete!
echo ============================================================
echo.
pause
