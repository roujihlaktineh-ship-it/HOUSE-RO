@echo off
REM Complete environment reset script
REM This script will remove old venv and create a fresh one

echo.
echo ============================================================
echo HOUSE PRICES - ENVIRONMENT RESET
echo ============================================================
echo.

REM Change to the correct directory
cd /d "%~dp0"
echo Current directory: %cd%

REM Check if Python is available
echo.
echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

REM Remove old .venv
echo.
echo Removing old .venv directory...
if exist .venv (
    rmdir /s /q .venv
    if exist .venv (
        echo ERROR: Could not remove .venv - it may be locked
        pause
        exit /b 1
    )
    echo OK: .venv removed
)

REM Create new venv
echo.
echo Creating new virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: Could not create venv
    pause
    exit /b 1
)
echo OK: .venv created

REM Activate venv
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Could not activate venv
    pause
    exit /b 1
)
echo OK: venv activated

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo WARNING: pip upgrade failed, continuing anyway...
)

REM Install requirements
echo.
echo Installing dependencies...
if exist requirements.txt (
    pip install -r requirements.txt
) else (
    echo Installing default packages...
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter streamlit flask fastapi uvicorn pydantic
)

if errorlevel 1 (
    echo ERROR: Package installation failed
    pause
    exit /b 1
)

REM Verify installation
echo.
echo Verifying installation...
python diagnose.py

echo.
echo ============================================================
echo SETUP COMPLETE!
echo ============================================================
echo.
echo To activate the environment in the future, run:
echo   .venv\Scripts\activate.bat
echo.
pause
