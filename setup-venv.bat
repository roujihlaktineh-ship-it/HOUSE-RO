@echo off
REM Batch script to create and setup Python virtual environment
REM Usage: setup-venv.bat

echo Setting up Python virtual environment...

REM Remove existing .venv if it exists
if exist .venv (
    echo Removing existing .venv folder...
    rmdir /s /q .venv
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv .venv

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing dependencies from requirements.txt...
if exist requirements.txt (
    pip install -r requirements.txt
) else (
    echo requirements.txt not found. Installing default packages...
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter streamlit flask fastapi uvicorn pydantic
)

echo.
echo Virtual environment setup complete!
echo To activate in future sessions, run: .venv\Scripts\activate.bat
pause
