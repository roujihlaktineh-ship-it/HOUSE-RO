# PowerShell script to create and setup Python virtual environment
# Usage: .\setup-venv.ps1

Write-Host "Setting up Python virtual environment..." -ForegroundColor Cyan

# Remove existing .venv if it exists
if (Test-Path ".venv") {
    Write-Host "Removing existing .venv folder..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".venv"
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Cyan
python -m venv .venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Cyan
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
} else {
    Write-Host "requirements.txt not found. Installing default packages..." -ForegroundColor Yellow
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter streamlit flask fastapi uvicorn pydantic
}

Write-Host "`nVirtual environment setup complete!" -ForegroundColor Green
Write-Host "To activate in future sessions, run: .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
