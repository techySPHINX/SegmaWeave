# Setup Script for CNN Project
# Run this in PowerShell

Write-Host "=====================================" -ForegroundColor Green
Write-Host "CNN Project Setup" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green

# Activate virtual environment
Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install packages
Write-Host "`nInstalling required packages..." -ForegroundColor Yellow
pip install torch torchvision matplotlib numpy tqdm

Write-Host "`n=====================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host "`nNext Steps:" -ForegroundColor Cyan
Write-Host "1. Open 'main.py' in VS Code" -ForegroundColor White
Write-Host "2. Run each cell to see the outputs" -ForegroundColor White
Write-Host "3. Enjoy your CNN project!" -ForegroundColor White
Write-Host "`n" -ForegroundColor White
