# RobustCBRN Eval - Setup Script for Windows PowerShell using uv

Write-Host "🚀 RobustCBRN Eval Setup Script (Windows)" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Check Python version
Write-Host "📋 Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($pythonVersion -match "Python (\d+)\.(\d+)") {
    $major = [int]$matches[1]
    $minor = [int]$matches[2]
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
        Write-Host "❌ Error: Python 3.10 or higher is required (found $pythonVersion)" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ $pythonVersion found" -ForegroundColor Green
} else {
    Write-Host "❌ Error: Python not found. Please install Python 3.10+" -ForegroundColor Red
    exit 1
}

# Install uv if not already installed
$uvPath = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvPath) {
    Write-Host "📦 Installing uv (fast Python package manager)..." -ForegroundColor Yellow
    # Using PowerShell to download and run the installer
    irm https://astral.sh/uv/install.ps1 | iex

    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

    Write-Host "✅ uv installed successfully" -ForegroundColor Green
} else {
    $uvVersion = uv --version
    Write-Host "✅ uv is already installed ($uvVersion)" -ForegroundColor Green
}

# Create virtual environment with uv
Write-Host "🔧 Creating virtual environment with uv..." -ForegroundColor Yellow
uv venv --python python3.10

# Activate virtual environment
Write-Host "🔌 Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Install dependencies with uv
Write-Host "📚 Installing dependencies with uv (this is 10x faster than pip)..." -ForegroundColor Yellow
uv pip install -r requirements.txt

# Check for CUDA and install PyTorch with CUDA support if available
$nvidiaCmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaCmd) {
    Write-Host "🎮 CUDA detected. Installing PyTorch with CUDA 11.8 support..." -ForegroundColor Yellow
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
}

# Create necessary directories
Write-Host "📁 Creating project directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path data, cache, results, logs | Out-Null

Write-Host ""
Write-Host "✨ Setup complete! Environment is ready." -ForegroundColor Green
Write-Host ""
Write-Host "📌 Quick start commands:" -ForegroundColor Cyan
Write-Host "  - Activate environment: .\.venv\Scripts\Activate.ps1"
Write-Host "  - Run tests: python -m unittest"
Write-Host "  - Run pipeline: python cli.py load data\wmdp_bio_sample_100.jsonl"
Write-Host ""
Write-Host "💡 Tip: uv commands for package management:" -ForegroundColor Cyan
Write-Host "  - Install package: uv pip install <package>"
Write-Host "  - Sync exact deps: uv pip sync requirements.txt"
Write-Host "  - Compile deps: uv pip compile requirements.in -o requirements.txt"