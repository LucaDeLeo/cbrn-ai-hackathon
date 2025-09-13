@echo off
REM RobustCBRN Eval - Setup Script for Windows CMD using uv

echo.
echo RobustCBRN Eval Setup Script (Windows CMD)
echo ==========================================
echo.

REM Check Python version
echo Checking Python version...
python --version 2>&1 | findstr /R "Python 3\.[1-9][0-9]" >nul
if errorlevel 1 (
    echo Error: Python 3.10 or higher is required
    echo Please install Python from https://www.python.org/
    exit /b 1
)
echo Python 3.10+ found

REM Check if uv is installed
where uv >nul 2>&1
if errorlevel 1 (
    echo Installing uv (fast Python package manager)...
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

    REM Refresh PATH
    call refreshenv

    echo uv installed successfully
) else (
    echo uv is already installed
)

REM Create virtual environment with uv
echo Creating virtual environment with uv...
uv venv --python python3.10

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install dependencies with uv
echo Installing dependencies with uv (this is 10x faster than pip)...
uv pip install -r requirements.txt

REM Check for CUDA and install PyTorch with CUDA support if available
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo CUDA detected. Installing PyTorch with CUDA 11.8 support...
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)

REM Create necessary directories
echo Creating project directories...
if not exist data mkdir data
if not exist cache mkdir cache
if not exist results mkdir results
if not exist logs mkdir logs

echo.
echo Setup complete! Environment is ready.
echo.
echo Quick start commands:
echo   - Activate environment: .venv\Scripts\activate.bat
echo   - Run tests: python -m unittest
echo   - Run pipeline: python cli.py load data\wmdp_bio_sample_100.jsonl
echo.
echo Tip: uv commands for package management:
echo   - Install package: uv pip install ^<package^>
echo   - Sync exact deps: uv pip sync requirements.txt
echo   - Compile deps: uv pip compile requirements.in -o requirements.txt
echo.