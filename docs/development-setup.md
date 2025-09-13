# Development Setup Guide

This guide provides detailed instructions for setting up the RobustCBRN Eval development environment.

## Prerequisites

Before starting, ensure you have the following installed:

### Required
- **Python 3.10+**: Download from [python.org](https://www.python.org/)
- **Git**: Version control system ([git-scm.com](https://git-scm.com/))

### Optional
- **CUDA 11.8+**: For GPU acceleration ([NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads))
- **Docker**: For containerized development ([docker.com](https://www.docker.com/))

## Environment Setup

### Quick Setup (Recommended)

We use `uv` for fast Python package management (10x faster than pip).

#### Linux/macOS

```bash
# Clone the repository
git clone https://github.com/apart-research/robustcbrn-eval.git
cd robustcbrn-eval

# Run the setup script
./scripts/setup.sh
```

#### Windows (PowerShell)

```powershell
# Clone the repository
git clone https://github.com/apart-research/robustcbrn-eval.git
cd robustcbrn-eval

# Run the setup script
.\scripts\setup.ps1
```

#### Windows (CMD)

```batch
REM Clone the repository
git clone https://github.com/apart-research/robustcbrn-eval.git
cd robustcbrn-eval

REM Run the setup script
scripts\setup.bat
```

### Manual Setup

If you prefer to set up manually or the scripts don't work:

#### 1. Install uv

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

#### 2. Create Virtual Environment

```bash
uv venv --python python3.10
```

#### 3. Activate Virtual Environment

```bash
# Linux/macOS
source .venv/bin/activate

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat
```

#### 4. Install Dependencies

```bash
# Core dependencies
uv pip install -r requirements.txt

# Development dependencies
uv pip install -r requirements-dev.txt
```

## GPU Setup (Optional)

If you have an NVIDIA GPU and want to use GPU acceleration:

### Check CUDA Installation

```bash
nvidia-smi
```

### Install PyTorch with CUDA Support

```bash
# CUDA 11.8
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## IDE Configuration

### Visual Studio Code

#### Recommended Extensions

Install these extensions for the best development experience:

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-python.black-formatter",
    "charliermarsh.ruff",
    "ms-python.mypy-type-checker",
    "njpwerner.autodocstring",
    "streetsidesoftware.code-spell-checker",
    "redhat.vscode-yaml",
    "tamasfe.even-better-toml",
    "davidanson.vscode-markdownlint",
    "yzhang.markdown-all-in-one",
    "github.vscode-pull-request-github"
  ]
}
```

#### Settings

Add to `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "100"],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "python.testing.pytestEnabled": false,
  "python.testing.unittestEnabled": true,
  "python.testing.unittestArgs": [
    "-v",
    "-s",
    "./tests",
    "-p",
    "test_*.py"
  ]
}
```

### PyCharm

1. Open the project in PyCharm
2. Configure Python Interpreter:
   - File → Settings → Project → Python Interpreter
   - Add Interpreter → Existing Environment
   - Select `.venv/bin/python` (Linux/macOS) or `.venv\Scripts\python.exe` (Windows)
3. Configure Code Style:
   - File → Settings → Editor → Code Style → Python
   - Set line length to 100
   - Enable "Optimize imports"
4. Configure File Watchers (optional):
   - Install Black and Ruff file watchers for automatic formatting

## Development Workflow

### Running Tests

```bash
# Run all tests
python -m unittest

# Run specific test file
python -m unittest tests.test_pipeline

# Run with coverage
coverage run -m unittest
coverage report
coverage html  # Generate HTML report
```

### Code Quality Checks

```bash
# Format code with Black
black src/ tests/

# Lint with Ruff
ruff check src/ tests/

# Type check with mypy
mypy src/

# Run all checks
make lint  # If Makefile is available
```

### Pre-commit Hooks

Set up pre-commit hooks to automatically check code quality:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hook scripts
pre-commit install

# Run against all files (optional)
pre-commit run --all-files
```

## Troubleshooting

### Common Issues

#### Python Version Error

**Problem**: "Python 3.10 or higher is required"

**Solution**:
```bash
# Check your Python version
python --version

# If < 3.10, install a newer version:
# - Use pyenv (recommended): https://github.com/pyenv/pyenv
# - Or download from python.org
```

#### uv Installation Fails

**Problem**: "uv: command not found" after installation

**Solution**:
```bash
# Add to PATH manually
export PATH="$HOME/.cargo/bin:$PATH"  # Linux/macOS
# Or restart your terminal
```

#### CUDA Not Detected

**Problem**: PyTorch doesn't detect GPU

**Solution**:
```bash
# Verify CUDA installation
nvidia-smi

# Check PyTorch CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with correct CUDA version
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Virtual Environment Activation Issues

**Problem**: "No module named 'src'" or similar import errors

**Solution**:
```bash
# Ensure virtual environment is activated
which python  # Should show .venv/bin/python

# If not, activate it:
source .venv/bin/activate  # Linux/macOS
```

#### Permission Denied on Scripts

**Problem**: "Permission denied" when running setup scripts

**Solution**:
```bash
# Make scripts executable
chmod +x scripts/setup.sh  # Linux/macOS
```

### Getting Help

If you encounter issues not covered here:

1. Check existing [GitHub Issues](https://github.com/apart-research/robustcbrn-eval/issues)
2. Search the [Discussions](https://github.com/apart-research/robustcbrn-eval/discussions)
3. Create a new issue with:
   - Your OS and Python version
   - Complete error message
   - Steps to reproduce

## Next Steps

Once your environment is set up:

1. Run the test suite to verify everything works:
   ```bash
   python -m unittest
   ```

2. Try loading sample data:
   ```bash
   python cli.py load data/wmdp_bio_sample_100.jsonl
   ```

3. Read the [Architecture Documentation](architecture/index.md)

4. Check the [Contributing Guidelines](../CONTRIBUTING.md)

## Useful Commands Reference

```bash
# Package Management (with uv)
uv pip install <package>          # Install a package
uv pip uninstall <package>         # Uninstall a package
uv pip list                        # List installed packages
uv pip freeze > requirements.txt   # Export dependencies

# Testing
python -m unittest                 # Run all tests
python -m unittest discover        # Discover and run tests
coverage run -m unittest           # Run with coverage

# Code Quality
black src/                         # Format code
ruff check src/                    # Lint code
mypy src/                          # Type check

# Git Operations
git status                         # Check status
git add .                          # Stage changes
git commit -m "message"            # Commit changes
git push origin branch-name        # Push to remote

# Project
python cli.py --help               # Show CLI help
python cli.py load <file>          # Load dataset
python cli.py analyze <file>       # Run analysis
```