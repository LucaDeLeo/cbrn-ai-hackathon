# RobustCBRN Evaluation Pipeline - Temporary Scripts

This directory contains improved shell scripts for the RobustCBRN evaluation pipeline. These scripts have been refactored from the original Python files to provide better integration capabilities.

## Scripts Overview

### Main Wrapper Script
- **`run_pipeline.sh`** - Main orchestrator script that runs the complete pipeline

### Individual Pipeline Steps
1. **`validate_platform.sh`** - Platform detection and dependency validation
2. **`setup_env.sh`** - Environment setup and dependency installation
3. **`discover_entry_points.sh`** - Discover and validate evaluation entry points
4. **`run_sample_evaluation.sh`** - Run sample evaluation to validate the pipeline
5. **`run_full_evaluation.sh`** - Run the complete evaluation suite
6. **`aggregate_results.sh`** - Aggregate results from multiple evaluations
7. **`generate_figures.sh`** - Generate figures and visualizations
8. **`run_tests_and_security.sh`** - Run tests and security validation
9. **`generate_report.sh`** - Generate the final evaluation report
10. **`final_verification.sh`** - Final verification of all pipeline outputs
11. **`platform_compat.sh`** - Cross-platform compatibility layer

## Usage

### Run Complete Pipeline
```bash
./run_pipeline.sh
```

### Run Specific Steps
```bash
./run_pipeline.sh --steps validate,setup,sample,aggregate
```

### Custom Configuration
```bash
./run_pipeline.sh \
  --dataset data/custom.jsonl \
  --subset 256 \
  --models "model1;model2" \
  --seeds "123;456"
```

### Run Individual Steps
```bash
./validate_platform.sh
./setup_env.sh
./discover_entry_points.sh
./run_sample_evaluation.sh
# ... etc
```

## Configuration

All scripts support configuration through environment variables or command-line arguments:

- `VENV_DIR` - Virtual environment directory (default: .venv)
- `DATASET` - Dataset file path (default: data/sample_sanitized.jsonl)
- `LOGS_DIR` - Logs directory (default: logs)
- `RESULTS_DIR` - Results directory (default: artifacts/results)
- `FIGURES_DIR` - Figures directory (default: artifacts/figs)
- `REPORT_DIR` - Report directory (default: docs/results)
- `LOG_LEVEL` - Log level (default: INFO)
- `SUBSET_SIZE` - Subset size for evaluation (default: 512)
- `CONSENSUS_K` - Consensus K value (default: 2)
- `MODELS` - Semicolon-separated list of models
- `SEEDS` - Semicolon-separated list of seeds
- `DEVICE` - Device for evaluation (default: cuda)
- `DTYPE` - Data type for evaluation (default: bfloat16)
- `CLOZE_MODE` - Cloze evaluation mode (default: fallback)

## Improvements Made

### 1. Proper Shell Scripts
- Converted from .py files with shell commands to proper .sh files
- Fixed shell syntax and added proper error handling
- Made all scripts executable

### 2. Configurable Paths
- All paths are now configurable via environment variables
- Default values provided for all configuration options
- Command-line argument support in the main wrapper script

### 3. Comprehensive Error Handling
- Added proper error handling with `set -euo pipefail`
- Error trapping with line number reporting
- Graceful failure handling with appropriate exit codes

### 4. Single Entry Point
- Created `run_pipeline.sh` as the main orchestrator
- Supports running individual steps or the complete pipeline
- Comprehensive help and usage information

### 5. Logging System
- Consistent logging format across all scripts
- Configurable log levels
- Timestamped log messages
- Clear success/failure indicators

### 6. Platform Compatibility
- Cross-platform support for Windows, macOS, and Linux
- Automatic platform detection and configuration
- Platform-specific command handling
- Virtual environment path detection for different OS

### 7. Robust Dependency Validation
- Comprehensive dependency checking for all required tools
- Python package validation with version checking
- Platform-specific validation (GPU, memory, disk space)
- Optional dependency detection with warnings

## Integration Readiness

✅ **FULLY READY FOR INTEGRATION** - All scripts have been updated and are ready for integration into the main project.

### Recent Updates Made:
1. **Fixed Shebang**: Changed from `#!/bin/bash` to `#!/usr/bin/env bash` for better portability
2. **Enhanced Windows Support**: Added `win32` OSTYPE detection for better Windows compatibility
3. **Fixed Path Handling**: Corrected virtual environment activation paths for Windows
4. **Made Executable**: All scripts now have proper executable permissions
5. **Comprehensive Error Handling**: All scripts implement robust error handling
6. **Cross-Platform Testing**: Scripts tested on Windows, macOS, and Linux

### Integration Steps:
1. Move scripts from `temporary/` to `scripts/` directory
2. Update `Makefile` to include new pipeline targets
3. Update main `README.md` to reference new pipeline
4. Test the pipeline on your target platform

See `INTEGRATION_GUIDE.md` for detailed integration instructions.

## Next Steps for Integration

1. Move scripts to `scripts/` directory in the main project
2. Update Makefile to use the new scripts
3. Add the scripts to the project's CI/CD pipeline
4. Update documentation to reference the new scripts
5. Test the scripts in the actual project environment

## Dependencies

### Required Dependencies
- Bash shell (version 4.0 or higher)
- Python 3.7+
- Virtual environment support
- Git (for version control)
- Standard Unix utilities (find, grep, wc, etc.)

### Platform-Specific Dependencies
- **Windows**: Git Bash or Cygwin, PowerShell (optional)
- **macOS**: Xcode command line tools, Homebrew (optional)
- **Linux**: Standard package manager (apt, yum, pacman)

### Python Packages
- **Required**: torch, transformers, datasets, numpy, pandas, matplotlib, scikit-learn
- **Optional**: inspect-ai, accelerate, evaluate, seaborn, plotly, typer, rich, pytest

## Security Considerations

The scripts include security validation:
- Check for sensitive data in public artifacts
- Validate JSON structure
- Look for hardcoded paths or credentials
- Ensure no per-item exploit labels are exposed
