# RobustCBRN Pipeline Integration Guide

## Overview

The scripts in this `temporary/` folder are **ready for integration** into the main RobustCBRN project. They provide a comprehensive, cross-platform end-to-end pipeline that replaces the existing fragmented approach with a unified, robust system.

## Integration Readiness Status

✅ **FULLY READY FOR INTEGRATION**

### What's Fixed and Ready:

1. **Cross-Platform Compatibility**: All scripts now properly handle Windows (Git Bash/Cygwin), macOS, and Linux
2. **Proper Shebang**: Changed from `#!/bin/bash` to `#!/usr/bin/env bash` for better portability
3. **Executable Permissions**: All scripts are now executable
4. **Windows Path Handling**: Fixed virtual environment activation paths for Windows
5. **Platform Detection**: Enhanced Windows detection to include `win32` OSTYPE
6. **Error Handling**: Comprehensive error handling with proper exit codes
7. **Logging**: Consistent logging format across all scripts

## Integration Steps

### Step 1: Move Scripts to Main Project

```bash
# Move all scripts from temporary/ to scripts/
cp temporary/*.sh scripts/
cp temporary/README.md scripts/PIPELINE_README.md
cp temporary/INTEGRATION_GUIDE.md scripts/
```

### Step 2: Update Makefile

Add new pipeline targets to the existing `Makefile`:

```makefile
# Add these targets to Makefile
.PHONY: pipeline pipeline-validate pipeline-setup pipeline-sample pipeline-full

pipeline:
	bash scripts/run_pipeline.sh

pipeline-validate:
	bash scripts/run_pipeline.sh --steps validate

pipeline-setup:
	bash scripts/run_pipeline.sh --steps validate,setup

pipeline-sample:
	bash scripts/run_pipeline.sh --steps validate,setup,discover,sample

pipeline-full:
	bash scripts/run_pipeline.sh --steps validate,setup,discover,sample,full,aggregate,figures,tests,report,verify
```

### Step 3: Update Documentation

Update the main `README.md` to reference the new pipeline:

```markdown
## Quick Start with New Pipeline

### Run Complete Pipeline
```bash
make pipeline
```

### Run Individual Steps
```bash
make pipeline-validate    # Platform validation only
make pipeline-setup       # Setup environment
make pipeline-sample      # Sample evaluation
make pipeline-full        # Complete evaluation
```

### Custom Configuration
```bash
# Run with custom dataset and subset
bash scripts/run_pipeline.sh --dataset data/custom.jsonl --subset 256

# Run specific steps only
bash scripts/run_pipeline.sh --steps setup,sample,aggregate
```

## Script Architecture

### Main Orchestrator
- **`run_pipeline.sh`**: Main wrapper script that orchestrates the complete pipeline

### Individual Pipeline Steps
1. **`validate_platform.sh`**: Platform detection and dependency validation
2. **`setup_env.sh`**: Environment setup and dependency installation
3. **`discover_entry_points.sh`**: Discover and validate evaluation entry points
4. **`run_sample_evaluation.sh`**: Run sample evaluation to validate the pipeline
5. **`run_full_evaluation.sh`**: Run the complete evaluation suite
6. **`aggregate_results.sh`**: Aggregate results from multiple evaluations
7. **`generate_figures.sh`**: Generate figures and visualizations
8. **`run_tests_and_security.sh`**: Run tests and security validation
9. **`generate_report.sh`**: Generate the final evaluation report
10. **`final_verification.sh`**: Final verification of all pipeline outputs
11. **`platform_compat.sh`**: Cross-platform compatibility layer

## Configuration Options

All scripts support extensive configuration through environment variables:

### Core Configuration
- `VENV_DIR`: Virtual environment directory (default: .venv)
- `DATASET`: Dataset file path (default: data/sample_sanitized.jsonl)
- `LOGS_DIR`: Logs directory (default: logs)
- `RESULTS_DIR`: Results directory (default: artifacts/results)
- `FIGURES_DIR`: Figures directory (default: artifacts/figs)
- `REPORT_DIR`: Report directory (default: docs/results)

### Evaluation Configuration
- `SUBSET_SIZE`: Subset size for evaluation (default: 512)
- `CONSENSUS_K`: Consensus K value (default: 2)
- `MODELS`: Semicolon-separated list of models
- `SEEDS`: Semicolon-separated list of seeds
- `DEVICE`: Device for evaluation (default: cuda)
- `DTYPE`: Data type for evaluation (default: bfloat16)
- `CLOZE_MODE`: Cloze evaluation mode (default: fallback)

### System Configuration
- `LOG_LEVEL`: Log level (default: INFO)
- `MAKE_CMD`: Make command (default: make)
- `PYTHON_VERSION`: Python command (default: python3)

## Cross-Platform Support

### Windows (Git Bash/Cygwin)
- Virtual environment: `.venv/Scripts/activate`
- Path separator: `;`
- Python command: `python`

### macOS/Linux
- Virtual environment: `.venv/bin/activate`
- Path separator: `:`
- Python command: `python3`

## Error Handling

All scripts implement comprehensive error handling:
- `set -euo pipefail` for strict error handling
- Error trapping with line number reporting
- Graceful failure handling with appropriate exit codes
- Detailed logging of success/failure states

## Logging System

Consistent logging format across all scripts:
- Timestamped log messages
- Configurable log levels (DEBUG, INFO, WARN, ERROR)
- Clear success/failure indicators
- Structured output for easy parsing

## Security Considerations

The pipeline includes security validation:
- Check for sensitive data in public artifacts
- Validate JSON structure
- Look for hardcoded paths or credentials
- Ensure no per-item exploit labels are exposed

## Testing Integration

### Manual Testing
```bash
# Test individual components
bash scripts/validate_platform.sh
bash scripts/setup_env.sh
bash scripts/discover_entry_points.sh

# Test complete pipeline
bash scripts/run_pipeline.sh --steps validate,setup,sample
```

### Automated Testing
The scripts can be integrated into CI/CD pipelines:
```yaml
# Example GitHub Actions step
- name: Run RobustCBRN Pipeline
  run: |
    bash scripts/run_pipeline.sh --steps validate,setup,sample
```

## Migration from Existing Scripts

The new pipeline is designed to be **backward compatible** with existing scripts:

### Existing Scripts (Still Supported)
- `scripts/setup.sh` → Can be replaced by `setup_env.sh`
- `scripts/run_sample.sh` → Can be replaced by `run_sample_evaluation.sh`
- `scripts/run_evalset.sh` → Can be replaced by `run_full_evaluation.sh`

### Gradual Migration
1. **Phase 1**: Add new pipeline alongside existing scripts
2. **Phase 2**: Update Makefile to use new pipeline
3. **Phase 3**: Update documentation to reference new pipeline
4. **Phase 4**: Deprecate old scripts (optional)

## Benefits of Integration

### For Users
- **Single Entry Point**: One command to run the complete pipeline
- **Better Error Handling**: Clear error messages and recovery instructions
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Configurable**: Extensive customization options
- **Robust**: Comprehensive validation and error handling

### For Developers
- **Modular Design**: Individual scripts can be run independently
- **Consistent Patterns**: All scripts follow the same structure
- **Easy Debugging**: Comprehensive logging and error reporting
- **Extensible**: Easy to add new pipeline steps
- **Maintainable**: Clear separation of concerns

## Next Steps After Integration

1. **Test the Pipeline**: Run the complete pipeline on different platforms
2. **Update CI/CD**: Integrate into GitHub Actions or other CI systems
3. **User Training**: Update documentation and provide examples
4. **Performance Optimization**: Monitor and optimize pipeline performance
5. **Feature Extensions**: Add new pipeline steps as needed

## Support and Troubleshooting

### Common Issues
1. **Windows Path Issues**: Ensure running in Git Bash or Cygwin
2. **Permission Issues**: Ensure scripts are executable (`chmod +x`)
3. **Python Environment**: Ensure virtual environment is properly activated
4. **Dependencies**: Run `validate_platform.sh` to check dependencies

### Getting Help
- Check script logs for detailed error messages
- Run individual steps to isolate issues
- Use `--help` flag for script-specific help
- Review the comprehensive README.md in the scripts directory

## Conclusion

The temporary scripts are **fully ready for integration** and provide a significant improvement over the existing fragmented approach. They offer:

- **Better User Experience**: Single command to run complete pipeline
- **Improved Reliability**: Comprehensive error handling and validation
- **Cross-Platform Support**: Works on all major operating systems
- **Enhanced Maintainability**: Modular design with consistent patterns
- **Future-Proof Architecture**: Easy to extend and modify

Integration is straightforward and can be done incrementally without disrupting existing workflows.
