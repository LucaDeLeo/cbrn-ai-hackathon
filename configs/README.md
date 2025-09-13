# Configuration Files

This directory contains JSON configuration files for different deployment scenarios.

## Available Configurations

### default.json
Standard configuration for most use cases. Includes:
- Balanced settings for CPU/GPU environments
- Standard logging configuration
- Default analysis parameters

### minimal.json
Lightweight configuration for CPU-only environments:
- No GPU dependencies
- Reduced memory footprint
- Basic feature set

### full.json
Full feature set with all components enabled:
- GPU acceleration
- All analysis modules
- Extended logging
- Advanced caching

## Configuration Structure

```json
{
  "logging": {
    "level": "INFO",
    "dir": "logs",
    "format": "json"
  },
  "determinism": {
    "seed": 42,
    "enabled": true
  },
  "data": {
    "batch_size": 32,
    "csv_mapping": {}
  },
  "analysis": {
    "confidence_level": 0.95,
    "bootstrap_samples": 1000
  }
}
```

## Usage

```bash
# Use specific configuration
python cli.py load data/sample.jsonl --config configs/minimal.json

# Configuration is loaded automatically from configs/default.json if not specified
python cli.py load data/sample.jsonl
```

## Custom Configurations

To create a custom configuration:
1. Copy an existing config file
2. Modify settings as needed
3. Use with `--config` flag

## Environment Variables

Configuration values can be overridden with environment variables:
- `ROBUSTCBRN_LOG_LEVEL`
- `ROBUSTCBRN_CACHE_DIR`
- `ROBUSTCBRN_PROJECT_SALT`