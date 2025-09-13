# Technical Assumptions

## Repository Structure: Monorepo
Single repository containing all components - core robustification engine, statistical analysis modules, model inference pipelines, CLI interface, and optional Streamlit demo. Clear module separation with `src/models/`, `src/analysis/`, `src/utils/`, `tests/`, and `configs/` directories.

## Service Architecture
**CRITICAL DECISION**: Modular pipeline architecture with pluggable components. Each robustification technique (consensus detection, cloze scoring, statistical heuristics) implemented as independent modules that can be enabled/disabled/combined. Pipeline stages: Data Loading → Preprocessing → Model Inference (cached) → Analysis → Filtering → Report Generation. NOT microservices - single process with multi-threading for parallel model inference.

## Testing Requirements
**CRITICAL DECISION**: Comprehensive testing pyramid given evaluation accuracy is critical:
- **Unit tests**: Every statistical function, bias detection algorithm, and data transformation
- **Integration tests**: End-to-end pipeline with small synthetic datasets
- **Regression tests**: Ensure deterministic results across runs (critical for reproducibility claims)
- **Performance tests**: Validate <4 hour processing time on reference hardware
- **Validation tests**: Agreement with 50-item human-labeled calibration subset
- **Manual testing utilities**: Scripts to spot-check individual questions and debug edge cases

## Additional Technical Assumptions and Requests

• **Radical dependency minimization**: Core functionality uses ONLY PyTorch, Transformers, NumPy, tqdm
• **Built-in Python優先**: Use standard library (json, csv, sqlite3, argparse) over external packages
• **Statistical implementations**: Write our own bootstrap CI and bias metrics (educational + no dependencies)
• **Visualization strategy**: Generate data, plot later if time allows (not critical for MVP)
• **CLI framework**: argparse over click/typer (built-in, sufficient, no learning curve)
• **Data handling**: Direct JSON/CSV parsing without pandas (we're not doing complex transforms)
• **Cache layer**: sqlite3 directly, no ORM needed for simple key-value storage
• **Streamlit**: Only add if we finish core early (it's a "nice to have" not "must have")
• **Testing**: unittest (built-in) over pytest initially (can migrate later)
• **Type hints**: Use them but no mypy/pydantic validation (runtime checking not critical for hackathon)
• **Python 3.10+** as primary language for ML ecosystem compatibility
• **PyTorch 2.0+** with HuggingFace Transformers for model inference
• **Model loading strategy**: Dynamic loading with transformers.AutoModelForCausalLM to support diverse architectures
• **Quantization**: Support for 4-bit/8-bit quantization via bitsandbytes for compute-constrained scenarios
• **Caching infrastructure**: Multi-level with SQLite for metadata, disk for model outputs (JSON), memory for active logits
• **Batch processing**: Dynamic batch sizing based on available GPU memory with torch.cuda.max_memory_allocated() monitoring
• **GPU optimization**: Flash Attention 2, torch.compile() for inference optimization, mixed precision (fp16/bf16)
• **Checkpoint system**: Save state every 100 questions with automatic recovery from interruption
• **Deterministic execution**: Fixed random seeds, sorted data structures, controlled model initialization
• **Configuration management**: JSON configs for all parameters with CLI overrides (stdlib-only)
• **Logging**: Structured logging with levels (DEBUG may include full model outputs in internal runs; public logs are aggregated per policy)
• **Data formats**: Native support for JSONL and CSV, extensible to other formats via adapters
• **Security**: Two-tier artifact policy; no plaintext question content in public artifacts; SHA-256 hashing with private and public salts; configurable redaction levels
• **Container support**: Dockerfile for reproducible environment (not required for MVP but good for distribution)
• **Result formats**: JSON for machine parsing, Markdown for human reports, optional CSV export
• **Progress tracking**: tqdm for CLI progress bars with ETA and throughput metrics
• **Error handling**: Graceful degradation - if one model fails, continue with others
• **Memory management**: Explicit GPU cache clearing between models, memory mapping for large datasets
• **Parallel execution**: concurrent.futures for CPU-bound statistical analyses
• **Documentation**: Docstrings for all public methods, README with examples, architecture diagram
• **Radical transparency principle**: Every algorithm implemented in readable Python with extensive comments
• **Verification-first design**: Judges should be able to audit our statistical methods in 5 minutes
• **"Show Your Work" documentation**: Each statistical test includes mathematical formulation in docstring
• **Security-conscious implementation**: Hash-based anonymization visible in code (shows we understand info-hazards)
• **Fail-gracefully architecture**: If GPU fails, statistical analysis still runs (shows robustness thinking)
• **One-command reproducibility**: Single command replicates all results (judges can verify during review)
• **Educational code style**: Variable names and structure that teach the algorithm (shows mastery)
• **Conservative claims**: Under-promise in docs, over-deliver in demo (builds trust)
• **Appendix honesty**: Security Considerations section that genuinely discusses limitations (shows maturity)
