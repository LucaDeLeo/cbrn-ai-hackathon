# RobustCBRN Eval Product Requirements Document (PRD)

## Goals and Background Context

### Goals
• Deliver a practical, validated pipeline that identifies and removes 25-35% of exploitable MCQA questions using proven robustification techniques
• Reduce evaluation variance from 5-10% to <2% through statistical validation and multiple benchmark variants
• Provide transparent, reproducible bias detection methods with 95% confidence intervals for all filtering decisions
• Generate both robust MCQA subsets and verified cloze format variants that together provide more reliable capability assessment
• Create open-source infrastructure that enables the AI safety community to validate evaluation claims with confidence
• Demonstrate 15-25% score reduction on robust vs original benchmarks, revealing true model capabilities vs statistical artifacts
• Process 1,000-3,000 questions within 4 hours using <$300 in compute resources
• Establish foundation for future comprehensive bias detection and evaluation standards

### Background Context

Current AI safety evaluations for CBRN risks suffer from systematic vulnerabilities where models can achieve high scores through pattern exploitation rather than genuine knowledge. The WMDP benchmark reveals that 31.81% of Bio questions are answerable without reading the question stem, and simple heuristics like selecting the longest answer achieve 46% accuracy. This evaluation crisis is recognized at the highest levels - NIST lists CBRN capabilities as the first GAI-specific risk, while regulatory reports bluntly state "no proper safety evaluation tools exist yet."

RobustCBRN Eval implements the validated Deep Ignorance §D.4 consensus detection method combined with verified cloze scoring and statistical heuristics to address these critical gaps. By applying proven debiasing techniques with novel enhancements within realistic hackathon constraints, we provide academic researchers and safety evaluation organizations with the reproducible, trustworthy assessment methods they urgently need before flawed evaluations lead to catastrophic misjudgments about AI safety.

### Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2025-01-12 | v1.0 | Initial PRD creation based on Project Brief | PM |

## Requirements

### Functional

• **FR1**: The system shall detect exploitable MCQA questions using multi-model consensus voting with 2-3 diverse open-source model families (e.g., Meta/Llama, Mistral, Qwen)
• **FR2**: The system shall implement choices-only screening where models answer questions without seeing the question stem
• **FR3**: The system shall calculate statistical heuristics including longest-answer selection, position bias, and lexical pattern detection without requiring model inference
• **FR4**: The system shall generate verified cloze format variants using length-normalized log-probabilities with one forward pass per answer option
• **FR5**: The system shall cache all model outputs to disk with checkpoint recovery for interrupted processing (internal artifacts may include plaintext; public artifacts must follow the Artifacts Release Policy)
• **FR6**: The system shall process JSON format with fields: question, choices (list), answer (index/letter) and CSV with standard columns
• **FR7**: The system shall output robust question subsets removing 25-35% flagged as exploitable with confidence intervals
• **FR8**: The system shall provide SHA-256 hashing with salts for question IDs (private per-project salt for internal artifacts; fixed public salt for sanitized subset)
• **FR9**: The system shall generate comprehensive bias reports showing statistical confidence intervals via bootstrap over items
• **FR10**: The system shall support batch processing of up to 10,000 questions per run with configurable batch sizes
• **FR11**: The system shall implement permutation sensitivity testing with 0-1 random reorders per item
• **FR12**: The system shall maintain <2% variance across three independent seeded runs on 1,000-item samples

• **FR13**: The system shall store and publish exact prompt templates and rendered prompts for each evaluation mode (choices-only, cloze) in a Prompts Appendix with versioned hashes

• **FR14**: The system shall implement a two-tier artifacts model (internal vs public) and enforce it at report time per the Artifacts Release Policy

### Non Functional

• **NFR1**: Processing throughput must complete 1,000-3,000 questions within 4 hours on Lambda GPU infrastructure
• **NFR2**: Total compute cost must remain under $300 for full WMDP-Bio evaluation (3,000 questions)
• **NFR3**: System must run on Linux (Ubuntu 20.04+) with CUDA 11.8+ and PyTorch 2.0+
• **NFR4**: Minimum GPU requirement of 24GB VRAM with optimal performance on 40GB+ (A100/H100)
• **NFR5**: Memory usage must not exceed 64GB RAM for caching and parallel processing
• **NFR6**: Storage must handle 100–150GB for MVP (models + caches); 500GB recommended for larger runs and extended caching
• **NFR7**: All processing must be deterministic with fixed seeds and ordered execution; enable PyTorch deterministic algorithms and cuDNN deterministic settings; document all env/runtime flags
• **NFR8**: No plaintext question content in public outputs; internal runs may temporarily store plaintext under access controls; public artifacts must follow the Artifacts Release Policy
• **NFR9**: System must support fallback to 1–2 models or quantized versions if compute-constrained; quantized fallback may require a minimal extra dependency (e.g., bitsandbytes)
• **NFR10**: CLI must be primary interface with optional Streamlit demo for visualization
• **NFR11**: Code must follow safety-aware release practices with redaction helpers and release checklist
• **NFR12**: System must maintain audit trail logging every decision for full reproducibility

• **NFR13**: Two-tier artifact policy (internal vs public) must be enforced with automated redaction checks prior to publishing

### Implementation Notes
**Potential Computational Constraints to Verify During Development:**
- Cloze scoring (FR4) multiplies compute by 4x - may need to validate on subset if models are larger than expected
- Multi-model consensus (FR1) memory requirements depend on actual model choices and quantization
- Permutation testing (FR11) doubles inference - could use stratified sampling if needed
- The 25-35% removal target (FR7) is based on WMDP-Bio findings and may vary by benchmark
- Actual compute budget utilization will depend on final model selection and batch optimization

*These constraints will be validated during architecture design and initial implementation. Requirements remain as specified pending empirical testing.*

## User Interface Design Goals

### Overall UX Vision
The interface prioritizes clarity and efficiency for technical users (AI safety researchers and evaluation teams). The primary CLI provides transparent progress tracking with real-time statistics, while the optional Streamlit demo offers visual exploration of bias detection results. All interactions emphasize reproducibility and auditability over aesthetic polish.

### Key Interaction Paradigms
Command-line driven batch processing with clear progress indicators, checkpoint recovery for interrupted runs, and comprehensive logging. The optional web interface provides read-only visualization of results with interactive filtering and statistical exploration. All outputs include confidence intervals and methodological transparency.

### Core Screens and Views
• **CLI Progress Display**: Real-time processing status with question count, model consensus, and running statistics
• **Results Summary Terminal Output**: Comprehensive bias report with exploitability percentages, statistical metrics, and confidence intervals
• **Streamlit Dashboard** (optional): Interactive visualization of removed vs retained questions, bias distribution charts, and model agreement heatmaps
• **Audit Log Viewer**: Detailed trace of all decisions for reproducibility verification

### Accessibility: None
Technical tool for expert users; standard terminal accessibility applies

### Branding
No specific branding requirements - focus on clear, professional presentation of technical data

### Target Device and Platforms: Desktop Only
Linux command-line environment (Ubuntu 20.04+) with optional web browser for Streamlit dashboard

## Technical Assumptions

### Repository Structure: Monorepo
Single repository containing all components - core robustification engine, statistical analysis modules, model inference pipelines, CLI interface, and optional Streamlit demo. Clear module separation with `src/models/`, `src/analysis/`, `src/utils/`, `tests/`, and `configs/` directories.

### Service Architecture
**CRITICAL DECISION**: Modular pipeline architecture with pluggable components. Each robustification technique (consensus detection, cloze scoring, statistical heuristics) implemented as independent modules that can be enabled/disabled/combined. Pipeline stages: Data Loading → Preprocessing → Model Inference (cached) → Analysis → Filtering → Report Generation. NOT microservices - single process with multi-threading for parallel model inference.

### Testing Requirements
**CRITICAL DECISION**: Comprehensive testing pyramid given evaluation accuracy is critical:
- **Unit tests**: Every statistical function, bias detection algorithm, and data transformation
- **Integration tests**: End-to-end pipeline with small synthetic datasets
- **Regression tests**: Ensure deterministic results across runs (critical for reproducibility claims)
- **Performance tests**: Validate <4 hour processing time on reference hardware
- **Validation tests**: Agreement with 50-item human-labeled calibration subset
- **Manual testing utilities**: Scripts to spot-check individual questions and debug edge cases

### Additional Technical Assumptions and Requests

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
• **Configuration management**: YAML configs for all parameters with CLI overrides
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

### Models Shortlist (MVP)
- Llama 3.1 8B Instruct (`meta-llama/Llama-3.1-8B-Instruct`)
  - VRAM (approx, batch=1, short prompts): bf16 ~18–22GB; int8 ~9–11GB; 4‑bit ~6–8GB
  - Notes: Strong open baseline; good tokenizer support for English; stable logits for cloze scoring
- Mistral 7B Instruct v0.3 (`mistralai/Mistral-7B-Instruct-v0.3`)
  - VRAM: bf16 ~14–16GB; int8 ~8–10GB; 4‑bit ~5–6GB
  - Notes: Efficient inference; robust with short prompts; widely used open model
- Qwen2.5 7B Instruct (`Qwen/Qwen2.5-7B-Instruct`)
  - VRAM: bf16 ~14–16GB; int8 ~8–10GB; 4‑bit ~5–6GB
  - Notes: Strong multiple‑choice behavior; good alternative architecture for diversity

General guidance
- Hardware: A100 40GB on Lambda GPU cloud (target); use bf16 where available for speed/stability
- Quantization: permit `bitsandbytes` for `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_use_double_quant=True`, compute dtype fp16/bf16
- Attention: enable Flash‑Attention 2 if available; set `attn_implementation="flash_attention_2"`
- Inference hygiene: `model.eval()`, `torch.inference_mode()`; disable grad
- Batch sizing: dynamic based on `torch.cuda.mem_get_info()`; keep headroom for logits
- Diversity: ensure pairwise Jaccard agreement on choices‑only predictions ≤85% on a 200‑item subset (warn otherwise)

Notes
- Footprints vary with context length and batch size; the above estimates assume short choices‑only/cloze prompts typical for MCQA.
- If memory constrained, start with 4‑bit quantized loads and reduce `--max-items` for cloze runs.

## Epic List

**Epic 1: Foundation & Core Pipeline** - *Establish minimal infrastructure with working data flow and first bias detection*
Get the basic pipeline running end-to-end with simple longest-answer heuristic to prove the architecture works. This gives us a working demo within the first 6 hours.

**Epic 2: Statistical Analysis Engine** - *Implement transparent bias detection algorithms with bootstrap confidence intervals*
Build our "from-scratch" statistical implementations that judges can read and trust. This is our differentiation - every algorithm transparent and documented.

**Epic 3: Model Consensus Detection** - *Add multi-model consensus voting for exploitable question identification*
Implement the Deep Ignorance §D.4 approach with 2-3 models. This is our primary technical contribution and highest-impact feature.

**Epic 4: Robust Splitting & Reporting** - *Generate validated subsets with comprehensive bias documentation*
Create the final outputs that users actually need - robust question sets, detailed reports, and security-conscious result presentation.

## Epic 1: Foundation & Core Pipeline

**Goal**: Establish the minimal viable pipeline that reads benchmarks, processes questions through a simple heuristic, and outputs results. This creates the skeleton that all other features plug into while delivering a working demo within 6 hours.

### Story 1.1: Project Setup & Data Pipeline

**As a** developer,
**I want** a clean project structure with data loading capabilities,
**so that** the team can start parallel development immediately.

**Acceptance Criteria:**
1. Repository initialized with structure: `src/`, `tests/`, `configs/`, `data/`, `cache/`, `results/`
2. Minimal requirements.txt with only: torch, transformers, numpy, tqdm
3. Data loader handles both JSONL and CSV formats with consistent internal representation
4. Configuration system using Python dataclasses (no external deps) with YAML serialization
5. Hash-based question ID generation working with configurable salt
6. Basic logging setup writing to both console and file
7. Successfully loads and parses WMDP-Bio sample (first 100 questions)
8. Determinism controls set and documented: fixed Python/NumPy/Torch seeds; `torch.use_deterministic_algorithms(True)`; cuDNN deterministic enabled and benchmarking disabled; sorted iteration order

### Story 1.2: Simple Heuristic Implementation

**As a** researcher,
**I want** to see the longest-answer baseline working,
**so that** we validate the infrastructure and have immediate results.

**Acceptance Criteria:**
1. Longest-answer selector implemented in <20 lines of readable code
2. Processes 1,000 questions in <10 seconds (no model needed)
3. Outputs accuracy score matching reported 46% baseline on WMDP-Bio
4. Results saved to JSON with timestamp and configuration
5. Progress bar shows questions processed with ETA
6. Memory usage stays under 1GB for 10,000 questions

### Story 1.3: CLI Interface Foundation

**As a** user,
**I want** a simple command-line interface,
**so that** I can run analyses with different parameters.

**Acceptance Criteria:**
1. Basic argparse CLI with --input, --output, --config flags
2. --dry-run flag that validates input without processing
3. --verbose flag that shows detailed progress
4. Graceful error handling with helpful messages
5. Exit codes: 0=success, 1=error, 2=partial completion
6. Help text includes concrete usage examples
7. Additional flags: `--models`, `--max-items`, `--time-limit`, `--budget`, `--public-report {on,off}`

## Epic 2: Statistical Analysis Engine

**Goal**: Implement transparent, from-scratch statistical methods that reveal benchmark biases without requiring model inference. Every implementation must be readable and mathematically documented.

### Story 2.1: Position Bias Analysis

**As a** researcher,
**I want** to detect position preference patterns,
**so that** I can identify if models systematically favor certain option positions.

**Acceptance Criteria:**
1. Calculate frequency distribution of correct answers across positions A,B,C,D
2. Chi-square test for uniform distribution (implemented from scratch with NumPy)
3. Identify questions where correct answer position is predictive
4. Generate position swap variants (A→D, B→C, etc.) for testing
5. Output includes observed vs expected frequencies with p-values
6. Docstring includes mathematical formula for chi-square calculation
7. Automated correctness checks validate option reindexing and label remapping after permutations (unit test with checksums)

### Story 2.2: Bootstrap Confidence Intervals

**As a** researcher,
**I want** bootstrap confidence intervals for all statistics,
**so that** I can quantify uncertainty in bias measurements.

**Acceptance Criteria:**
1. Bootstrap CI implementation in pure NumPy (<50 lines)
2. Support for any statistic function (mean, median, custom)
3. Configurable iterations (default 10,000) and confidence level (default 95%)
4. BCa (bias-corrected accelerated) variant attempted if time allows
5. Performance target: ~10,000 iterations on 3,000 items in seconds on CPU; provide adaptive iteration mode that early-stops when CI width stabilizes; report wall-clock
6. Unit tests verify against known statistical examples
7. Clear documentation of percentile vs BCa methods

### Story 2.3: Lexical Pattern Detection

**As a** researcher,
**I want** to identify textual patterns in exploitable questions,
**so that** I can understand what makes questions vulnerable.

**Acceptance Criteria:**
1. Detect common phrases in correct vs incorrect answers
2. Length distribution analysis with statistical significance
3. Identify "all of the above" / "none of the above" patterns
4. Technical term density comparison
5. Results include top-10 discriminative patterns with effect sizes
6. No external NLP libraries - pure string operations

### Story 2.4: Statistical Battery Integration

**As a** researcher,
**I want** all statistical tests run as a suite,
**so that** I get comprehensive bias analysis in one pass.

**Acceptance Criteria:**
1. Single function runs all implemented statistical tests
2. Consistent output format across all tests (JSON schema)
3. Summary statistics with traffic-light indicators (green/yellow/red)
4. Parallel execution of independent tests
5. Total runtime <30 seconds for 3,000 questions
6. Option to run specific tests only

## Epic 3: Model Consensus Detection

**Goal**: Implement the Deep Ignorance §D.4 consensus approach using 2-3 diverse models to identify questions answerable without stems. This is our primary technical contribution.

### Story 3.1: Model Loading & Caching Infrastructure

**As a** developer,
**I want** efficient model management,
**so that** we can run multiple models within memory constraints.

**Acceptance Criteria:**
1. Dynamic model loading with automatic cache directory management
2. Support for loading models sequentially to manage memory
3. Automatic fallback to quantized versions if memory constrained (optional minimal dependency permitted, e.g., bitsandbytes)
4. Model diversity check (warns if models too similar)
5. Clear logging of model loading progress and memory usage
6. Checkpoint system saves model outputs every 100 questions
7. Diversity metric: compute pairwise Jaccard agreement on choices-only predictions over a 200-item random subset; warn if any pair >85% agreement

### Story 3.2: Choices-Only Evaluation

**As a** researcher,
**I want** models to answer questions using only answer choices,
**so that** I can identify exploitable questions per Deep Ignorance §D.4.

**Acceptance Criteria:**
1. Format choices-only prompts correctly for each model architecture
2. Handle varying model prompt formats (Llama vs Mistral vs Qwen)
3. Batch processing with dynamic batch size based on available memory
4. Process 1,000 questions in <2 hours with 2 models
5. Cache all model outputs to disk in JSON format
6. Include model confidence scores when available
7. Store exact rendered prompts for each item and model in the internal artifact store; publish sanitized prompt templates in the Prompts Appendix

### Story 3.3: Consensus Voting Mechanism

**As a** researcher,
**I want** configurable consensus strategies,
**so that** I can balance sensitivity vs specificity in detection.

**Acceptance Criteria:**
1. Unanimous consensus (all models agree) implementation
2. Majority voting (2/3 models) implementation
3. Weighted voting based on model confidence defined as length-normalized log-probability scores computed via Transformers logit access; standardize across models by z-scoring within-model before combining
4. Agreement matrix showing model-by-model consensus
5. Identify questions with high disagreement for manual review
6. Output includes per-question voting details

### Story 3.4: Cloze Scoring Variant

**As a** researcher,
**I want** to compare MCQA vs cloze format performance,
**so that** I can quantify format-induced bias.

**Acceptance Criteria:**
1. Convert MCQA to cloze format with proper templates
2. Access log probabilities via HuggingFace Transformers
3. Length-normalized scoring for fair comparison
4. Process subset of 500 questions (compute budget limit); may expand if runtime allows on Lambda GPU within budget
5. Report score differential with confidence intervals
6. Document any residual biases in cloze format

## Epic 4: Robust Splitting & Reporting

**Goal**: Generate final outputs that the safety community can actually use - validated question subsets, comprehensive reports, and security-conscious documentation.

### Story 4.1: Intelligent Question Filtering

**As a** researcher,
**I want** multiple filtering strategies,
**so that** I can choose the appropriate trade-off for my use case.

**Acceptance Criteria:**
1. Conservative mode (high confidence, fewer removals)
2. Balanced mode (default, 25-35% removal target)
3. Aggressive mode (remove anything suspicious)
4. Preserve topic/difficulty balance in filtered sets
5. Generate both "robust" and "excluded" question lists; publish excluded IDs only for the sanitized calibration subset using the public salt; keep full excluded lists internal
6. Include filtering rationale for each removed question

### Story 4.2: Comprehensive Report Generation

**As a** user,
**I want** detailed analysis reports,
**so that** I can understand and trust the robustification results.

**Acceptance Criteria:**
1. Executive summary with key statistics and confidence intervals
2. Detailed methodology section explaining each technique
3. Results broken down by detection method
4. Comparison table: original vs robust benchmark scores
5. Markdown format for readability, JSON for machine parsing
6. Include timestamp, version, configuration, and reproducibility info
7. Include a Prompts Appendix with exact templates, rendering logic, and versioned hashes; redact any sensitive content per policy (see `docs/prompts_appendix.md`)
8. Provide a Reproducibility Bundle: commands, seeds, package versions, model SHAs, and artifact checksums
9. Release bundle must satisfy the `docs/release_checklist.md`

### Story 4.3: Security & Safety Measures

**As a** safety researcher,
**I want** responsible handling of sensitive content,
**so that** results don't create new info-hazards.

**Acceptance Criteria:**
1. No plaintext question content in public outputs
2. SHA-256 hashes for question identification (two salts: private internal salt; public fixed salt for sanitized subset)
3. Aggregate statistics only in public outputs (no per-item details outside sanitized subset)
4. Security considerations appendix documenting limitations
5. Optional local-only mode for maximum security
6. Redaction helpers for sensitive patterns
7. Artifacts Release Policy implemented with automated redaction checks on publish
8. Public release must pass the `docs/release_checklist.md`

### Story 4.5: Safety‑Relevance Validation (Sanitized Subset)

**As a** safety researcher,
**I want** a small, sanitized conceptual vs procedural subset,
**so that** I can demonstrate that robustified scoring suppresses spurious gains more on procedural-like items without exposing sensitive content.

**Acceptance Criteria:**
1. Construct a 50-item non-actionable, sanitized subset partitioned into conceptual vs procedural categories with labeling protocol documented
2. Run the full pipeline on this subset; report robust vs original score deltas with 95% CIs and topic breakdowns
3. Publish only hashed IDs using a public salt, aggregate metrics, and safe exemplar snippets if applicable
4. Include caveats that results are a proxy for safety-relevance, not a deployment clearance

### Story 4.4: Demo & Documentation Package

**As a** judge,
**I want** clear demonstration of the tool's capabilities,
**so that** I can evaluate its effectiveness and usability.

**Acceptance Criteria:**
1. One-line command that runs full pipeline on sample data
2. README with installation, usage, and methodology
3. Example output on WMDP-Bio subset (sanitized)
4. Performance benchmarks (time, memory, cost)
5. Video script prepared (even if not recorded)
6. Submission report following hackathon template
7. Include link to `docs/release_checklist.md` and `docs/prompts_appendix.md` in repository documentation

## Artifacts Release Policy

To reconcile transparency with safety, we adopt a two-tier artifact model:

| Artifact Class | Internal (private) | Public (release) |
| --- | --- | --- |
| Raw question text | Allowed (access-controlled) | Not allowed |
| Model outputs/logs | Allowed (full, DEBUG) | Aggregated statistics only |
| Item IDs | Allowed with private salt | Allowed for sanitized subset with public salt |
| Prompt templates | Exact, rendered per item | Exact templates and rendering logic; no raw hazardous content |
| Excluded item lists | Full lists allowed | Only sanitized subset IDs (public salt) |
| Reports | Full detail | Redacted aggregate metrics with CIs |

Automated redaction checks must pass before publishing public artifacts. The Prompts Appendix includes all prompt templates and rendering logic with versioned hashes.

## Checklist Results Report

### Executive Summary
- **Overall PRD completeness**: 85%
- **MVP scope appropriateness**: Just Right (focused on core robustification techniques)
- **Readiness for architecture phase**: Nearly Ready (minor gaps in operational details)
- **Most critical gaps**: Limited user research documentation, operational requirements need expansion

### Category Analysis

| Category                         | Status  | Critical Issues |
| -------------------------------- | ------- | --------------- |
| 1. Problem Definition & Context  | PASS    | None - Excellent problem articulation with quantified impact |
| 2. MVP Scope Definition          | PASS    | None - Clear MVP with fallback strategies |
| 3. User Experience Requirements  | PARTIAL | CLI-focused, limited end-user journey mapping |
| 4. Functional Requirements       | PASS    | None - Comprehensive with clear acceptance criteria |
| 5. Non-Functional Requirements   | PASS    | None - Detailed performance and security requirements |
| 6. Epic & Story Structure        | PASS    | None - Well-structured with parallel execution plan |
| 7. Technical Guidance            | PASS    | None - Radical minimalism approach well-documented |
| 8. Cross-Functional Requirements | PARTIAL | Data retention and operational monitoring underspecified |
| 9. Clarity & Communication       | PASS    | None - Clear language with strategic reasoning |

### Key Strengths
- Clear problem statement with quantified impact (31.81% exploitable questions)
- Radical dependency minimization strategy reduces implementation risk
- Parallel development plan maximizes 4-person team efficiency
- Comprehensive fallback strategies at every level
- Security-conscious design with hash-based anonymization

### Areas for Improvement
- **HIGH**: Add operational monitoring and deployment specifications
- **MEDIUM**: Finalize model selection and quantization plan (note: minimal dependency permitted)
- **LOW**: Include architecture and artifact-flow diagrams (internal vs public)

### MVP Validation
The MVP scope is appropriately sized for a 48-hour hackathon with clear priorities:
1. Core statistical analysis (CPU-only fallback)
2. Model consensus detection (primary contribution)
3. Optional enhancements (cloze scoring, visualization)

### Technical Readiness
Technical constraints are well-defined with explicit compute budgets, memory requirements, and fallback strategies. The radical transparency approach with minimal dependencies is a strategic advantage.

## Next Steps

### UX Expert Prompt

Given that RobustCBRN Eval is primarily a CLI-based evaluation tool for AI safety researchers, the UX focus should be on clarity of output and ease of use rather than traditional interface design. Please review the PRD at docs/prd.md and provide guidance on:

1. **CLI Output Design**: Optimal formatting for progress indicators, statistical reports, and error messages that researchers can easily interpret
2. **Result Visualization**: If time permits, design considerations for the optional Streamlit dashboard showing bias detection results
3. **Documentation UX**: Structure for README and help text that enables researchers to quickly understand and trust the tool
4. **Reproducibility Flow**: User journey for researchers wanting to verify and reproduce results

Focus on functional clarity over aesthetics, as our users are technical researchers who value transparency and reproducibility above visual polish.

### Architect Prompt

Please review the RobustCBRN Eval PRD at docs/prd.md and create a technical architecture that implements our radical dependency minimization strategy using open-source model families only. Key architectural considerations:

1. **Modular Pipeline**: Design pluggable components for consensus detection, statistical analysis, and cloze scoring that can fail independently
2. **Caching Strategy**: Multi-level caching system (SQLite metadata, JSON model outputs, memory logits) with checkpoint recovery and two-tier artifact handling (internal vs public)
3. **Memory Management**: Sequential model loading strategy to handle 2-3 7B models on single A100 GPU
4. **Statistical Implementations**: Pure NumPy implementations of bootstrap CI, chi-square tests, and bias metrics (no scipy)
5. **Security Design**: Hash-based anonymization for sensitive content with configurable redaction levels; two salts (internal/private and public for sanitized subset)
6. **Parallel Execution**: Thread pool for statistical analyses, GPU optimization for model inference
7. **Fail-Graceful Architecture**: Ensure statistical analysis works even if GPU fails

Prioritize code readability and auditability - judges should understand our algorithms in minutes. Use only torch, transformers, numpy, and tqdm as core dependencies. Structure should support 4-person parallel development with minimal integration overhead. Note: a minimal quantization dependency (e.g., bitsandbytes) is permitted if required for memory constraints.

### Measurement & Pre-Registration Plan

Before running large-scale experiments, pre-register key hypotheses as targets (not guarantees) and commit to reporting measured deltas with 95% CIs:
- Flagged fraction target 25–35%; report measured value with bootstrap CI
- Variance target <2% across three seeded runs on 1,000-item sample; report measured variance
- Robust vs original score delta target 15–25%; report measured delta with CI
- Longest-answer baseline degradation from ~46% to materially lower; report absolute delta with CI

Document exact commands, seeds, versions, model SHAs, and runtime/budget usage in the Reproducibility Bundle.
