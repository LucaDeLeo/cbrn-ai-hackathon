# Component Architecture

## Core Pipeline Components

### 1. Data Loader Module
**Responsibility:** Load and normalize input data from various formats

**Key Interfaces:**
- `load_dataset(path: str, format: str) -> Dataset`
- `validate_schema(data: Dict) -> bool`
- `normalize_format(data: Any) -> Dataset`

**Dependencies:** None (uses built-in json/csv)

**Implementation:**
```python
@dataclass
class Question:
    id: str  # SHA-256 hash
    choices: List[str]
    answer_index: int
    metadata: Dict[str, Any]
```

### 1a. Model Diversity Policy & Checker
**Responsibility:** Enforce heterogeneous model families per PRD

**Policy:**
- Require 2–3 models from distinct open-source families (e.g., Llama/Meta, OLMo/AllenAI, Mistral, Qwen, Phi). Closed APIs are out-of-scope for MVP to preserve determinism, logging, and safety posture. Select local models that maximize architectural diversity and document the selection in the audit log.
- At run start, compute family signatures from model identifiers/configs and fail-fast or warn if insufficient diversity.

**Key Interfaces:**
- `detect_model_family(model_name_or_path: str) -> Literal['meta','allenai','mistral','qwen','phi','other']`
- `validate_diversity(families: List[str], required: int = 2) -> None`

### 2. Cache Manager
**Responsibility:** Multi-level caching with checkpoint recovery

**Key Interfaces:**
- `cache_model_output(question_id: str, model_id: str, output: Any)`
- `get_cached_output(question_id: str, model_id: str) -> Optional[Any]`
- `save_checkpoint(state: PipelineState)`
- `recover_from_checkpoint() -> Optional[PipelineState]`

**Dependencies:** SQLite3 for metadata, JSON for outputs

**Implementation:**
- SQLite: Question metadata, processing status
- JSON files: Model outputs (one file per 100 questions)
- Memory: Active batch logits (cleared between models)
 - Content policy: No plaintext question or prompt content in cache; keys are hashed IDs

### 3. Statistical Analysis Engine
**Responsibility:** Pure NumPy implementations of all statistical tests

**Key Interfaces:**
- `calculate_bootstrap_ci(data: np.ndarray, statistic: Callable, n_iterations: int = 10000) -> Tuple[float, float]`
- `chi_square_test(observed: np.ndarray, expected: np.ndarray) -> ChiSquareResult`
- `detect_position_bias(questions: List[Question]) -> BiasReport`
- `analyze_lexical_patterns(questions: List[Question]) -> PatternReport`

**Dependencies:** NumPy only

**Mathematical Documentation:**
```python
def bootstrap_ci(data, statistic, n_iterations=10000, confidence=0.95):
    """
    Bootstrap Confidence Interval (Percentile Method)
    
    Mathematical formulation:
    Given data X = {x₁, x₂, ..., xₙ} and statistic θ
    1. Generate B bootstrap samples X*ᵢ by sampling with replacement
    2. Calculate θ*ᵢ = statistic(X*ᵢ) for i = 1...B
    3. CI = [θ*_(α/2), θ*_(1-α/2)] where α = 1 - confidence
    
    Implementation uses vectorized NumPy operations for efficiency.
    """
    n = len(data)
    bootstrap_stats = np.empty(n_iterations)
    
    for i in range(n_iterations):
        sample = data[np.random.choice(n, n, replace=True)]
        bootstrap_stats[i] = statistic(sample)
    
    lower = np.percentile(bootstrap_stats, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 + confidence) / 2 * 100)
    
    return lower, upper
```

**Performance Notes:**
- Bootstrap iterations configurable; default 10k, may auto-reduce for large datasets to meet runtime budgets (e.g., 2k–5k) while reporting the actual iterations used in the report.
- Vectorize where feasible and reuse RNG states; parallelize independent tests via the thread pool.

### 4. Model Consensus Detector
**Responsibility:** Implement Deep Ignorance §D.4 choices-only evaluation

**Key Interfaces:**
- `evaluate_choices_only(model: AutoModelForCausalLM, choices: List[str]) -> int`
- `calculate_consensus(predictions: List[int]) -> ConsensusResult`
- `identify_exploitable(consensus_results: List[ConsensusResult]) -> List[str]`
 - `store_rendered_prompt(question_id: str, model_id: str, prompt: str, internal: bool) -> None`

**Dependencies:** PyTorch, Transformers

**Memory Management:**
```python
class SequentialModelLoader:
    def load_model(self, model_name: str):
        # Clear any existing model
        if self.current_model:
            del self.current_model
            torch.cuda.empty_cache()
        
        # Load with appropriate precision
        self.current_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=self.quantize
        )
```

**Choices-Only Scoring (Methodology):**
- Construct prompts that present answer options without the stem; template per architecture with minimal special tokens.
- Score each option with a single forward pass per option using token log-probabilities over the option string (no generation), then select argmax of length-normalized logprob: `score = sum(logp(tokens)) / len(tokens)`.
- Batch options across questions to maximize GPU utilization; cache per-question per-model scores (not raw text).
 - Store exact rendered prompts internally (access-controlled) for reproducibility; publish only templates and rendering logic in the Prompts Appendix.

```python
@torch.inference_mode()
def score_option_texts(model, tokenizer, prefix: str, options: List[str]) -> List[float]:
    """Return length-normalized logprob for each option given a prefix.
    Stores no plaintext in cache; call-site maps back to hashed IDs.
    """
    device = model.device
    scores = []
    for opt in options:
        text = prefix + opt
        enc = tokenizer(text, return_tensors='pt').to(device)
        # labels equal to input_ids to compute token-wise NLL
        labels = enc.input_ids.clone()
        # Mask prefix tokens so only option tokens contribute
        prefix_len = len(tokenizer(prefix)['input_ids'])
        labels[:, :prefix_len] = -100
        out = model(**enc, labels=labels)
        # Negative loss is mean logprob over unmasked tokens; recover sum/len
        # out.loss = -mean(logprob)
        unmasked = (labels != -100).sum().item()
        avg_logprob = -out.loss.item()
        scores.append(avg_logprob)  # already mean over option tokens
    return scores
```

### 4b. Cloze Scoring Module (Verified)
**Responsibility:** Convert MCQA to cloze form and compute length-normalized logprob per option

**Key Interfaces:**
- `to_cloze(question: Question, template: str) -> List[str]  # per-option text`
- `score_options(model, tokenizer, option_texts: List[str]) -> List[float]  # length-normalized`

**Methodology:**
- Use deterministic templates per domain; compute log-prob of the option completion given the cloze context via one forward pass per option; normalize by token count to mitigate length bias.
- Limit to configured subset size (e.g., 500 items) to meet compute budget.

### 5. Pipeline Orchestrator
**Responsibility:** Coordinate component execution with fail-graceful behavior

**Key Interfaces:**
- `run_pipeline(config: PipelineConfig) -> PipelineResult`
- `handle_component_failure(component: str, error: Exception) -> bool`
- `generate_reports(results: PipelineResult) -> None`

**Dependencies:** All other components

**Fail-Graceful Logic:**
```python
def run_analysis(self, questions: List[Question]) -> AnalysisResult:
    results = AnalysisResult()

    # Validate model family diversity before GPU work
    try:
        families = [detect_model_family(m) for m in self.config.models]
        validate_diversity(families, required=2)
    except Exception as e:
        self.logger.warning(f"Diversity validation: {e}")
    
    # Always run statistical analysis (CPU-only, reliable)
    try:
        results.statistical = self.stat_engine.analyze(questions)
    except Exception as e:
        self.logger.error(f"Statistical analysis failed: {e}")
        results.statistical = None
    
    # Try GPU-based analyses if available
    if torch.cuda.is_available():
        try:
            results.consensus = self.consensus_detector.analyze(questions)
        except Exception as e:
            self.logger.warning(f"Consensus detection failed: {e}, continuing...")
            results.consensus = None
    
    # Continue with whatever succeeded
    return results
```

### 6. Permutation Sensitivity Tester
**Responsibility:** Measure answer stability to random option order changes

**Key Interfaces:**
- `permute_once(choices: List[str], seed: int) -> Tuple[List[str], PermutationMap]`
- `evaluate_permutation_effect(question: Question, models: List[str]) -> PermutationResult`

**Methodology:**
- For each question, perform 0–1 random permutation of options (controlled by seed), re-run choices-only (or cloze) scoring, and compute delta in predictions/scores.
- Aggregate permutation sensitivity metrics with bootstrap CIs and include in bias report.

### 7. CI‑Backed Filtering & Reporting
**Responsibility:** Produce robust subsets using consensus + statistical tests with quantified uncertainty

**Decision Logic:**
- Flag exploitable items via configurable consensus strategies (unanimous/majority/weighted) on choices-only results, plus corroborating heuristics (e.g., longest-answer) and permutation sensitivity.
- Compute removal rate and key statistics with bootstrap 95% CIs (percentile method). Include per-question decision rationale (e.g., `consensus=3/3`, `longest_answer=match`, `perm_delta>τ`).

**Outputs:**
- Internal (access-controlled):
  - `robust_subset.jsonl` with private-salt hashed IDs and keep/remove decisions; no raw text
  - `bias_report.internal.json` including CIs, p-values, effect sizes, permutation metrics, and run metadata
  - `audit_log.jsonl` with IDs, timestamps, seeds, model families, decision rationale
- Public (sanitized via ReleaseGate):
  - `bias_report.public.json` with aggregate metrics and CIs; no per-item details beyond the sanitized calibration subset
  - `calibration_subset_ids.txt` (public-salt IDs only)
  - `prompts_appendix.md` (templates and rendering logic; no hazardous content)
  - `reproducibility_bundle.json` (commands, seeds, versions, model SHAs, checksums)
