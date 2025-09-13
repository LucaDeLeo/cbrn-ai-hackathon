# Epic 3: Model Consensus Detection

**Goal**: Implement the Deep Ignorance §D.4 consensus approach using 2-3 diverse models to identify questions answerable without stems. This is our primary technical contribution.

## Story 3.1: Model Loading & Caching Infrastructure

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

## Story 3.2: Choices-Only Evaluation

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

## Story 3.3: Consensus Voting Mechanism

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

## Story 3.4: Cloze Scoring Variant

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

## Story 3.5: Pilot Run & Budget Check

**As a** project lead,
**I want** a 200-item pilot with cost/runtime estimates,
**so that** we can adjust parameters to meet time and budget constraints.

**Acceptance Criteria:**
1. Run 200 items end-to-end (choices-only + stats; cloze optional) and capture wall-clock
2. Record GPU time and estimated cost using `--gpu-hourly-price`
3. Verify VRAM headroom and batch sizing stability
4. Adjust `--max-items`, batch sizes, and cloze subset if needed
5. Publish pilot metrics in the internal audit log

## Story 3.6: Consensus Threshold Ablation

**As a** researcher,
**I want** to compare consensus strategies and thresholds,
**so that** we pick a default with strong precision/recall on the calibration subset.

**Acceptance Criteria:**
1. Evaluate unanimous, majority, and weighted strategies on calibration subset
2. Report precision/recall and confusion breakdown with 95% CIs
3. Select default threshold per target operating point; document rationale
4. Include ablation table in the report (IDs only)
