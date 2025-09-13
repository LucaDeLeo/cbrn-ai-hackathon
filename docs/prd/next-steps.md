# Next Steps

## Measurement & Pre-Registration Plan

Before running large-scale experiments, pre-register key hypotheses as targets (not guarantees) and commit to reporting measured deltas with 95% CIs:
- Flagged fraction target 25–35%; report measured value with bootstrap CI
- Variance target <2% across three seeded runs on 1,000-item sample; report measured variance
- Robust vs original score delta target 15–25%; report measured delta with CI
- Longest-answer baseline degradation from ~46% to materially lower; report absolute delta with CI

Document exact commands, seeds, versions, model SHAs, and runtime/budget usage in the Reproducibility Bundle.
## Story 3.0: Model Shortlist & Prompt Template Finalization

**As a** team,
**I want** a locked shortlist of open-source models and finalized prompt templates,
**so that** inference is deterministic and reproducible across runs.

**Acceptance Criteria:**
1. Model shortlist confirmed (e.g., Llama‑3.1‑8B‑Instruct, Mistral‑7B‑v0.3, Qwen2.5‑7B‑Instruct)
2. VRAM profiling recorded for bf16/int8/4-bit per model on A100 40GB
3. Choices-only and cloze templates finalized (see Prompts Appendix)
4. Tokenization quirks documented and fixed (e.g., special tokens, whitespace)
5. Commit template hashes and model revisions to Reproducibility Bundle
