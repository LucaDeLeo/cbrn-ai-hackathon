# Implementation Sequencing

## Story Execution Order
Based on Product Owner validation, stories should be executed in the following order to minimize blockers and ensure proper dependencies:

**Phase 1: Foundation (Hour 0-2)**
1. Story 1.0 - Repository & Project Initialization (NEW)
2. Story 1.1 - Project Setup & Data Pipeline
3. Story 1.7 - Basic CI/CD Pipeline (NEW)
4. Story 1.2 - Simple Heuristic Implementation
5. Story 1.3 - CLI Interface Foundation
6. Story 1.6 - Test Skeleton & Sample Data
7. Story 1.4 - Schema Validation & Stratified Sampling
8. Story 1.5 - Cache Integrity & Config Hashing

**Phase 2: Statistical Engine (Hour 2-4)**
1. Story 2.2 - Bootstrap Confidence Intervals (foundational)
2. Story 2.1 - Position Bias Analysis
3. Story 2.3 - Lexical Pattern Detection
4. Story 2.4 - Statistical Battery Integration
5. Story 2.5 - Heuristic Degradation Measurement
6. Story 2.6 - Stratified Bootstrap & CI Width
7. Story 2.7 - Confidence-Aware Metrics

**Phase 3: Model Consensus (Hour 4-8)**
1. Story 3.0 - Model Registry & Download Strategy (NEW)
2. Story 3.5 - Pilot Run & Budget Check (validate early)
3. Story 3.1 - Model Loading & Caching Infrastructure
4. Story 3.2 - Choices-Only Evaluation
5. Story 3.3 - Consensus Voting Mechanism
6. Story 3.6 - Consensus Threshold Ablation
7. Story 3.4 - Cloze Scoring Variant (optional if time permits)

**Phase 4: Reporting & Release (Hour 8-10)**
1. Story 4.3 - Security & Safety Measures (security-first)
2. Story 4.0 - Balance Preservation Validator
3. Story 4.1 - Intelligent Question Filtering
4. Story 4.2 - Comprehensive Report Generation
5. Story 4.7 - Audit Log & Run Manifest
6. Story 4.5 - Safety-Relevance Validation
7. Story 4.6 - Release Gate Automation
8. Story 4.4 - Demo & Documentation Package

**Phase 5: Final Submission (Hour 10-12)**
1. Story 5.1 - Public Bundle Assembly
2. Story 5.2 - Reproducibility Bundle & Runbook
3. Story 5.3 - Demo Script & Video Outline
4. Story 5.4 - Judges Quick-Verify Guide

## Critical Dependencies
- Story 1.0 (repository init) blocks all others
- Story 2.2 (bootstrap CIs) required by most statistical analyses
- Story 3.0 (model registry) required before model loading
- Story 4.3 (security) should be implemented before any reporting

## Parallel Work Opportunities
- After Hour 2: Statistical engine and model preparation can proceed in parallel
- After Hour 6: Report generation can start while final analyses complete

## Models Shortlist (MVP)
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
