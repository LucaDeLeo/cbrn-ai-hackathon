# Requirements

## Functional

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

• **FR15**: The system shall implement confidence-aware scoring with configurable thresholds (t=0, 0.5, 0.75, 0.9) that penalizes incorrect answers proportionally (-t/(1-t)) while assigning zero penalty to abstentions ("I don't know", "uncertain")

• **FR16**: The system shall generate four benchmark variants with different confidence thresholds, enabling safety-aware evaluation that rewards appropriate abstention over confident hallucination

• **FR17**: The system shall analyze abstention patterns to identify knowledge gaps (questions with consistent abstention), overconfident questions (high error despite confidence), and calculate calibration metrics (Brier scores, calibration curves)

• **FR18**: The system shall integrate confidence scoring with Deep Ignorance detection, marking questions flagged by either method as exploitable, targeting 30-40% removal rate (exceeding the original 25-35% target)

## Non Functional

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

## Implementation Notes
**Potential Computational Constraints to Verify During Development:**
- Cloze scoring (FR4) multiplies compute by 4x - may need to validate on subset if models are larger than expected
- Multi-model consensus (FR1) memory requirements depend on actual model choices and quantization
- Permutation testing (FR11) doubles inference - could use stratified sampling if needed
- The 25-35% removal target (FR7) is based on WMDP-Bio findings and may vary by benchmark
- Confidence scoring (FR15-16) adds minimal overhead - only scoring function, no additional inference
- Abstention pattern analysis (FR17) is post-processing only, no GPU requirements
- Combined Deep Ignorance + confidence scoring (FR18) targets 30-40% removal, improving robustness
- Actual compute budget utilization will depend on final model selection and batch optimization

*These constraints will be validated during architecture design and initial implementation. Requirements remain as specified pending empirical testing.*
