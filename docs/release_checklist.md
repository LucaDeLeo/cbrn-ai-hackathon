# Release Checklist

This checklist ensures public artifacts are safe, reproducible, and aligned with hackathon requirements.

## Safety & Redaction
- [ ] Apply two-tier artifacts policy: no plaintext question content in public outputs
- [ ] Redaction checks pass (no raw stems, options, or hazardous strings)
- [ ] Public-salt IDs used only for sanitized calibration subset
- [ ] No per-item exploit labels in public artifacts (aggregate metrics only)

## Reproducibility Bundle
- [ ] Commands used (full CLI invocations with flags)
- [ ] Seeds and determinism settings (env vars and code-level flags)
- [ ] Package versions; PyTorch/Transformers versions
- [ ] Model identifiers and SHAs (weights commit/revision)
- [ ] Hardware and runtime metrics (GPU type, VRAM, wall-clock)
- [ ] Artifact checksums (reports, bundles)

## Prompts Appendix
- [ ] Exact templates published (choices-only and cloze)
- [ ] Template version and SHA documented
- [ ] Rendered prompts stored internally per item/model (not public)

## Diversity & Measurement
- [ ] Model family diversity check run; pairwise Jaccard ≤85% on 200-item subset or variance explained
- [ ] Measurement & Pre-Registration Plan followed; report measured deltas with 95% CIs
- [ ] Bootstrap iteration count and CI widths reported

## Performance & Budget
- [ ] Throughput and total runtime recorded
- [ ] Cost estimate computed for full run (based on cloud rates)
- [ ] Cloze subset size justified vs budget and runtime

## Submission Package
- [ ] Public bias report (aggregate metrics + CIs)
- [ ] Calibration subset IDs (public-salt)
- [ ] Reproducibility bundle JSON
- [ ] Prompts Appendix
- [ ] Security Considerations section conforms to sprint guidance

