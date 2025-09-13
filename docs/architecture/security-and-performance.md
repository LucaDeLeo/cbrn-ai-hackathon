# Security and Performance

## Security Requirements

**Input Validation:**
- Schema validation for all inputs
- Size limits on input files (max 100MB)
- Sanitization of file paths

**Data Protection:**
- No plaintext question storage
- Hash-based question IDs only
- Configurable redaction levels
- Audit logs exclude sensitive content

**Access Control:**
- Read-only access to benchmark files
- Write access only to designated output directories
- No network access during processing

## Performance Optimization

**Target Metrics:**
- Process 1,000 questions in <2 hours with 2 models
- Process 3,000 questions in <4 hours total
- Memory usage <64GB RAM
- GPU memory usage <40GB VRAM

**Optimization Strategies:**
- Batch processing with adaptive sizing
- Sequential model loading
- Multi-level caching
- Parallel statistical analysis
- Mixed precision inference (disabled in deterministic mode if needed)
- Prefer token-level scoring over generation for choices/cloze
- Flash Attention when available (disabled in deterministic mode)
