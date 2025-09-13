# Output Schemas

## Robust Subset (`robust_subset.jsonl`)
- Fields per line: `{"id": str, "keep": bool, "rationale": {"consensus": str, "heuristics": Dict, "perm_delta": float}}`
- No question text or choices. `id` is SHA-256 with salt (private salt internally; public salt only for sanitized calibration subset).

## Bias Report (`bias_report.json`)
- Top-level keys: `{"summary": Dict, "position_bias": Dict, "length_bias": Dict, "lexical_patterns": Dict, "permutation": Dict, "confidence_intervals": Dict}`
- Each statistic includes point estimate and 95% CI bounds. No raw text.

## Audit Log (`audit_log.jsonl`)
- Fields per line: `{"ts": str, "id": str, "component": str, "decision": str, "models": List[str], "families": List[str], "seed": int, "config_hash": str}`
- Optional: `{"cost_estimate": float}` appended at end of run.

## Error Categories and Responses

```python
class ErrorHandler:
    """Centralized error handling with appropriate responses"""
    
    ERROR_RESPONSES = {
        'GPU_OOM': ('Reduce batch size or enable quantization', 'warning'),
        'MODEL_LOAD_FAIL': ('Skip model or try quantized version', 'warning'),
        'CACHE_CORRUPT': ('Clear cache and restart', 'error'),
        'INPUT_INVALID': ('Check input format and schema', 'error'),
        'STATISTICAL_FAIL': ('Use fallback statistics', 'warning'),
    }
    
    def handle(self, error: Exception, context: str) -> bool:
        """Handle error and determine if pipeline should continue"""
        
        error_type = self._classify_error(error)
        response, level = self.ERROR_RESPONSES.get(
            error_type,
            ('Unknown error', 'error')
        )
        
        logger.log(level, f"{context}: {error} - {response}")
        
        # Determine if recoverable
        if level == 'warning':
            return True  # Continue pipeline
        else:
            return False  # Stop pipeline
```
