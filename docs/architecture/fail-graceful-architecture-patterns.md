# Fail-Graceful Architecture Patterns

## Component Isolation and Fallback

```python
class FailGracefulPipeline:
    """
    Implements fail-graceful patterns ensuring pipeline continues
    even when components fail.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.components = self._initialize_components()
        self.failures = []
    
    def run(self, questions: List[Question]) -> PipelineResult:
        """Run pipeline with graceful degradation"""
        
        result = PipelineResult()
        
        # Level 1: Always-available statistical analysis
        result.statistical = self._run_safe(
            self.components['statistical'],
            questions,
            fallback=BasicStatistics()
        )
        
        # Level 2: GPU-dependent consensus (optional)
        if self._check_gpu_available():
            result.consensus = self._run_safe(
                self.components['consensus'],
                questions,
                fallback=None  # No fallback, just skip
            )
        else:
            logger.warning("GPU unavailable, skipping consensus detection")
            self.failures.append("consensus: no GPU")
        
        # Level 3: Memory-intensive cloze (optional)
        if result.consensus and self._check_memory_available(threshold_gb=30):
            result.cloze = self._run_safe(
                self.components['cloze'],
                questions,
                fallback=None
            )
        
        # Generate report with available results
        result.report = self._generate_adaptive_report(result)
        
        return result
    
    def _run_safe(self, component: Any, input_data: Any, fallback: Any = None):
        """Execute component with error handling"""
        try:
            return component.process(input_data)
        except Exception as e:
            logger.error(f"Component {component.__class__.__name__} failed: {e}")
            self.failures.append(f"{component.__class__.__name__}: {str(e)}")
            
            if fallback:
                logger.info(f"Using fallback for {component.__class__.__name__}")
                return fallback.process(input_data)
            
            return None
    
    def _generate_adaptive_report(self, result: PipelineResult) -> Report:
        """Generate report adapting to available results"""
        
        report = Report()
        report.metadata['failures'] = self.failures
        
        # Always include statistical results
        if result.statistical:
            report.add_section("Statistical Analysis", result.statistical)
        
        # Include GPU results if available
        if result.consensus:
            report.add_section("Model Consensus", result.consensus)
        else:
            report.add_note("Model consensus skipped (GPU unavailable)")
        
        if result.cloze:
            report.add_section("Cloze Format Comparison", result.cloze)
        
        # Adjust confidence based on available components
        confidence = 0.95 if result.consensus else 0.80
        report.metadata['confidence'] = confidence
        
        return report
```
