# Parallel Execution Strategy

## Thread Pool for Statistical Analysis

```python
class ParallelStatisticalEngine:
    """
    Parallelizes independent statistical tests using thread pool.
    CPU-bound operations distributed across cores.
    """
    
    def __init__(self, max_workers: int = None):
        if max_workers is None:
            max_workers = min(8, (os.cpu_count() or 1))
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def analyze_all(self, questions: List[Question]) -> StatisticalReport:
        """Run all statistical tests in parallel"""
        
        # Define independent test functions
        tests = {
            'position_bias': self._test_position_bias,
            'length_bias': self._test_length_bias,
            'lexical_patterns': self._test_lexical_patterns,
            'answer_distribution': self._test_answer_distribution,
        }
        
        # Submit all tests
        futures = {
            name: self.executor.submit(test_func, questions)
            for name, test_func in tests.items()
        }
        
        # Collect results
        results = {}
        for name, future in futures.items():
            try:
                results[name] = future.result(timeout=30)
            except Exception as e:
                logger.error(f"Test {name} failed: {e}")
                results[name] = None
        
        return StatisticalReport(**results)
    
    def _test_position_bias(self, questions: List[Question]) -> BiasResult:
        """Chi-square test for answer position distribution"""
        positions = [q.answer_position for q in questions]
        observed = np.bincount(positions, minlength=4)
        expected = np.full(4, len(questions) / 4)
        
        # Chi-square statistic
        chi2 = np.sum((observed - expected) ** 2 / expected)
        
        # Bootstrap CI for effect size
        def effect_size(data):
            counts = np.bincount(data, minlength=4)
            return np.std(counts) / np.mean(counts)
        
        ci_lower, ci_upper = bootstrap_ci(
            np.array(positions),
            effect_size,
            n_iterations=10000
        )
        
        return BiasResult(
            statistic=chi2,
            p_value=self._chi2_p_value(chi2, df=3),
            effect_size_ci=(ci_lower, ci_upper),
            flagged=chi2 > 7.815  # Critical value at p=0.05
        )
```

## GPU Optimization for Model Inference

```python
class OptimizedGPUInference:
    """
    GPU optimization strategies for model inference.
    """
    
    def __init__(self, deterministic: bool = False):
        self.deterministic = deterministic
        self.use_flash_attention = (not deterministic) and self._check_flash_attention()
        self.compile_model = (not deterministic) and (torch.__version__ >= "2.0")
    
    def optimize_model(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """Apply optimizations to model"""
        
        # Flash Attention 2 if available (disabled in deterministic mode)
        if self.use_flash_attention:
            model = model.to_bettertransformers()
        
        # Torch compile for inference optimization (disabled in deterministic mode)
        if self.compile_model:
            model = torch.compile(
                model,
                mode="reduce-overhead",
                fullgraph=True
            )
        
        # Mixed precision
        model = model.half()  # FP16 (use bf16 where supported)
        
        return model
    
    @torch.inference_mode()
    def batch_inference(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompts: List[str],
        batch_size: int = 16
    ) -> List[int]:
        """Efficient batched inference"""
        
        predictions = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            
            # Tokenize with padding
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)
            
            # Score efficiently (prefer scoring over generation for choices/cloze)
            with torch.cuda.amp.autocast():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Decode predictions
            batch_preds = self._decode_predictions(outputs, tokenizer)
            predictions.extend(batch_preds)
        
        return predictions
```
