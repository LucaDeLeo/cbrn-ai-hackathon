# Memory Management Strategy

## Sequential Model Loading

```python
class MemoryEfficientModelManager:
    """
    Manages sequential loading of multiple 7B models on single GPU.
    Key strategies:
    - Load one model at a time
    - Use 8-bit quantization when needed
    - Clear cache between models
    - Monitor memory usage
    """
    
    def __init__(self, max_memory_gb: int = 40, deterministic: bool = True, quantize: bool = False):
        self.max_memory_gb = max_memory_gb
        self.current_model = None
        self.current_tokenizer = None
        self.deterministic = deterministic
        self.quantize = quantize
    
    def process_with_models(
        self, 
        model_names: List[str], 
        questions: List[Question]
    ) -> Dict[str, List[int]]:
        results = {}
        
        for model_name in model_names:
            # Check available memory
            available_gb = self._get_available_memory()
            
            # Determine loading strategy
            if available_gb < 15:
                logger.info(f"Low memory, using 8-bit for {model_name}")
                quantize = True
            else:
                quantize = False
            
            # Load model
            self._load_model(model_name, quantize=quantize)
            
            # Process in batches
            batch_size = self._adaptive_batch_size()
            predictions = []
            
            for i in range(0, len(questions), batch_size):
                batch = questions[i:i+batch_size]
                batch_preds = self._process_batch(batch)
                predictions.extend(batch_preds)
                
                # Save intermediate results
                if i % 100 == 0:
                    self._save_checkpoint(model_name, predictions)
            
            results[model_name] = predictions
            
            # Cleanup for next model
            self._cleanup()
        
        return results
    
    def _adaptive_batch_size(self) -> int:
        """Dynamically adjust batch size based on available memory"""
        available_gb = self._get_available_memory()
        
        if available_gb > 20:
            return 32
        elif available_gb > 10:
            return 16
        elif available_gb > 5:
            return 8
        else:
            return 4
    
    def _cleanup(self):
        """Free memory between models"""
        del self.current_model
        del self.current_tokenizer
        self.current_model = None
        self.current_tokenizer = None
        
        torch.cuda.empty_cache()
        gc.collect()
```
