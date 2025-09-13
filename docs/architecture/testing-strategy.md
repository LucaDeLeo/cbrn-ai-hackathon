# Testing Strategy

## Testing Pyramid

        E2E Tests (5%)
       /              \
    Integration Tests (25%)
   /                      \
Unit Tests (70% - Statistical & Core Logic)

## Test Organization

    # tests/test_statistical.py
    class TestBootstrapCI(unittest.TestCase):
        def test_bootstrap_ci_known_distribution(self):
            """Test bootstrap CI against known normal distribution"""
            np.random.seed(42)
            data = np.random.normal(100, 15, 1000)
            
            lower, upper = bootstrap_ci(data, np.mean)
            
            # 95% CI for mean should contain true mean
            self.assertLess(lower, 100)
            self.assertGreater(upper, 100)
            
            # Check interval width
            width = upper - lower
            self.assertAlmostEqual(width, 1.96 * 15 / np.sqrt(1000), delta=1)

    # tests/test_pipeline.py  
    class TestFailGraceful(unittest.TestCase):
        def test_pipeline_continues_without_gpu(self):
            """Test pipeline completes with statistical analysis only"""
            config = PipelineConfig(components=['statistical'])
            pipeline = FailGracefulPipeline(config)
            
            # Mock GPU unavailable
            with patch('torch.cuda.is_available', return_value=False):
                result = pipeline.run(self.sample_questions)
            
            self.assertIsNotNone(result.statistical)
            self.assertIsNone(result.consensus)
            self.assertIn('no GPU', result.report.metadata['failures'])

    class TestDeterminism(unittest.TestCase):
        def test_deterministic_mode_reproducibility(self):
            enable_determinism(1234)
            result1 = run_pipeline(PipelineConfig(seed=1234, deterministic=True))
            enable_determinism(1234)
            result2 = run_pipeline(PipelineConfig(seed=1234, deterministic=True))
            self.assertEqual(result1.hash(), result2.hash())

    class TestSecurity(unittest.TestCase):
        def test_no_plaintext_in_cache(self):
            # Inspect cache outputs for forbidden fields
            forbidden_keys = {'question', 'choices', 'prompt'}
            for path in Path('cache/outputs').glob('*.json'):
                obj = json.load(open(path))
                self.assertTrue(all(k not in obj for k in forbidden_keys))
