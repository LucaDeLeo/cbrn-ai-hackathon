# Caching Strategy

## Multi-Level Cache Architecture

```python
class CacheHierarchy:
    """
    Three-level caching system optimized for different access patterns and safety:
    
    L1 (Memory): Active batch logits - Cleared between models
    L2 (SQLite): Question metadata, status - Persistent, fast queries
    L3 (JSON): Model outputs - Persistent, space-efficient, NO PLAINTEXT CONTENT (IDs only, numeric scores)
    
    Two-tier policy: caches never contain plaintext; internal plaintext may exist in access-controlled logs only.
    """
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.memory_cache = {}  # L1: Active logits
        self.db_path = cache_dir / "metadata.db"  # L2: SQLite
        self.output_dir = cache_dir / "outputs"  # L3: JSON files
        
        self._init_db()
        self.output_dir.mkdir(exist_ok=True)

        # Content policy (two-tier): cache only non-sensitive artifacts
        # - keys: hashed question IDs (private salt internally, public salt for sanitized subset)
        # - values: indices, numeric scores/logits, configuration hashes
        # - never: question text, choices text, prompts, completions
    
    def get(self, key: str, level: int = 3) -> Optional[Any]:
        # Try L1 (memory)
        if level >= 1 and key in self.memory_cache:
            return self.memory_cache[key]
        
        # Try L2 (SQLite) for metadata
        if level >= 2:
            result = self._query_db(key)
            if result:
                return result
        
        # Try L3 (JSON) for full outputs
        if level >= 3:
            json_path = self.output_dir / f"{key[:8]}.json"
            if json_path.exists():
                with open(json_path) as f:
                    data = json.load(f)
                    return data.get(key)
        
        return None

    def redact_for_release(self, report_paths: List[Path]) -> Path:
        """
        Apply artifacts policy to produce a sanitized public bundle.
        - Remove any residual plaintext fields
        - Replace private-salt IDs with public-salt IDs for sanitized subset
        - Include checksums and metadata
        Returns path to the tarball/zip of public artifacts.
        """
        # Implementation detail deferred; ensure checks are part of release gate
        ...
```

## Checkpoint Recovery System

```python
@dataclass
class PipelineState:
    """Checkpoint state for recovery"""
    timestamp: datetime
    questions_processed: int
    questions_total: int
    components_completed: List[str]
    results_so_far: Dict[str, Any]
    config_hash: str  # Detect config changes
    
    def save(self, path: Path):
        """Save checkpoint atomically"""
        temp_path = path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(asdict(self), f, default=str)
        temp_path.replace(path)  # Atomic on POSIX
    
    @classmethod
    def load(cls, path: Path) -> Optional['PipelineState']:
        """Load checkpoint with validation"""
        if not path.exists():
            return None
        
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            logger.warning(f"Invalid checkpoint: {e}")
            return None
```
