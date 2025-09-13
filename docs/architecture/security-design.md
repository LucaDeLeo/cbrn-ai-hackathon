# Security Design

## Hash-Based Anonymization System

```python
class SecurityManager:
    """
    Implements hash-based anonymization with configurable redaction.
    Two-tier salts:
    - private_salt: per-project (internal artifacts)
    - public_salt: fixed (for sanitized calibration subset IDs only)
    No plaintext question content in public artifacts.
    """
    
    def __init__(self, private_salt: str = None, public_salt: str = None):
        if private_salt is None:
            private_salt = secrets.token_hex(32)
        if public_salt is None:
            public_salt = "public-sanitized-salt-v1"
        
        self.private_salt = private_salt.encode()
        self.public_salt = public_salt.encode()
        self.redaction_level = "moderate"  # low, moderate, high
    
    def _content_signature(self, question: Dict) -> bytes:
        """Canonical content signature used for hashing (not stored)."""
        return json.dumps(
            {"q": question.get("question", ""),
             "c": question.get("choices", [])},
            sort_keys=True
        ).encode()

    def internal_id(self, question: Dict) -> str:
        """Generate SHA-256 ID with private salt for internal artifacts."""
        content = self._content_signature(question)
        digest = hashlib.sha256(self.private_salt + content).hexdigest()
        return digest[:32]

    def public_id_for_calibration(self, question: Dict) -> str:
        """Generate SHA-256 ID with public salt (sanitized subset only)."""
        content = self._content_signature(question)
        digest = hashlib.sha256(self.public_salt + content).hexdigest()
        return digest[:32]

    def anonymize_question(self, question: Dict) -> Dict:
        return {
            "id": self.internal_id(question),
            "choice_count": len(question.get("choices", [])),
            "answer_position": question.get("answer", -1),
            # No question text stored
        }
    
    def redact_output(self, content: str) -> str:
        """Apply redaction based on security level"""
        if self.redaction_level == "high":
            # Remove all specific values
            content = re.sub(r'\b\d+\.\d+\b', '[REDACTED]', content)
            content = re.sub(r'"(question|choices|prompt)":\s*".*?"', '"\\1":"[REDACTED]"', content)
        elif self.redaction_level == "moderate":
            # Keep aggregate statistics only (IDs, indices, scores)
            content = re.sub(r'"(question|choices|prompt)":\s*".*?"', '"\\1":"[REDACTED]"', content)
        
        return content
```
