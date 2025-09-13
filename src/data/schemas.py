from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    choices: List[str]
    answer: int  # 0-based index
    topic: Optional[str] = None
    difficulty: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

