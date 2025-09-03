# tfg_ml/context.py
"""
Shared in-memory context for conversation history and system decisions.

This module exposes:
- `Context`: A simple, thread-safe store for chat history and pipeline decisions.
- `CTX`: A global singleton used across UI, crew, and agents.

"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional
import threading


class Context:
    """
    Thread-safe container for chat history and decisions.

    Attributes:
        data: Internal dictionary with:
            - "history": list of dicts {"question": str, "answer": str}
            - "last_topic": optional label for the current topic
            - "decisions": list of dicts {"stage": str, "text": str, "meta": dict}
    """

    def __init__(self, max_interactions: int = 8) -> None:
        self._lock = threading.RLock()
        self.history: Deque[Dict[str, str]] = deque(maxlen=max_interactions)

    def add(self, user: str, assistant: str) -> None:
        self.history.append({"user": user, "assistant": assistant})

    def as_prompt(self) -> str:
        """
        Return a text block with the last interactions
        """
        lines: List[str] = []
        for h in self.history:
            lines.append(f"User: {h['user']}\nAssistant: {h['assistant']}")
        return "\n\n".join(lines)

# Global singleton shared across modules
CTX = Context(max_interactions= 8)
