# tfg_ml/context.py
"""
Shared in-memory context for conversation history and system decisions.

This module exposes:
- `Context`: A simple, thread-safe store for chat history and pipeline decisions.
- `CTX`: A global singleton used across UI, crew, and agents.

"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
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

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.data: Dict[str, Any] = {
            "history": [],        # type: List[Dict[str, str]]
            "last_topic": None,    # type: Optional[str]
            "decisions": [],       # type: List[Dict[str, Any]]
        }

    # -----------------------------
    # Mutators
    # -----------------------------
    def add_interaction(self, question: str, answer: str) -> None:
        """Append a user–assistant interaction to the history."""
        with self._lock:
            self.data["history"].append({"question": question, "answer": answer})

    def add_decision(self, stage: str, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """Record a system decision (stage, description, optional metadata)."""
        with self._lock:
            self.data["decisions"].append({"stage": stage, "text": text, "meta": (meta or {})})

    def set(self, key: str, value: Any) -> None:
        """Set an arbitrary key in the context."""
        with self._lock:
            self.data[key] = value

    # -----------------------------
    # Accessors
    # -----------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """Get an arbitrary key from the context."""
        with self._lock:
            return self.data.get(key, default)

    def summary_history(self, n: int = 20) -> str:
        """
        Return a plain-text summary of the last `n` interactions, formatted for the coordinator.
        """
        with self._lock:
            history: List[Dict[str, str]] = self.data["history"][-max(0, n):]
            lines = [
                f"User: {h.get('question', '')}\nAnswer: {h.get('answer', '')}"
                for h in history
            ]
        return "\n".join(lines)

    def summary(self, n: int = 10) -> str:
        """
        Return a compact one-line-per-decision summary of the last `n` decisions.
        """
        with self._lock:
            decs: List[Dict[str, Any]] = self.data["decisions"][-max(0, n):]
            lines = [f"- [{d.get('stage', '')}] {d.get('text', '')}" for d in decs]
        return "\n".join(lines)


# ✅ Global singleton shared across modules
CTX = Context()
