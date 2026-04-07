# utils/conversation_memory.py
"""
Conversation Memory Manager

Stores and retrieves conversation history for:
1. Query contextualization (multi-turn support)
2. User personality/preferences (semantic memory)
3. Session analytics (what was asked, answered)
"""

import json
import os
from typing import List, Dict, Optional
from datetime import datetime
from pipeline_logger import get_logger
from config.settings import (
    ENABLE_CONVERSATION_MEMORY,
    CONVERSATION_HISTORY_LIMIT,
    MEMORY_STORE_PATH,
)

logger = get_logger("conversation_memory")


class ConversationMemory:
    """Thread-safe conversation history storage"""

    def __init__(self, memory_path: str = MEMORY_STORE_PATH):
        self.memory_path = memory_path
        self.enabled = ENABLE_CONVERSATION_MEMORY
        self.history_limit = max(1, int(CONVERSATION_HISTORY_LIMIT))
        self.conversation_history: List[Dict] = []

        if self.enabled:
            self._load_from_disk()
            logger.info(
                "ConversationMemory initialized | path=%s | limit=%d | loaded=%d",
                self.memory_path,
                self.history_limit,
                len(self.conversation_history),
            )

    def add_turn(self, question: str, answer: str, metadata: Dict = None) -> None:
        """Add a Q&A turn to history."""
        if not self.enabled or not question:
            return

        turn = {
            "timestamp": datetime.now().isoformat(),
            "question": question.strip(),
            "answer": (answer or "").strip()[:500],  # Truncate long answers
            "metadata": metadata or {},
        }

        self.conversation_history.append(turn)

        # Trim to limit
        if len(self.conversation_history) > self.history_limit:
            self.conversation_history = self.conversation_history[-self.history_limit :]

        self._save_to_disk()
        logger.debug(
            "Turn added to memory | total=%d | q_len=%d",
            len(self.conversation_history),
            len(question),
        )

    def get_history(self, num_turns: int = None) -> List[Dict]:
        """Get last N turns from history."""
        if not self.enabled:
            return []

        if num_turns is None:
            return self.conversation_history

        return self.conversation_history[-max(1, num_turns) :]

    def get_last_question(self) -> Optional[str]:
        """Get the most recent question."""
        if not self.enabled or not self.conversation_history:
            return None

        return self.conversation_history[-1].get("question")

    def get_context_summary(self, max_chars: int = 500) -> str:
        """Get a summary of recent conversation for LLM context."""
        if not self.enabled or not self.conversation_history:
            return ""

        lines = []
        total_chars = 0

        # Go backwards through history
        for turn in reversed(self.conversation_history):
            q = (turn.get("question") or "").strip()
            a = (turn.get("answer") or "")[:100].strip()  # Snippet
            line = f"Q: {q}\nA: {a}\n"

            if total_chars + len(line) > max_chars:
                break

            lines.append(line)
            total_chars += len(line)

        # Reverse to get chronological order
        lines.reverse()
        return "\n".join(lines).strip()

    def clear_history(self) -> None:
        """Clear all history."""
        self.conversation_history = []
        self._save_to_disk()
        logger.info("Conversation history cleared")

    def _save_to_disk(self) -> None:
        """Persist history to JSON file."""
        if not self.enabled or not self.memory_path:
            return

        try:
            os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
            logger.debug("Conversation history saved | count=%d", len(self.conversation_history))
        except Exception as ex:
            logger.warning("Failed to save conversation memory | error=%s", ex)

    def _load_from_disk(self) -> None:
        """Load history from JSON file."""
        if not self.enabled or not self.memory_path or not os.path.exists(
            self.memory_path
        ):
            self.conversation_history = []
            return

        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                self.conversation_history = json.load(f)
            logger.info(
                "Conversation history loaded | count=%d",
                len(self.conversation_history),
            )
        except Exception as ex:
            logger.warning(
                "Failed to load conversation memory | error=%s | starting fresh", ex
            )
            self.conversation_history = []

    def get_user_preferences(self) -> Dict:
        """Infer user preferences from history (for future personalization)."""
        if not self.conversation_history:
            return {}

        preferences = {
            "avg_question_length": 0,
            "total_questions": len(self.conversation_history),
            "common_topics": [],
        }

        q_lengths = [len(t.get("question", "")) for t in self.conversation_history]
        if q_lengths:
            preferences["avg_question_length"] = sum(q_lengths) / len(q_lengths)

        return preferences
