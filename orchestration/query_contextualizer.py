# orchestration/query_contextualizer.py
"""
Contextual Retrieval RAG Pattern

Rewrites user queries using conversation history to resolve pronouns,
implicit references, and maintain multi-turn context.

Example:
  User Turn 1: "Tell me about the Premium plan"
  User Turn 2: "How much does it cost?"
  
  Without contextualization:
    Search for: "How much does it cost?" ← No context, might find wrong docs
  
  With contextualization:
    Rewritten: "How much does the Premium plan cost?" ← Search finds pricing docs
"""

from typing import List, Dict, Tuple, Optional
from llm.groq_client import GroqClient
from pipeline_logger import get_logger
from config.settings import (
    ENABLE_CONTEXTUAL_RETRIEVAL,
    CONTEXT_HISTORY_TURNS,
    MAX_CONTEXT_HISTORY_CHARS,
)

logger = get_logger("query_contextualizer")


class QueryContextualizer:
    """Rewrites queries using conversation history for better retrievalDuring multi-turn conversations"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client or GroqClient()
        self.enable_contextual_retrieval = ENABLE_CONTEXTUAL_RETRIEVAL
        self.history_turns = max(1, int(CONTEXT_HISTORY_TURNS))
        self.max_history_chars = max(500, int(MAX_CONTEXT_HISTORY_CHARS))

    def contextualize_query(
        self,
        current_query: str,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Tuple[str, bool, str]:
        """
        Rewrite the current query using conversation history.

        Returns:
            (rewritten_query: str, was_rewritten: bool, reason: str)
        """
        if not self.enable_contextual_retrieval:
            return current_query, False, "contextual_retrieval_disabled"

        if not conversation_history or len(conversation_history) == 0:
            return current_query, False, "no_history"

        # Use last N turns
        relevant_history = conversation_history[-self.history_turns :]
        if not relevant_history:
            return current_query, False, "no_relevant_history"

        # Build history context (limit to max chars)
        history_text = self._build_history_context(relevant_history)
        if not history_text:
            return current_query, False, "empty_history"

        rewritten = self._rewrite_with_llm(current_query, history_text)
        if rewritten and rewritten.lower().strip() != current_query.lower().strip():
            logger.info(
                "Query contextualized | original=%d chars | rewritten=%d chars",
                len(current_query),
                len(rewritten),
            )
            logger.debug(
                "Query rewrite | original: %s | rewritten: %s",
                current_query,
                rewritten,
            )
            return rewritten, True, "rewritten_from_history"

        return current_query, False, "no_changes_needed"

    def _build_history_context(self, history: List[Dict]) -> str:
        """Build a concise history summary within char limit."""
        context_parts = []
        current_chars = 0

        for turn in history:
            question = (turn.get("question") or "").strip()
            answer_snippet = (turn.get("answer") or "")[:150].strip()

            if not question:
                continue

            turn_text = f"Q: {question}\nA: {answer_snippet}\n"
            if current_chars + len(turn_text) > self.max_history_chars:
                break

            context_parts.append(turn_text)
            current_chars += len(turn_text)

        return "\n".join(context_parts).strip()

    def _rewrite_with_llm(self, query: str, history: str) -> str:
        """Use LLM to rewrite query with history context."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You rewrite incomplete user queries using conversation history. "
                    "Make the query standalone and self-contained. "
                    "Return ONLY the rewritten query, nothing else. "
                    "Preserve the core intent and question structure."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{history}\n\n"
                    f"Current query: {query}\n\n"
                    "Rewrite this query to be standalone using history context if needed.\n"
                    "If already standalone, return it unchanged."
                ),
            },
        ]

        try:
            response = self.llm_client.generate(
                messages, max_tokens=150, temperature=0.1
            )
            rewritten = (response or "").strip()
            if rewritten:
                return rewritten
        except Exception as ex:
            logger.warning(
                "Query contextualization LLM call failed | error=%s", ex
            )

        return query

    def get_rewrite_reason(self, original: str, rewritten: str) -> str:
        """Diagnostic: why was the query rewritten?"""
        if not original or not rewritten:
            return ""

        # Simple heuristics
        if "it" in original.lower() and "it" not in rewritten.lower():
            return "resolved_pronoun_it"
        if "that" in original.lower() and "that" not in rewritten.lower():
            return "resolved_pronoun_that"
        if len(rewritten) > len(original) * 1.2:
            return "added_context"
        if len(rewritten) < len(original) * 0.8:
            return "simplified"

        return "content_updated"
