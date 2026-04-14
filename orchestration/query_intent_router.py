import re
from typing import Any, Dict

from config.settings import (
    ENABLE_QUERY_INTENT_ROUTING,
    INTENT_ROUTER_COMPLEX_TOP_K,
    INTENT_ROUTER_DEFAULT_TOP_K,
    INTENT_ROUTER_SUMMARY_TOP_K,
)


class QueryIntentRouter:
    """Lightweight intent router for adaptive retrieval strategy."""

    def route(self, query: str) -> Dict[str, Any]:
        q = (query or "").strip().lower()
        if not ENABLE_QUERY_INTENT_ROUTING or not q:
            return self._policy("qa")

        if self._is_summary(q):
            return self._policy("summary")
        if self._is_comparison(q):
            return self._policy("comparison")
        if self._is_mcq(q):
            return self._policy("mcq")
        if self._is_translation(q):
            return self._policy("translation")
        if self._is_complex(q):
            return self._policy("complex_qa")
        return self._policy("qa")

    def _policy(self, intent: str) -> Dict[str, Any]:
        if intent == "summary":
            return {
                "intent": intent,
                "top_k": int(INTENT_ROUTER_SUMMARY_TOP_K),
                "notes": "summary_needs_broader_context",
            }
        if intent in {"comparison", "complex_qa"}:
            return {
                "intent": intent,
                "top_k": int(INTENT_ROUTER_COMPLEX_TOP_K),
                "notes": "complex_query_needs_more_evidence",
            }
        if intent in {"translation", "mcq"}:
            return {
                "intent": intent,
                "top_k": int(INTENT_ROUTER_DEFAULT_TOP_K),
                "notes": "format_task",
            }
        return {
            "intent": "qa",
            "top_k": int(INTENT_ROUTER_DEFAULT_TOP_K),
            "notes": "default",
        }

    def _is_summary(self, q: str) -> bool:
        return any(k in q for k in ["summarize", "summary", "gist", "overall", "brief"])

    def _is_comparison(self, q: str) -> bool:
        return any(k in q for k in ["compare", "difference", "vs", "versus", "pros and cons"])

    def _is_mcq(self, q: str) -> bool:
        return bool(re.search(r"\b(mcq|multiple\s*choice)\b", q))

    def _is_translation(self, q: str) -> bool:
        return any(k in q for k in ["translate", "translation"])

    def _is_complex(self, q: str) -> bool:
        conjunctions = len(re.findall(r"\b(and|or|however|also|additionally|further)\b", q))
        return conjunctions >= 2 or len(q.split()) >= 18
