from typing import Dict, Any


class GraphRAGRouter:
    """Lightweight routing scaffold to choose graph/global vs local retrieval flow."""

    GLOBAL_HINTS = {
        "overall",
        "across",
        "main themes",
        "trend",
        "summary of all",
        "big picture",
        "compare all",
        "dataset-wide",
        "holistic",
    }

    def route(self, query: str, intent: str) -> Dict[str, Any]:
        q = (query or "").lower().strip()
        intent_key = (intent or "qa").strip().lower()

        is_global = intent_key in {"summary", "comparison"}
        if not is_global:
            for hint in self.GLOBAL_HINTS:
                if hint in q:
                    is_global = True
                    break

        return {
            "mode": "graph_global" if is_global else "baseline_local",
            "reason": "global_query_detected" if is_global else "local_query_detected",
            "graph_enabled": is_global,
        }
