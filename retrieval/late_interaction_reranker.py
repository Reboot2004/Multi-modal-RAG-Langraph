import re
from typing import Dict, List, Any


class LateInteractionReranker:
    """Token-level heuristic reranker that approximates late-interaction MaxSim behavior."""

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
        q_tokens = self._tokens(query)
        if not q_tokens:
            return docs[:top_k]

        reranked = []
        for item in docs:
            text = item.get("text", "")
            d_tokens = self._tokens(text)
            li_score = self._late_interaction_score(q_tokens, d_tokens)

            base = float(item.get("rerank_score", item.get("score", 0.0)))
            blended = (0.65 * base) + (0.35 * li_score)

            updated = dict(item)
            updated["late_interaction_score"] = float(li_score)
            updated["rerank_score"] = float(blended)
            reranked.append(updated)

        reranked.sort(key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)
        return reranked[:max(1, int(top_k))]

    def _tokens(self, text: str) -> List[str]:
        raw = re.findall(r"[a-zA-Z0-9_\-]+", (text or "").lower())
        return [t for t in raw if len(t) > 1]

    def _late_interaction_score(self, q_tokens: List[str], d_tokens: List[str]) -> float:
        if not d_tokens:
            return 0.0

        d_set = set(d_tokens)
        hit = 0.0
        for q in q_tokens:
            if q in d_set:
                hit += 1.0
                continue

            # soft token overlap proxy for multilingual/typo robustness
            soft = 0.0
            for d in d_set:
                prefix = self._common_prefix_ratio(q, d)
                if prefix > soft:
                    soft = prefix
            hit += soft

        return max(0.0, min(1.0, hit / max(1, len(q_tokens))))

    def _common_prefix_ratio(self, a: str, b: str) -> float:
        n = min(len(a), len(b))
        if n == 0:
            return 0.0
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i / n
