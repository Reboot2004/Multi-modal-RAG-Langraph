import re
from typing import Any, Dict, List


class GroundingVerifier:
    """Heuristic grounding verifier: checks whether answer sentences are supported by retrieval context."""

    def evaluate(self, answer: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer or "") if s.strip()]
        if not sentences:
            return {
                "support_ratio": 0.0,
                "supported_sentences": 0,
                "total_sentences": 0,
                "unsupported_examples": [],
            }

        contexts = []
        for item in (retrieved_docs or [])[:6]:
            txt = (item.get("text") or item.get("content") or "").lower()
            if txt:
                contexts.append(txt)

        supported = 0
        unsupported_examples = []

        for sent in sentences:
            token_hits = self._token_support_score(sent.lower(), contexts)
            if token_hits >= 0.45:
                supported += 1
            elif len(unsupported_examples) < 3:
                unsupported_examples.append(sent[:200])

        total = len(sentences)
        ratio = supported / max(1, total)

        return {
            "support_ratio": float(ratio),
            "supported_sentences": int(supported),
            "total_sentences": int(total),
            "unsupported_examples": unsupported_examples,
        }

    def _token_support_score(self, sentence: str, contexts: List[str]) -> float:
        tokens = [t for t in re.findall(r"\w+", sentence) if len(t) > 2]
        if not tokens or not contexts:
            return 0.0

        best = 0.0
        token_set = set(tokens)
        for ctx in contexts:
            hits = sum(1 for t in token_set if t in ctx)
            score = hits / max(1, len(token_set))
            if score > best:
                best = score
        return best
