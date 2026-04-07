# orchestration/fallback_retrieval.py
"""
Fallback Retrieval Strategies

When primary retrieval fails or returns low-confidence results,
try alternative strategies:
1. BM25-only (lexical only)
2. Re-expand and retry
3. Broader semantic search
4. Full document retrieval
"""

from typing import List, Dict, Optional
from pipeline_logger import get_logger

logger = get_logger("fallback_retrieval")


class FallbackRetrievalHandler:
    """Graceful fallback when retrieval quality is low"""

    def __init__(self, retriever=None):
        self.retriever = retriever
        self.min_results_threshold = 2  # Need at least this many results
        self.min_confidence_threshold = 0.3

    def should_fallback(
        self,
        results: List[Dict],
        top_score: float,
        query_length: int,
    ) -> Tuple[bool, str, Dict]:
        """
        Determine if we should trigger fallback strategy.
        
        Returns:
            (should_fallback: bool, reason: str, fallback_params: Dict)
        """
        reasons = []

        # Check 1: Too few results
        if len(results) < self.min_results_threshold:
            reasons.append("too_few_results")

        # Check 2: All results have low confidence
        if results and top_score < self.min_confidence_threshold:
            reasons.append("low_confidence")

        # Check 3: Very short queries might need help
        if query_length < 3:
            reasons.append("very_short_query")

        if reasons:
            logger.info(
                "Fallback triggered | reasons=%s | result_count=%d | top_score=%.2f",
                reasons,
                len(results),
                top_score,
            )
            return True, "|".join(reasons), self._get_fallback_params(reasons)
        else:
            return False, "no_fallback_needed", {}

    def _get_fallback_params(self, reasons: List[str]) -> Dict:
        """Determine fallback parameters based on failure reasons."""
        params = {
            "strategy": "reexpand",  # Default to query re-expansion
            "increase_k": 3,
            "use_lexical_only": False,
            "use_broader_semantic": False,
        }

        if "too_few_results" in reasons:
            params["strategy"] = "reexpand"
            params["increase_k"] = 5

        if "low_confidence" in reasons:
            params["strategy"] = "lexical_fallback"
            params["use_lexical_only"] = True

        if "very_short_query" in reasons:
            params["strategy"] = "broader_semantic"
            params["use_broader_semantic"] = True

        return params

    def execute_fallback(
        self,
        query: str,
        original_results: List[Dict],
        fallback_params: Dict,
    ) -> List[Dict]:
        """
        Execute fallback retrieval strategy.
        
        Returns:
            Combined results (original + fallback) deduplicated
        """
        if not self.retriever:
            logger.warning("Fallback retriever not configured")
            return original_results

        strategy = fallback_params.get("strategy", "reexpand")
        logger.info("Fallback execution | strategy=%s | original_count=%d", strategy, len(original_results))

        fallback_results = []

        if strategy == "reexpand":
            # Try with expanded query variants
            fallback_results = self._fallback_reexpand(query)

        elif strategy == "lexical_fallback":
            # Try BM25-only search
            fallback_results = self._fallback_lexical(query)

        elif strategy == "broader_semantic":
            # Try with lower similarity threshold
            fallback_results = self._fallback_broader_semantic(query)

        # Combine and deduplicate
        combined = self._deduplicate_results(original_results + fallback_results)

        logger.info(
            "Fallback complete | strategy=%s | fallback_count=%d | combined=%d",
            strategy,
            len(fallback_results),
            len(combined),
        )

        return combined

    def _fallback_reexpand(self, query: str) -> List[Dict]:
        """Re-expand query and try retrieval again with broader keywords."""
        try:
            # Try query without modifiers (broader search)
            broader_query = self._broaden_query(query)
            results = self.retriever.retrieve(broader_query, top_k=7)
            return results.get("results", [])
        except Exception as ex:
            logger.warning("Fallback reexpand failed | error=%s", ex)
            return []

    def _fallback_lexical(self, query: str) -> List[Dict]:
        """Try lexical (BM25) search only."""
        try:
            # Force lexical-only search
            results = self.retriever.retrieve(query, top_k=5)
            lexical_only = [r for r in results.get("results", []) if r.get("lexical_score", 0) > 0.1]
            return lexical_only
        except Exception as ex:
            logger.warning("Fallback lexical failed | error=%s", ex)
            return []

    def _fallback_broader_semantic(self, query: str) -> List[Dict]:
        """Try semantic search with broader k."""
        try:
            results = self.retriever.retrieve(query, top_k=8)
            return results.get("results", [])
        except Exception as ex:
            logger.warning("Fallback broader semantic failed | error=%s", ex)
            return []

    def _broaden_query(self, query: str) -> str:
        """Remove modifiers from query to make it broader."""
        import re

        # Remove words like "only", "just", "exactly"
        modifiers = {
            r'\bonly\b': '', r'\bjust\b': '', r'\bexactly\b': '', r'\bspecific\b': '',
            r'\brecent\b': '', r'\blatest\b': '',
        }

        broader = query
        for modifier, replacement in modifiers.items():
            broader = re.sub(modifier, replacement, broader, flags=re.IGNORECASE)

        # Clean extra spaces
        broader = re.sub(r'\s+', ' ', broader).strip()

        logger.debug("Query broadened | original=%d chars | broader=%d chars", len(query), len(broader))
        return broader

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results keeping highest score."""
        seen = {}
        dedup = []

        for result in results:
            text = (result.get("text") or "")[:100]  # Use first 100 chars as key
            metadata = result.get("metadata", {})
            key = f"{metadata.get('source')}_{metadata.get('page')}_{text}"

            if key not in seen:
                seen[key] = result
                dedup.append(result)
            else:
                # Keep result with higher score
                existing_score = seen[key].get("score", 0)
                new_score = result.get("score", 0)
                if new_score > existing_score:
                    # Replace in list
                    idx = dedup.index(seen[key])
                    dedup[idx] = result
                    seen[key] = result

        logger.debug("Fallback deduplication | input=%d | output=%d", len(results), len(dedup))
        return dedup


# Type hints
from typing import Tuple
