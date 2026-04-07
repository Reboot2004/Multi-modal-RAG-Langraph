# retrieval/llm_reranker.py
"""
LLM-based Re-ranking

Use Groq to intelligently re-score retrieved chunks based on:
- Semantic relevance to query
- Answer completeness
- Information density
- Source authority/credibility

More accurate than embedding-based scoring alone.
"""

from typing import List, Dict, Tuple
from llm.groq_client import GroqClient
from pipeline_logger import get_logger

logger = get_logger("llm_reranker")


class LLMReranker:
    """Use LLM to intelligently rerank retrieved chunks"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client or GroqClient()
        self.batch_size = 5  # Re-score top 5 at a time to save tokens

    def rerank_with_llm(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Re-rank candidates using LLM scoring.
        
        Args:
            query: User's question
            candidates: List of retrieved chunks with scores
            top_k: Return top K after re-ranking
        
        Returns:
            Re-ranked list of candidates with new 'llm_rerank_score'
        """
        if not candidates:
            return []

        # Only re-rank top batch_size candidates (cost optimization)
        batch = candidates[: self.batch_size]
        if len(batch) <= 1:
            # Single result, no need to rerank
            logger.debug("LLM reranker | only_one_result=true | returning_as_is")
            return candidates[:top_k]

        # Build scoring prompt
        chunk_texts = []
        for i, chunk in enumerate(batch, 1):
            text = (chunk.get("text") or "")[:300]  # Truncate for token budget
            source = chunk.get("metadata", {}).get("source", "unknown")
            chunk_texts.append(f"[{i}] {source}: {text}")

        chunks_str = "\n".join(chunk_texts)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert information retrieval specialist. "
                    "Score snippets by relevance to the query (0-100). "
                    "Higher score = more relevant and useful. "
                    "Return ONLY a JSON array of scores: [score1, score2, ...]. "
                    "No explanations."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Snippets:\n{chunks_str}\n\n"
                    "Return scores as JSON array (one score per snippet in order):\n"
                    "[...scores...]"
                ),
            },
        ]

        try:
            response = self.llm_client.generate(
                messages, max_tokens=100, temperature=0.0
            )

            # Parse scores from response
            scores = self._parse_scores(response, len(batch))

            # Apply scores to batch
            for i, chunk in enumerate(batch):
                chunk["llm_rerank_score"] = scores[i] / 100.0  # Normalize to 0-1
                chunk["llm_rerank_score_raw"] = scores[i]  # Keep raw 0-100

            # Re-sort by LLM score
            reranked_batch = sorted(
                batch,
                key=lambda x: x.get("llm_rerank_score", 0.0),
                reverse=True,
            )

            # Append non-batch candidates
            remaining = candidates[self.batch_size :]
            result = reranked_batch + remaining

            logger.info(
                "LLM reranker | candidates=%d | batch=%d | scores=%s",
                len(candidates),
                len(batch),
                scores[:3],  # Log first 3 scores
            )

            return result[:top_k]

        except Exception as ex:
            logger.warning(
                "LLM reranking failed; returning original order | error=%s", ex
            )
            return candidates[:top_k]

    def _parse_scores(self, response: str, expected_count: int) -> List[int]:
        """Parse JSON array of scores from LLM response."""
        try:
            import json

            # Try to extract JSON array from response
            response_clean = response.strip()

            # Try direct JSON parse
            if response_clean.startswith("["):
                scores = json.loads(response_clean)
                if isinstance(scores, list) and len(scores) == expected_count:
                    return [min(100, max(0, int(s))) for s in scores]

            # Try to find JSON array in response
            import re

            match = re.search(r"\[[\d\s,\.]+\]", response_clean)
            if match:
                scores = json.loads(match.group())
                if isinstance(scores, list) and len(scores) == expected_count:
                    return [min(100, max(0, int(s))) for s in scores]

            # Fallback: return equal scores
            logger.warning("LLM reranker | parse_failed | using_defaults")
            return [75] * expected_count

        except Exception as ex:
            logger.warning("LLM reranker | parse_error=%s | using_defaults", ex)
            return [75] * expected_count

    def rerank_with_threshold(
        self,
        query: str,
        candidates: List[Dict],
        threshold: float = 0.5,
        top_k: int = 5,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Re-rank and split by threshold into high-confidence and low-confidence.
        
        Returns:
            (high_confidence_chunks, low_confidence_chunks)
        """
        reranked = self.rerank_with_llm(query, candidates, top_k=len(candidates))

        high_conf = []
        low_conf = []

        for chunk in reranked:
            score = chunk.get("llm_rerank_score", 0.5)
            if score >= threshold:
                high_conf.append(chunk)
            else:
                low_conf.append(chunk)

            if len(high_conf) >= top_k:
                break

        logger.info(
            "LLM reranker threshold | high_conf=%d | low_conf=%d | threshold=%.2f",
            len(high_conf),
            len(low_conf),
            threshold,
        )

        return high_conf[:top_k], low_conf
