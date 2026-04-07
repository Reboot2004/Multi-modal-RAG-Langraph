# reranker.py
# retrieval/reranker.py

from typing import List, Dict
from sentence_transformers import CrossEncoder
from pipeline_logger import get_logger


logger = get_logger("reranker")


class Reranker:
    def __init__(self):
        """
        Multilingual cross-encoder reranker.
        """
        self.model = CrossEncoder(
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
        )
        logger.info("Reranker initialized")

    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank retrieved results using cross-encoder scoring.

        results: list of {
            "score": float,
            "text": str,
            "metadata": {...}
        }

        Returns top_k reranked results.
        """

        if not results:
            logger.info("Rerank skipped: empty retrieval result set")
            return []

        # Prepare pairs
        pairs = [(query, item["text"]) for item in results]

        scores = self.model.predict(pairs)
        logger.debug("Reranker produced %d scores", len(scores))

        # Attach rerank scores
        for item, score in zip(results, scores):
            item["rerank_score"] = float(score)

        # Sort by rerank_score descending
        reranked = sorted(
            results,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        logger.info("Rerank completed | input=%d | output=%d", len(results), min(top_k, len(reranked)))

        return reranked[:top_k]