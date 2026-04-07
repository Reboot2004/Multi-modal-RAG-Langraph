# retriever.py
# retrieval/retriever.py

from typing import List, Dict
from embeddings.embedder import MultilingualEmbedder
from embeddings.vector_store import VectorStore
from processing.language_detector import LanguageDetector
from config.settings import (
    TOP_K,
    LEXICAL_TOP_K,
    ENABLE_HYBRID_DB,
    ENABLE_HYPE,
    HYPE_QUERY_TOP_K,
    SEMANTIC_SCORE_WEIGHT,
    HYPE_SCORE_WEIGHT,
    LEXICAL_SCORE_WEIGHT,
)
from pipeline_logger import get_logger


logger = get_logger("retriever")


class Retriever:
    def __init__(self):
        self.embedder = MultilingualEmbedder()
        self.vector_store = VectorStore()
        self.lang_detector = LanguageDetector()
        logger.info("Retriever initialized | HyPE enabled=%s", ENABLE_HYPE)

    def retrieve(self, query: str, top_k: int = TOP_K) -> Dict:
        """
        Returns:
        {
            "query_language": "hi",
            "results": [
                {
                    "score": float,
                    "text": str,
                    "metadata": {...}
                }
            ]
        }
        """

        # Detect query language
        query_language = self.lang_detector.detect_language(query)
        logger.info("Retrieve called | query_language=%s | top_k=%d", query_language, top_k)

        # Embed query
        query_embedding = self.embedder.embed_query(query)

        # Search FAISS
        semantic_results = self.vector_store.search(query_embedding, max(top_k * 2, HYPE_QUERY_TOP_K))
        logger.info("Semantic retrieval results=%d", len(semantic_results))

        if ENABLE_HYBRID_DB:
            lexical_results = self.vector_store.search_lexical(query=query, top_k=max(top_k * 2, LEXICAL_TOP_K))
        else:
            lexical_results = []
        logger.info("Lexical retrieval results=%d", len(lexical_results))

        if ENABLE_HYPE:
            hype_results = self.vector_store.search_hype(query_embedding, max(top_k * 2, HYPE_QUERY_TOP_K))
        else:
            hype_results = []
        logger.info("HyPE retrieval results=%d", len(hype_results))

        results = self._fuse_results(
            semantic_results=semantic_results,
            hype_results=hype_results,
            lexical_results=lexical_results,
            top_k=top_k,
        )

        return {
            "query_language": query_language,
            "results": results,
        }

    def _fuse_results(
        self,
        semantic_results: List[Dict],
        hype_results: List[Dict],
        lexical_results: List[Dict],
        top_k: int,
    ) -> List[Dict]:
        fused = {}

        def ensure_entry(item: Dict):
            chunk_id = item.get("metadata", {}).get("chunk_id")
            if chunk_id is None:
                return None

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "score": 0.0,
                    "semantic_score": 0.0,
                    "hype_score": 0.0,
                    "lexical_score": 0.0,
                    "text": item.get("text", ""),
                    "metadata": item.get("metadata", {}),
                }

            return chunk_id

        for item in semantic_results:
            chunk_id = ensure_entry(item)
            if chunk_id is None:
                continue
            fused[chunk_id]["semantic_score"] = max(
                fused[chunk_id]["semantic_score"],
                float(item.get("score", 0.0)),
            )

        for item in hype_results:
            chunk_id = ensure_entry(item)
            if chunk_id is None:
                continue
            fused[chunk_id]["hype_score"] = max(
                fused[chunk_id]["hype_score"],
                float(item.get("score", 0.0)),
            )

        for item in lexical_results:
            chunk_id = ensure_entry(item)
            if chunk_id is None:
                continue
            fused[chunk_id]["lexical_score"] = max(
                fused[chunk_id]["lexical_score"],
                float(item.get("score", 0.0)),
            )

        final_results = []
        for entry in fused.values():
            fused_score = (
                entry["semantic_score"] * SEMANTIC_SCORE_WEIGHT
                + entry["hype_score"] * HYPE_SCORE_WEIGHT
                + entry["lexical_score"] * LEXICAL_SCORE_WEIGHT
            )

            final_results.append(
                {
                    "score": float(fused_score),
                    "semantic_score": float(entry["semantic_score"]),
                    "hype_score": float(entry["hype_score"]),
                    "lexical_score": float(entry["lexical_score"]),
                    "text": entry["text"],
                    "metadata": entry["metadata"],
                }
            )

        final_results.sort(key=lambda item: item["score"], reverse=True)
        logger.debug(
            "Fusion completed | semantic_in=%d | hype_in=%d | lexical_in=%d | final=%d",
            len(semantic_results),
            len(hype_results),
            len(lexical_results),
            min(top_k, len(final_results)),
        )
        return final_results[:top_k]