"""
Enhanced Query Orchestrator integrating Tier 1 & 2 features:
- Adaptive Retrieval (adjust k based on query complexity)
- LLM-based Re-ranking (use Groq to re-score chunks)
- Semantic Query Caching (avoid redundant retrievals)
- Citation Tracking (map sentences to sources)
- Fallback Retrieval (graceful degradation)
"""

from typing import List, Dict, Any

from config.settings import (
    ENABLE_ADAPTIVE_RETRIEVAL,
    ENABLE_LLM_RERANKING,
    ENABLE_SEMANTIC_CACHE,
    ENABLE_CITATION_TRACKING,
    ENABLE_FALLBACK_RETRIEVAL,
    TOP_K,
)
from retrieval.enhancements import (
    AdaptiveRetrieval,
    LLMReranker,
    FallbackRetrieval,
)
from utils.semantic_cache import SemanticCache
from utils.citation_tracker import CitationTracker
from embeddings.embedder import MultilingualEmbedder
from pipeline_logger import get_logger


logger = get_logger("enhanced_orchestrator")


class EnhancedRetrieverWrapper:
    """
    Wraps existing retriever with all Tier 1 & 2 enhancements.
    Acts as a drop-in replacement for standard retriever.
    """

    def __init__(self, base_retriever, llm_client):
        """
        Initialize enhanced wrapper.
        
        Args:
            base_retriever: Existing Retriever instance
            llm_client: LLM client for re-ranking and citation extraction
        """
        self.base_retriever = base_retriever
        self.llm_client = llm_client
        self.embedder = MultilingualEmbedder()
        
        # Initialize feature components
        self.adaptive = AdaptiveRetrieval() if ENABLE_ADAPTIVE_RETRIEVAL else None
        self.reranker = LLMReranker(llm_client) if ENABLE_LLM_RERANKING else None
        self.semantic_cache = SemanticCache() if ENABLE_SEMANTIC_CACHE else None
        self.fallback = FallbackRetrieval(self.base_retriever) if ENABLE_FALLBACK_RETRIEVAL else None
        self.citation_tracker = CitationTracker() if ENABLE_CITATION_TRACKING else None
        
        logger.info(
            "EnhancedRetrieverWrapper initialized | "
            "adaptive=%s | reranking=%s | cache=%s | fallback=%s | citations=%s",
            bool(self.adaptive),
            bool(self.reranker),
            bool(self.semantic_cache),
            bool(self.fallback),
            bool(self.citation_tracker),
        )

    def retrieve_with_all_features(
        self, query: str, base_k: int = TOP_K, **kwargs
    ) -> Dict[str, Any]:
        """
        Full-featured retrieval pipeline combining all tiers.
        
        Args:
            query: User query
            base_k: Base top-k for retrieval
            **kwargs: Additional args to pass to base retriever
            
        Returns:
            {
                "results": [...],
                "citation_map": {...},  # If citations enabled
                "fallback_info": {...},  # If fallback used
                "cache_hit": bool,
                "adaptive_k": int,
                "confidence": {...},
            }
        """
        metadata = {
            "cache_hit": False,
            "adaptive_k": base_k,
            "fallback_used": False,
        }

        # Step 0: Semantic cache lookup
        cache_result = None
        if self.semantic_cache:
            try:
                query_embedding = self.embedder.embed_query(query)
                cache_result = self.semantic_cache.get(query, query_embedding)
                
                if cache_result:
                    logger.info("[CACHE HIT] Query reused with %d results", len(cache_result))
                    metadata["cache_hit"] = True
                    return self._wrap_results(
                        cache_result, query, metadata, from_cache=True
                    )
            except Exception as e:
                logger.warning("[CACHE] Lookup failed: %s", e)

        # Step 1: Adaptive retrieval (adjust k based on query complexity)
        effective_k = base_k
        if self.adaptive:
            effective_k = self.adaptive.adaptive_k(base_k, query)
            metadata["adaptive_k"] = effective_k
            logger.debug(f"Adaptive k: {base_k} -> {effective_k}")

        # Step 2: Retrieve with fallback strategies
        if self.fallback:
            retrieval_data, fallback_meta = self.fallback.retrieve_with_fallback(query, effective_k)
            metadata.update(fallback_meta)
            results = retrieval_data
        else:
            retrieval_data = self.base_retriever.retrieve(query, top_k=effective_k, **kwargs)
            if isinstance(retrieval_data, dict):
                results = retrieval_data.get("results", [])
            else:
                results = retrieval_data

        # Step 3: LLM-based re-ranking
        if self.reranker and results:
            try:
                reranked = self.reranker.rerank(query, results, top_k=min(effective_k, len(results)))
                results = reranked
                logger.debug(f"Re-ranked {len(results)} results")
            except Exception as e:
                logger.warning(f"Re-ranking failed: {e}, using original results")

        # Step 4: Semantic caching - store for future use
        if self.semantic_cache and not metadata["cache_hit"]:
            try:
                query_embedding = self.embedder.embed_query(query)
                self.semantic_cache.set(query, query_embedding, results)
                logger.debug("Results cached for future queries")
            except Exception as e:
                logger.warning(f"Cache write failed: {e}")

        # Step 5: Citation tracking
        if self.citation_tracker and results:
            for chunk in results:
                chunk_id = chunk.get("id", chunk.get("chunk_id", ""))
                self.citation_tracker.add_chunk(
                    chunk_id,
                    chunk.get("content", ""),
                    chunk.get("metadata", {})
                )

        return self._wrap_results(results, query, metadata)

    def retrieve(self, query: str, top_k: int = TOP_K, **kwargs) -> Dict[str, Any]:
        """
        Retriever-compatible API used by LangGraphQueryOrchestrator.
        """
        wrapped = self.retrieve_with_all_features(query=query, base_k=top_k, **kwargs)
        query_language = "en"
        try:
            if hasattr(self.base_retriever, "lang_detector"):
                query_language = self.base_retriever.lang_detector.detect_language(query)
        except Exception:
            query_language = "en"

        return {
            "query_language": query_language,
            "results": wrapped.get("results", []),
            "_metadata": wrapped.get("_metadata", {}),
        }

    def _wrap_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        metadata: Dict[str, Any],
        from_cache: bool = False,
    ) -> Dict[str, Any]:
        """Wrap results with metadata."""
        wrapped = {
            "results": results,
            "_metadata": {
                **metadata,
                "total_results": len(results),
                "from_cache": from_cache,
            }
        }

        # Add citation map if available
        if self.citation_tracker and results:
            try:
                citation_map = self.citation_tracker.map_answer_to_chunks(
                    "", results
                )
                wrapped["citation_map"] = citation_map
            except Exception as e:
                logger.warning(f"Citation tracking failed: {e}")

        return wrapped

    def get_citation_prompt_suffix(self) -> str:
        """Get prompt instruction for including citations."""
        if not self.citation_tracker:
            return ""
        return self.citation_tracker.build_citation_prompt_suffix()

    def extract_answer_citations(self, answer: str) -> Dict[str, Any]:
        """Extract and map citations from generated answer."""
        if not self.citation_tracker:
            return {"citations": [], "clean_answer": answer}

        try:
            citations = self.citation_tracker.extract_citations_from_answer(answer)
            clean_answer, spans = self.citation_tracker.generate_span_citations(answer)
            
            return {
                "citations": citations,
                "citation_spans": spans,
                "clean_answer": clean_answer,
            }
        except Exception as e:
            logger.warning(f"Answer citation extraction failed: {e}")
            return {"citations": [], "clean_answer": answer}

    def cleanup_cache(self):
        """Cleanup expired cache entries."""
        if self.semantic_cache:
            try:
                self.semantic_cache.cleanup_expired()
            except Exception as e:
                logger.warning(f"Cache cleanup failed: {e}")


class EnhancedLangGraphQueryOrchestrator:
    """
    Wrapper for existing LangGraphQueryOrchestrator that injects enhanced features.
    Minimal changes to existing code - just wraps the retriever.
    """

    def __init__(self, base_orchestrator, llm_client):
        """
        Initialize enhanced orchestrator.
        
        Args:
            base_orchestrator: Existing LangGraphQueryOrchestrator
            llm_client: LLM client
        """
        self.base_orchestrator = base_orchestrator
        self.llm_client = llm_client
        
        # Replace retriever with enhanced version
        self.enhanced_retriever = EnhancedRetrieverWrapper(
            base_orchestrator.retriever,
            llm_client
        )
        self.base_orchestrator.retriever = self.enhanced_retriever

        logger.info("EnhancedLangGraphQueryOrchestrator initialized")

    def retrieve(self, query: str, top_k: int = TOP_K, **kwargs) -> Dict[str, Any]:
        """
        Retrieve using base orchestrator + enhancements.
        
        Returns combined results with enhancement metadata.
        """
        base_results = self.base_orchestrator.retrieve(query, top_k=top_k, **kwargs)

        base_results["_enhancements"] = {
            "calls_adaptive": ENABLE_ADAPTIVE_RETRIEVAL,
            "calls_reranking": ENABLE_LLM_RERANKING,
            "calls_caching": ENABLE_SEMANTIC_CACHE,
            "calls_fallback": ENABLE_FALLBACK_RETRIEVAL,
            "calls_citations": ENABLE_CITATION_TRACKING,
        }

        return base_results

    def get_citation_prompt_suffix(self) -> str:
        """Get citation prompt for LLM."""
        return self.enhanced_retriever.get_citation_prompt_suffix()

    def extract_answer_citations(self, answer: str) -> Dict[str, Any]:
        """Extract citations from generated answer."""
        return self.enhanced_retriever.extract_answer_citations(answer)

    def cleanup(self):
        """Cleanup resources."""
        self.enhanced_retriever.cleanup_cache()
