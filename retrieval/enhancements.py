"""
Retrieval Enhancements:
- Adaptive Retrieval: Adjust k based on query complexity
- LLM Re-ranking: Use Groq to re-score retrieved chunks
- Fallback Strategies: Graceful degradation when retrieval fails
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import json

import numpy as np
from config.settings import (
    ENABLE_ADAPTIVE_RETRIEVAL,
    ADAPTIVE_SIMPLE_QUERY_K_OFFSET,
    ADAPTIVE_COMPLEX_QUERY_K_OFFSET,
    ENABLE_LLM_RERANKING,
    LLM_RERANK_TOP_K,
    LLM_RERANK_BATCH_SIZE,
    ENABLE_FALLBACK_RETRIEVAL,
    FALLBACK_MIN_RESULTS,
    FALLBACK_MIN_CONFIDENCE,
    TOP_K,
)


class AdaptiveRetrieval:
    """Dynamically adjust retrieve k based on query characteristics."""

    @staticmethod
    def estimate_query_complexity(query: str) -> str:
        """
        Classify query complexity.
        
        Args:
            query: User query text
            
        Returns:
            'simple', 'moderate', or 'complex'
        """
        # Simple heuristics
        question_words = len(re.findall(r'\b(what|where|who|when)\b', query.lower()))
        conjunctions = len(re.findall(r'\b(and|or|but|however)\b', query.lower()))
        avg_word_length = np.mean([len(w) for w in query.split()]) if query else 0
        
        if question_words >= 2 or conjunctions >= 2:
            return "complex"
        elif conjunctions >= 1 or avg_word_length > 6:
            return "moderate"
        else:
            return "simple"

    @staticmethod
    def adaptive_k(base_k: int = TOP_K, query: str = "") -> int:
        """
        Compute effective k for retrieval.
        
        Args:
            base_k: Default top-k value
            query: User query (optional, for complexity estimation)
            
        Returns:
            Adjusted k value
        """
        if not ENABLE_ADAPTIVE_RETRIEVAL:
            return base_k
        
        if not query:
            return base_k
        
        complexity = AdaptiveRetrieval.estimate_query_complexity(query)
        
        if complexity == "simple":
            adjusted_k = max(1, base_k + ADAPTIVE_SIMPLE_QUERY_K_OFFSET)
        elif complexity == "complex":
            adjusted_k = base_k + ADAPTIVE_COMPLEX_QUERY_K_OFFSET
        else:
            adjusted_k = base_k
        
        print(f"[DEBUG] Query complexity: {complexity}, adjusted k: {adjusted_k}")
        return adjusted_k


class LLMReranker:
    """Use LLM to re-score and re-rank retrieval results."""

    def __init__(self, groq_client):
        """
        Initialize reranker.
        
        Args:
            groq_client: Groq client instance
        """
        self.groq_client = groq_client
        self.enabled = ENABLE_LLM_RERANKING

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = LLM_RERANK_TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank chunks using LLM relevance judgment.
        
        Args:
            query: User query
            chunks: Retrieved chunks with scores
            top_k: How many to re-rank (re-rank top 2x-3x of final k)
            
        Returns:
            Re-ranked chunks with updated scores
        """
        if not self.enabled or not chunks:
            return chunks

        # Limit reranking to top candidates for efficiency
        candidates_to_rerank = chunks[:min(len(chunks), top_k * 2)]
        
        # Batch process
        reranked = []
        
        for i in range(0, len(candidates_to_rerank), LLM_RERANK_BATCH_SIZE):
            batch = candidates_to_rerank[i:i + LLM_RERANK_BATCH_SIZE]
            batch_reranked = self._rerank_batch(query, batch)
            reranked.extend(batch_reranked)
        
        # Sort by new score and return top_k
        reranked = sorted(reranked, key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        return reranked[:top_k]

    def _rerank_batch(
        self, query: str, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Re-rank a batch of chunks.
        
        Args:
            query: User query
            chunks: Chunk batch
            
        Returns:
            Chunks with rerank_score added
        """
        if not chunks:
            return []
        
        try:
            # Build prompt
            chunk_texts = "\n\n".join(
                [f"[CHUNK {i+1}]\n{c.get('text', c.get('content', ''))[:500]}" for i, c in enumerate(chunks)]
            )
            
            rerank_prompt = f"""Given the query: "{query}"

Rank these chunks by relevance (1 = most relevant, {len(chunks)} = least relevant).
Respond ONLY with a JSON object mapping chunk indices to scores (0.0-1.0):
{{"0": 0.95, "1": 0.60, ...}}

Chunks:
{chunk_texts}"""
            
            # Call Groq
            response = self.groq_client.generate(
                [{"role": "user", "content": rerank_prompt}],
                max_tokens=220,
                temperature=0.0,
            )
            
            # Parse response
            try:
                scores_dict = json.loads(response)
                
                for i, chunk in enumerate(chunks):
                    score_key = str(i)
                    score = scores_dict.get(score_key, 0.5)
                    chunk["rerank_score"] = float(score)
                    chunk["original_score"] = chunk.get("score", 0.5)
                
                return chunks
                
            except (json.JSONDecodeError, ValueError):
                print(f"[WARN] Failed to parse rerank scores: {response}")
                for chunk in chunks:
                    chunk["rerank_score"] = chunk.get("score", 0.5)
                return chunks
        
        except Exception as e:
            print(f"[WARN] LLM reranking failed: {e}, using original scores")
            for chunk in chunks:
                chunk["rerank_score"] = chunk.get("score", 0.5)
            return chunks


class FallbackRetrieval:
    """Graceful degradation strategies when retrieval fails."""

    def __init__(self, retriever):
        """
        Initialize fallback handler.
        
        Args:
            retriever: Main retriever instance (must have fallback methods)
        """
        self.retriever = retriever
        self.enabled = ENABLE_FALLBACK_RETRIEVAL

    def retrieve_with_fallback(
        self, query: str, base_k: int = TOP_K
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Attempt retrieval with fallback strategies.
        
        Returns:
            (results, metadata with fallback_strategy info)
        """
        metadata = {"fallback_triggered": False, "strategy_used": "primary"}
        
        if not self.enabled:
            results = self._primary_retrieve(query, base_k)
            return results, metadata
        
        # Try primary retrieval
        results = self._primary_retrieve(query, base_k)
        
        # Check if fallback needed
        if self._should_trigger_fallback(results):
            metadata["fallback_triggered"] = True
            print(f"[DEBUG] Triggering fallback for query: {query[:50]}")
            
            # Strategy 1: Expand query and retry
            expanded_results = self._fallback_expand_query(query, base_k)
            if expanded_results:
                results = expanded_results
                metadata["strategy_used"] = "query_expansion"
            
            # Strategy 2: Increase k and retry
            if not expanded_results or len(expanded_results) < FALLBACK_MIN_RESULTS:
                increased_k_results = self._primary_retrieve(query, base_k * 2)
                if increased_k_results and len(increased_k_results) > len(results):
                    results = increased_k_results
                    metadata["strategy_used"] = "increased_k"
            
            # Strategy 3: Use lexical fallback if available
            if hasattr(self.retriever, 'retrieve_lexical'):
                lexical_results = self.retriever.retrieve_lexical(query, base_k)
                if lexical_results:
                    results.extend(lexical_results)
                    metadata["strategy_used"] = "hybrid_lexical"
        
        return results, metadata

    def _should_trigger_fallback(self, results: List[Dict[str, Any]]) -> bool:
        """Check if fallback is needed."""
        if not results:
            return True
        
        if len(results) < FALLBACK_MIN_RESULTS:
            return True
        
        max_score = max((r.get("score", 0) for r in results), default=0)
        if max_score < FALLBACK_MIN_CONFIDENCE:
            return True
        
        return False

    def _primary_retrieve(
        self, query: str, k: int
    ) -> List[Dict[str, Any]]:
        """Execute primary retrieval."""
        try:
            raw = self.retriever.retrieve(query, top_k=k)
            if isinstance(raw, dict):
                return raw.get("results", [])
            return raw
        except Exception as e:
            print(f"[WARN] Primary retrieval failed: {e}")
            return []

    def _fallback_expand_query(
        self, query: str, k: int
    ) -> Optional[List[Dict[str, Any]]]:
        """Fallback: Expand query and retry."""
        try:
            if not hasattr(self.retriever, 'expand_query'):
                return None
            
            expanded = self.retriever.expand_query(query)
            expanded_results = []
            
            for exp_query in expanded:
                results = self._primary_retrieve(exp_query, k // len(expanded))
                expanded_results.extend(results)
            
            return expanded_results if expanded_results else None
        
        except Exception as e:
            print(f"[WARN] Query expansion fallback failed: {e}")
            return None

    def retrieve_with_confidence(
        self, query: str, base_k: int = TOP_K
    ) -> Dict[str, Any]:
        """
        Return results with confidence metadata.
        
        Returns:
            {
                "results": [...],
                "confidence": {
                    "overall": 0.0-1.0,
                    "reasoning": str,
                    "fallback_triggered": bool,
                    "strategy": str,
                },
            }
        """
        results, metadata = self.retrieve_with_fallback(query, base_k)
        
        # Calculate confidence
        if not results:
            confidence = 0.0
            reasoning = "No results found"
        elif len(results) < FALLBACK_MIN_RESULTS:
            confidence = 0.5
            reasoning = f"Low result count: {len(results)}"
        else:
            avg_score = np.mean([r.get("score", 0) for r in results])
            confidence = min(0.95, avg_score)  # Cap at 0.95
            reasoning = f"Average relevance: {avg_score:.2f}"
        
        if metadata.get("fallback_triggered"):
            confidence *= 0.8  # Reduce confidence after fallback
            reasoning += f" (fallback: {metadata['strategy_used']})"
        
        return {
            "results": results,
            "confidence": {
                "overall": confidence,
                "reasoning": reasoning,
                "fallback_triggered": metadata["fallback_triggered"],
                "strategy": metadata["strategy_used"],
            },
        }
