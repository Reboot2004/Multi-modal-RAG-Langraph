# hierarchical_retriever.py
"""
Hierarchical Retrieval Module
Implements two-stage retrieval: document-level first, then chunk-level within relevant docs.
Reduces noise and improves retrieval coherence.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from embeddings.vector_store import VectorStore


class HierarchicalRetriever:
    """
    Two-stage retrieval strategy:
    1. Document-level: Find N most relevant documents (by aggregating chunk scores)
    2. Chunk-level: Within top N docs, retrieve top-k chunks
    """

    def __init__(self, vector_store: VectorStore):
        """
        Initialize with vector store instance.
        
        Args:
            vector_store: VectorStore with FAISS index
        """
        self.vector_store = vector_store

    def retrieve_hierarchical(
        self,
        query_embedding: np.ndarray,
        top_documents: int = 5,
        top_chunks_per_doc: int = 3,
        total_chunks: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform hierarchical retrieval.
        
        Args:
            query_embedding: Query embedding vector
            top_documents: Number of documents to retrieve at first stage
            top_chunks_per_doc: Number of chunks to retrieve from each doc
            total_chunks: Total limit on returned chunks (if None, computed from k*n)
            
        Returns:
            Dict with keys:
            - results: List of chunk dicts (with scores and doc info)
            - stage1_docs: List of doc identifiers from stage 1
            - retrieval_method: str, "hierarchical"
            - total_chunks_returned: int
        """
        if total_chunks is None:
            total_chunks = top_documents * top_chunks_per_doc

        # Stage 1: Document-level retrieval (aggregate chunks by document source)
        stage1_results = self._retrieve_documents(query_embedding, top_documents)
        
        # Extract unique document sources
        top_doc_sources = [doc["source"] for doc in stage1_results]
        
        # Stage 2: Chunk-level retrieval within top documents
        all_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=max(total_chunks * 2, top_chunks_per_doc * max(1, len(top_doc_sources))),
        )
        
        filtered_results = [
            result for result in all_results
            if result.get("metadata", {}).get("source") in top_doc_sources
        ][:total_chunks]
        
        return {
            "results": filtered_results,
            "stage1_docs": top_doc_sources,
            "retrieval_method": "hierarchical",
            "total_chunks_returned": len(filtered_results),
            "stage1_doc_count": len(stage1_results),
            "stage2_chunks_per_doc": top_chunks_per_doc,
        }

    def _retrieve_documents(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top documents by aggregating chunk relevance scores.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of documents to return
            
        Returns:
            List of dicts with 'source' and 'avg_score' keys
        """
        # First, get many chunks to ensure we cover multiple documents
        expanded_k = max(100, top_k * 20)  # Retrieve many chunks for aggregation
        chunk_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=expanded_k,
        )
        
        # Aggregate scores by document source
        doc_scores = {}
        for result in chunk_results:
            source = result.get("metadata", {}).get("source", "unknown")
            score = float(result.get("score", 0.0))
            
            if source not in doc_scores:
                doc_scores[source] = []
            doc_scores[source].append(score)
        
        # Compute average score per document
        doc_stats = []
        for source, scores in doc_scores.items():
            avg_score = np.mean(scores) if scores else 0.0
            max_score = np.max(scores) if scores else 0.0
            doc_stats.append({
                "source": source,
                "avg_score": float(avg_score),
                "max_score": float(max_score),
                "chunk_count": len(scores),
            })
        
        # Sort by average score (descending) and return top-k
        doc_stats.sort(key=lambda x: x["avg_score"], reverse=True)
        return doc_stats[:top_k]

    def retrieve_with_diversity(
        self,
        query_embedding: np.ndarray,
        top_documents: int = 5,
        top_chunks_per_doc: int = 3,
        diversity_penalty: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Hierarchical retrieval with diversity penalty to reduce redundancy.
        
        Args:
            query_embedding: Query embedding
            top_documents: Number of documents at stage 1
            top_chunks_per_doc: Number of chunks per document
            diversity_penalty: Penalty multiplier for similar embeddings (0-1)
            
        Returns:
            Hierarchical retrieval result with diversity applied
        """
        base_result = self.retrieve_hierarchical(
            query_embedding=query_embedding,
            top_documents=top_documents,
            top_chunks_per_doc=top_chunks_per_doc,
        )
        
        # Apply diversity re-ranking
        diversified = self._apply_diversity_penalty(
            base_result["results"],
            query_embedding,
            diversity_penalty,
        )
        
        base_result["results"] = diversified
        base_result["diversity_penalty_applied"] = True
        
        return base_result

    def _apply_diversity_penalty(
        self,
        results: List[Dict[str, Any]],
        query_embedding: np.ndarray,
        penalty: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results to penalize similar chunks (reduce redundancy).
        
        Args:
            results: List of retrieved chunks
            query_embedding: Query embedding for reference
            penalty: Similarity penalty multiplier
            
        Returns:
            Re-ranked results list
        """
        if not results or len(results) < 2:
            return results
        
        # Try to extract embeddings if available
        selected = []
        
        for result in results:
            # If we don't have chunk embeddings, use positional diversity
            # (prioritize chunks from different fragments of same doc)
            unique_enough = self._is_diverse_from_selected(result, selected, penalty)
            if unique_enough:
                selected.append(result)
        
        # Return selected results in original order
        return [r for r in results if r in selected]

    def _is_diverse_from_selected(
        self,
        candidate: Dict[str, Any],
        selected: List[Dict[str, Any]],
        penalty: float,
    ) -> bool:
        """Check if candidate is diverse enough from already selected items."""
        if not selected:
            return True
        
        # Simple heuristic: if same source as recent selection, skip
        candidate_source = candidate.get("metadata", {}).get("source")
        for recent in selected[-3:]:  # Check last 3 selected
            if recent.get("metadata", {}).get("source") == candidate_source:
                # Same source and similar page = probably redundant
                if recent.get("metadata", {}).get("page") == candidate.get("metadata", {}).get("page"):
                    return False
        
        return True

    def get_hierarchy_stats(self, retrieval_result: Dict[str, Any]) -> Dict[str, Any]:
        """Compute statistics about the hierarchical retrieval."""
        results = retrieval_result.get("results", [])
        stage1_docs = retrieval_result.get("stage1_docs", [])
        
        # Group by source
        by_source = {}
        for r in results:
            source = r.get("metadata", {}).get("source")
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(r)
        
        return {
            "stage1_doc_count": len(stage1_docs),
            "total_chunks": len(results),
            "docs_with_chunks": len(by_source),
            "chunks_per_doc_avg": len(results) / len(by_source) if by_source else 0,
            "chunks_per_doc_dist": {src: len(chunks) for src, chunks in by_source.items()},
        }
