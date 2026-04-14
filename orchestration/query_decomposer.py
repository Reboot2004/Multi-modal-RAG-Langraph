# query_decomposer.py
"""
Query Decomposition Module
Decomposes multi-part questions into independent sub-queries for targeted retrieval.
"""

from typing import List, Dict, Any
import re


class QueryDecomposer:
    """
    Decomposes complex, multi-part queries into simpler sub-queries.
    Uses LLM to identify logical question boundaries and reformulate each part.
    """

    def __init__(self, llm_client):
        """
        Initialize decomposer with an LLM client for query analysis.
        
        Args:
            llm_client: Configured LLM client for query decomposition
        """
        self.llm_client = llm_client

    def detect_multi_part(self, query: str) -> bool:
        """
        Heuristically detect if query is multi-part (contains multiple questions).
        
        Args:
            query: Input query string
            
        Returns:
            True if query appears to have multiple parts
        """
        if not query or len(query) < 20:
            return False
        
        # Count question marks, "and", "also", "additionally"
        q_count = query.count("?")
        logical_conjuncts = sum([
            len(re.findall(r"\band\b", query, re.IGNORECASE)),
            len(re.findall(r"\balso\b", query, re.IGNORECASE)),
            len(re.findall(r"\badditionally\b", query, re.IGNORECASE)),
            len(re.findall(r"\bfurthermore\b", query, re.IGNORECASE)),
            len(re.findall(r"\bmoreover\b", query, re.IGNORECASE)),
        ])
        
        # Signal if multiple questions or multiple logical parts detected
        return q_count > 1 or logical_conjuncts > 1

    def decompose(
        self,
        query: str,
        max_sub_queries: int = 5,
    ) -> Dict[str, Any]:
        """
        Decompose query into sub-queries using LLM.
        
        Args:
            query: Input query to decompose
            max_sub_queries: Maximum number of sub-queries to generate
            
        Returns:
            Dict with keys:
            - is_multi_part: bool, whether decomposition was applied
            - original_query: str, original query
            - sub_queries: List[str], list of decomposed queries (empty if not multi-part)
            - decomposition_notes: str, explanation of decomposition logic
        """
        if not self.detect_multi_part(query):
            return {
                "is_multi_part": False,
                "original_query": query,
                "sub_queries": [],
                "decomposition_notes": "Single-part query; no decomposition needed.",
            }
        
        # Build decomposition prompt
        prompt = self._build_decomposition_prompt(query, max_sub_queries)
        
        try:
            response = self.llm_client.generate(
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You are an expert at analyzing complex questions and breaking them down into logically independent sub-questions. "
                            "Each sub-question should be self-contained and retrievable independently.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_tokens=1000,
                temperature=0.1,  # Deterministic decomposition
            )
            
            sub_queries = self._parse_decomposition_response(response)
            
            return {
                "is_multi_part": True,
                "original_query": query,
                "sub_queries": sub_queries,
                "decomposition_notes": f"Decomposed into {len(sub_queries)} sub-queries for targeted retrieval.",
            }
        except Exception as e:
            return {
                "is_multi_part": True,
                "original_query": query,
                "sub_queries": [],
                "decomposition_notes": f"Decomposition failed: {e}. Using original query.",
            }

    def _build_decomposition_prompt(self, query: str, max_sub_queries: int) -> str:
        """Build the LLM prompt for query decomposition."""
        return (
            f"Analyze the following complex question and break it down into {max_sub_queries} or fewer "
            f"logically independent sub-questions. Each sub-question should be retrievable independently.\n\n"
            f"Original question:\n{query}\n\n"
            f"Return ONLY a numbered list of sub-questions, one per line. Example format:\n"
            f"1. First sub-question?\n"
            f"2. Second sub-question?\n"
            f"3. Third sub-question?\n\n"
            f"Sub-questions:"
        )

    def _parse_decomposition_response(self, response: str) -> List[str]:
        """Parse LLM sub-queries from response."""
        if not response:
            return []
        
        lines = response.strip().split("\n")
        sub_queries = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering (e.g., "1. ", "1) ")
            cleaned = re.sub(r"^[\d]+[.\)]\s+", "", line)
            # Remove trailing question mark if present (will be added back if needed)
            cleaned = cleaned.rstrip("?").strip()
            if cleaned:
                sub_queries.append(cleaned)
        
        return sub_queries[:5]  # Limit to 5 sub-queries max

    def fuse_sub_results(
        self,
        original_query: str,
        sub_queries: List[str],
        sub_results: List[List[Dict[str, Any]]],
        llm_client,
    ) -> str:
        """
        Fuse sub-query results into a cohesive final answer.
        
        Args:
            original_query: Original multi-part query
            sub_queries: List of sub-queries
            sub_results: List of retrieved doc lists (one per sub-query)
            llm_client: LLM client for answer fusion
            
        Returns:
            Fused answer addressing all sub-questions
        """
        if not sub_queries or not sub_results:
            return ""
        
        # Build context from all retrieved docs
        context_parts = []
        for i, (sub_q, docs) in enumerate(zip(sub_queries, sub_results)):
            if docs:
                context_parts.append(f"\n### Part {i+1}: {sub_q}")
                for doc in docs[:3]:  # Top 3 per sub-query
                    content = (doc.get("text", "") or doc.get("page_content", ""))[:300]
                    source = doc.get("metadata", {}).get("source", "unknown")
                    context_parts.append(f"**Source: {source}**\n{content}\n")
        
        full_context = "".join(context_parts)
        
        # Build fusion prompt
        fusion_prompt = (
            f"You have received search results for multiple related questions:\n\n"
            f"Original query: {original_query}\n\n"
            f"Sub-questions:\n"
            + "\n".join(f"{i+1}. {sq}" for i, sq in enumerate(sub_queries))
            + f"\n\nRetrieved context:\n{full_context}\n\n"
            f"Now synthesize a comprehensive answer that addresses all sub-questions cohesively. "
            f"Structure your answer to cover each part clearly. Use the retrieved context to support each part."
        )
        
        try:
            fused_answer = llm_client.generate(
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You are an expert at synthesizing information from multiple sources to provide comprehensive, cohesive answers.",
                    },
                    {
                        "role": "user",
                        "content": fusion_prompt,
                    },
                ],
                max_tokens=2000,
                temperature=0.3,
            )
            return fused_answer or ""
        except Exception as e:
            return f"Error fusing results: {e}"
