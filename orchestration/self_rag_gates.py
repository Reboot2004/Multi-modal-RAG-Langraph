# orchestration/self_rag_gates.py
"""
Self-RAG Quality Control Gates

Implements four quality checkpoints to prevent hallucinations and improve confidence:
1. Retrieval Gate: Should we even retrieve for this query?
2. Relevance Gate: Are the retrieved docs actually relevant?
3. Faithfulness Gate: Does the answer stick to retrieved facts?
4. Usefulness Gate: Is the answer actually useful for the question?
"""

from typing import List, Dict, Tuple
from llm.groq_client import GroqClient
from pipeline_logger import get_logger
from config.settings import (
    SELF_RAG_DOC_RELEVANCE_THRESHOLD,
    SELF_RAG_FAITHFULNESS_THRESHOLD,
    SELF_RAG_USEFULNESS_THRESHOLD,
    SELF_RAG_HARD_REFUSAL_ENABLED,
    SELF_RAG_HARD_REFUSAL_THRESHOLD,
    SELF_RAG_WEIGHT_RETRIEVAL,
    SELF_RAG_WEIGHT_RELEVANCE,
    SELF_RAG_WEIGHT_FAITHFULNESS,
    SELF_RAG_WEIGHT_USEFULNESS,
)

logger = get_logger("self_rag_gates")


class SelfRAGGates:
    """Quality control gates for RAG pipeline"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client or GroqClient()
        # Load thresholds from config
        self.retrieval_threshold = float(SELF_RAG_DOC_RELEVANCE_THRESHOLD)
        self.relevance_threshold = float(SELF_RAG_DOC_RELEVANCE_THRESHOLD)
        self.faithfulness_threshold = float(SELF_RAG_FAITHFULNESS_THRESHOLD)
        self.usefulness_threshold = float(SELF_RAG_USEFULNESS_THRESHOLD)
        
        # Hard refusal config
        self.hard_refusal_enabled = SELF_RAG_HARD_REFUSAL_ENABLED
        self.hard_refusal_threshold = float(SELF_RAG_HARD_REFUSAL_THRESHOLD)
        
        # Weight config
        self.weight_retrieval = float(SELF_RAG_WEIGHT_RETRIEVAL)
        self.weight_relevance = float(SELF_RAG_WEIGHT_RELEVANCE)
        self.weight_faithfulness = float(SELF_RAG_WEIGHT_FAITHFULNESS)
        self.weight_usefulness = float(SELF_RAG_WEIGHT_USEFULNESS)

    def gate_retrieval_needed(self, query: str) -> Tuple[bool, float, str]:
        """
        Gate 1: Determine if this query needs retrieval.
        
        Some queries (greetings, API info, status) don't benefit from retrieval.
        Others (factual, code, domain-specific) require document context.
        
        Returns:
            (retrieval_needed: bool, confidence: float, reason: str)
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a query classifier. Determine if a query needs document retrieval. "
                    "Respond with ONLY: RETRIEVE or SKIP, followed by confidence (0.0-1.0)"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    "Respond in format: RETRIEVE|SKIP <confidence> <reason>\n"
                    "Examples:\n"
                    "- RETRIEVE 0.95 specific factual information needed\n"
                    "- SKIP 0.9 greeting requires no context\n"
                    "- RETRIEVE 0.8 might refer to specific domain knowledge"
                ),
            },
        ]

        try:
            response = self.llm_client.generate(
                messages, max_tokens=50, temperature=0.1
            )
            parts = response.strip().split()
            if len(parts) < 2:
                return True, 0.8, "default_retrieve"

            decision = parts[0].upper()
            confidence = float(parts[1]) if len(parts) > 1 else 0.5
            reason = " ".join(parts[2:]) if len(parts) > 2 else decision.lower()

            needs_retrieval = decision == "RETRIEVE"
            confidence = max(0.0, min(1.0, confidence))

            logger.info(
                "Gate 1: Retrieval needed | query_len=%d | needs=%s | confidence=%.2f | reason=%s",
                len(query),
                needs_retrieval,
                confidence,
                reason,
            )

            return needs_retrieval, confidence, reason

        except Exception as ex:
            logger.warning("Gate 1 failed; defaulting to retrieve | error=%s", ex)
            return True, 0.5, f"error: {ex}"

    def gate_doc_relevance(
        self, query: str, retrieved_docs: List[Dict]
    ) -> Tuple[bool, float, str]:
        """
        Gate 2: Check if retrieved documents are relevant to the query.
        
        Returns:
            (all_relevant: bool, avg_confidence: float, reason: str)
        """
        if not retrieved_docs:
            return False, 0.0, "no_documents_retrieved"

        # Sample up to 2 docs to avoid excessive LLM calls
        sample_docs = retrieved_docs[:2]
        doc_texts = [
            f"[{d['metadata'].get('source', '?')}] {d['text'][:200]}"
            for d in sample_docs
        ]
        doc_context = "\n---\n".join(doc_texts)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a relevance judge. For each document, determine if it's relevant to the query. "
                    "Respond with RELEVANT or IRRELEVANT followed by confidence (0.0-1.0)"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Documents:\n{doc_context}\n\n"
                    "After reviewing, respond:\n"
                    "RELEVANT <confidence> OR IRRELEVANT <confidence>"
                ),
            },
        ]

        try:
            response = self.llm_client.generate(
                messages, max_tokens=50, temperature=0.1
            )
            parts = response.strip().split()
            if len(parts) < 2:
                return True, 0.7, "parse_failed"

            decision = parts[0].upper()
            confidence = float(parts[1]) if len(parts) > 1 else 0.5
            confidence = max(0.0, min(1.0, confidence))

            is_relevant = decision == "RELEVANT"

            logger.info(
                "Gate 2: Doc relevance | docs_count=%d | relevant=%s | confidence=%.2f",
                len(retrieved_docs),
                is_relevant,
                confidence,
            )

            return is_relevant, confidence, decision.lower()

        except Exception as ex:
            logger.warning("Gate 2 failed; assuming relevant | error=%s", ex)
            return True, 0.5, f"error: {ex}"

    def gate_answer_faithfulness(
        self, query: str, answer: str, retrieved_docs: List[Dict]
    ) -> Tuple[bool, float, str]:
        """
        Gate 3: Check if the answer is faithful to retrieved documents.
        
        Evaluates:
        - Does the answer come from the retrieved context?
        - Are claims grounded in the documents?
        - Are there hallucinations or contradictions?
        
        Returns:
            (faithful: bool, confidence: float, reason: str)
        """
        if not retrieved_docs:
            return False, 0.0, "no_docs_available"

        # Use top doc as reference
        ref_doc = retrieved_docs[0]
        doc_text = ref_doc["text"][:500]
        source = ref_doc["metadata"].get("source", "unknown")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a faithfulness judge. Determine if an answer is faithful to source documents. "
                    "Respond with FAITHFUL or UNFAITHFUL followed by confidence (0.0-1.0)"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Source document (from {source}):\n{doc_text}\n\n"
                    f"Question: {query}\n\n"
                    f"Answer: {answer}\n\n"
                    "Is the answer faithful to the source? "
                    "Respond: FAITHFUL <confidence> OR UNFAITHFUL <confidence>"
                ),
            },
        ]

        try:
            response = self.llm_client.generate(
                messages, max_tokens=50, temperature=0.0
            )
            parts = response.strip().split()
            if len(parts) < 2:
                return True, 0.6, "parse_failed"

            decision = parts[0].upper()
            confidence = float(parts[1]) if len(parts) > 1 else 0.5
            confidence = max(0.0, min(1.0, confidence))

            is_faithful = decision == "FAITHFUL"

            logger.info(
                "Gate 3: Answer faithfulness | faithful=%s | confidence=%.2f | source=%s",
                is_faithful,
                confidence,
                source,
            )

            return is_faithful, confidence, decision.lower()

        except Exception as ex:
            logger.warning("Gate 3 failed; assuming faithful | error=%s", ex)
            return True, 0.5, f"error: {ex}"

    def gate_answer_usefulness(
        self, query: str, answer: str
    ) -> Tuple[bool, float, str]:
        """
        Gate 4: Check if the answer is useful for the query.
        
        Evaluates:
        - Does the answer address the question?
        - Is it complete or partial?
        - Would a user find this helpful?
        
        Returns:
            (useful: bool, confidence: float, reason: str)
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a usefulness evaluator. Rate if an answer is useful for the question. "
                    "Respond with USEFUL or NOT_USEFUL followed by confidence (0.0-1.0)"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"Answer: {answer}\n\n"
                    "Is this answer useful and relevant? "
                    "Respond: USEFUL <confidence> OR NOT_USEFUL <confidence>"
                ),
            },
        ]

        try:
            response = self.llm_client.generate(
                messages, max_tokens=50, temperature=0.1
            )
            parts = response.strip().split()
            if len(parts) < 2:
                return True, 0.7, "parse_failed"

            decision = parts[0].upper()
            confidence = float(parts[1]) if len(parts) > 1 else 0.5
            confidence = max(0.0, min(1.0, confidence))

            is_useful = decision == "USEFUL"

            logger.info(
                "Gate 4: Answer usefulness | useful=%s | confidence=%.2f",
                is_useful,
                confidence,
            )

            return is_useful, confidence, decision.lower()

        except Exception as ex:
            logger.warning("Gate 4 failed; assuming useful | error=%s", ex)
            return True, 0.6, f"error: {ex}"

    def compute_overall_confidence(
        self,
        retrieval_confidence: float = 1.0,
        doc_relevance_confidence: float = 1.0,
        faithfulness_confidence: float = 1.0,
        usefulness_confidence: float = 1.0,
    ) -> float:
        """
        Compute overall confidence score (0-1) as weighted average of all gates.
        
        Weights are configurable in settings.py (SELF_RAG_WEIGHT_*)
        """
        # Use configured weights
        overall = (
            retrieval_confidence * self.weight_retrieval
            + doc_relevance_confidence * self.weight_relevance
            + faithfulness_confidence * self.weight_faithfulness
            + usefulness_confidence * self.weight_usefulness
        )
        return max(0.0, min(1.0, overall))

    def should_refuse_answer(self, confidence: float) -> Tuple[bool, str]:
        """
        Determine if answer should be refused due to low confidence (hard refusal).
        
        Returns:
            (should_refuse: bool, reason: str)
        """
        if not self.hard_refusal_enabled:
            return False, "hard_refusal_disabled"
        
        if confidence < self.hard_refusal_threshold:
            return True, f"low_confidence_{confidence:.2f}"
        
        return False, "confidence_acceptable"

    def get_confidence_badge(self, confidence: float) -> Tuple[str, str]:
        """
        Return (emoji, label) for confidence score.
        
        Returns:
            (badge: str, level: str)
        """
        if confidence >= 0.8:
            return "🟢", "HIGH"  # Green - high confidence
        elif confidence >= 0.6:
            return "🟡", "MEDIUM"  # Yellow - medium confidence
        else:
            return "🔴", "LOW"  # Red - low confidence
