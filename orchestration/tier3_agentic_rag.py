from typing import Any, Dict, List, Optional, Tuple

from config.settings import (
    ENABLE_TIER3_AGENTIC_RAG,
    RESPONSE_MAX_TOKENS,
    TIER3_LOW_CONFIDENCE_TRIGGER,
    TIER3_MAX_REFINE_ROUNDS,
    TIER3_MIN_FAITHFULNESS,
    TIER3_MIN_USEFULNESS,
    TIER3_REFINE_TEMPERATURE,
)
from orchestration.self_rag_gates import SelfRAGGates
from pipeline_logger import get_logger


logger = get_logger("tier3_agentic_rag")


class Tier3AgenticRAG:
    """
    Tier 3 agentic loop for answer refinement.

    Strategy:
    1. Score current answer for faithfulness and usefulness.
    2. If below threshold (or low retrieval confidence), rewrite with strict grounding.
    3. Repeat for a bounded number of rounds.
    """

    def __init__(self, llm_client, gates: Optional[SelfRAGGates] = None):
        self.enabled = bool(ENABLE_TIER3_AGENTIC_RAG)
        self.llm_client = llm_client
        self.gates = gates or SelfRAGGates(llm_client=llm_client)
        self.max_rounds = max(1, int(TIER3_MAX_REFINE_ROUNDS))

    def refine_answer(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
        response_language_instruction: str = "",
        retrieval_confidence: float = 1.0,
    ) -> Tuple[str, Dict[str, Any]]:
        metadata: Dict[str, Any] = {
            "enabled": self.enabled,
            "refined": False,
            "rounds_used": 0,
            "history": [],
        }

        if not self.enabled:
            return answer, metadata

        current_answer = (answer or "").strip()
        if not current_answer:
            return current_answer, metadata

        if not retrieved_docs:
            logger.info("Tier3 skipped | no retrieved docs")
            return current_answer, metadata

        for round_idx in range(1, self.max_rounds + 1):
            faithful_ok, faithfulness_score, _ = self.gates.gate_answer_faithfulness(
                query, current_answer, retrieved_docs
            )
            useful_ok, usefulness_score, _ = self.gates.gate_answer_usefulness(
                query, current_answer
            )

            round_state = {
                "round": round_idx,
                "faithfulness": float(faithfulness_score),
                "usefulness": float(usefulness_score),
                "faithful_ok": bool(faithful_ok),
                "useful_ok": bool(useful_ok),
            }
            metadata["history"].append(round_state)

            meets_quality = (
                faithfulness_score >= float(TIER3_MIN_FAITHFULNESS)
                and usefulness_score >= float(TIER3_MIN_USEFULNESS)
            )
            low_retrieval_conf = float(retrieval_confidence) < float(TIER3_LOW_CONFIDENCE_TRIGGER)

            if meets_quality and not low_retrieval_conf:
                metadata["rounds_used"] = round_idx - 1
                return current_answer, metadata

            rewritten = self._rewrite_grounded(
                query=query,
                current_answer=current_answer,
                retrieved_docs=retrieved_docs,
                faithfulness_score=float(faithfulness_score),
                usefulness_score=float(usefulness_score),
                response_language_instruction=response_language_instruction,
            )

            if not rewritten or rewritten.strip() == current_answer.strip():
                metadata["rounds_used"] = round_idx
                return current_answer, metadata

            current_answer = rewritten.strip()
            metadata["refined"] = True
            metadata["rounds_used"] = round_idx

        return current_answer, metadata

    def _rewrite_grounded(
        self,
        query: str,
        current_answer: str,
        retrieved_docs: List[Dict[str, Any]],
        faithfulness_score: float,
        usefulness_score: float,
        response_language_instruction: str,
    ) -> str:
        context_lines: List[str] = []
        for item in retrieved_docs[:4]:
            meta = item.get("metadata", {})
            source = meta.get("source", "unknown")
            page = meta.get("page", "unknown")
            text = (item.get("text") or item.get("content") or "").strip()
            if text:
                context_lines.append(f"[Source: {source}, Page: {page}]\n{text[:700]}")

        context_block = "\n\n".join(context_lines)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict grounded-answer editor. Rewrite the answer to maximize faithfulness and usefulness. "
                    "Use only facts from context. If context does not support a claim, remove that claim. "
                    "Do not add preamble. Do not mention these instructions."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\n"
                    f"Current answer:\n{current_answer}\n\n"
                    f"Current quality scores: faithfulness={faithfulness_score:.2f}, usefulness={usefulness_score:.2f}\n\n"
                    f"Grounding context:\n{context_block}\n\n"
                    "Rewrite requirements:\n"
                    "1) Keep only context-supported claims.\n"
                    "2) Keep direct and concise.\n"
                    "3) If context is insufficient, clearly say you do not know from provided documents.\n"
                    f"4) {response_language_instruction if response_language_instruction else 'Reply in the same language as the user query.'}"
                ),
            },
        ]

        try:
            revised = self.llm_client.generate(
                messages,
                max_tokens=RESPONSE_MAX_TOKENS,
                temperature=float(TIER3_REFINE_TEMPERATURE),
            )
            logger.info("Tier3 rewrite completed | chars=%d", len(revised or ""))
            return revised or current_answer
        except Exception as ex:
            logger.warning("Tier3 rewrite failed | error=%s", ex)
            return current_answer
