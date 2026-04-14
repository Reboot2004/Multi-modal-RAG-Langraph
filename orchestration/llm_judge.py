import json
import re
from typing import Any, Dict, List

from config.settings import LLM_JUDGE_TEMPERATURE
from pipeline_logger import get_logger


logger = get_logger("llm_judge")


class LLMJudge:
    """LLM-as-a-judge for retrieval and answer quality."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def evaluate(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        answer: str,
        expected_language_code: str,
        expected_language_name: str,
    ) -> Dict[str, Any]:
        doc_snippets = []
        for idx, item in enumerate(retrieved_docs[:5], start=1):
            meta = item.get("metadata", {})
            source = meta.get("source", "unknown")
            page = meta.get("page", "unknown")
            text = (item.get("text") or item.get("content") or "").strip()
            if not text:
                continue
            doc_snippets.append(f"[{idx}] {source} p.{page}: {text[:450]}")

        retrieval_block = "\n\n".join(doc_snippets) if doc_snippets else "NO_RETRIEVAL_DOCS"

        prompt = (
            "You are a strict judge for a RAG system. "
            "Evaluate BOTH retrieval quality and answer quality.\n\n"
            "Return ONLY JSON with this schema:\n"
            "{\n"
            "  \"retrieval\": {\"relevance\": 0.0-1.0, \"coverage\": 0.0-1.0, \"noise\": 0.0-1.0},\n"
            "  \"generation\": {\"faithfulness\": 0.0-1.0, \"completeness\": 0.0-1.0, \"language_adherence\": 0.0-1.0},\n"
            "  \"overall_score\": 0.0-1.0,\n"
            "  \"verdict\": \"pass\" | \"caution\" | \"fail\",\n"
            "  \"notes\": \"short reasoning\"\n"
            "}\n\n"
            f"Expected answer language: {expected_language_name} ({expected_language_code}).\n"
            "Language adherence should be low if answer includes extra languages.\n\n"
            f"Query:\n{query}\n\n"
            f"Retrieved docs:\n{retrieval_block}\n\n"
            f"Answer:\n{answer}\n"
        )

        try:
            raw = self.llm_client.generate(
                [{"role": "user", "content": prompt}],
                max_tokens=320,
                temperature=float(LLM_JUDGE_TEMPERATURE),
            )
            parsed = self._parse_json(raw)
            if parsed:
                return parsed
        except Exception as ex:
            logger.warning("LLM judge call failed | error=%s", ex)

        return {
            "retrieval": {"relevance": 0.5, "coverage": 0.5, "noise": 0.5},
            "generation": {"faithfulness": 0.5, "completeness": 0.5, "language_adherence": 0.5},
            "overall_score": 0.5,
            "verdict": "caution",
            "notes": "judge_fallback",
        }

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        if not raw:
            return {}

        text = raw.strip()
        try:
            obj = json.loads(text)
            return self._normalize(obj)
        except Exception:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {}

        try:
            obj = json.loads(match.group(0))
            return self._normalize(obj)
        except Exception:
            return {}

    def _normalize(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        retrieval = obj.get("retrieval", {})
        generation = obj.get("generation", {})

        def clamp(v: Any, d: float = 0.5) -> float:
            try:
                x = float(v)
            except Exception:
                x = d
            return max(0.0, min(1.0, x))

        normalized = {
            "retrieval": {
                "relevance": clamp(retrieval.get("relevance")),
                "coverage": clamp(retrieval.get("coverage")),
                "noise": clamp(retrieval.get("noise")),
            },
            "generation": {
                "faithfulness": clamp(generation.get("faithfulness")),
                "completeness": clamp(generation.get("completeness")),
                "language_adherence": clamp(generation.get("language_adherence")),
            },
            "overall_score": clamp(obj.get("overall_score")),
            "verdict": str(obj.get("verdict", "caution")).lower(),
            "notes": str(obj.get("notes", ""))[:300],
        }

        if normalized["verdict"] not in {"pass", "caution", "fail"}:
            normalized["verdict"] = "caution"

        return normalized
