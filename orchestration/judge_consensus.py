from typing import Dict, Any, List

from llm.client_factory import build_llm_client
from orchestration.llm_judge import LLMJudge


class JudgeConsensus:
    """Runs multiple judge passes and aggregates into a consensus score."""

    def __init__(self, llm_client, judges: int = 2, judge_specs: List[Dict[str, str]] = None):
        self.primary_client = llm_client
        self.judges = max(1, int(judges))
        self.judge_specs = judge_specs or []

    def _build_judge_clients(self):
        clients = []

        if self.primary_client is not None:
            clients.append(self.primary_client)

        for spec in self.judge_specs:
            provider = (spec or {}).get("provider")
            model = (spec or {}).get("model")
            if not provider:
                continue
            try:
                client = build_llm_client(provider=provider, model_name=model)
                clients.append(client)
            except Exception:
                continue

        # Deduplicate by provider/model identity if possible.
        unique = []
        seen = set()
        for c in clients:
            provider = getattr(c, "__class__", type(c)).__name__
            model = getattr(c, "model_name", "")
            key = f"{provider}:{model}"
            if key in seen:
                continue
            seen.add(key)
            unique.append(c)

        return unique[: self.judges]

    def evaluate(self, **kwargs) -> Dict[str, Any]:
        runs: List[Dict[str, Any]] = []

        clients = self._build_judge_clients()
        if not clients and self.primary_client is not None:
            clients = [self.primary_client]

        for client in clients:
            judge = LLMJudge(llm_client=client)
            runs.append(judge.evaluate(**kwargs))

        overall_scores = [float(r.get("overall_score", 0.5)) for r in runs]
        avg_score = sum(overall_scores) / len(overall_scores)
        disagreement = max(overall_scores) - min(overall_scores) if overall_scores else 0.0

        verdict = "pass"
        if avg_score < 0.5:
            verdict = "fail"
        elif avg_score < 0.7:
            verdict = "caution"

        consensus = dict(runs[0]) if runs else {
            "retrieval": {"relevance": 0.5, "coverage": 0.5, "noise": 0.5},
            "generation": {"faithfulness": 0.5, "completeness": 0.5, "language_adherence": 0.5},
        }
        consensus["overall_score"] = float(avg_score)
        consensus["verdict"] = verdict
        consensus["consensus_meta"] = {
            "judges": len(runs),
            "disagreement": float(disagreement),
            "scores": overall_scores,
        }
        return consensus
