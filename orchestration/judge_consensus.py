from typing import Dict, Any, List

from orchestration.llm_judge import LLMJudge


class JudgeConsensus:
    """Runs multiple judge passes and aggregates into a consensus score."""

    def __init__(self, llm_client, judges: int = 2):
        self.llm_client = llm_client
        self.judges = max(1, int(judges))

    def evaluate(self, **kwargs) -> Dict[str, Any]:
        runs: List[Dict[str, Any]] = []

        for _ in range(self.judges):
            judge = LLMJudge(llm_client=self.llm_client)
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
            "judges": self.judges,
            "disagreement": float(disagreement),
            "scores": overall_scores,
        }
        return consensus
