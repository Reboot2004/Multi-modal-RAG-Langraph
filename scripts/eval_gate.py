import argparse
import json
import os
import sys
from typing import Dict, Any, List


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "count": 0,
            "avg_confidence": 0.0,
            "avg_judge": 0.0,
            "refusal_rate": 1.0,
            "language_adherence": 0.0,
        }

    count = len(rows)
    avg_conf = sum(float(r.get("confidence", 0.0)) for r in rows) / count
    avg_judge = sum(float(r.get("judge", {}).get("overall_score", 0.0)) for r in rows) / count
    refusal_rate = sum(1 for r in rows if r.get("refused", False)) / count

    lang_ok = 0
    for r in rows:
        response_lang = str(r.get("response_language", "")).strip().lower()
        detected = str(r.get("language_compliance", {}).get("detected", response_lang)).strip().lower()
        if response_lang and detected == response_lang:
            lang_ok += 1
    lang_adherence = lang_ok / count

    return {
        "count": count,
        "avg_confidence": avg_conf,
        "avg_judge": avg_judge,
        "refusal_rate": refusal_rate,
        "language_adherence": lang_adherence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline quality gate for RAG eval logs")
    parser.add_argument("--log-path", default="data/processed/rag_eval_log.jsonl")
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--min-judge", type=float, default=0.55)
    parser.add_argument("--max-refusal-rate", type=float, default=0.35)
    parser.add_argument("--min-language-adherence", type=float, default=0.90)
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.log_path)
    metrics = compute_metrics(rows)

    print("Eval Gate Metrics:")
    for k, v in metrics.items():
        print(f"- {k}: {v}")

    if metrics["count"] == 0 and args.allow_empty:
        print("No eval rows found; passing due to --allow-empty")
        return 0

    if metrics["count"] < args.min_samples:
        print(f"Not enough samples for strict gate: {metrics['count']} < {args.min_samples}")
        return 0

    failures = []
    if metrics["avg_confidence"] < args.min_confidence:
        failures.append("avg_confidence")
    if metrics["avg_judge"] < args.min_judge:
        failures.append("avg_judge")
    if metrics["refusal_rate"] > args.max_refusal_rate:
        failures.append("refusal_rate")
    if metrics["language_adherence"] < args.min_language_adherence:
        failures.append("language_adherence")

    if failures:
        print("QUALITY GATE FAILED:", ", ".join(failures))
        return 2

    print("QUALITY GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
