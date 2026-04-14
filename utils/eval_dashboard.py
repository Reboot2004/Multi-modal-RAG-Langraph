# eval_dashboard.py
"""
Eval Dashboard Utilities
Reads and visualizes per-query RAG evaluation telemetry from JSONL logs.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
from collections import defaultdict, Counter


class EvalDashboard:
    """Reader and analyzer for RAG evaluation telemetry logs."""

    def __init__(self, log_path: str = "data/processed/rag_eval_log.jsonl"):
        """Initialize dashboard with path to JSONL log file."""
        self.log_path = Path(log_path)

    def read_logs(self) -> List[Dict[str, Any]]:
        """Read all records from JSONL log file."""
        records = []
        if not self.log_path.exists():
            return records
        
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"Warning: JSON parse error at line {line_num}: {e}")
        except Exception as e:
            print(f"Error reading log file: {e}")
        
        return records

    def get_summary_stats(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary statistics from records."""
        if not records:
            return {
                "total_queries": 0,
                "avg_confidence": 0.0,
                "refusal_rate": 0.0,
                "languages": {},
                "judge_verdict_dist": {},
                "intents": {},
            }

        total = len(records)
        confidences = [float(r.get("confidence", 0.5)) for r in records]
        refusals = sum(1 for r in records if r.get("refused", False))
        
        # Language distribution
        languages = Counter(r.get("response_language", "unknown") for r in records)
        
        # Judge verdict distribution
        judge_verdicts = Counter()
        for r in records:
            judge = r.get("judge", {})
            if isinstance(judge, dict) and "verdict" in judge:
                judge_verdicts[judge["verdict"]] += 1
        
        # Intent distribution
        intents = Counter(r.get("intent", "qa") for r in records)
        
        # Confidence by intent
        confidence_by_intent = defaultdict(list)
        for r in records:
            intent = r.get("intent", "qa")
            conf = float(r.get("confidence", 0.5))
            confidence_by_intent[intent].append(conf)
        
        avg_conf_by_intent = {
            intent: sum(confs) / len(confs)
            for intent, confs in confidence_by_intent.items()
        }
        
        # Average scores
        self_rag_scores = [r.get("self_rag", {}) for r in records]
        avg_faithfulness = sum(
            float(s.get("faithfulness", 0.5)) for s in self_rag_scores
        ) / len(self_rag_scores) if self_rag_scores else 0.5
        avg_usefulness = sum(
            float(s.get("usefulness", 0.5)) for s in self_rag_scores
        ) / len(self_rag_scores) if self_rag_scores else 0.5
        
        # Judge scores
        judge_scores = [
            float(r.get("judge", {}).get("overall_score", 0.5))
            for r in records
            if r.get("judge", {}).get("overall_score") is not None
        ]
        avg_judge_score = sum(judge_scores) / len(judge_scores) if judge_scores else 0.5
        
        # Grounding scores
        grounding_scores = [
            float(r.get("grounding", {}).get("support_ratio", 0.5))
            for r in records
            if r.get("grounding", {}).get("support_ratio") is not None
        ]
        avg_grounding_score = sum(grounding_scores) / len(grounding_scores) if grounding_scores else 0.5
        
        return {
            "total_queries": total,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.5,
            "refusal_count": refusals,
            "refusal_rate": refusals / total if total > 0 else 0.0,
            "languages": dict(languages),
            "judge_verdict_dist": dict(judge_verdicts),
            "intents": dict(intents),
            "avg_confidence_by_intent": avg_conf_by_intent,
            "avg_faithfulness": avg_faithfulness,
            "avg_usefulness": avg_usefulness,
            "avg_judge_score": avg_judge_score,
            "avg_grounding_score": avg_grounding_score,
        }

    def get_failed_queries(self, records: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
        """Get queries that were refused or had low confidence."""
        failed = [
            r for r in records
            if r.get("refused", False) or float(r.get("confidence", 0.5)) < 0.5
        ]
        # Sort by confidence (ascending) and return
        failed.sort(key=lambda x: float(x.get("confidence", 0.5)))
        return failed[:limit]

    def get_language_distribution_records(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create DataFrame of query counts by language and intent."""
        data = defaultdict(lambda: defaultdict(int))
        for r in records:
            lang = r.get("response_language", "unknown")
            intent = r.get("intent", "qa")
            data[lang][intent] += 1
        
        df_dict = {}
        for lang, intents_dict in data.items():
            df_dict[lang] = intents_dict
        
        df = pd.DataFrame(df_dict).fillna(0).astype(int).T
        return df

    def get_confidence_trend(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create DataFrame of confidence over time (sliding window)."""
        if not records:
            return pd.DataFrame()
        
        confidences = [float(r.get("confidence", 0.5)) for r in records]
        window_size = max(5, len(records) // 10)  # ~10 windows
        
        windows = []
        for i in range(0, len(confidences), window_size):
            window_confs = confidences[i:i+window_size]
            avg_conf = sum(window_confs) / len(window_confs) if window_confs else 0.5
            windows.append({"window": i // window_size + 1, "avg_confidence": avg_conf})
        
        return pd.DataFrame(windows)

    def get_refusal_reasons(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        """Aggregate refusal reasons."""
        reasons = Counter()
        for r in records:
            if r.get("refused", False):
                reason = r.get("refusal_reason", "unknown")
                reasons[reason] += 1
        return dict(reasons)

    def to_dataframe(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert records to flat DataFrame for detailed view."""
        flattened = []
        for r in records:
            flat = {
                "query": r.get("query", "")[:50],  # Truncate for readability
                "intent": r.get("intent", "qa"),
                "language": r.get("response_language", "unknown"),
                "confidence": float(r.get("confidence", 0.5)),
                "self_rag_faith": float(r.get("self_rag", {}).get("faithfulness", 0.5)),
                "self_rag_useful": float(r.get("self_rag", {}).get("usefulness", 0.5)),
                "judge_overall": float(r.get("judge", {}).get("overall_score", 0.5)),
                "judge_verdict": r.get("judge", {}).get("verdict", "caution"),
                "grounding_support": float(r.get("grounding", {}).get("support_ratio", 0.5)),
                "refused": r.get("refused", False),
                "refusal_reason": r.get("refusal_reason", ""),
            }
            flattened.append(flat)
        
        return pd.DataFrame(flattened)
