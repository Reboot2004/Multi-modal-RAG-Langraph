import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

from config.settings import ENABLE_EVAL_LOGGING, EVAL_LOG_PATH


class EvalLogger:
    """Append-only JSONL logger for per-query RAG evaluation telemetry."""

    def __init__(self, log_path: str = EVAL_LOG_PATH):
        self.enabled = bool(ENABLE_EVAL_LOGGING)
        self.log_path = log_path

    def write(self, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        envelope = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(envelope, ensure_ascii=False) + "\n")
