import json
import os
from datetime import datetime, timezone
from typing import Dict, Any

from config.settings import ENABLE_HUMAN_FEEDBACK, FEEDBACK_LOG_PATH


class FeedbackStore:
    def __init__(self):
        self.enabled = bool(ENABLE_HUMAN_FEEDBACK)
        self.path = FEEDBACK_LOG_PATH

    def write(self, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        event = {"ts_utc": datetime.now(timezone.utc).isoformat(), **payload}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
