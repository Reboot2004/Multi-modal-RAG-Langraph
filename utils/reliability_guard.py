import json
import os
import time
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Any

from config.settings import (
    ENABLE_CIRCUIT_BREAKER,
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    RELIABILITY_STATE_PATH,
    ENABLE_QUALITY_ROLLBACK_GUARD,
    ROLLBACK_MIN_AVG_CONFIDENCE,
    ROLLBACK_MIN_AVG_JUDGE_SCORE,
    ROLLBACK_WINDOW_SIZE,
    ROLLBACK_SIGNAL_PATH,
)


class CircuitBreaker:
    def __init__(self, key: str):
        self.enabled = bool(ENABLE_CIRCUIT_BREAKER)
        self.key = key
        self.failure_threshold = int(CIRCUIT_BREAKER_FAILURE_THRESHOLD)
        self.cooldown = int(CIRCUIT_BREAKER_COOLDOWN_SECONDS)
        self.path = RELIABILITY_STATE_PATH

    def _read(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write(self, state: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def allow_request(self) -> bool:
        if not self.enabled:
            return True
        state = self._read()
        item = state.get(self.key, {})
        opened_at = float(item.get("opened_at", 0.0))
        is_open = bool(item.get("is_open", False))
        if not is_open:
            return True
        if (time.time() - opened_at) >= self.cooldown:
            item["is_open"] = False
            item["failures"] = 0
            state[self.key] = item
            self._write(state)
            return True
        return False

    def record_success(self) -> None:
        if not self.enabled:
            return
        state = self._read()
        state[self.key] = {"failures": 0, "is_open": False, "opened_at": 0.0}
        self._write(state)

    def record_failure(self) -> None:
        if not self.enabled:
            return
        state = self._read()
        item = state.get(self.key, {"failures": 0, "is_open": False, "opened_at": 0.0})
        item["failures"] = int(item.get("failures", 0)) + 1
        if item["failures"] >= self.failure_threshold:
            item["is_open"] = True
            item["opened_at"] = time.time()
        state[self.key] = item
        self._write(state)


class QualityRollbackGuard:
    def __init__(self):
        self.enabled = bool(ENABLE_QUALITY_ROLLBACK_GUARD)
        self.window = deque(maxlen=max(5, int(ROLLBACK_WINDOW_SIZE)))

    def record(self, confidence: float, judge_score: float) -> Dict[str, Any]:
        if not self.enabled:
            return {"triggered": False}
        self.window.append((float(confidence), float(judge_score)))
        if len(self.window) < self.window.maxlen:
            return {"triggered": False, "ready": False}

        avg_conf = sum(v[0] for v in self.window) / len(self.window)
        avg_judge = sum(v[1] for v in self.window) / len(self.window)
        triggered = avg_conf < float(ROLLBACK_MIN_AVG_CONFIDENCE) or avg_judge < float(ROLLBACK_MIN_AVG_JUDGE_SCORE)
        if triggered:
            os.makedirs(os.path.dirname(ROLLBACK_SIGNAL_PATH), exist_ok=True)
            payload = {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "avg_confidence": avg_conf,
                "avg_judge_score": avg_judge,
                "window": len(self.window),
                "action": "rollback_recommended",
            }
            with open(ROLLBACK_SIGNAL_PATH, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        return {
            "triggered": triggered,
            "ready": True,
            "avg_confidence": avg_conf,
            "avg_judge_score": avg_judge,
        }
