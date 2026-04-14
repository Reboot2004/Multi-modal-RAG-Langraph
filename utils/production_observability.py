import json
import os
import time
from datetime import datetime, timezone
from statistics import quantiles
from typing import Dict, Any, List

from config.settings import (
    ENABLE_PRODUCTION_TELEMETRY,
    PRODUCTION_TELEMETRY_LOG_PATH,
    GROQ_INPUT_COST_PER_1K,
    GROQ_OUTPUT_COST_PER_1K,
    OPENROUTER_INPUT_COST_PER_1K,
    OPENROUTER_OUTPUT_COST_PER_1K,
    SLO_P95_LATENCY_MS,
    SLO_MIN_ANSWER_SUCCESS_RATE,
    SLO_MIN_LANGUAGE_ADHERENCE,
    SLO_ALERT_LOG_PATH,
)


def estimate_tokens(text: str) -> int:
    return max(1, int(len((text or "").strip()) / 4))


def estimate_cost_usd(provider: str, prompt_tokens: int, completion_tokens: int) -> float:
    provider_key = (provider or "groq").lower()
    if provider_key == "openrouter":
        in_rate = float(OPENROUTER_INPUT_COST_PER_1K)
        out_rate = float(OPENROUTER_OUTPUT_COST_PER_1K)
    else:
        in_rate = float(GROQ_INPUT_COST_PER_1K)
        out_rate = float(GROQ_OUTPUT_COST_PER_1K)
    return ((prompt_tokens / 1000.0) * in_rate) + ((completion_tokens / 1000.0) * out_rate)


class ProductionTelemetry:
    def __init__(self):
        self.enabled = bool(ENABLE_PRODUCTION_TELEMETRY)
        self.path = PRODUCTION_TELEMETRY_LOG_PATH

    def write(self, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        event = {"ts_utc": datetime.now(timezone.utc).isoformat(), **payload}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


class SLOMonitor:
    def __init__(self):
        self.latencies: List[float] = []
        self.successes: List[int] = []
        self.language_ok: List[int] = []

    def record(self, latency_ms: float, success: bool, language_adherence: bool) -> Dict[str, Any]:
        self.latencies.append(float(latency_ms))
        self.successes.append(1 if success else 0)
        self.language_ok.append(1 if language_adherence else 0)

        p95 = self._p95_latency()
        success_rate = sum(self.successes) / len(self.successes)
        lang_rate = sum(self.language_ok) / len(self.language_ok)

        breached = (
            p95 > float(SLO_P95_LATENCY_MS)
            or success_rate < float(SLO_MIN_ANSWER_SUCCESS_RATE)
            or lang_rate < float(SLO_MIN_LANGUAGE_ADHERENCE)
        )

        result = {
            "p95_latency_ms": p95,
            "success_rate": success_rate,
            "language_adherence": lang_rate,
            "breached": breached,
        }
        if breached:
            os.makedirs(os.path.dirname(SLO_ALERT_LOG_PATH), exist_ok=True)
            with open(SLO_ALERT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts_utc": datetime.now(timezone.utc).isoformat(), **result}, ensure_ascii=False) + "\n")
        return result

    def _p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        if len(self.latencies) < 2:
            return float(self.latencies[0])
        return float(quantiles(self.latencies, n=100)[94])


def now_ms() -> float:
    return time.perf_counter() * 1000.0
