from typing import Dict, List


class AVSegmenter:
    def __init__(self, window_seconds: float, overlap_seconds: float):
        self.window_seconds = max(4.0, float(window_seconds))
        self.overlap_seconds = max(0.0, min(float(overlap_seconds), self.window_seconds / 2.0))

    def build_segments(self, duration_seconds: float, max_seconds: float = 0.0) -> List[Dict]:
        effective_duration = float(duration_seconds)
        if max_seconds and max_seconds > 0:
            effective_duration = min(effective_duration, float(max_seconds))

        if effective_duration <= 0:
            return []

        segments = []
        step = max(1.0, self.window_seconds - self.overlap_seconds)
        start = 0.0
        idx = 1

        while start < effective_duration:
            end = min(start + self.window_seconds, effective_duration)
            segments.append(
                {
                    "idx": idx,
                    "start": round(start, 3),
                    "end": round(end, 3),
                }
            )
            if end >= effective_duration:
                break
            start += step
            idx += 1

        return segments
