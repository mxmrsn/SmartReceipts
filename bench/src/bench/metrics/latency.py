"""Latency aggregates over a pipeline's per-sample timings."""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass
class LatencyStats:
    sample_count: int
    p50_ms: int
    p95_ms: int
    mean_ms: int

    @classmethod
    def from_samples(cls, ms_values: list[int]) -> "LatencyStats":
        if not ms_values:
            return cls(0, 0, 0, 0)
        sorted_vals = sorted(ms_values)
        return cls(
            sample_count=len(ms_values),
            p50_ms=int(statistics.median(sorted_vals)),
            p95_ms=int(_percentile(sorted_vals, 0.95)),
            mean_ms=int(statistics.mean(sorted_vals)),
        )


def _percentile(sorted_vals: list[int], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = pct * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac
