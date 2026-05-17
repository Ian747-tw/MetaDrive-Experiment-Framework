from __future__ import annotations


def clamp_risk(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)
