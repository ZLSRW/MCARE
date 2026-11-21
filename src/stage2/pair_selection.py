"""Preference ranking, semantic eligibility gating, and length-matched hard-negative selection.

Implements Section III-B2 (Eqs. 8-11) and Section IV-C (Eqs. 31-32):
  - Tie-corrected within-context percentile mapping P_x(.)
  - Semantic gate V_x = {i : c_hat_i == 0 and w_hat_i >= w_0} (w_0 = 3)
  - Length-matched negative pool L_x within delta = 0.25 of positive word count
  - Violation-prioritized hard negative B_x, falling back to lowest-scoring
    candidate in L_x when B_x is empty
"""

from __future__ import annotations

import math
from typing import Any

from stage2.annotate_candidates import BOOLS

DEFAULT_WEIGHTS = {"quality": 0.50, "warmth": 0.25, "support": 0.10, "reference": 0.15}


def percentile(values: list[float]) -> list[float]:
    n = len(values)
    assert n >= 2 and all(math.isfinite(v) for v in values)
    denom = float(n - 1)
    return [
        (sum(y < x - 1e-10 for y in values) + 0.5 * (sum(abs(y - x) <= 1e-10 for y in values) - 1)) / denom
        for x in values
    ]


def values_for(h: dict[str, Any], weights: dict[str, float]) -> list[float]:
    n = len(h["cs"])
    denominator = 1.0 if h.get("has_reference") else (1.0 - weights["reference"])
    return [
        (
            weights["quality"] * h["quality"][i]
            + weights["warmth"] * h["warmth"][i]
            + weights["support"] * h["future"][i]
            + weights["reference"] * h["reference"][i]
        )
        / denominator
        for i in range(n)
    ]


def pick(
    h: dict[str, Any],
    values: list[float],
    window: float,
    chosen: int,
    banned: tuple[int, ...] = (),
    hard_window: float = 0.25,
) -> tuple[int | None, str | None]:
    """Select the length-matched negative sample per Eqs. (10)-(11)."""
    cs, costs, valid = h["cs"], h["costs"], h["valid"]
    length = len(cs[chosen]["text"].split())

    def within(i: int, w: float) -> bool:
        if i == chosen or cs[i]["text"] == cs[chosen]["text"]:
            return False
        words = len(cs[i]["text"].split())
        return (1.0 - w) * length <= words <= (1.0 + w) * length

    near = [i for i in valid if within(i, window)]
    bad = [i for i in valid if within(i, hard_window) and costs[i] > 0 and i not in banned]
    if not near and not bad:
        return None, "no_length_matched_negative"
    if bad:
        return max(bad, key=lambda i: (values[i], -i)), None
    if not near:
        return None, "no_length_matched_negative"
    eligible_near = [i for i in near if i not in banned]
    if not eligible_near:
        return None, "no_distinct_negative"
    rejected = min(eligible_near, key=lambda i: (values[i], i))
    if values[chosen] <= values[rejected] + 1e-10:
        return None, "no_strict_score_gap"
    return rejected, None
