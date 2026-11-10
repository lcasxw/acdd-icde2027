from __future__ import annotations

"""Runtime-only acceleration for the frozen prequential EM policy.

The scientific estimand and numerical algorithm are unchanged.  The frozen
implementation estimates the target prior for every strict prefix using at
most 100 EM updates and tolerance 1e-8.  This module performs those identical
updates with NumPy vector arithmetic instead of a Python scalar list
comprehension.  No labels are accepted.
"""

from typing import Sequence

import numpy as np

from .source_calibration import apply_calibrator
from .threshold_policies import ThresholdPolicy, _finite_scores


def fast_platt_em_prequential_decisions(
    policy: ThresholdPolicy,
    scores: Sequence[float],
    *,
    start: int,
) -> list[int]:
    """Return strict-past decisions using the frozen EM update exactly."""

    if policy.method != "platt_em":
        raise ValueError("runtime-v2 acceleration is defined only for platt_em")
    if policy.access_contract.target_score_scope != "strict_past":
        raise ValueError("runtime-v2 acceleration requires strict-past access")
    values = _finite_scores(scores)
    if start < 1 or start >= len(values):
        raise ValueError("prequential start must leave a non-empty strict past and suffix")
    probabilities = np.clip(
        apply_calibrator(policy.calibrator, values), 1e-12, 1.0 - 1e-12
    )
    source_prior = min(max(float(policy.source_prior), 1e-8), 1.0 - 1e-8)
    decisions: list[int] = []
    estimate = source_prior
    for index in range(start, len(values)):
        previous = estimate
        past = probabilities[:index]
        for _ in range(policy.em_maximum_iterations):
            positive = past * previous / source_prior
            negative = (1.0 - past) * (1.0 - previous) / (1.0 - source_prior)
            updated = min(max(float(np.mean(positive / (positive + negative))), 0.05), 0.95)
            if abs(updated - previous) <= policy.em_tolerance:
                previous = updated
                break
            previous = updated
        estimate = previous
        current = probabilities[index]
        positive = current * estimate / source_prior
        negative = (1.0 - current) * (1.0 - estimate) / (1.0 - source_prior)
        decisions.append(int(positive / (positive + negative) >= 0.5))
    return decisions


def install_runtime_v2_patch() -> object:
    """Patch only the expensive frozen method and return the original method."""

    original = ThresholdPolicy.predict_prequential_sequence

    def accelerated(
        policy: ThresholdPolicy, scores: Sequence[float], *, start: int
    ) -> list[int]:
        if (
            policy.method == "platt_em"
            and policy.access_contract.target_score_scope == "strict_past"
        ):
            return fast_platt_em_prequential_decisions(policy, scores, start=start)
        return original(policy, scores, start=start)

    ThresholdPolicy.predict_prequential_sequence = accelerated
    return original
