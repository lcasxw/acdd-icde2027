from __future__ import annotations

"""Access-matched threshold policies for frozen binary logits.

The public API deliberately separates labelled source validation data from
unlabelled target scores.  Deployable policies never accept target labels.
Oracle diagnostics live in :mod:`acdd.strengthening` and therefore cannot be
accidentally selected through this interface.
"""

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Sequence

import numpy as np

from .metrics import (
    _prequential_thresholds_and_decisions,
    oracle_logit_threshold,
    prior_quantile_logit_threshold,
)
from .source_calibration import CalibrationModel, apply_calibrator, fit_calibrator, sigmoid


SOURCE_ONLY_METHODS = {
    "fixed_0.5",
    "source_youden",
    "source_temperature",
    "source_intercept",
    "source_platt",
    "source_isotonic",
}
TARGET_SCORE_METHODS = {"target_prior_quantile", "platt_em", "rlls_bbse"}
ALL_METHODS = SOURCE_ONLY_METHODS | TARGET_SCORE_METHODS


@lru_cache(maxsize=256)
def _cached_calibrator(
    raw_logits: tuple[float, ...], labels: tuple[int, ...], kind: str
) -> CalibrationModel:
    return fit_calibrator(raw_logits, labels, kind)


@lru_cache(maxsize=128)
def _cached_source_youden(
    raw_logits: tuple[float, ...], labels: tuple[int, ...]
) -> float:
    return oracle_logit_threshold(labels, raw_logits)[0]


@dataclass(frozen=True)
class AccessContract:
    """Information available when a policy is fitted and queried."""

    access_group: str
    endpoint: str
    target_score_scope: str
    declared_positive_prior: float = 0.5
    uses_future_target_information: bool = False

    def __post_init__(self) -> None:
        allowed = {"none", "fixed_available", "strict_past"}
        if self.target_score_scope not in allowed:
            raise ValueError(f"unknown target score scope: {self.target_score_scope}")
        if not 0.0 < self.declared_positive_prior < 1.0:
            raise ValueError("declared prior must lie strictly between zero and one")
        if self.target_score_scope == "strict_past" and self.uses_future_target_information:
            raise ValueError("strict-past access cannot use future target information")
        if self.access_group == "source_only" and self.target_score_scope != "none":
            raise ValueError("source-only access cannot expose target scores")


@dataclass(frozen=True)
class SourceValidationData:
    raw_logits: tuple[float, ...]
    labels: tuple[int, ...]

    @classmethod
    def from_sequences(
        cls, raw_logits: Sequence[float], labels: Sequence[int]
    ) -> "SourceValidationData":
        scores = tuple(float(value) for value in raw_logits)
        truth = tuple(int(value) for value in labels)
        if len(scores) != len(truth) or not scores:
            raise ValueError("source validation scores and labels must have equal length")
        if set(truth) != {0, 1} or not all(math.isfinite(value) for value in scores):
            raise ValueError("source validation requires finite logits and both classes")
        return cls(scores, truth)


def _finite_scores(values: Sequence[float], *, allow_empty: bool = False) -> tuple[float, ...]:
    scores = tuple(float(value) for value in values)
    if (not scores and not allow_empty) or not all(math.isfinite(value) for value in scores):
        raise ValueError("target access requires finite unlabelled scores")
    return scores


def _prior_adjusted_probability(probability: float, source_prior: float, target_prior: float) -> float:
    probability = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    source_prior = min(max(float(source_prior), 1e-8), 1.0 - 1e-8)
    target_prior = min(max(float(target_prior), 0.05), 0.95)
    positive = probability * target_prior / source_prior
    negative = (1.0 - probability) * (1.0 - target_prior) / (1.0 - source_prior)
    return positive / (positive + negative)


def _em_target_prior(
    probabilities: Sequence[float],
    source_prior: float,
    *,
    maximum_iterations: int,
    tolerance: float,
) -> float:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("EM label shift requires target scores")
    estimate = min(max(float(source_prior), 0.05), 0.95)
    for _ in range(maximum_iterations):
        adjusted = np.asarray(
            [_prior_adjusted_probability(value, source_prior, estimate) for value in values]
        )
        updated = min(max(float(adjusted.mean()), 0.05), 0.95)
        if abs(updated - estimate) <= tolerance:
            estimate = updated
            break
        estimate = updated
    return estimate


def _bbse_target_prior(
    source_probabilities: Sequence[float],
    source_labels: Sequence[int],
    target_probabilities: Sequence[float],
    *,
    ridge: float,
) -> float:
    if ridge < 0.0:
        raise ValueError("RLLS ridge must be non-negative")
    source_p = np.asarray(source_probabilities, dtype=np.float64)
    source_y = np.asarray(source_labels, dtype=np.int64)
    target_p = np.asarray(target_probabilities, dtype=np.float64)
    if target_p.ndim != 1 or len(target_p) == 0:
        raise ValueError("RLLS/BBSE requires target scores")
    source_h = (source_p >= 0.5).astype(np.int64)
    target_h = (target_p >= 0.5).astype(np.int64)
    confusion = np.zeros((2, 2), dtype=np.float64)
    for label in (0, 1):
        mask = source_y == label
        if not mask.any():
            raise ValueError("RLLS/BBSE requires both source classes")
        confusion[:, label] = np.bincount(source_h[mask], minlength=2) / int(mask.sum())
    observed = np.bincount(target_h, minlength=2).astype(np.float64) / len(target_h)
    source_prior = np.asarray([np.mean(source_y == 0), np.mean(source_y == 1)])
    system = confusion.T @ confusion + float(ridge) * np.eye(2)
    rhs = confusion.T @ observed + float(ridge) * source_prior
    estimate = np.linalg.solve(system, rhs)
    estimate = np.clip(estimate, 0.05, 0.95)
    estimate /= estimate.sum()
    return float(estimate[1])


@dataclass(frozen=True)
class ThresholdPolicy:
    method: str
    access_contract: AccessContract
    source: SourceValidationData
    source_prior: float
    source_threshold: float = 0.0
    calibrator: CalibrationModel | None = None
    fitted_target_prior: float | None = None
    fitted_target_threshold: float | None = None
    em_maximum_iterations: int = 100
    em_tolerance: float = 1e-8
    rlls_ridge: float = 1e-3

    def _target_state(self, past_scores_only: Sequence[float]) -> tuple[float | None, float | None]:
        if self.method in SOURCE_ONLY_METHODS:
            return None, None
        if self.access_contract.target_score_scope == "strict_past":
            scores = _finite_scores(past_scores_only)
            if self.method == "target_prior_quantile":
                threshold = prior_quantile_logit_threshold(
                    scores, self.access_contract.declared_positive_prior
                )
                return None, threshold
            probabilities = apply_calibrator(self.calibrator, scores)
            if self.method == "platt_em":
                prior = _em_target_prior(
                    probabilities,
                    self.source_prior,
                    maximum_iterations=self.em_maximum_iterations,
                    tolerance=self.em_tolerance,
                )
            else:
                source_probabilities = apply_calibrator(
                    self.calibrator, self.source.raw_logits
                )
                prior = _bbse_target_prior(
                    source_probabilities,
                    self.source.labels,
                    probabilities,
                    ridge=self.rlls_ridge,
                )
            return prior, None
        return self.fitted_target_prior, self.fitted_target_threshold

    def predict(self, current_score: float, past_scores_only: Sequence[float] = ()) -> int:
        """Predict using only the score and the explicitly supplied strict past."""

        score = float(current_score)
        if not math.isfinite(score):
            raise ValueError("current score must be finite")
        if self.method in {"fixed_0.5", "source_temperature"}:
            return int(score >= 0.0)
        if self.method == "source_youden":
            return int(score >= self.source_threshold)
        if self.method in {"source_intercept", "source_platt", "source_isotonic"}:
            probability = float(apply_calibrator(self.calibrator, [score])[0])
            return int(probability >= 0.5)
        prior, threshold = self._target_state(past_scores_only)
        if self.method == "target_prior_quantile":
            if threshold is None:
                raise AssertionError("target quantile policy lacks a threshold")
            return int(score >= threshold)
        probability = float(apply_calibrator(self.calibrator, [score])[0])
        if prior is None:
            raise AssertionError("label-shift policy lacks a target-prior estimate")
        return int(_prior_adjusted_probability(probability, self.source_prior, prior) >= 0.5)

    def predict_prequential_sequence(
        self, scores: Sequence[float], *, start: int
    ) -> list[int]:
        """Evaluate every decision with the exact preceding score prefix only."""

        values = _finite_scores(scores)
        if self.access_contract.target_score_scope != "strict_past":
            return [self.predict(values[index]) for index in range(start, len(values))]
        if start < 1 or start >= len(values):
            raise ValueError("prequential start must leave a non-empty strict past and suffix")
        if self.method == "target_prior_quantile":
            _, decisions = _prequential_thresholds_and_decisions(
                values, self.access_contract.declared_positive_prior
            )
            return decisions[start:]
        if self.method == "rlls_bbse":
            source_probabilities = apply_calibrator(self.calibrator, self.source.raw_logits)
            source_labels = np.asarray(self.source.labels, dtype=np.int64)
            source_hard = (source_probabilities >= 0.5).astype(np.int64)
            confusion = np.zeros((2, 2), dtype=np.float64)
            for label in (0, 1):
                mask = source_labels == label
                confusion[:, label] = np.bincount(
                    source_hard[mask], minlength=2
                ) / int(mask.sum())
            source_prior_vector = np.asarray(
                [1.0 - self.source_prior, self.source_prior], dtype=np.float64
            )
            system = confusion.T @ confusion + self.rlls_ridge * np.eye(2)
            inverse = np.linalg.inv(system)
            target_probabilities = apply_calibrator(self.calibrator, values)
            target_hard = (target_probabilities >= 0.5).astype(np.int64)
            cumulative_zeros = np.cumsum(target_hard == 0)
            cumulative_ones = np.cumsum(target_hard == 1)
            decisions = []
            for index in range(start, len(values)):
                observed = np.asarray(
                    [cumulative_zeros[index - 1], cumulative_ones[index - 1]],
                    dtype=np.float64,
                ) / index
                estimate = inverse @ (
                    confusion.T @ observed + self.rlls_ridge * source_prior_vector
                )
                estimate = np.clip(estimate, 0.05, 0.95)
                estimate /= estimate.sum()
                adjusted = _prior_adjusted_probability(
                    target_probabilities[index], self.source_prior, float(estimate[1])
                )
                decisions.append(int(adjusted >= 0.5))
            return decisions
        probabilities = apply_calibrator(self.calibrator, values)
        decisions = []
        estimate = self.source_prior
        for index in range(start, len(values)):
            # Warm starts do not change the fixed point; the frozen convergence
            # criterion remains 1e-8 with at most 100 EM updates.
            previous = estimate
            for _ in range(self.em_maximum_iterations):
                adjusted = np.asarray(
                    [
                        _prior_adjusted_probability(value, self.source_prior, previous)
                        for value in probabilities[:index]
                    ]
                )
                updated = min(max(float(adjusted.mean()), 0.05), 0.95)
                if abs(updated - previous) <= self.em_tolerance:
                    previous = updated
                    break
                previous = updated
            estimate = previous
            decisions.append(
                int(
                    _prior_adjusted_probability(
                        probabilities[index], self.source_prior, estimate
                    )
                    >= 0.5
                )
            )
        return decisions


def fit(
    source_validation_scores: Sequence[float],
    source_validation_labels: Sequence[int],
    available_unlabeled_target_scores: Sequence[float],
    access_contract: AccessContract,
    *,
    method: str,
    em_maximum_iterations: int = 100,
    em_tolerance: float = 1e-8,
    rlls_ridge: float = 1e-3,
) -> ThresholdPolicy:
    """Fit an access-matched decision policy without accepting target labels."""

    normalized = method.lower().replace("-", "_")
    if normalized not in ALL_METHODS:
        raise KeyError(f"unknown threshold policy: {method}")
    source = SourceValidationData.from_sequences(
        source_validation_scores, source_validation_labels
    )
    target = _finite_scores(available_unlabeled_target_scores, allow_empty=True)
    if access_contract.target_score_scope == "none" and target:
        raise PermissionError("the access contract forbids target-score fitting")
    if normalized in TARGET_SCORE_METHODS and access_contract.target_score_scope == "none":
        raise PermissionError(f"{normalized} requires an unlabelled target-score access group")
    if access_contract.target_score_scope == "fixed_available" and normalized in TARGET_SCORE_METHODS:
        _finite_scores(target)
    if access_contract.target_score_scope == "strict_past" and target:
        raise PermissionError("prequential policies must receive scores only through past_scores_only")
    if em_maximum_iterations != 100 or not math.isclose(em_tolerance, 1e-8):
        raise ValueError("EM settings are frozen at 100 iterations and tolerance 1e-8")

    source_prior = float(np.mean(source.labels))
    calibrator: CalibrationModel | None = None
    source_threshold = 0.0
    fitted_target_prior: float | None = None
    fitted_target_threshold: float | None = None
    if normalized == "source_youden":
        source_threshold = _cached_source_youden(source.raw_logits, source.labels)
    elif normalized == "source_temperature":
        calibrator = _cached_calibrator(source.raw_logits, source.labels, "temperature_only")
    elif normalized == "source_intercept":
        calibrator = _cached_calibrator(source.raw_logits, source.labels, "intercept_only")
    elif normalized in {"source_platt", "platt_em", "rlls_bbse"}:
        calibrator = _cached_calibrator(source.raw_logits, source.labels, "platt")
    elif normalized == "source_isotonic":
        calibrator = _cached_calibrator(source.raw_logits, source.labels, "isotonic")

    if normalized == "target_prior_quantile" and target:
        fitted_target_threshold = prior_quantile_logit_threshold(
            target, access_contract.declared_positive_prior
        )
    elif normalized in {"platt_em", "rlls_bbse"} and target:
        target_probabilities = apply_calibrator(calibrator, target)
        if normalized == "platt_em":
            fitted_target_prior = _em_target_prior(
                target_probabilities,
                source_prior,
                maximum_iterations=em_maximum_iterations,
                tolerance=em_tolerance,
            )
        else:
            fitted_target_prior = _bbse_target_prior(
                apply_calibrator(calibrator, source.raw_logits),
                source.labels,
                target_probabilities,
                ridge=rlls_ridge,
            )
    return ThresholdPolicy(
        method=normalized,
        access_contract=access_contract,
        source=source,
        source_prior=source_prior,
        source_threshold=source_threshold,
        calibrator=calibrator,
        fitted_target_prior=fitted_target_prior,
        fitted_target_threshold=fitted_target_threshold,
        em_maximum_iterations=em_maximum_iterations,
        em_tolerance=em_tolerance,
        rlls_ridge=rlls_ridge,
    )
