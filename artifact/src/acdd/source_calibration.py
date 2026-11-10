from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CalibrationModel:
    kind: str
    slope: float | None = None
    intercept: float | None = None
    isotonic_x: tuple[float, ...] = ()
    isotonic_y: tuple[float, ...] = ()


def sigmoid(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)
    positive = array >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponential = np.exp(array[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _validate(raw_logits: Sequence[float], labels: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(raw_logits, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 1 or truth.shape != scores.shape or len(scores) == 0:
        raise ValueError("calibration requires one logit and label per sample")
    if not np.isfinite(scores).all() or set(truth.tolist()) != {0, 1}:
        raise ValueError("calibration requires finite logits and both binary classes")
    return scores, truth


def binary_nll(labels: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def _golden_section(objective: object, lower: float, upper: float, iterations: int = 160) -> float:
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left, right = float(lower), float(upper)
    c = right - ratio * (right - left)
    d = left + ratio * (right - left)
    fc, fd = objective(c), objective(d)
    for _ in range(iterations):
        if fc <= fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = objective(c)
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = objective(d)
    return (left + right) / 2.0


def _fit_platt(scores: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    parameters = np.asarray([0.0, 0.0], dtype=np.float64)  # log(slope), intercept
    objective = lambda values: binary_nll(
        truth, sigmoid(math.exp(float(values[0])) * scores + float(values[1]))
    )
    for _ in range(300):
        slope = math.exp(float(parameters[0]))
        probabilities = sigmoid(slope * scores + parameters[1])
        residual = probabilities - truth
        gradient = np.asarray(
            [float(np.mean(residual * slope * scores)), float(np.mean(residual))]
        )
        if float(np.linalg.norm(gradient)) < 1e-10:
            break
        current = objective(parameters)
        step = 1.0
        accepted = False
        for _ in range(40):
            proposal = parameters - step * gradient
            proposal[0] = np.clip(proposal[0], -8.0, 8.0)
            proposal[1] = np.clip(proposal[1], -20.0, 20.0)
            if objective(proposal) <= current - 1e-4 * step * float(gradient @ gradient):
                parameters = proposal
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    return math.exp(float(parameters[0])), float(parameters[1])


def _fit_isotonic(scores: np.ndarray, truth: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores, sorted_truth = scores[order], truth[order].astype(np.float64)
    unique, inverse, counts = np.unique(sorted_scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(unique), dtype=np.float64)
    np.add.at(sums, inverse, sorted_truth)
    blocks = [
        {"start": index, "stop": index + 1, "weight": float(counts[index]), "sum": float(sums[index])}
        for index in range(len(unique))
    ]
    index = 0
    while index < len(blocks) - 1:
        left = blocks[index]
        right = blocks[index + 1]
        if left["sum"] / left["weight"] <= right["sum"] / right["weight"]:
            index += 1
            continue
        merged = {
            "start": left["start"],
            "stop": right["stop"],
            "weight": left["weight"] + right["weight"],
            "sum": left["sum"] + right["sum"],
        }
        blocks[index : index + 2] = [merged]
        index = max(0, index - 1)
    fitted = np.empty(len(unique), dtype=np.float64)
    for block in blocks:
        fitted[int(block["start"]) : int(block["stop"])] = block["sum"] / block["weight"]
    return tuple(float(value) for value in unique), tuple(float(value) for value in fitted)


def fit_calibrator(
    raw_logits: Sequence[float], labels: Sequence[int], kind: str
) -> CalibrationModel:
    scores, truth = _validate(raw_logits, labels)
    normalized = kind.lower().replace("-", "_")
    if normalized == "temperature_only":
        objective = lambda log_t: binary_nll(truth, sigmoid(scores / math.exp(float(log_t))))
        fitted = _golden_section(objective, -4.0, 4.0)
        return CalibrationModel(normalized, slope=1.0 / math.exp(fitted), intercept=0.0)
    if normalized == "intercept_only":
        objective = lambda intercept: binary_nll(truth, sigmoid(scores + float(intercept)))
        fitted = _golden_section(objective, -20.0, 20.0)
        return CalibrationModel(normalized, slope=1.0, intercept=fitted)
    if normalized == "platt":
        slope, intercept = _fit_platt(scores, truth)
        return CalibrationModel(normalized, slope=slope, intercept=intercept)
    if normalized == "isotonic":
        knots, values = _fit_isotonic(scores, truth)
        return CalibrationModel(
            normalized,
            isotonic_x=knots,
            isotonic_y=values,
        )
    raise KeyError(f"unknown calibration model: {kind}")


def apply_calibrator(model: CalibrationModel, raw_logits: Sequence[float]) -> np.ndarray:
    scores = np.asarray(raw_logits, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("calibrator input must be a finite logit vector")
    if model.kind == "isotonic":
        if not model.isotonic_x:
            raise ValueError("isotonic model has no fitted knots")
        return np.interp(scores, model.isotonic_x, model.isotonic_y)
    if model.slope is None or model.intercept is None or model.slope <= 0:
        raise ValueError("parametric calibrator is incomplete or non-monotone")
    return sigmoid(model.slope * scores + model.intercept)


def probability_decisions(probabilities: Sequence[float]) -> list[int]:
    return [int(float(value) >= 0.5) for value in probabilities]
