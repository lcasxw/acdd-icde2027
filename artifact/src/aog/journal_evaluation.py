from __future__ import annotations

from collections import defaultdict
import csv
import math
from pathlib import Path
import random
from statistics import mean
from typing import Iterable, Sequence

from .rank_calibration import (
    Prediction,
    balanced_accuracy,
    brier_score,
    expected_calibration_error,
    group_subjects,
    oracle_threshold,
    prior_threshold,
    quantile,
    roc_auc,
    shifted_probabilities,
)


POLICY_ACCESS_GROUP = {
    "fixed_threshold": "source_only",
    "source_validation_threshold": "source_only",
    "prior_quantile_full": "full_unlabeled_score",
    "prior_quantile_prefix10": "prefix10_unlabeled_score",
    "oracle_threshold": "oracle_diagnostic",
}


def read_prediction_table(path: str | Path) -> list[Prediction]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"subject_key", "dataset_id", "y_true", "p1"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"prediction file lacks required columns: {path}")
    result = []
    for fallback_order, row in enumerate(rows):
        probability = float(row["p1"])
        label = int(row["y_true"])
        if label not in {0, 1} or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"invalid prediction row in {path}")
        result.append(
            Prediction(
                subject_key=row["subject_key"],
                dataset_id=row["dataset_id"],
                y_true=label,
                p1=probability,
                order=int(row.get("trial_order") or fallback_order),
            )
        )
    return result


def _confusion(rows: Sequence[Prediction], threshold: float) -> tuple[int, int, int, int]:
    true_positive = false_positive = true_negative = false_negative = 0
    for row in rows:
        predicted = int(row.p1 >= threshold)
        if row.y_true == 1 and predicted == 1:
            true_positive += 1
        elif row.y_true == 0 and predicted == 1:
            false_positive += 1
        elif row.y_true == 0 and predicted == 0:
            true_negative += 1
        else:
            false_negative += 1
    return true_positive, false_positive, true_negative, false_negative


def macro_f1(rows: Sequence[Prediction], threshold: float) -> float:
    tp, fp, tn, fn = _confusion(rows, threshold)
    positive = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    negative = 0.0 if 2 * tn + fp + fn == 0 else 2 * tn / (2 * tn + fp + fn)
    return (positive + negative) / 2.0


def cohen_kappa(rows: Sequence[Prediction], threshold: float) -> float:
    tp, fp, tn, fn = _confusion(rows, threshold)
    total = tp + fp + tn + fn
    if total == 0:
        raise ValueError("kappa requires predictions")
    observed = (tp + tn) / total
    truth_positive = (tp + fn) / total
    predicted_positive = (tp + fp) / total
    expected = truth_positive * predicted_positive + (1 - truth_positive) * (
        1 - predicted_positive
    )
    return 0.0 if math.isclose(expected, 1.0) else (observed - expected) / (1 - expected)


def _threshold(
    rows: Sequence[Prediction],
    policy: str,
    positive_prior: float,
    prefix_trials: int,
    source_threshold: float | None,
) -> float:
    if policy == "fixed_threshold":
        return 0.5
    if policy == "source_validation_threshold":
        if source_threshold is None:
            raise ValueError("source-validation policy requires a frozen source threshold")
        return float(source_threshold)
    if policy == "prior_quantile_full":
        return prior_threshold(rows, positive_prior)
    if policy == "prior_quantile_prefix10":
        if len(rows) < prefix_trials:
            raise ValueError(
                f"prefix policy requires at least {prefix_trials} trials for every subject"
            )
        ordered = sorted(rows, key=lambda row: row.order)
        return prior_threshold(ordered[:prefix_trials], positive_prior)
    if policy == "oracle_threshold":
        return oracle_threshold(rows)[0]
    raise KeyError(f"unknown decision policy: {policy}")


def evaluate_access_policy(
    rows: Iterable[Prediction],
    policy: str,
    positive_prior: float = 0.5,
    prefix_trials: int = 10,
    source_threshold: float | None = None,
) -> dict[str, object]:
    if policy not in POLICY_ACCESS_GROUP:
        raise KeyError(f"unregistered access policy: {policy}")
    grouped = group_subjects(rows)
    subject_rows = []
    for subject_key, subject in sorted(grouped.items()):
        threshold = _threshold(
            subject, policy, positive_prior, prefix_trials, source_threshold
        )
        shifted = shifted_probabilities(subject, threshold)
        subject_rows.append(
            {
                "subject_key": subject_key,
                "dataset_id": subject[0].dataset_id,
                "n_trials": len(subject),
                "threshold": threshold,
                "balanced_accuracy": balanced_accuracy(subject, threshold),
                "roc_auc": roc_auc(subject),
                "macro_f1": macro_f1(subject, threshold),
                "cohen_kappa": cohen_kappa(subject, threshold),
                "ece": expected_calibration_error(subject, shifted),
                "brier": brier_score(subject, shifted),
            }
        )
    if not subject_rows:
        raise ValueError("evaluation requires at least one subject")
    metrics = [
        "balanced_accuracy",
        "roc_auc",
        "macro_f1",
        "cohen_kappa",
        "ece",
        "brier",
    ]
    aggregate = {
        f"mean_subject_{metric}": mean(float(row[metric]) for row in subject_rows)
        for metric in metrics
    }
    balanced = [float(row["balanced_accuracy"]) for row in subject_rows]
    aggregate["worst_subject_balanced_accuracy"] = min(balanced)
    aggregate["p10_subject_balanced_accuracy"] = quantile(balanced, 0.10)
    return {
        "policy": policy,
        "access_group": POLICY_ACCESS_GROUP[policy],
        "positive_prior": positive_prior if "prior" in policy else None,
        "prefix_trials": prefix_trials if "prefix" in policy else None,
        "subject_metrics": subject_rows,
        "aggregate": aggregate,
    }


def rank_within_access_group(evaluations: Iterable[dict[str, object]]) -> dict[str, list[str]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for evaluation in evaluations:
        grouped[str(evaluation["access_group"])].append(evaluation)
    result = {}
    for access_group, values in grouped.items():
        ordered = sorted(
            values,
            key=lambda item: float(item["aggregate"]["mean_subject_balanced_accuracy"]),
            reverse=True,
        )
        result[access_group] = [str(item["policy"]) for item in ordered]
    return result


def paired_bootstrap_gain(
    baseline: dict[str, float],
    proposed: dict[str, float],
    samples: int = 10000,
    seed: int = 20260811,
) -> dict[str, float]:
    subjects = sorted(set(baseline) & set(proposed))
    if not subjects or set(baseline) != set(proposed):
        raise ValueError("paired bootstrap requires identical non-empty subject keys")
    gains = [proposed[key] - baseline[key] for key in subjects]
    rng = random.Random(seed)
    estimates = [mean(rng.choice(gains) for _ in gains) for _ in range(samples)]
    estimates.sort()
    return {
        "mean_gain": mean(gains),
        "ci_lower": estimates[max(0, int(0.025 * samples) - 1)],
        "ci_upper": estimates[min(samples - 1, int(0.975 * samples))],
        "n_subjects": len(subjects),
    }


def confirmation_gate_report(
    mean_subject_auc: float,
    full_prior_gain: dict[str, float],
    group_norm_gain: dict[str, float],
    prefix_gain: float,
    full_gain: float,
    gates: dict[str, object],
) -> dict[str, object]:
    recovery = prefix_gain / full_gain if full_gain > 0 else float("nan")
    endpoints = {
        "mean_subject_auc": mean_subject_auc >= float(gates["mean_subject_auc_minimum"]),
        "full_prior_gain": (
            full_prior_gain["mean_gain"] >= float(gates["full_prior_ba_gain_minimum"])
            and full_prior_gain["ci_lower"]
            > float(gates["full_prior_gain_ci_lower_strictly_above"])
        ),
        "group_norm_gain": (
            group_norm_gain["mean_gain"] >= float(gates["group_norm_ba_gain_minimum"])
            and group_norm_gain["ci_lower"]
            > float(gates["group_norm_gain_ci_lower_strictly_above"])
        ),
        "prefix10_recovery_secondary": (
            math.isfinite(recovery)
            and recovery >= float(gates["prefix10_full_gain_recovery_minimum"])
        ),
    }
    return {
        "primary_confirmation_passed": all(
            endpoints[key] for key in ("mean_subject_auc", "full_prior_gain", "group_norm_gain")
        ),
        "endpoints": endpoints,
        "prefix10_recovery": recovery,
        "failure_policy": "report_without_reselection",
    }
