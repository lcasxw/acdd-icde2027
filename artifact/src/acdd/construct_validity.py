from __future__ import annotations

"""Vectorized Monte Carlo construct-validity checks for ACDD."""

import math
from typing import Any, Mapping

import numpy as np

from .metrics import oracle_logit_threshold


def _ensure_two_classes(rng: np.random.Generator, labels: np.ndarray, prior: float) -> None:
    invalid = (labels.sum(axis=1) == 0) | (labels.sum(axis=1) == labels.shape[1])
    while invalid.any():
        labels[invalid] = rng.binomial(1, prior, size=(int(invalid.sum()), labels.shape[1]))
        invalid = (labels.sum(axis=1) == 0) | (labels.sum(axis=1) == labels.shape[1])


def _balanced_accuracy(
    labels: np.ndarray, scores: np.ndarray, thresholds: float | np.ndarray
) -> np.ndarray:
    threshold_array = np.asarray(thresholds, dtype=np.float64)
    if threshold_array.ndim == 1:
        threshold_array = threshold_array[:, None]
    predictions = scores >= threshold_array
    positives = labels.sum(axis=1)
    negatives = labels.shape[1] - positives
    true_positive = (predictions & (labels == 1)).sum(axis=1)
    true_negative = ((~predictions) & (labels == 0)).sum(axis=1)
    return 0.5 * (true_positive / positives + true_negative / negatives)


def _oracle_balanced_accuracy(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, axis=1, kind="stable")
    sorted_labels = np.take_along_axis(labels, order, axis=1)
    cumulative_positive = np.cumsum(sorted_labels, axis=1)
    cumulative_negative = np.cumsum(1 - sorted_labels, axis=1)
    positives = labels.sum(axis=1)[:, None]
    negatives = (labels.shape[1] - labels.sum(axis=1))[:, None]
    after_cut = 0.5 * (
        cumulative_negative / negatives + (positives - cumulative_positive) / positives
    )
    before_all = np.full((labels.shape[0], 1), 0.5, dtype=np.float64)
    return np.max(np.concatenate((before_all, after_cut), axis=1), axis=1)


def _auc_no_ties(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, axis=1, kind="stable")
    ranks = np.argsort(order, axis=1, kind="stable") + 1
    positives = labels.sum(axis=1)
    negatives = labels.shape[1] - positives
    rank_sum = (ranks * labels).sum(axis=1)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> np.ndarray:
    result = np.zeros(labels.shape[0], dtype=np.float64)
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        mask = (probabilities >= lower) & (
            (probabilities < upper) | ((index == bins - 1) & (probabilities == 1.0))
        )
        count = mask.sum(axis=1)
        safe = np.maximum(count, 1)
        confidence = (probabilities * mask).sum(axis=1) / safe
        accuracy = (labels * mask).sum(axis=1) / safe
        result += count / labels.shape[1] * np.abs(confidence - accuracy)
    return result


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _true_decision_loss(separation: float, intercept: float) -> float:
    mean0 = -separation / 2.0 + intercept
    mean1 = separation / 2.0 + intercept

    def population_ba(threshold: float) -> float:
        tnr = _normal_cdf(threshold - mean0)
        tpr = _normal_cdf(mean1 - threshold)
        return 0.5 * (tnr + tpr)

    return population_ba(intercept) - population_ba(0.0)


def _paired_decision_loss_interval(
    labels: np.ndarray,
    scores: np.ndarray,
    oracle_threshold: float,
    z_value: float = 1.959963984540054,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    oracle_correct = (scores >= oracle_threshold) == labels
    default_correct = (scores >= 0.0) == labels
    differences = oracle_correct.astype(np.float64) - default_correct.astype(np.float64)
    means = []
    variances = []
    counts = []
    for label in (0, 1):
        mask = labels == label
        count = mask.sum(axis=1).astype(np.float64)
        total = (differences * mask).sum(axis=1)
        total_square = ((differences**2) * mask).sum(axis=1)
        group_mean = total / count
        group_variance = np.maximum(
            (total_square - total * total / count) / np.maximum(count - 1.0, 1.0), 0.0
        )
        means.append(group_mean)
        variances.append(group_variance)
        counts.append(count)
    estimate = 0.5 * (means[0] + means[1])
    standard_error = 0.5 * np.sqrt(
        variances[0] / counts[0] + variances[1] / counts[1]
    )
    return estimate, estimate - z_value * standard_error, estimate + z_value * standard_error


def simulate_cell(
    *,
    sample_size: int,
    positive_prior: float,
    separation: float,
    repetitions: int,
    seed: int,
    intercept_shift: float,
    slope_compression: float,
    rank_noise_standard_deviation: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    labels = rng.binomial(1, positive_prior, size=(repetitions, sample_size)).astype(np.int8)
    _ensure_two_classes(rng, labels, positive_prior)
    latent = rng.normal(
        loc=(labels * 2.0 - 1.0) * separation / 2.0,
        scale=1.0,
        size=(repetitions, sample_size),
    )
    shifted = latent + intercept_shift
    compressed = slope_compression * shifted
    corrupted = shifted + rng.normal(
        0.0, rank_noise_standard_deviation, size=shifted.shape
    )

    auc_latent = _auc_no_ties(labels, latent)
    auc_shifted = _auc_no_ties(labels, shifted)
    auc_compressed = _auc_no_ties(labels, compressed)
    auc_corrupted = _auc_no_ties(labels, corrupted)
    oracle_latent = _oracle_balanced_accuracy(labels, latent)
    oracle_shifted = _oracle_balanced_accuracy(labels, shifted)
    oracle_corrupted = _oracle_balanced_accuracy(labels, corrupted)
    ba_shifted_default = _balanced_accuracy(labels, shifted, 0.0)
    ba_compressed_default = _balanced_accuracy(labels, compressed, 0.0)
    ba_corrupted_default = _balanced_accuracy(labels, corrupted, 0.0)

    decision_loss_shifted = oracle_shifted - ba_shifted_default
    discrimination_change_intercept = oracle_shifted - oracle_latent
    discrimination_change_rank = oracle_corrupted - oracle_shifted
    decision_recovery_rank = decision_loss_shifted - (
        oracle_corrupted - ba_corrupted_default
    )
    performance_change_rank = ba_corrupted_default - ba_shifted_default
    residual = performance_change_rank - (
        discrimination_change_rank + decision_recovery_rank
    )

    probabilities = _sigmoid(shifted)
    compressed_probabilities = _sigmoid(compressed)
    ece = _ece(labels, probabilities)
    compressed_ece = _ece(labels, compressed_probabilities)

    estimate, lower, upper = _paired_decision_loss_interval(
        labels, shifted, intercept_shift
    )
    truth = _true_decision_loss(separation, intercept_shift)
    coverage = np.mean((lower <= truth) & (truth <= upper))
    return {
        "seed": seed,
        "sample_size": sample_size,
        "positive_prior": positive_prior,
        "class_separation": separation,
        "repetitions": repetitions,
        "mean_raw_logit_auc": float(auc_shifted.mean()),
        "maximum_auc_difference_positive_monotone": float(
            np.max(
                np.maximum(
                    np.abs(auc_latent - auc_shifted),
                    np.abs(auc_shifted - auc_compressed),
                )
            )
        ),
        "mean_auc_change_rank_corruption": float((auc_corrupted - auc_shifted).mean()),
        "mean_intercept_discrimination_change": float(
            discrimination_change_intercept.mean()
        ),
        "maximum_absolute_intercept_discrimination_change": float(
            np.max(np.abs(discrimination_change_intercept))
        ),
        "mean_intercept_decision_loss": float(decision_loss_shifted.mean()),
        "mean_slope_ece_change": float((compressed_ece - ece).mean()),
        "mean_absolute_slope_ece_change": float(np.abs(compressed_ece - ece).mean()),
        "maximum_slope_decision_change": float(
            np.max(np.abs(ba_compressed_default - ba_shifted_default))
        ),
        "mean_discrimination_change_rank_corruption": float(
            discrimination_change_rank.mean()
        ),
        "maximum_absolute_acdd_identity_residual": float(np.max(np.abs(residual))),
        "decision_loss_interval_coverage": float(coverage),
        "mean_decision_loss_interval_estimate": float(estimate.mean()),
        "population_decision_loss": float(truth),
    }


def nonidentifiability_witness() -> dict[str, Any]:
    scores = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    labels_a = [0, 0, 0, 0, 1, 1, 1, 1]
    labels_b = [0, 0, 1, 1, 1, 1, 1, 1]
    threshold_a, ba_a, _ = oracle_logit_threshold(labels_a, scores)
    threshold_b, ba_b, _ = oracle_logit_threshold(labels_b, scores)
    return {
        "unlabelled_score_marginal_identical": True,
        "joint_a_oracle_threshold": threshold_a,
        "joint_b_oracle_threshold": threshold_b,
        "joint_a_oracle_balanced_accuracy": ba_a,
        "joint_b_oracle_balanced_accuracy": ba_b,
        "different_optimal_thresholds": not math.isclose(threshold_a, threshold_b),
    }


def run_grid(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for seed in config["seeds"]:
        for sample_size in config["sample_sizes"]:
            for prior in config["positive_priors"]:
                for separation in config["class_separations"]:
                    rows.append(
                        simulate_cell(
                            sample_size=int(sample_size),
                            positive_prior=float(prior),
                            separation=float(separation),
                            repetitions=int(config["repetitions_per_seed_cell"]),
                            seed=int(seed),
                            intercept_shift=float(config["intercept_shift"]),
                            slope_compression=float(config["positive_slope_compression"]),
                            rank_noise_standard_deviation=float(
                                config["rank_noise_standard_deviation"]
                            ),
                        )
                    )
    minimum_n = int(config["coverage_minimum_n"])
    covered = [
        row["decision_loss_interval_coverage"]
        for row in rows
        if row["sample_size"] >= minimum_n
    ]
    witness = nonidentifiability_witness()
    gates = {
        "positive_monotone_auc_invariance": max(
            row["maximum_auc_difference_positive_monotone"] for row in rows
        )
        <= 1e-12,
        "intercept_selectivity": max(
            row["maximum_absolute_intercept_discrimination_change"] for row in rows
        )
        <= 1e-12
        and min(row["mean_intercept_decision_loss"] for row in rows) > 0.0,
        "slope_changes_ece_not_point_five_decision": max(
            row["maximum_slope_decision_change"] for row in rows
        )
        <= 1e-12
        and min(row["mean_absolute_slope_ece_change"] for row in rows) > 0.0,
        "rank_corruption_changes_discrimination": max(
            row["mean_discrimination_change_rank_corruption"] for row in rows
        )
        < 0.0,
        "acdd_identity": max(
            row["maximum_absolute_acdd_identity_residual"] for row in rows
        )
        <= float(config["identity_residual_tolerance"]),
        "coverage": min(covered) >= float(config["coverage_gate"]),
        "unlabelled_nonidentifiability": witness["different_optimal_thresholds"],
    }
    report = {
        "schema_version": 1,
        "status": "passed" if all(gates.values()) else "failed",
        "cells": len(rows),
        "total_repeated_datasets": sum(row["repetitions"] for row in rows),
        "minimum_coverage_n_ge_threshold": min(covered),
        "maximum_identity_residual": max(
            row["maximum_absolute_acdd_identity_residual"] for row in rows
        ),
        "nonidentifiability_witness": witness,
        "gates": gates,
    }
    return rows, report

