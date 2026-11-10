from __future__ import annotations

"""Post-confirmation validity analyses for the TNSRE strengthening study."""

from dataclasses import asdict, dataclass
import math
from statistics import mean
from typing import Any, Sequence

from .metrics import (
    TrialScore,
    balanced_accuracy_from_predictions,
    oracle_logit_threshold,
    roc_auc_from_raw_logits,
)
from .threshold_policies import (
    SOURCE_ONLY_METHODS,
    AccessContract,
    ThresholdPolicy,
    fit,
)


@dataclass(frozen=True)
class CrossFittedOracleResult:
    participant_id: str
    protocol_id: str
    n_sessions: int
    n_valid_folds: int
    n_trials: int
    balanced_accuracy_cross_fitted_oracle: float
    balanced_accuracy_default: float
    roc_auc_raw_logit: float
    cross_fitted_headroom: float
    cross_fitted_decision_loss: float
    in_sample_oracle_balanced_accuracy: float
    in_sample_headroom: float
    in_sample_decision_loss: float


def _macro_f1(labels: Sequence[int], predictions: Sequence[int]) -> float:
    values = []
    for label in (0, 1):
        true_positive = sum(y == label and p == label for y, p in zip(labels, predictions))
        false_positive = sum(y != label and p == label for y, p in zip(labels, predictions))
        false_negative = sum(y == label and p != label for y, p in zip(labels, predictions))
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(2 * true_positive / denominator if denominator else 0.0)
    return mean(values)


def _weighted_recall(
    labels: Sequence[int], predictions: Sequence[int], positive_weight: float = 0.75
) -> float:
    if not 0.0 < positive_weight < 1.0:
        raise ValueError("positive weight must lie between zero and one")
    recalls = []
    for label in (0, 1):
        indices = [index for index, value in enumerate(labels) if value == label]
        if not indices:
            raise ValueError("weighted recall requires both classes")
        recalls.append(mean(predictions[index] == label for index in indices))
    return (1.0 - positive_weight) * recalls[0] + positive_weight * recalls[1]


def leave_one_session_out_oracle(
    trials: Sequence[TrialScore],
) -> tuple[CrossFittedOracleResult, list[dict[str, Any]], list[dict[str, Any]]]:
    """Estimate an oracle threshold on other sessions and evaluate held sessions.

    This function is explicitly label-informed and diagnostic.  It is kept out
    of the deployable threshold-policy module to make accidental use impossible.
    """

    rows = sorted(trials, key=lambda row: row.order)
    if not rows or len({row.participant_id for row in rows}) != 1:
        raise ValueError("cross-fitted oracle requires one participant")
    sessions = list(dict.fromkeys(row.session_id for row in rows if row.session_id))
    if len(sessions) < 2:
        raise ValueError("leave-one-session-out requires at least two sessions")
    fold_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    held_labels: list[int] = []
    held_scores: list[float] = []
    held_predictions: list[int] = []
    for session in sessions:
        training = [row for row in rows if row.session_id != session]
        held = [row for row in rows if row.session_id == session]
        if len({row.y_true for row in training}) < 2:
            exclusions.append({"session_id": session, "reason": "training_fold_missing_class"})
            continue
        if len({row.y_true for row in held}) < 2:
            exclusions.append({"session_id": session, "reason": "held_fold_missing_class"})
            continue
        threshold, training_oracle_ba, _ = oracle_logit_threshold(
            [row.y_true for row in training], [row.raw_logit for row in training]
        )
        labels = [row.y_true for row in held]
        scores = [row.raw_logit for row in held]
        predictions = [int(score >= threshold) for score in scores]
        fold_rows.append(
            {
                "participant_id": rows[0].participant_id,
                "dataset_id": rows[0].dataset_id,
                "protocol_id": rows[0].protocol_id,
                "held_session_id": session,
                "n_training_trials": len(training),
                "n_held_trials": len(held),
                "training_oracle_threshold": threshold,
                "training_oracle_balanced_accuracy": training_oracle_ba,
                "held_balanced_accuracy": balanced_accuracy_from_predictions(
                    labels, predictions
                ),
            }
        )
        held_labels.extend(labels)
        held_scores.extend(scores)
        held_predictions.extend(predictions)
    if len(fold_rows) < 2:
        raise ValueError("participant has fewer than two valid cross-fitted folds")
    default_predictions = [int(score >= 0.0) for score in held_scores]
    default_ba = balanced_accuracy_from_predictions(held_labels, default_predictions)
    crossfit_ba = balanced_accuracy_from_predictions(held_labels, held_predictions)
    _, in_sample_oracle_ba, _ = oracle_logit_threshold(held_labels, held_scores)
    result = CrossFittedOracleResult(
        participant_id=rows[0].participant_id,
        protocol_id=rows[0].protocol_id,
        n_sessions=len(sessions),
        n_valid_folds=len(fold_rows),
        n_trials=len(held_labels),
        balanced_accuracy_cross_fitted_oracle=crossfit_ba,
        balanced_accuracy_default=default_ba,
        roc_auc_raw_logit=roc_auc_from_raw_logits(held_labels, held_scores),
        cross_fitted_headroom=crossfit_ba - 0.5,
        cross_fitted_decision_loss=crossfit_ba - default_ba,
        in_sample_oracle_balanced_accuracy=in_sample_oracle_ba,
        in_sample_headroom=in_sample_oracle_ba - 0.5,
        in_sample_decision_loss=in_sample_oracle_ba - default_ba,
    )
    return result, fold_rows, exclusions


def _access_contract(method: str, endpoint: str) -> AccessContract:
    if method in SOURCE_ONLY_METHODS:
        return AccessContract("source_only", endpoint, "none")
    mapping = {
        "prefix_suffix": ("prefix_unlabeled_score", "fixed_available", False),
        "prequential": ("prequential_unlabeled_score", "strict_past", False),
        "session1_to_later": ("session1_unlabeled_score", "fixed_available", False),
        "full_prior": ("full_unlabeled_score", "fixed_available", True),
    }
    try:
        group, scope, future = mapping[endpoint]
    except KeyError as exc:
        raise KeyError(f"unknown strengthening endpoint: {endpoint}") from exc
    return AccessContract(group, endpoint, scope, 0.5, future)


def _endpoint_partition(
    rows: Sequence[TrialScore], endpoint: str, prefix_length: int
) -> tuple[list[float], list[int]]:
    if endpoint in {"prefix_suffix", "prequential"}:
        if len(rows) <= prefix_length:
            raise ValueError("endpoint requires trials after the prefix")
        return [row.raw_logit for row in rows[:prefix_length]], list(
            range(prefix_length, len(rows))
        )
    if endpoint == "session1_to_later":
        sessions = list(dict.fromkeys(row.session_id for row in rows if row.session_id))
        if len(sessions) < 2:
            raise ValueError("session1 endpoint requires at least two sessions")
        first = sessions[0]
        return (
            [row.raw_logit for row in rows if row.session_id == first],
            [index for index, row in enumerate(rows) if row.session_id != first],
        )
    if endpoint == "full_prior":
        return [row.raw_logit for row in rows], list(range(len(rows)))
    raise KeyError(f"unknown strengthening endpoint: {endpoint}")


def policy_endpoint_metrics(
    trials: Sequence[TrialScore],
    source_validation_scores: Sequence[float],
    source_validation_labels: Sequence[int],
    *,
    method: str,
    endpoint: str,
    prefix_length: int = 10,
    rlls_ridge: float = 1e-3,
) -> dict[str, Any]:
    """Evaluate one access-matched policy on one participant and endpoint."""

    rows = sorted(trials, key=lambda row: row.order)
    if not rows or len({row.participant_id for row in rows}) != 1:
        raise ValueError("policy evaluation requires one participant")
    normalized = method.lower().replace("-", "_")
    contract = _access_contract(normalized, endpoint)
    available, indices = _endpoint_partition(rows, endpoint, prefix_length)
    fit_target = available if contract.target_score_scope == "fixed_available" else ()
    policy: ThresholdPolicy = fit(
        source_validation_scores,
        source_validation_labels,
        fit_target,
        contract,
        method=normalized,
        rlls_ridge=rlls_ridge,
    )
    if endpoint == "prequential":
        predictions = policy.predict_prequential_sequence(
            [row.raw_logit for row in rows], start=prefix_length
        )
    else:
        predictions = [policy.predict(rows[index].raw_logit) for index in indices]
    labels = [rows[index].y_true for index in indices]
    scores = [rows[index].raw_logit for index in indices]
    default_predictions = [int(score >= 0.0) for score in scores]
    access_ba = balanced_accuracy_from_predictions(labels, predictions)
    default_ba = balanced_accuracy_from_predictions(labels, default_predictions)
    _, oracle_ba, _ = oracle_logit_threshold(labels, scores)
    decision_loss = oracle_ba - default_ba
    gain = access_ba - default_ba
    return {
        "participant_id": rows[0].participant_id,
        "dataset_id": rows[0].dataset_id,
        "protocol_id": rows[0].protocol_id,
        "method": normalized,
        "access_group": contract.access_group,
        "endpoint": endpoint,
        "prefix_length": prefix_length if endpoint in {"prefix_suffix", "prequential"} else "",
        "uses_future_target_information": contract.uses_future_target_information,
        "target_labels_available_to_fit": False,
        "n_trials": len(labels),
        "balanced_accuracy_default": default_ba,
        "balanced_accuracy": access_ba,
        "roc_auc_raw_logit": roc_auc_from_raw_logits(labels, scores),
        "in_sample_oracle_balanced_accuracy": oracle_ba,
        "in_sample_headroom": oracle_ba - 0.5,
        "decision_loss": decision_loss,
        "gain": gain,
        "recovery_fraction": gain / decision_loss if decision_loss >= 0.01 else "",
        "harm_any_negative": gain < 0.0,
        "harm_gt_2pp": gain < -0.02,
        "macro_f1": _macro_f1(labels, predictions),
        "weighted_recall_0.25_0.75": _weighted_recall(labels, predictions, 0.75),
    }


def crossfit_result_dict(result: CrossFittedOracleResult) -> dict[str, Any]:
    return asdict(result)
