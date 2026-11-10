from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
import random
from statistics import mean
from typing import Iterable, Mapping, Sequence


RECOVERY_DENOMINATOR_MINIMUM = 0.01


@dataclass(frozen=True)
class TrialScore:
    """One frozen binary prediction on the unbounded logit-difference scale."""

    participant_id: str
    dataset_id: str
    protocol_id: str
    y_true: int
    raw_logit: float
    order: int
    session_id: str = ""


@dataclass(frozen=True)
class UnlabeledScore:
    """The only target information visible to a deployable threshold rule."""

    participant_id: str
    dataset_id: str
    protocol_id: str
    raw_logit: float
    order: int
    session_id: str = ""


@dataclass(frozen=True)
class ACDDResult:
    endpoint: str
    n_trials: int
    balanced_accuracy_default: float
    balanced_accuracy_access: float
    balanced_accuracy_oracle: float
    roc_auc_raw_logit: float
    rank_headroom: float
    decision_loss: float
    absolute_access_gain: float
    access_recovery_fraction: float | None
    recovery_fraction_reportable: bool
    harmed: bool
    oracle_discrimination_headroom: float | None = None
    access_decision_loss: float | None = None
    decision_loss_recovered: float | None = None
    rank_preserving: bool = True
    practically_harmed_2pp: bool = False


@dataclass(frozen=True)
class GeneralizedACDDResult:
    """Decompose a method change even when the method changes score ordering.

    The exact identity is

    ``delta_BA = delta_discrimination + decision_loss_recovery``.

    A normalized recovery fraction is defined only for methods that declare a
    rank-preserving score relationship and pass the empirical order check.
    """

    n_trials: int
    balanced_accuracy_baseline: float
    balanced_accuracy_method: float
    discrimination_headroom_baseline: float
    discrimination_headroom_method: float
    decision_loss_baseline: float
    decision_loss_method: float
    performance_change: float
    discrimination_change: float
    decision_loss_recovery: float
    decomposition_residual: float
    declared_rank_preserving: bool
    empirical_rank_preserving: bool
    recovery_fraction: float | None
    recovery_fraction_reportable: bool


@dataclass(frozen=True)
class AccessCondition:
    """Machine-readable target-information contract for one access stratum."""

    name: str
    target_signal_scope: str
    target_score_scope: str
    target_labels_available: bool
    declared_prior_available: bool
    uses_future_target_information: bool
    deployable_online: bool


ACCESS_EDGES = {
    "source_only": {
        "sample_local",
        "prefix_unlabeled_score",
        "session1_unlabeled_score",
        "online_unlabeled_signal",
    },
    "sample_local": {"online_unlabeled_signal"},
    "prefix_unlabeled_score": {"prequential_unlabeled_score", "full_unlabeled_score"},
    "prequential_unlabeled_score": {"full_unlabeled_score"},
    "session1_unlabeled_score": {"full_unlabeled_score"},
    "online_unlabeled_signal": {"full_unlabeled_signal"},
    "full_unlabeled_score": {"full_unlabeled_signal"},
    "full_unlabeled_signal": {"oracle_diagnostic"},
    "oracle_diagnostic": set(),
}


ACCESS_CONDITIONS = {
    row.name: row
    for row in (
        AccessCondition("source_only", "none", "none", False, False, False, True),
        AccessCondition("sample_local", "current", "none", False, False, False, True),
        AccessCondition(
            "prefix_unlabeled_score", "none", "fixed_prefix", False, True, False, True
        ),
        AccessCondition(
            "prequential_unlabeled_score",
            "none",
            "strict_past",
            False,
            True,
            False,
            True,
        ),
        AccessCondition(
            "session1_unlabeled_score", "none", "first_session", False, True, False, True
        ),
        AccessCondition(
            "online_unlabeled_signal", "past_and_current", "none", False, False, False, True
        ),
        AccessCondition(
            "full_unlabeled_score", "none", "complete_record", False, True, True, False
        ),
        AccessCondition(
            "full_unlabeled_signal", "complete_record", "none", False, False, True, False
        ),
        AccessCondition(
            "oracle_diagnostic", "complete_record", "complete_record", True, True, True, False
        ),
    )
}


def _check_labels(labels: Sequence[int]) -> None:
    if not labels or set(labels) - {0, 1}:
        raise ValueError("binary labels must be a non-empty sequence of zeros and ones")
    if len(set(labels)) != 2:
        raise ValueError("the metric requires both classes")


def _check_scores(scores: Sequence[float], labels: Sequence[int]) -> None:
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must be non-empty and have equal length")
    if not all(math.isfinite(float(value)) for value in scores):
        raise ValueError("scores must be finite")
    _check_labels(labels)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid quantile request")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def prior_quantile_logit_threshold(scores: Sequence[float], positive_prior: float) -> float:
    if not 0.0 < positive_prior < 1.0:
        raise ValueError("the declared positive prior must lie strictly between zero and one")
    if not scores or not all(math.isfinite(float(value)) for value in scores):
        raise ValueError("threshold estimation requires finite scores")
    return _quantile(scores, 1.0 - positive_prior)


class _FenwickOrderStatistics:
    """Incremental exact order statistics over a fixed finite value set."""

    def __init__(self, values: Sequence[float]) -> None:
        self.values = sorted(set(float(value) for value in values))
        self.ranks = {value: index + 1 for index, value in enumerate(self.values)}
        self.tree = [0] * (len(self.values) + 1)
        self.size = 0

    def add(self, value: float) -> None:
        index = self.ranks[float(value)]
        self.size += 1
        while index < len(self.tree):
            self.tree[index] += 1
            index += index & -index

    def kth(self, order: int) -> float:
        """Return the zero-based ``order``-th value, including ties."""

        if order < 0 or order >= self.size:
            raise IndexError("order statistic is outside the populated prefix")
        index = 0
        bit = 1 << (len(self.tree).bit_length() - 1)
        remaining = order + 1
        while bit:
            candidate = index + bit
            if candidate < len(self.tree) and self.tree[candidate] < remaining:
                remaining -= self.tree[candidate]
                index = candidate
            bit >>= 1
        return self.values[index]


def _prequential_thresholds_and_decisions(
    scores: Sequence[float], positive_prior: float
) -> tuple[list[float], list[int]]:
    """Compute all strict-past quantiles once in ``O(n log n)`` time."""

    if not 0.0 < positive_prior < 1.0:
        raise ValueError("the declared positive prior must lie strictly between zero and one")
    if not scores or not all(math.isfinite(float(value)) for value in scores):
        raise ValueError("threshold estimation requires finite scores")
    order_statistics = _FenwickOrderStatistics(scores)
    thresholds = [math.nan] * len(scores)
    decisions = [0] * len(scores)
    order_statistics.add(float(scores[0]))
    probability = 1.0 - positive_prior
    for index in range(1, len(scores)):
        position = probability * (index - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        lower_value = order_statistics.kth(lower)
        if lower == upper:
            threshold = lower_value
        else:
            upper_value = order_statistics.kth(upper)
            weight = position - lower
            threshold = lower_value * (1.0 - weight) + upper_value * weight
        thresholds[index] = threshold
        decisions[index] = int(float(scores[index]) >= threshold)
        order_statistics.add(float(scores[index]))
    return thresholds, decisions


def balanced_accuracy_from_predictions(
    labels: Sequence[int], predictions: Sequence[int]
) -> float:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have equal length")
    _check_labels(labels)
    recalls = []
    for label in (0, 1):
        indices = [index for index, value in enumerate(labels) if value == label]
        recalls.append(mean(int(predictions[index] == label) for index in indices))
    return mean(recalls)


def balanced_accuracy_from_scores(
    labels: Sequence[int], scores: Sequence[float], threshold: float = 0.0
) -> float:
    _check_scores(scores, labels)
    return balanced_accuracy_from_predictions(
        labels, [int(float(score) >= threshold) for score in scores]
    )


def roc_auc_from_raw_logits(labels: Sequence[int], raw_logits: Sequence[float]) -> float:
    """Tie-aware Mann--Whitney ROC AUC computed directly from raw logits."""

    _check_scores(raw_logits, labels)
    ordered = sorted(zip(raw_logits, labels), key=lambda item: item[0])
    positives = sum(label == 1 for _, label in ordered)
    negatives = len(ordered) - positives
    rank_sum = 0.0
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and ordered[stop][0] == ordered[start][0]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        rank_sum += average_rank * sum(label == 1 for _, label in ordered[start:stop])
        start = stop
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def youden_j(labels: Sequence[int], predictions: Sequence[int]) -> float:
    return 2.0 * balanced_accuracy_from_predictions(labels, predictions) - 1.0


def oracle_logit_threshold(
    labels: Sequence[int], raw_logits: Sequence[float]
) -> tuple[float, float, float]:
    _check_scores(raw_logits, labels)
    grouped: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    for score_value, label in zip(raw_logits, labels):
        grouped[float(score_value)][int(label)] += 1
    values = sorted(grouped)
    candidates = [math.nextafter(values[0], -math.inf)]
    candidates.extend((left + right) / 2.0 for left, right in zip(values, values[1:]))
    candidates.append(math.nextafter(values[-1], math.inf))
    best_threshold = candidates[0]
    negatives = sum(label == 0 for label in labels)
    positives = len(labels) - negatives
    true_negatives = 0
    true_positives = positives

    def current_ba() -> float:
        return mean((true_negatives / negatives, true_positives / positives))

    best_ba = current_ba()
    for index, value in enumerate(values):
        true_negatives += grouped[value][0]
        true_positives -= grouped[value][1]
        candidate_ba = current_ba()
        threshold = candidates[index + 1]
        if candidate_ba > best_ba:
            best_threshold, best_ba = threshold, candidate_ba
    best_predictions = [int(score >= best_threshold) for score in raw_logits]
    best_j = youden_j(labels, best_predictions)
    if not math.isclose(best_ba, (1.0 + best_j) / 2.0, abs_tol=1e-12):
        raise AssertionError("BA=(1+J)/2 identity failed")
    return best_threshold, best_ba, best_j


def _rank_signature(scores: Sequence[float]) -> tuple[float, ...]:
    """Return tie-aware empirical ranks without consulting labels."""

    if not scores or not all(math.isfinite(float(value)) for value in scores):
        raise ValueError("rank comparison requires finite scores")
    indexed = sorted(enumerate(float(value) for value in scores), key=lambda item: item[1])
    ranks = [0.0] * len(indexed)
    start = 0
    while start < len(indexed):
        stop = start + 1
        while stop < len(indexed) and indexed[stop][1] == indexed[start][1]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        for original_index, _ in indexed[start:stop]:
            ranks[original_index] = average_rank
        start = stop
    return tuple(ranks)


def generalized_acdd_comparison(
    labels: Sequence[int],
    baseline_scores: Sequence[float],
    baseline_predictions: Sequence[int],
    method_scores: Sequence[float],
    method_predictions: Sequence[int],
    *,
    declared_rank_preserving: bool,
    denominator_minimum: float = RECOVERY_DENOMINATOR_MINIMUM,
) -> GeneralizedACDDResult:
    """Decompose a method comparison into discrimination and decision terms.

    Methods such as target-score threshold transport should set
    ``declared_rank_preserving=True``. Representation-changing adaptation
    methods must set it to ``False`` and therefore never receive a normalized
    recovery fraction, even if one finite evaluation happens to retain the
    empirical order.
    """

    _check_scores(baseline_scores, labels)
    _check_scores(method_scores, labels)
    if len(baseline_predictions) != len(labels) or len(method_predictions) != len(labels):
        raise ValueError("predictions must match the evaluation trials")
    baseline_ba = balanced_accuracy_from_predictions(labels, baseline_predictions)
    method_ba = balanced_accuracy_from_predictions(labels, method_predictions)
    _, baseline_oracle, _ = oracle_logit_threshold(labels, baseline_scores)
    _, method_oracle, _ = oracle_logit_threshold(labels, method_scores)
    baseline_headroom = baseline_oracle - 0.5
    method_headroom = method_oracle - 0.5
    baseline_loss = baseline_oracle - baseline_ba
    method_loss = method_oracle - method_ba
    performance_change = method_ba - baseline_ba
    discrimination_change = method_headroom - baseline_headroom
    decision_loss_recovery = baseline_loss - method_loss
    residual = performance_change - (discrimination_change + decision_loss_recovery)
    if not math.isclose(residual, 0.0, abs_tol=1e-12):
        raise AssertionError("generalized ACDD identity failed")
    empirical_rank_preserving = _rank_signature(baseline_scores) == _rank_signature(
        method_scores
    )
    if declared_rank_preserving and not empirical_rank_preserving:
        raise ValueError("declared rank-preserving method changed the empirical score order")
    reportable = (
        declared_rank_preserving
        and empirical_rank_preserving
        and baseline_loss >= denominator_minimum
    )
    recovery = decision_loss_recovery / baseline_loss if reportable else None
    return GeneralizedACDDResult(
        n_trials=len(labels),
        balanced_accuracy_baseline=baseline_ba,
        balanced_accuracy_method=method_ba,
        discrimination_headroom_baseline=baseline_headroom,
        discrimination_headroom_method=method_headroom,
        decision_loss_baseline=baseline_loss,
        decision_loss_method=method_loss,
        performance_change=performance_change,
        discrimination_change=discrimination_change,
        decision_loss_recovery=decision_loss_recovery,
        decomposition_residual=residual,
        declared_rank_preserving=declared_rank_preserving,
        empirical_rank_preserving=empirical_rank_preserving,
        recovery_fraction=recovery,
        recovery_fraction_reportable=reportable,
    )


def _acdd_reference(
    labels: Sequence[int], raw_logits: Sequence[float]
) -> tuple[float, float, float]:
    _check_scores(raw_logits, labels)
    default_predictions = [int(score >= 0.0) for score in raw_logits]
    default_ba = balanced_accuracy_from_predictions(labels, default_predictions)
    _, oracle_ba, _ = oracle_logit_threshold(labels, raw_logits)
    return default_ba, oracle_ba, roc_auc_from_raw_logits(labels, raw_logits)


def _decompose_decision_from_reference(
    labels: Sequence[int],
    access_predictions: Sequence[int],
    endpoint: str,
    reference: tuple[float, float, float],
    denominator_minimum: float = RECOVERY_DENOMINATOR_MINIMUM,
) -> ACDDResult:
    if len(access_predictions) != len(labels):
        raise ValueError("access predictions must match the evaluation trials")
    default_ba, oracle_ba, roc_auc = reference
    access_ba = balanced_accuracy_from_predictions(labels, access_predictions)
    decision_loss = oracle_ba - default_ba
    access_decision_loss = oracle_ba - access_ba
    gain = access_ba - default_ba
    reportable = decision_loss >= denominator_minimum
    recovery = gain / decision_loss if reportable else None
    return ACDDResult(
        endpoint=endpoint,
        n_trials=len(labels),
        balanced_accuracy_default=default_ba,
        balanced_accuracy_access=access_ba,
        balanced_accuracy_oracle=oracle_ba,
        roc_auc_raw_logit=roc_auc,
        rank_headroom=oracle_ba - 0.5,
        decision_loss=decision_loss,
        absolute_access_gain=gain,
        access_recovery_fraction=recovery,
        recovery_fraction_reportable=reportable,
        harmed=access_ba < default_ba,
        oracle_discrimination_headroom=oracle_ba - 0.5,
        access_decision_loss=access_decision_loss,
        decision_loss_recovered=decision_loss - access_decision_loss,
        rank_preserving=True,
        practically_harmed_2pp=gain < -0.02,
    )


def decompose_decision(
    labels: Sequence[int],
    raw_logits: Sequence[float],
    access_predictions: Sequence[int],
    endpoint: str,
    denominator_minimum: float = RECOVERY_DENOMINATOR_MINIMUM,
) -> ACDDResult:
    reference = _acdd_reference(labels, raw_logits)
    return _decompose_decision_from_reference(
        labels,
        access_predictions,
        endpoint,
        reference,
        denominator_minimum,
    )


def _ordered(trials: Sequence[TrialScore], order: str) -> list[TrialScore]:
    if order == "acquisition":
        return sorted(trials, key=lambda row: row.order)
    if order == "class0_then_class1_diagnostic":
        return sorted(trials, key=lambda row: (row.y_true, row.order))
    if order == "class1_then_class0_diagnostic":
        return sorted(trials, key=lambda row: (-row.y_true, row.order))
    raise KeyError(f"unknown order: {order}")


def _validate_one_participant(trials: Sequence[TrialScore]) -> None:
    if not trials:
        raise ValueError("endpoint evaluation requires trials")
    if len({row.participant_id for row in trials}) != 1:
        raise ValueError("evaluate endpoints separately for each participant")
    if len({row.dataset_id for row in trials}) != 1:
        raise ValueError("one participant cannot span datasets")
    if any(row.y_true not in {0, 1} or not math.isfinite(row.raw_logit) for row in trials):
        raise ValueError("invalid trial score")


def _strip_labels(rows: Sequence[TrialScore]) -> list[UnlabeledScore]:
    return [
        UnlabeledScore(
            participant_id=row.participant_id,
            dataset_id=row.dataset_id,
            protocol_id=row.protocol_id,
            raw_logit=row.raw_logit,
            order=row.order,
            session_id=row.session_id,
        )
        for row in rows
    ]


def _validate_unlabeled_one_participant(rows: Sequence[UnlabeledScore]) -> None:
    if not rows:
        raise ValueError("endpoint evaluation requires unlabeled scores")
    if len({row.participant_id for row in rows}) != 1:
        raise ValueError("evaluate endpoints separately for each participant")
    if len({row.dataset_id for row in rows}) != 1:
        raise ValueError("one participant cannot span datasets")
    if any(not math.isfinite(row.raw_logit) for row in rows):
        raise ValueError("unlabeled scores must be finite")


def _endpoint_decisions_from_sequence(
    rows: Sequence[UnlabeledScore],
    endpoint: str,
    prefix_length: int,
    positive_prior: float,
) -> tuple[list[int], list[float], list[int]]:
    """Threshold logic whose type signature physically excludes target labels."""

    _validate_unlabeled_one_participant(rows)
    scores = [row.raw_logit for row in rows]
    if prefix_length < 1:
        raise ValueError("prefix length must be positive")
    if endpoint in {"prefix_full", "prefix_suffix", "prequential"} and len(rows) <= prefix_length:
        raise ValueError("endpoint requires trials after the prefix")
    if endpoint == "prefix_full":
        threshold = prior_quantile_logit_threshold(scores[:prefix_length], positive_prior)
        indices = list(range(len(rows)))
        thresholds = [threshold] * len(indices)
    elif endpoint == "prefix_suffix":
        threshold = prior_quantile_logit_threshold(scores[:prefix_length], positive_prior)
        indices = list(range(prefix_length, len(rows)))
        thresholds = [threshold] * len(indices)
    elif endpoint == "prequential":
        indices = list(range(prefix_length, len(rows)))
        all_thresholds, all_decisions = _prequential_thresholds_and_decisions(
            scores, positive_prior
        )
        thresholds = [all_thresholds[index] for index in indices]
    elif endpoint == "full_prior":
        threshold = prior_quantile_logit_threshold(scores, positive_prior)
        indices = list(range(len(rows)))
        thresholds = [threshold] * len(indices)
    elif endpoint == "full_prior_suffix":
        if len(rows) <= prefix_length:
            raise ValueError("full_prior_suffix requires trials after the prefix")
        threshold = prior_quantile_logit_threshold(scores, positive_prior)
        indices = list(range(prefix_length, len(rows)))
        thresholds = [threshold] * len(indices)
    elif endpoint == "session1_to_later":
        sessions = []
        for row in rows:
            if row.session_id and row.session_id not in sessions:
                sessions.append(row.session_id)
        if len(sessions) < 2:
            raise ValueError("session1_to_later requires at least two sessions")
        calibration = [row.raw_logit for row in rows if row.session_id == sessions[0]]
        threshold = prior_quantile_logit_threshold(calibration, positive_prior)
        indices = [index for index, row in enumerate(rows) if row.session_id != sessions[0]]
        thresholds = [threshold] * len(indices)
    else:
        raise KeyError(f"unknown endpoint: {endpoint}")
    if endpoint == "prequential":
        decisions = [all_decisions[index] for index in indices]
    else:
        decisions = [int(scores[index] >= threshold) for index, threshold in zip(indices, thresholds)]
    return indices, thresholds, decisions


def endpoint_decisions_from_unlabeled(
    rows: Sequence[UnlabeledScore],
    endpoint: str,
    prefix_length: int = 10,
    positive_prior: float = 0.5,
) -> tuple[list[int], list[float], list[int]]:
    """Evaluate a deployable endpoint in recorded acquisition order."""

    ordered = sorted(rows, key=lambda row: row.order)
    return _endpoint_decisions_from_sequence(
        ordered, endpoint, prefix_length=prefix_length, positive_prior=positive_prior
    )


def endpoint_decisions(
    trials: Sequence[TrialScore],
    endpoint: str,
    prefix_length: int = 10,
    positive_prior: float = 0.5,
    order: str = "acquisition",
) -> tuple[list[int], list[float], list[int]]:
    """Return evaluation indices, causal thresholds and access decisions.

    Threshold estimation never receives labels. Label-created orders are named
    diagnostics and must not be presented as deployable access conditions.
    """

    _validate_one_participant(trials)
    rows = _ordered(trials, order)
    return _endpoint_decisions_from_sequence(
        _strip_labels(rows),
        endpoint,
        prefix_length=prefix_length,
        positive_prior=positive_prior,
    )


def evaluate_endpoint(
    trials: Sequence[TrialScore],
    endpoint: str,
    prefix_length: int = 10,
    positive_prior: float = 0.5,
    order: str = "acquisition",
) -> ACDDResult:
    rows = _ordered(trials, order)
    indices, _, decisions = endpoint_decisions(
        trials, endpoint, prefix_length, positive_prior, order
    )
    labels = [rows[index].y_true for index in indices]
    scores = [rows[index].raw_logit for index in indices]
    return decompose_decision(labels, scores, decisions, endpoint)


def _target_information_response_surface_legacy(
    trials: Sequence[TrialScore],
    prefix_lengths: Sequence[int | str],
    positive_priors: Sequence[float],
    orders: Sequence[str],
    evaluation_scopes: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    endpoint_by_scope = {
        "full": "prefix_full",
        "suffix": "prefix_suffix",
        "prequential": "prequential",
    }
    for order in orders:
        for prior in positive_priors:
            for prefix in prefix_lengths:
                for scope in evaluation_scopes:
                    endpoint = "full_prior" if prefix == "full" else endpoint_by_scope[scope]
                    if prefix == "full" and scope != "full":
                        continue
                    try:
                        result = evaluate_endpoint(
                            trials,
                            endpoint,
                            prefix_length=len(trials) if prefix == "full" else int(prefix),
                            positive_prior=float(prior),
                            order=order,
                        )
                    except ValueError as exc:
                        exclusions.append(
                            {
                                "participant_id": trials[0].participant_id,
                                "k": prefix,
                                "positive_prior": prior,
                                "order": order,
                                "evaluation_scope": scope,
                                "reason": str(exc),
                            }
                        )
                        continue
                    rows.append(
                        {
                            "participant_id": trials[0].participant_id,
                            "dataset_id": trials[0].dataset_id,
                            "protocol_id": trials[0].protocol_id,
                            "k": prefix,
                            "positive_prior": prior,
                            "order": order,
                            "order_uses_labels": order != "acquisition",
                            "evaluation_scope": scope,
                            **asdict(result),
                        }
                    )
    return rows, exclusions


def target_information_response_surface(
    trials: Sequence[TrialScore],
    prefix_lengths: Sequence[int | str],
    positive_priors: Sequence[float],
    orders: Sequence[str],
    evaluation_scopes: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Equivalent response surface with cached order, quantiles and rank metrics."""

    _validate_one_participant(trials)
    output_rows: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    endpoint_by_scope = {
        "full": "prefix_full",
        "suffix": "prefix_suffix",
        "prequential": "prequential",
    }
    ordered = {order: _ordered(trials, order) for order in orders}
    scores = {order: [row.raw_logit for row in rows] for order, rows in ordered.items()}
    references: dict[tuple[str, str, int], tuple[float, float, float]] = {}
    prequential: dict[tuple[str, float], tuple[list[float], list[int]]] = {}
    for order in orders:
        current_rows = ordered[order]
        current_scores = scores[order]
        for prior in positive_priors:
            prior_value = float(prior)
            for prefix in prefix_lengths:
                for scope in evaluation_scopes:
                    endpoint = "full_prior" if prefix == "full" else endpoint_by_scope[scope]
                    if prefix == "full" and scope != "full":
                        continue
                    prefix_length = len(current_rows) if prefix == "full" else int(prefix)
                    try:
                        if prefix_length < 1:
                            raise ValueError("prefix length must be positive")
                        if (
                            endpoint in {"prefix_full", "prefix_suffix", "prequential"}
                            and len(current_rows) <= prefix_length
                        ):
                            raise ValueError("endpoint requires trials after the prefix")
                        if endpoint == "prefix_full":
                            threshold = prior_quantile_logit_threshold(
                                current_scores[:prefix_length], prior_value
                            )
                            indices = list(range(len(current_rows)))
                            decisions = [
                                int(current_scores[index] >= threshold) for index in indices
                            ]
                            reference_key = (order, "full", 0)
                        elif endpoint == "prefix_suffix":
                            threshold = prior_quantile_logit_threshold(
                                current_scores[:prefix_length], prior_value
                            )
                            indices = list(range(prefix_length, len(current_rows)))
                            decisions = [
                                int(current_scores[index] >= threshold) for index in indices
                            ]
                            reference_key = (order, "suffix", prefix_length)
                        elif endpoint == "prequential":
                            prequential_key = (order, prior_value)
                            if prequential_key not in prequential:
                                prequential[prequential_key] = (
                                    _prequential_thresholds_and_decisions(
                                        current_scores, prior_value
                                    )
                                )
                            _, all_decisions = prequential[prequential_key]
                            indices = list(range(prefix_length, len(current_rows)))
                            decisions = [all_decisions[index] for index in indices]
                            reference_key = (order, "suffix", prefix_length)
                        elif endpoint == "full_prior":
                            threshold = prior_quantile_logit_threshold(
                                current_scores, prior_value
                            )
                            indices = list(range(len(current_rows)))
                            decisions = [
                                int(current_scores[index] >= threshold) for index in indices
                            ]
                            reference_key = (order, "full", 0)
                        else:  # pragma: no cover - guarded by endpoint_by_scope
                            raise KeyError(f"unknown endpoint: {endpoint}")
                        labels = [current_rows[index].y_true for index in indices]
                        evaluation_scores = [current_scores[index] for index in indices]
                        if reference_key not in references:
                            references[reference_key] = _acdd_reference(
                                labels, evaluation_scores
                            )
                        result = _decompose_decision_from_reference(
                            labels,
                            decisions,
                            endpoint,
                            references[reference_key],
                        )
                    except ValueError as exc:
                        exclusions.append(
                            {
                                "participant_id": trials[0].participant_id,
                                "k": prefix,
                                "positive_prior": prior,
                                "order": order,
                                "evaluation_scope": scope,
                                "reason": str(exc),
                            }
                        )
                        continue
                    output_rows.append(
                        {
                            "participant_id": trials[0].participant_id,
                            "dataset_id": trials[0].dataset_id,
                            "protocol_id": trials[0].protocol_id,
                            "k": prefix,
                            "positive_prior": prior,
                            "order": order,
                            "order_uses_labels": order != "acquisition",
                            "evaluation_scope": scope,
                            **asdict(result),
                        }
                    )
    return output_rows, exclusions


def access_leq(left: str, right: str) -> bool:
    if left not in ACCESS_EDGES or right not in ACCESS_EDGES:
        raise KeyError("unknown access group")
    if left == right:
        return True
    frontier = list(ACCESS_EDGES[left])
    visited = set()
    while frontier:
        node = frontier.pop()
        if node == right:
            return True
        if node not in visited:
            visited.add(node)
            frontier.extend(ACCESS_EDGES[node])
    return False


def access_condition(name: str) -> AccessCondition:
    try:
        return ACCESS_CONDITIONS[name]
    except KeyError as exc:
        raise KeyError(f"unknown access group: {name}") from exc


def assert_online_deployable_access(name: str) -> None:
    condition = access_condition(name)
    if condition.target_labels_available:
        raise PermissionError(f"{name} uses target labels and is diagnostic only")
    if condition.uses_future_target_information or not condition.deployable_online:
        raise PermissionError(f"{name} is transductive rather than online deployable")


def assert_within_access_ranking(rows: Iterable[Mapping[str, object]]) -> None:
    groups = {str(row["access_group"]) for row in rows}
    if len(groups) > 1:
        raise ValueError("ranking across access groups is prohibited; report a partial order or frontier")


def access_pareto_frontier(
    rows: Sequence[Mapping[str, object]], performance_key: str
) -> list[Mapping[str, object]]:
    """Keep points not dominated by a no-more-access point with better performance."""

    frontier = []
    for candidate in rows:
        dominated = False
        for alternative in rows:
            if alternative is candidate:
                continue
            no_more_access = access_leq(
                str(alternative["access_group"]), str(candidate["access_group"])
            )
            no_worse = float(alternative[performance_key]) >= float(candidate[performance_key])
            strictly_better = (
                str(alternative["access_group"]) != str(candidate["access_group"])
                or float(alternative[performance_key]) > float(candidate[performance_key])
            )
            if no_more_access and no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier


def _percentile(values: Sequence[float], probability: float) -> float:
    return _quantile(values, probability)


def hierarchical_bootstrap(
    records: Sequence[Mapping[str, object]],
    value_key: str,
    samples: int = 10_000,
    seed: int = 20260813,
) -> dict[str, dict[str, float]]:
    """Resample dataset -> protocol -> participant after repeat aggregation.

    Repeats (for example seeds) are first averaged in each
    dataset/protocol/participant cell. Participant- and dataset-weighted
    estimands then average sampled protocol directions within each participant,
    so a participant is not counted once per source direction. The
    protocol-balanced estimand intentionally retains one mean per protocol.
    """

    if not records or samples < 1:
        raise ValueError("hierarchical bootstrap requires records and positive samples")
    repeated: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in records:
        key = (str(row["dataset_id"]), str(row["protocol_id"]), str(row["participant_id"]))
        value = float(row[value_key])
        if not math.isfinite(value):
            raise ValueError("bootstrap values must be finite")
        repeated[key].append(value)
    aggregated = {key: mean(values) for key, values in repeated.items()}
    hierarchy: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (dataset, protocol, participant), value in aggregated.items():
        hierarchy[dataset][protocol][participant] = value
    datasets = sorted(hierarchy)
    rng = random.Random(seed)
    estimates = {name: [] for name in ("participant_weighted", "dataset_balanced", "protocol_balanced")}
    for _ in range(samples):
        sampled_datasets = [rng.choice(datasets) for _ in datasets]
        dataset_means = []
        protocol_means_all = []
        participant_values_all = []
        for dataset in sampled_datasets:
            protocols = sorted(hierarchy[dataset])
            sampled_protocols = [rng.choice(protocols) for _ in protocols]
            protocol_means = []
            sampled_direction_values: dict[str, list[float]] = defaultdict(list)
            for protocol in sampled_protocols:
                participant_map = hierarchy[dataset][protocol]
                participant_ids = sorted(participant_map)
                sampled_ids = [rng.choice(participant_ids) for _ in participant_ids]
                sampled_participants = [participant_map[name] for name in sampled_ids]
                protocol_mean = mean(sampled_participants)
                protocol_means.append(protocol_mean)
                protocol_means_all.append(protocol_mean)
                for participant, value in participant_map.items():
                    sampled_direction_values[participant].append(value)
            participant_direction_means = [
                mean(values) for values in sampled_direction_values.values()
            ]
            sampled_participant_means = [
                rng.choice(participant_direction_means)
                for _ in participant_direction_means
            ]
            dataset_means.append(mean(sampled_participant_means))
            participant_values_all.extend(sampled_participant_means)
        estimates["participant_weighted"].append(mean(participant_values_all))
        estimates["dataset_balanced"].append(mean(dataset_means))
        estimates["protocol_balanced"].append(mean(protocol_means_all))
    participant_points: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (dataset, _, participant), value in aggregated.items():
        participant_points[(dataset, participant)].append(value)
    point_values = [mean(values) for values in participant_points.values()]
    dataset_point = mean(
        mean(
            value
            for (dataset, _), values in participant_points.items()
            if dataset == name
            for value in [mean(values)]
        )
        for name in datasets
    )
    protocol_keys = sorted({(dataset, protocol) for dataset, protocol, _ in aggregated})
    protocol_point = mean(
        mean(
            value
            for (dataset, protocol, _), value in aggregated.items()
            if (dataset, protocol) == key
        )
        for key in protocol_keys
    )
    points = {
        "participant_weighted": mean(point_values),
        "dataset_balanced": dataset_point,
        "protocol_balanced": protocol_point,
    }
    return {
        name: {
            "estimate": points[name],
            "ci_lower": _percentile(values, 0.025),
            "ci_upper": _percentile(values, 0.975),
        }
        for name, values in estimates.items()
    }


def tail_and_harm_summary(
    results: Sequence[ACDDResult], practical_harm_delta: float = 0.02
) -> dict[str, float]:
    if not results:
        raise ValueError("tail summary requires participant results")
    if practical_harm_delta < 0:
        raise ValueError("practical harm delta must be non-negative")
    access = [row.balanced_accuracy_access for row in results]
    gains = [row.balanced_accuracy_access - row.balanced_accuracy_default for row in results]
    return {
        "harm_rate_zero": mean(float(gain < 0.0) for gain in gains),
        "harm_rate": mean(float(gain < 0.0) for gain in gains),
        "practical_harm_delta": practical_harm_delta,
        "harm_rate_practical": mean(float(gain < -practical_harm_delta) for gain in gains),
        "median_gain": _percentile(gains, 0.5),
        "p05_gain": _percentile(gains, 0.05),
        "p10_gain": _percentile(gains, 0.10),
        "p05_access_balanced_accuracy": _percentile(access, 0.05),
        "p10_access_balanced_accuracy": _percentile(access, 0.10),
    }


def bootstrap_tail_and_harm_summary(
    results: Sequence[ACDDResult],
    *,
    samples: int = 10_000,
    seed: int = 20260815,
    practical_harm_delta: float = 0.02,
) -> dict[str, object]:
    """Participant-bootstrap intervals for central, tail and harm estimands."""

    if not results or samples < 1:
        raise ValueError("bootstrap tail summary requires results and positive samples")
    point = tail_and_harm_summary(results, practical_harm_delta)
    rng = random.Random(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        sampled = [rng.choice(results) for _ in results]
        summary = tail_and_harm_summary(sampled, practical_harm_delta)
        for key, value in summary.items():
            if key != "practical_harm_delta":
                draws[key].append(float(value))
    intervals = {
        key: {
            "estimate": float(point[key]),
            "ci_lower": _percentile(values, 0.025),
            "ci_upper": _percentile(values, 0.975),
        }
        for key, values in draws.items()
    }
    return {
        "resampling_unit": "participant",
        "samples": samples,
        "seed": seed,
        "practical_harm_delta": practical_harm_delta,
        "estimands": intervals,
    }
