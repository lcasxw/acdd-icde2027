from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Iterable, Sequence


EPSILON = 1e-7


@dataclass(frozen=True)
class Prediction:
    subject_key: str
    dataset_id: str
    y_true: int
    p1: float
    order: int


def read_predictions(path: str | Path) -> list[Prediction]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"subject_key", "dataset_id", "y_true", "p1"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"prediction file lacks required columns: {path}")
    result = []
    for fallback_order, row in enumerate(rows):
        label = int(row["y_true"])
        probability = float(row["p1"])
        if label not in {0, 1}:
            raise ValueError(f"non-binary label in {path}: {label}")
        if not 0.0 <= probability <= 1.0 or not math.isfinite(probability):
            raise ValueError(f"invalid probability in {path}: {probability}")
        result.append(
            Prediction(
                subject_key=str(row["subject_key"]),
                dataset_id=str(row["dataset_id"]),
                y_true=label,
                p1=probability,
                order=int(row.get("trial_order") or fallback_order),
            )
        )
    return result


def group_subjects(rows: Iterable[Prediction]) -> dict[str, list[Prediction]]:
    grouped: dict[str, list[Prediction]] = defaultdict(list)
    for row in rows:
        grouped[row.subject_key].append(row)
    return {key: sorted(values, key=lambda item: item.order) for key, values in grouped.items()}


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must be between zero and one")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def prior_threshold(rows: Sequence[Prediction], positive_prior: float) -> float:
    if not 0.0 < positive_prior < 1.0:
        raise ValueError("positive prior must lie strictly between zero and one")
    return quantile([row.p1 for row in rows], 1.0 - positive_prior)


def balanced_accuracy(rows: Sequence[Prediction], threshold: float) -> float:
    recalls = []
    for label in (0, 1):
        selected = [row for row in rows if row.y_true == label]
        if not selected:
            raise ValueError("balanced accuracy requires both classes")
        correct = sum(int(row.p1 >= threshold) == label for row in selected)
        recalls.append(correct / len(selected))
    return mean(recalls)


def roc_auc(rows: Sequence[Prediction]) -> float:
    ordered = sorted(rows, key=lambda row: row.p1)
    positives = sum(row.y_true == 1 for row in ordered)
    negatives = len(ordered) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC requires both classes")
    rank_sum = 0.0
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and ordered[stop].p1 == ordered[start].p1:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        rank_sum += average_rank * sum(row.y_true == 1 for row in ordered[start:stop])
        start = stop
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def brier_score(rows: Sequence[Prediction], probabilities: Sequence[float] | None = None) -> float:
    values = probabilities if probabilities is not None else [row.p1 for row in rows]
    return mean((float(value) - row.y_true) ** 2 for row, value in zip(rows, values))


def expected_calibration_error(
    rows: Sequence[Prediction], probabilities: Sequence[float] | None = None, bins: int = 10
) -> float:
    values = list(probabilities if probabilities is not None else [row.p1 for row in rows])
    result = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [
            position
            for position, value in enumerate(values)
            if lower <= value < upper or (index == bins - 1 and value == 1.0)
        ]
        if not selected:
            continue
        confidence = mean(max(values[position], 1.0 - values[position]) for position in selected)
        accuracy = mean(
            int(values[position] >= 0.5) == rows[position].y_true for position in selected
        )
        result += len(selected) / len(rows) * abs(accuracy - confidence)
    return result


def shifted_probabilities(rows: Sequence[Prediction], threshold: float) -> list[float]:
    threshold = min(1.0 - EPSILON, max(EPSILON, threshold))
    threshold_logit = math.log(threshold / (1.0 - threshold))
    result = []
    for row in rows:
        probability = min(1.0 - EPSILON, max(EPSILON, row.p1))
        shifted = math.log(probability / (1.0 - probability)) - threshold_logit
        result.append(1.0 / (1.0 + math.exp(-shifted)))
    return result


def oracle_threshold(rows: Sequence[Prediction]) -> tuple[float, float]:
    candidates = sorted({row.p1 for row in rows})
    if not candidates:
        raise ValueError("oracle threshold requires predictions")
    margins = [
        candidates[0] - EPSILON,
        *[(first + second) / 2.0 for first, second in zip(candidates, candidates[1:])],
        candidates[-1] + EPSILON,
    ]
    best_threshold = 0.5
    best_score = -1.0
    for threshold in margins:
        score = balanced_accuracy(rows, threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    return best_threshold, best_score


def select_source_threshold(rows: Sequence[Prediction], grid_size: int) -> tuple[float, float]:
    if grid_size < 3:
        raise ValueError("source threshold grid must contain at least three points")
    candidates = sorted(
        {
            quantile([row.p1 for row in rows], index / (grid_size - 1))
            for index in range(grid_size)
        }
    )
    grouped = group_subjects(rows)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        score = mean(balanced_accuracy(subject, threshold) for subject in grouped.values())
        if score > best_score:
            best_threshold = threshold
            best_score = score
    return best_threshold, best_score


def parse_run_path(path: Path, root: Path) -> tuple[str, str, int]:
    relative = path.resolve().relative_to(root.resolve())
    if len(relative.parts) < 5:
        raise ValueError(f"unexpected prediction path: {path}")
    direction = relative.parts[0]
    if "-to-" not in direction:
        raise ValueError(f"direction is missing '-to-': {direction}")
    source, target = direction.split("-to-", 1)
    seed_part = relative.parts[3]
    if not seed_part.startswith("seed-"):
        raise ValueError(f"unexpected seed directory: {seed_part}")
    seed_value = seed_part[5:] if seed_part.startswith("seed-") else seed_part
    return source, target, int(seed_value)


def stable_seed(base: int, *parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return (base + int.from_bytes(digest[:8], "big")) % (2**32)


def bootstrap_interval(values: Sequence[float], samples: int, seed: int) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires values")
    rng = random.Random(seed)
    estimates = [mean(rng.choice(values) for _ in values) for _ in range(samples)]
    estimates.sort()
    lower = estimates[max(0, int(0.025 * samples) - 1)]
    upper = estimates[min(samples - 1, int(0.975 * samples))]
    return lower, upper


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
