from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from acdd.metrics import hierarchical_bootstrap


POLICIES = {
    "prior_quantile_full": "full_unlabeled_score",
    "prior_quantile_prefix10": "prefix_unlabeled_score",
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty participant metric table: {path}")
    return rows


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires values")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _participant_contrasts(
    path: Path,
    cohort: str,
    candidate: str,
) -> list[dict[str, Any]]:
    selected = [row for row in _read(path) if row["candidate"] == candidate]
    grouped: dict[tuple[str, ...], dict[str, float]] = defaultdict(dict)
    for row in selected:
        key = (
            row["backbone"],
            row["dataset_id"],
            row["protocol_id"],
            row["subject_key"],
            row["seed"],
        )
        policy = row["decision_policy"]
        if policy in {"fixed_threshold", *POLICIES}:
            if policy in grouped[key]:
                raise ValueError(f"duplicate participant-policy row: {key} {policy}")
            grouped[key][policy] = float(row["balanced_accuracy"])

    contrasts: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        backbone, dataset, protocol, participant, seed = key
        required = {"fixed_threshold", *POLICIES}
        if set(values) != required:
            raise ValueError(f"incomplete policy set: {key}; got {sorted(values)}")
        baseline = values["fixed_threshold"]
        for policy, access_group in POLICIES.items():
            access = values[policy]
            contrasts.append(
                {
                    "cohort": cohort,
                    "backbone": backbone,
                    "candidate": candidate,
                    "policy": policy,
                    "access_group": access_group,
                    "dataset_id": dataset,
                    "protocol_id": protocol,
                    "participant_id": participant,
                    "seed": int(seed),
                    "balanced_accuracy_default": baseline,
                    "balanced_accuracy_access": access,
                    "absolute_access_gain": access - baseline,
                    "harmed": access < baseline,
                }
            )
    return contrasts


def _subject_level(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate seed and protocol before subject-tail and harm calculations."""

    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["cohort"],
            row["backbone"],
            row["policy"],
            row["dataset_id"],
            row["participant_id"],
        )
        grouped[key].append(row)
    output = []
    for key, members in sorted(grouped.items()):
        cohort, backbone, policy, dataset, participant = key
        baseline = mean(float(row["balanced_accuracy_default"]) for row in members)
        access = mean(float(row["balanced_accuracy_access"]) for row in members)
        output.append(
            {
                "cohort": cohort,
                "backbone": backbone,
                "policy": policy,
                "dataset_id": dataset,
                "participant_id": participant,
                "n_protocol_seed_observations": len(members),
                "balanced_accuracy_default": baseline,
                "balanced_accuracy_access": access,
                "absolute_access_gain": access - baseline,
                "harmed": access < baseline,
            }
        )
    return output


def _summary(rows: list[dict[str, Any]], samples: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    keys = sorted({(row["cohort"], row["backbone"], row["policy"]) for row in rows})
    for cohort, backbone, policy in keys:
        selected = [
            row
            for row in rows
            if (row["cohort"], row["backbone"], row["policy"])
            == (cohort, backbone, policy)
        ]
        subjects = _subject_level(selected)
        gains = [float(row["absolute_access_gain"]) for row in subjects]
        access = [float(row["balanced_accuracy_access"]) for row in subjects]
        name = f"{cohort}/{backbone}/{policy}"
        output[name] = {
            "cohort": cohort,
            "backbone": backbone,
            "policy": policy,
            "n_dataset_protocol_participants": len(
                {
                    (row["dataset_id"], row["protocol_id"], row["participant_id"])
                    for row in selected
                }
            ),
            "n_unique_participants": len(subjects),
            "estimands": hierarchical_bootstrap(
                selected,
                value_key="absolute_access_gain",
                samples=samples,
                seed=20260813,
            ),
            "subject_tail": {
                "gain_p05": _percentile(gains, 0.05),
                "gain_p10": _percentile(gains, 0.10),
                "access_balanced_accuracy_p05": _percentile(access, 0.05),
                "access_balanced_accuracy_p10": _percentile(access, 0.10),
                "harm_rate": mean(float(row["harmed"]) for row in subjects),
            },
        }
    return output


def run(discovery: Path, dreyer: Path, output: Path, samples: int) -> Path:
    contrasts = _participant_contrasts(discovery, "discovery", "raw-bn")
    contrasts += _participant_contrasts(dreyer, "dreyer_confirmation", "raw-bn")
    output.mkdir(parents=True, exist_ok=True)
    contrast_path = output / "historical_participant_contrasts.csv"
    subject_path = output / "historical_subject_aggregates.csv"
    report_path = output / "historical_estimands.json"
    _write(contrast_path, contrasts)
    _write(subject_path, _subject_level(contrasts))
    report = {
        "schema_version": 1,
        "status": "complete_from_frozen_v4_participant_tables",
        "aggregation": (
            "Seed is aggregated within dataset/protocol/participant by the hierarchical "
            "bootstrap. Subject tail and harm additionally aggregate protocols within "
            "participant before P5/P10 and P(BA_A < BA_0.5) are computed."
        ),
        "confirmation_selection_guard": (
            "Dreyer uses the prespecified raw-bn fixed-versus-prior decision-policy "
            "comparison from the frozen confirmation analysis; no candidate or endpoint "
            "is selected on Dreyer. The separate frozen-access-winner rows are an absolute "
            "preprocessing endpoint and do not contain the paired decision policies."
        ),
        "bootstrap_samples": samples,
        "inputs": {str(discovery): _sha256(discovery), str(dreyer): _sha256(dreyer)},
        "results": _summary(contrasts, samples),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = {
        path.name: _sha256(path) for path in (contrast_path, subject_path, report_path)
    }
    manifest_path = output / "historical_estimands_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--dreyer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    print(run(args.discovery, args.dreyer, args.output, args.bootstrap_samples))


if __name__ == "__main__":
    main()
