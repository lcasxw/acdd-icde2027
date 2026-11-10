from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

from aog.journal_evaluation import evaluate_access_policy, read_prediction_table


POLICIES = {
    "raw-bn": ("fixed_threshold", "prior_quantile_full", "prior_quantile_prefix10"),
    "raw-gn": ("fixed_threshold",),
    "frozen-access-winner": ("fixed_threshold",),
}


def _run_relative_path(unit: dict[str, Any]) -> Path:
    protocol = unit["protocol"]
    if protocol["kind"] == "lodo":
        base = Path("lodo") / protocol["target"]
    else:
        base = Path("single_source") / f"{protocol['source']}-to-{protocol['target']}"
    return base / unit["backbone"] / unit["candidate"] / f"seed-{unit['seed']}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty confirmation table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_subject_metrics(
    plan_path: Path,
    run_root: Path,
    protocol_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    units = plan["dreyer_confirmation"]
    missing: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for unit in units:
        run_dir = run_root / _run_relative_path(unit)
        required = [run_dir / "best.pt", run_dir / "summary.json", run_dir / "predictions.csv"]
        absent = [path.name for path in required if not path.is_file()]
        if absent:
            missing.append({"unit_id": unit["unit_id"], "missing": absent})
            continue
        predictions = read_prediction_table(required[-1])
        for policy in POLICIES[str(unit["candidate"])]:
            evaluation = evaluate_access_policy(
                predictions,
                policy,
                positive_prior=float(protocol["score_transport"]["positive_prior"]),
                prefix_trials=int(protocol["score_transport"]["prefix_trials"]),
            )
            for subject in evaluation["subject_metrics"]:
                rows.append(
                    {
                        "unit_id": unit["unit_id"],
                        "backbone": unit["backbone"],
                        "protocol_id": unit["protocol"]["id"],
                        "protocol_kind": unit["protocol"]["kind"],
                        "source": unit["protocol"]["source"] or "LODO",
                        "target": unit["protocol"]["target"],
                        "seed": int(unit["seed"]),
                        "candidate": unit["candidate"],
                        "decision_policy": policy,
                        "decision_access_group": evaluation["access_group"],
                        **subject,
                    }
                )
    completeness = {
        "schema_version": 1,
        "expected_units": len(units),
        "complete_units": len(units) - len(missing),
        "is_complete": not missing and len(units) == 90,
        "missing": missing,
        "dreyer2023_accessed": True,
    }
    if not completeness["is_complete"]:
        raise RuntimeError(
            f"Dreyer confirmation must be complete before analysis: "
            f"{completeness['complete_units']}/{completeness['expected_units']}"
        )
    return rows, completeness


def _key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row["backbone"]),
        str(row["protocol_id"]),
        int(row["seed"]),
        str(row["subject_key"]),
    )


def _select(
    rows: Iterable[dict[str, Any]], candidate: str, policy: str
) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    return {
        _key(row): row
        for row in rows
        if row["candidate"] == candidate and row["decision_policy"] == policy
    }


def _paired(
    rows: list[dict[str, Any]], comparison: str
) -> list[dict[str, Any]]:
    if comparison == "full_prior_vs_fixed":
        baseline = _select(rows, "raw-bn", "fixed_threshold")
        proposed = _select(rows, "raw-bn", "prior_quantile_full")
    elif comparison == "prefix10_vs_fixed":
        baseline = _select(rows, "raw-bn", "fixed_threshold")
        proposed = _select(rows, "raw-bn", "prior_quantile_prefix10")
    elif comparison == "raw_gn_vs_raw_bn":
        baseline = _select(rows, "raw-bn", "fixed_threshold")
        proposed = _select(rows, "raw-gn", "fixed_threshold")
    else:
        raise KeyError(comparison)
    if set(baseline) != set(proposed):
        raise ValueError(f"unpaired Dreyer comparison: {comparison}")
    return [
        {
            "comparison": comparison,
            "backbone": key[0],
            "protocol_id": key[1],
            "seed": key[2],
            "subject_key": key[3],
            "dataset_id": baseline[key]["dataset_id"],
            "baseline_balanced_accuracy": float(baseline[key]["balanced_accuracy"]),
            "proposed_balanced_accuracy": float(proposed[key]["balanced_accuracy"]),
            "balanced_accuracy_gain": float(proposed[key]["balanced_accuracy"])
            - float(baseline[key]["balanced_accuracy"]),
            "baseline_roc_auc": float(baseline[key]["roc_auc"]),
        }
        for key in sorted(baseline)
    ]


def _cluster(effects: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in effects:
        grouped[(str(row["backbone"]), str(row["subject_key"]))].append(row)
    return [
        {
            "backbone": key[0],
            "subject_key": key[1],
            "dataset_id": values[0]["dataset_id"],
            "n_repeated_observations": len(values),
            "baseline_balanced_accuracy": mean(
                float(row["baseline_balanced_accuracy"]) for row in values
            ),
            "proposed_balanced_accuracy": mean(
                float(row["proposed_balanced_accuracy"]) for row in values
            ),
            "balanced_accuracy_gain": mean(
                float(row["balanced_accuracy_gain"]) for row in values
            ),
            "baseline_roc_auc": mean(float(row["baseline_roc_auc"]) for row in values),
        }
        for key, values in sorted(grouped.items())
    ]


def _bootstrap(values: list[float], samples: int, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("confirmation bootstrap requires finite non-empty values")
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        estimates.append(array[indices].mean(axis=1))
    distribution = np.concatenate(estimates)
    return {
        "mean_gain": float(array.mean()),
        "ci_lower": float(np.quantile(distribution, 0.025)),
        "ci_upper": float(np.quantile(distribution, 0.975)),
        "n_subjects": int(len(array)),
    }


def _endpoint(clustered: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    baseline = np.asarray(
        [float(row["baseline_balanced_accuracy"]) for row in clustered], dtype=np.float64
    )
    proposed = np.asarray(
        [float(row["proposed_balanced_accuracy"]) for row in clustered], dtype=np.float64
    )
    return {
        **_bootstrap(
            [float(row["balanced_accuracy_gain"]) for row in clustered], samples, seed
        ),
        "baseline_mean": float(baseline.mean()),
        "proposed_mean": float(proposed.mean()),
        "baseline_worst_subject": float(baseline.min()),
        "proposed_worst_subject": float(proposed.min()),
        "baseline_p10_subject": float(np.quantile(baseline, 0.10)),
        "proposed_p10_subject": float(np.quantile(proposed, 0.10)),
        "mean_subject_auc": float(
            np.mean([float(row["baseline_roc_auc"]) for row in clustered])
        ),
    }


def analyze(
    subject_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    samples: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    comparisons = ("full_prior_vs_fixed", "prefix10_vs_fixed", "raw_gn_vs_raw_bn")
    effects = {name: _paired(subject_rows, name) for name in comparisons}
    gates = protocol["confirmation_gates"]
    backbones: dict[str, Any] = {}
    clustered_output: list[dict[str, Any]] = []
    for backbone_index, backbone in enumerate(sorted({row["backbone"] for row in subject_rows})):
        values: dict[str, Any] = {}
        for comparison_index, comparison in enumerate(comparisons):
            clustered = _cluster(
                row for row in effects[comparison] if row["backbone"] == backbone
            )
            clustered_output.extend({"comparison": comparison, **row} for row in clustered)
            values[comparison] = _endpoint(
                clustered, samples, seed + 1000 * backbone_index + 100 * comparison_index
            )
        full = values["full_prior_vs_fixed"]
        prefix = values["prefix10_vs_fixed"]
        group_norm = values["raw_gn_vs_raw_bn"]
        recovery = (
            float(prefix["mean_gain"]) / float(full["mean_gain"])
            if float(full["mean_gain"]) > 0
            else float("nan")
        )
        endpoints = {
            "mean_subject_auc": float(full["mean_subject_auc"])
            >= float(gates["mean_subject_auc_minimum"]),
            "full_prior_gain": float(full["mean_gain"])
            >= float(gates["full_prior_ba_gain_minimum"])
            and float(full["ci_lower"])
            > float(gates["full_prior_gain_ci_lower_strictly_above"]),
            "group_norm_gain": float(group_norm["mean_gain"])
            >= float(gates["group_norm_ba_gain_minimum"])
            and float(group_norm["ci_lower"])
            > float(gates["group_norm_gain_ci_lower_strictly_above"]),
            "prefix10_recovery_secondary": bool(
                np.isfinite(recovery)
                and recovery >= float(gates["prefix10_full_gain_recovery_minimum"])
            ),
        }
        values["gates"] = {
            **endpoints,
            "prefix10_recovery": recovery,
            "primary_confirmation_passed": all(
                endpoints[key]
                for key in ("mean_subject_auc", "full_prior_gain", "group_norm_gain")
            ),
        }
        backbones[str(backbone)] = values
    report = {
        "schema_version": 1,
        "status": "confirmation_complete",
        "confirmatory_dataset": "Dreyer2023",
        "resampling_unit": (
            "unique Dreyer target subject within backbone; four source directions, "
            "LODO and three seeds averaged before resampling"
        ),
        "bootstrap_samples": samples,
        "backbones": backbones,
        "all_backbones_primary_confirmation_passed": all(
            value["gates"]["primary_confirmation_passed"] for value in backbones.values()
        ),
        "failure_policy": "report_without_reselection",
        "dreyer2023_accessed": True,
    }
    return report, clustered_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    rows, completeness = collect_subject_metrics(args.plan, args.run_root, args.protocol)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    report, clustered = analyze(rows, protocol, args.bootstrap_samples, 20260812)
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "dreyer_subject_metrics.csv", rows)
    _write_csv(args.output / "dreyer_clustered_effects.csv", clustered)
    (args.output / "dreyer_completeness.json").write_text(
        json.dumps(completeness, indent=2), encoding="utf-8"
    )
    report_path = args.output / "dreyer_confirmation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
