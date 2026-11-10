from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from .metrics import TrialScore, evaluate_endpoint, target_information_response_surface
from aog.integrity import file_hash, source_tree_hash, stable_json_hash


EXPECTED_UNIT_COUNT = 90
REQUIRED_PREDICTION_COLUMNS = {
    "subject_key",
    "dataset_id",
    "session_id",
    "trial_order",
    "y_true",
    "logit0",
    "logit1",
    "raw_logit",
    "p0",
    "p1",
}


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_stieger_protocol(path: str | Path) -> dict[str, Any]:
    protocol = _read_json(path)
    if protocol.get("schema_version") != 2:
        raise ValueError("Stieger confirmation requires the prospectively amended v2 schema")
    if protocol.get("study_id") != "v5-journal-acdd-stieger-confirmation":
        raise ValueError("unexpected Stieger confirmation study ID")
    if protocol.get("status") != "prospectively_amended_before_target_parsing":
        raise PermissionError("Stieger protocol is not in its pre-analysis amended state")
    amendment = protocol.get("preanalysis_amendment", {})
    if amendment.get("triggered_by_target_results") is not False:
        raise PermissionError("the pre-analysis amendment cannot be informed by target results")
    if any(
        amendment.get(key) not in (False, 0)
        for key in (
            "target_signals_inspected",
            "target_labels_inspected",
            "completed_target_exports",
            "completed_target_prediction_units",
        )
    ):
        raise PermissionError("the ACDD v2 amendment must precede target parsing and prediction")
    if protocol["dataset"]["name"] != "Stieger2021":
        raise ValueError("the independent confirmation dataset must be Stieger2021")
    if protocol["dataset"].get("previously_inspected") is not False:
        raise PermissionError("independent confirmation requires a previously uninspected dataset")
    if protocol.get("allows_training") is not False:
        raise PermissionError("Stieger confirmation must be inference-only")
    endpoints = protocol["endpoints"]
    required = {
        "prefix10_full",
        "prefix10_suffix",
        "prequential",
        "session1_to_later",
        "full_prior_upper_bound",
    }
    if set(endpoints) != required:
        raise ValueError("Stieger endpoints changed from the frozen contract")
    expected = (
        len(protocol["source_protocols"])
        * len(protocol["backbones"])
        * len(protocol["candidates"])
        * len(protocol["seeds"])
    )
    if expected != EXPECTED_UNIT_COUNT:
        raise ValueError(f"Stieger plan must contain {EXPECTED_UNIT_COUNT} units, found {expected}")
    if protocol["prediction_contract"]["score_for_auc"] != "raw_logit=logit1-logit0":
        raise ValueError("Stieger AUC must use the raw logit difference")
    method = protocol.get("method_contract", {})
    if method.get("comparison_identity") != "Delta_BA=(D_m-D_0)+(E_0-E_m)":
        raise ValueError("generalized ACDD comparison identity changed")
    if "rank-preserving" not in method.get("normalized_recovery_scope", ""):
        raise ValueError("normalized recovery must be restricted to rank-preserving rules")
    statistics = protocol.get("statistics", {})
    if float(statistics.get("practical_harm_delta", -1)) != 0.02:
        raise ValueError("the practical harm delta must remain frozen at two percentage points")
    if statistics.get("tail_and_harm_bootstrap_required") is not True:
        raise ValueError("participant-bootstrap tail and harm intervals are required")
    guard = protocol.get("deployment_claim_guard", {})
    required_guard = {"median_gain_gt", "p10_gain_ge", "practical_harm_rate_le"}
    if not required_guard.issubset(guard):
        raise ValueError("deployment claim guard is incomplete")
    return protocol


def build_stieger_plan(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path)
    protocol = validate_stieger_protocol(protocol_path)
    units = []
    for source in protocol["source_protocols"]:
        for backbone in protocol["backbones"]:
            for candidate in protocol["candidates"]:
                for seed in protocol["seeds"]:
                    units.append(
                        {
                            "unit_id": f"STIEGER-INF-{len(units) + 1:03d}",
                            "source_protocol": source,
                            "backbone": backbone,
                            "candidate": candidate,
                            "seed": int(seed),
                            "training_allowed": False,
                            "requires_checkpoint_sha256": True,
                            "output_relative_path": (
                                Path(source["id"])
                                / backbone
                                / candidate["id"]
                                / f"seed-{seed}"
                            ).as_posix(),
                        }
                    )
    if len(units) != EXPECTED_UNIT_COUNT:
        raise AssertionError("internal Stieger unit-count mismatch")
    return {
        "schema_version": 2,
        "study_id": protocol["study_id"],
        "protocol_sha256": file_hash(protocol_path),
        "unit_count": len(units),
        "inference_only": True,
        "units": units,
    }


def write_stieger_plan(protocol_path: str | Path, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_stieger_plan(protocol_path), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output


def validate_checkpoint_manifest(
    manifest_path: str | Path, plan_path: str | Path, check_files: bool = True
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    plan = _read_json(plan_path)
    if manifest.get("status") != "checkpoint_manifest_frozen":
        raise PermissionError("checkpoint manifest is not frozen")
    if manifest.get("plan_sha256") != file_hash(plan_path):
        raise ValueError("checkpoint manifest does not match the Stieger plan")
    entries = manifest.get("checkpoints", [])
    by_unit = {entry["unit_id"]: entry for entry in entries}
    expected_ids = {unit["unit_id"] for unit in plan["units"]}
    if set(by_unit) != expected_ids or len(entries) != EXPECTED_UNIT_COUNT:
        raise ValueError("checkpoint manifest must cover exactly the 90 frozen units")
    manifest_root = Path(manifest_path).resolve().parent
    for unit in plan["units"]:
        entry = by_unit[unit["unit_id"]]
        for key in ("checkpoint_path", "checkpoint_sha256", "source_validation_predictions_path", "source_validation_predictions_sha256"):
            if not entry.get(key):
                raise ValueError(f"checkpoint entry lacks {key}: {unit['unit_id']}")
        if check_files:
            for path_key, hash_key in (
                ("checkpoint_path", "checkpoint_sha256"),
                ("source_validation_predictions_path", "source_validation_predictions_sha256"),
            ):
                path = Path(entry[path_key])
                path = path if path.is_absolute() else (manifest_root / path).resolve()
                if not path.is_file() or file_hash(path) != entry[hash_key]:
                    raise ValueError(f"frozen input missing or changed: {unit['unit_id']} {path_key}")
    return manifest


def freeze_stieger_confirmation(
    protocol_path: str | Path,
    plan_path: str | Path,
    checkpoint_manifest_path: str | Path,
    preanalysis_amendment_lock_path: str | Path,
    technical_amendment_lock_path: str | Path,
    compatibility_amendment_lock_path: str | Path,
    project_root: str | Path,
    output_path: str | Path,
) -> Path:
    protocol = validate_stieger_protocol(protocol_path)
    plan = _read_json(plan_path)
    if plan.get("protocol_sha256") != file_hash(protocol_path):
        raise ValueError("Stieger plan changed after it was materialized")
    validate_checkpoint_manifest(checkpoint_manifest_path, plan_path, check_files=True)
    amendment_lock = _read_json(preanalysis_amendment_lock_path)
    if amendment_lock.get("status") != "acdd_v2_preanalysis_amendment_frozen":
        raise PermissionError("ACDD v2 pre-analysis amendment is not frozen")
    amendment_payload = dict(amendment_lock)
    amendment_recorded = amendment_payload.pop("lock_payload_sha256", None)
    if amendment_recorded != stable_json_hash(amendment_payload):
        raise ValueError("pre-analysis amendment lock payload hash mismatch")
    if amendment_lock.get("amended_protocol_sha256") != file_hash(protocol_path):
        raise ValueError("pre-analysis amendment lock does not match the active protocol")
    root = Path(project_root).resolve()
    technical_lock = _read_json(technical_amendment_lock_path)
    technical_payload = dict(technical_lock)
    technical_recorded = technical_payload.pop("lock_payload_sha256", None)
    if technical_lock.get("status") != "stieger_local_cache_adapter_amendment_frozen":
        raise PermissionError("Stieger local-cache technical amendment is not frozen")
    if technical_recorded != stable_json_hash(technical_payload):
        raise ValueError("local-cache technical amendment payload hash mismatch")
    if technical_lock.get("triggered_by_target_results") is not False:
        raise PermissionError("local-cache adapter cannot be informed by target results")
    if technical_lock.get("scientific_fields_changed") != []:
        raise PermissionError("local-cache adapter changed a scientific field")
    if technical_lock.get("target_mat_files_opened_before_amendment") is not False:
        raise PermissionError("technical amendment occurred after target MAT parsing")
    compatibility_lock = _read_json(compatibility_amendment_lock_path)
    compatibility_payload = dict(compatibility_lock)
    compatibility_recorded = compatibility_payload.pop("lock_payload_sha256", None)
    if compatibility_lock.get("status") != "stieger_streaming_adapter_amendment_frozen":
        raise PermissionError("Stieger streaming compatibility amendment is not frozen")
    if compatibility_recorded != stable_json_hash(compatibility_payload):
        raise ValueError("streaming compatibility amendment payload hash mismatch")
    if compatibility_lock.get("triggered_by_target_results") is not False:
        raise PermissionError("streaming adapter cannot be informed by target results")
    if compatibility_lock.get("scientific_fields_changed") != []:
        raise PermissionError("streaming adapter changed a scientific field")
    if compatibility_lock.get("target_mat_files_opened_before_amendment") is not False:
        raise PermissionError("streaming amendment occurred after target MAT parsing")
    if compatibility_lock.get("adapter_source_sha256") != file_hash(
        root / "src" / "acdd" / "stieger_export.py"
    ):
        raise ValueError("streaming amendment does not match the active adapter source")
    execution_artifacts = {
        "configs/cloud_4090d_v5_stieger.json": root / "configs" / "cloud_4090d_v5_stieger.json",
        "configs/cloud_4090d_v5_stieger_autodl.json": root
        / "configs"
        / "cloud_4090d_v5_stieger_autodl.json",
        "analysis/analyze_v5_stieger_confirmation.py": root / "analysis" / "analyze_v5_stieger_confirmation.py",
        "analysis/analyze_v5_source_calibration_matrix.py": root / "analysis" / "analyze_v5_source_calibration_matrix.py",
        "analysis/reexport_v5_source_validation_logits.py": root / "analysis" / "reexport_v5_source_validation_logits.py",
        "analysis/build_v5_stieger_checkpoint_manifest.py": root / "analysis" / "build_v5_stieger_checkpoint_manifest.py",
        "analysis/freeze_v5_stieger_confirmation.py": root
        / "analysis"
        / "freeze_v5_stieger_confirmation.py",
        "analysis/freeze_v5_preanalysis_amendment.py": root
        / "analysis"
        / "freeze_v5_preanalysis_amendment.py",
        "analysis/run_acdd_v2_synthetic_controls.py": root
        / "analysis"
        / "run_acdd_v2_synthetic_controls.py",
        "analysis/preflight_v5_stieger_participant_major.py": root
        / "analysis"
        / "preflight_v5_stieger_participant_major.py",
        "analysis/preflight_v5_stieger_raw_cache.py": root
        / "analysis"
        / "preflight_v5_stieger_raw_cache.py",
        "analysis/freeze_v5_stieger_local_cache_amendment.py": root
        / "analysis"
        / "freeze_v5_stieger_local_cache_amendment.py",
        "analysis/preflight_v5_stieger_streaming_adapter.py": root
        / "analysis"
        / "preflight_v5_stieger_streaming_adapter.py",
        "analysis/freeze_v5_stieger_streaming_adapter_amendment.py": root
        / "analysis"
        / "freeze_v5_stieger_streaming_adapter_amendment.py",
        "analysis/finalize_v5_stieger_raw_prefetch.py": root
        / "analysis"
        / "finalize_v5_stieger_raw_prefetch.py",
        "outputs/v5-journal-stieger/protocol/raw-prefetch-completion-v4.json": root
        / "outputs"
        / "v5-journal-stieger"
        / "protocol"
        / "raw-prefetch-completion-v4.json",
        "outputs/v5-journal-stieger/protocol/participant-major-checkpoint-preflight.json": root
        / "outputs"
        / "v5-journal-stieger"
        / "protocol"
        / "participant-major-checkpoint-preflight.json",
        "outputs/v5-journal-stieger/protocol/raw-cache-preflight-v2.json": root
        / "outputs"
        / "v5-journal-stieger"
        / "protocol"
        / "raw-cache-preflight-v2.json",
        "outputs/v5-journal-stieger/protocol/prelocal-adapter-subject-001-export-failure.log": root
        / "outputs"
        / "v5-journal-stieger"
        / "protocol"
        / "prelocal-adapter-subject-001-export-failure.log",
        "outputs/v5-journal-stieger/protocol/prelocal-adapter-export-exclusions.jsonl": root
        / "outputs"
        / "v5-journal-stieger"
        / "protocol"
        / "prelocal-adapter-export-exclusions.jsonl",
        "outputs/v5-journal-stieger/protocol/prestreaming-v3-class-name-failure.log": root
        / "outputs"
        / "v5-journal-stieger"
        / "protocol"
        / "prestreaming-v3-class-name-failure.log",
        "outputs/v5-journal-stieger/protocol/streaming-adapter-preflight-v4.json": root
        / "outputs"
        / "v5-journal-stieger"
        / "protocol"
        / "streaming-adapter-preflight-v4.json",
        "locks/v5-journal-stieger-v2.lock.json": root
        / "locks"
        / "v5-journal-stieger-v2.lock.json",
        "locks/v5-journal-stieger-v3.lock.json": root
        / "locks"
        / "v5-journal-stieger-v3.lock.json",
        "scripts/cloud/run_v5_stieger_inference.py": root / "scripts" / "cloud" / "run_v5_stieger_inference.py",
        "scripts/cloud/run_v5_stieger_participant_major.py": root
        / "scripts"
        / "cloud"
        / "run_v5_stieger_participant_major.py",
        "scripts/cloud/run_v5_stieger_confirmation.sh": root / "scripts" / "cloud" / "run_v5_stieger_confirmation.sh",
    }
    lock = {
        "schema_version": 1,
        "status": "stieger_confirmation_frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "confirmatory_dataset": "Stieger2021",
        "raw_archive_transfer_before_amendment": True,
        "target_parsing_before_amendment": False,
        "target_predictions_before_amendment": 0,
        "allows_training": False,
        "protocol_sha256": file_hash(protocol_path),
        "plan_sha256": file_hash(plan_path),
        "checkpoint_manifest_sha256": file_hash(checkpoint_manifest_path),
        "preanalysis_amendment_lock_sha256": file_hash(preanalysis_amendment_lock_path),
        "technical_amendment_lock_sha256": file_hash(technical_amendment_lock_path),
        "compatibility_amendment_lock_sha256": file_hash(
            compatibility_amendment_lock_path
        ),
        "journal_source_tree_sha256": source_tree_hash(root / "src" / "aog"),
        "acdd_source_tree_sha256": source_tree_hash(root / "src" / "acdd"),
        "execution_artifact_sha256": {
            name: file_hash(path) for name, path in execution_artifacts.items()
        },
        "decision_branches": protocol["decision_branches"],
        "preanalysis_amendment": protocol["preanalysis_amendment"],
        "method_contract": protocol["method_contract"],
    }
    lock["lock_payload_sha256"] = stable_json_hash(lock)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def validate_stieger_lock(
    lock_path: str | Path,
    protocol_path: str | Path,
    plan_path: str | Path,
    checkpoint_manifest_path: str | Path,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    lock = _read_json(lock_path)
    if lock.get("status") != "stieger_confirmation_frozen":
        raise PermissionError("Stieger confirmation lock is not frozen")
    payload = dict(lock)
    recorded = payload.pop("lock_payload_sha256", None)
    if recorded != stable_json_hash(payload):
        raise ValueError("Stieger lock payload hash mismatch")
    checks = {
        "protocol_sha256": protocol_path,
        "plan_sha256": plan_path,
        "checkpoint_manifest_sha256": checkpoint_manifest_path,
    }
    for key, path in checks.items():
        if file_hash(path) != lock[key]:
            raise ValueError(f"frozen Stieger input changed: {key}")
    validate_checkpoint_manifest(checkpoint_manifest_path, plan_path, check_files=True)
    if project_root is not None:
        root = Path(project_root).resolve()
        current = {
            "journal_source_tree_sha256": source_tree_hash(root / "src" / "aog"),
            "acdd_source_tree_sha256": source_tree_hash(root / "src" / "acdd"),
        }
        changed = [key for key, value in current.items() if lock.get(key) != value]
        technical_lock_path = (
            root / "locks" / "v5-journal-stieger-local-cache-adapter-v3.lock.json"
        )
        if lock.get("technical_amendment_lock_sha256"):
            if (
                not technical_lock_path.is_file()
                or file_hash(technical_lock_path)
                != lock["technical_amendment_lock_sha256"]
            ):
                changed.append("technical_amendment_lock_sha256")
        compatibility_lock_path = (
            root / "locks" / "v5-journal-stieger-streaming-adapter-v4.lock.json"
        )
        if lock.get("compatibility_amendment_lock_sha256"):
            if (
                not compatibility_lock_path.is_file()
                or file_hash(compatibility_lock_path)
                != lock["compatibility_amendment_lock_sha256"]
            ):
                changed.append("compatibility_amendment_lock_sha256")
        execution_changed = [
            name
            for name, expected in lock.get("execution_artifact_sha256", {}).items()
            if file_hash(root / name) != expected
        ]
        if execution_changed:
            changed.append(f"execution_artifacts:{execution_changed}")
        if changed:
            raise ValueError(f"Stieger implementation changed after freeze: {changed}")
    return lock


def read_stieger_predictions(path: str | Path, protocol_id: str) -> list[TrialScore]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not REQUIRED_PREDICTION_COLUMNS.issubset(reader.fieldnames):
            missing = sorted(REQUIRED_PREDICTION_COLUMNS - set(reader.fieldnames or ()))
            raise ValueError(f"Stieger predictions lack mandatory raw-logit columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("Stieger prediction table is empty")
    result = []
    for row in rows:
        logit0, logit1, raw_logit = float(row["logit0"]), float(row["logit1"]), float(row["raw_logit"])
        if abs((logit1 - logit0) - raw_logit) > 1e-6:
            raise ValueError("raw_logit is not logit1-logit0")
        if row["dataset_id"] != "Stieger2021":
            raise ValueError("non-Stieger row in independent confirmation predictions")
        result.append(
            TrialScore(
                participant_id=row["subject_key"],
                dataset_id=row["dataset_id"],
                protocol_id=protocol_id,
                y_true=int(row["y_true"]),
                raw_logit=raw_logit,
                order=int(row["trial_order"]),
                session_id=row["session_id"],
            )
        )
    return result


def analyze_participant(
    trials: list[TrialScore], response_surface: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    endpoints = {
        "prefix10_full": "prefix_full",
        "prefix10_suffix": "prefix_suffix",
        "prequential": "prequential",
        "session1_to_later": "session1_to_later",
        "full_prior_upper_bound": "full_prior",
        "full_prior_suffix_diagnostic": "full_prior_suffix",
    }
    endpoint_rows = []
    exclusions = []
    for name, endpoint in endpoints.items():
        try:
            result = evaluate_endpoint(trials, endpoint, prefix_length=10, positive_prior=0.5)
            endpoint_rows.append({"participant_id": trials[0].participant_id, **asdict(result), "endpoint": name})
        except ValueError as exc:
            exclusions.append(
                {
                    "participant_id": trials[0].participant_id,
                    "analysis": name,
                    "reason": str(exc),
                }
            )
    surface, surface_exclusions = target_information_response_surface(
        trials,
        response_surface["prefix_lengths"],
        response_surface["positive_priors"],
        response_surface["orders"],
        response_surface["evaluation_scopes"],
    )
    exclusions.extend({"analysis": "response_surface", **row} for row in surface_exclusions)
    return endpoint_rows, surface, exclusions


def check_inference_completeness(plan_path: str | Path, run_root: str | Path) -> dict[str, Any]:
    plan = _read_json(plan_path)
    root = Path(run_root)
    complete, missing = 0, []
    for unit in plan["units"]:
        directory = root / unit["output_relative_path"]
        required = [directory / "predictions.csv", directory / "summary.json"]
        absent = [path.name for path in required if not path.is_file()]
        if absent:
            missing.append({"unit_id": unit["unit_id"], "missing": absent})
            continue
        read_stieger_predictions(required[0], unit["source_protocol"]["id"])
        complete += 1
    return {
        "expected": EXPECTED_UNIT_COUNT,
        "complete": complete,
        "missing": missing,
        "is_complete": complete == EXPECTED_UNIT_COUNT and not missing,
    }
