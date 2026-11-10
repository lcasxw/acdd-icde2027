from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aog.data.store import load_manifest  # noqa: E402
from aog.integrity import file_hash  # noqa: E402
from aog.normalization import fit_input_preprocessor, transform_input_signal  # noqa: E402
from acdd.stieger_protocol import (  # noqa: E402
    REQUIRED_PREDICTION_COLUMNS,
    validate_checkpoint_manifest,
    validate_stieger_lock,
)


PREDICTION_FIELDS = [
    "subject_key",
    "dataset_id",
    "session_id",
    "run_id",
    "trial_index",
    "trial_order",
    "y_true",
    "logit0",
    "logit1",
    "raw_logit",
    "p0",
    "p1",
    "y_pred",
]


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - cloud preflight checks this
        raise RuntimeError("PyTorch is required for frozen Stieger inference") from exc
    return torch


def _safe(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_checkpoint_payload(
    checkpoint: dict[str, Any], unit: dict[str, Any]
) -> None:
    candidate = unit["candidate"]
    required = {
        "model",
        "model_name",
        "n_channels",
        "n_times",
        "normalization",
        "preprocessing",
        "seed",
        "stage",
    }
    if not required.issubset(checkpoint):
        raise ValueError(
            f"checkpoint lacks frozen inference metadata: {sorted(required - set(checkpoint))}"
        )
    expected = {
        "model_name": unit["backbone"],
        "normalization": candidate["normalization"],
        "preprocessing": candidate["preprocessing"],
        "seed": unit["seed"],
        "stage": "final",
        "target_dataset": "Dreyer2023",
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    source = unit["source_protocol"]["source"]
    if checkpoint.get("source_dataset") != source:
        mismatches["source_dataset"] = (checkpoint.get("source_dataset"), source)
    if mismatches:
        raise ValueError(f"checkpoint metadata mismatch: {mismatches}")


def _load_checkpoint(
    path: Path, unit: dict[str, Any], device: Any
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    from aog.models import build_model

    checkpoint = torch.load(path, map_location="cpu")
    _validate_checkpoint_payload(checkpoint, unit)
    model = build_model(
        checkpoint["model_name"],
        int(checkpoint["n_channels"]),
        int(checkpoint["n_times"]),
        normalization=checkpoint["normalization"],
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def _crop(entry: Any, window: tuple[float, float]) -> np.ndarray:
    start = round((window[0] - entry.epoch_tmin) * entry.sampling_rate)
    stop = round((window[1] - entry.epoch_tmin) * entry.sampling_rate)
    signals = np.load(Path(entry.path) / "signals.npy", mmap_mode="r", allow_pickle=False)
    if start < 0 or stop > signals.shape[-1] or stop <= start:
        raise ValueError(f"invalid frozen model crop for {entry.subject_key}")
    return np.asarray(signals[:, :, start:stop], dtype=np.float32)


def _prepare(signals: np.ndarray, preprocessing: str) -> np.ndarray:
    fitted = fit_input_preprocessor(signals, preprocessing)
    return np.stack(
        [transform_input_signal(signal, preprocessing, fitted) for signal in signals]
    ).astype(np.float32)


def _predict(
    model: Any, signals: np.ndarray, device: Any, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    torch = _torch()
    logits, probabilities = [], []
    with torch.inference_mode():
        for start in range(0, len(signals), batch_size):
            batch = torch.from_numpy(signals[start : start + batch_size].copy()).to(device)
            batch_logits = model(batch)
            logits.append(batch_logits.cpu().numpy())
            probabilities.append(torch.softmax(batch_logits, dim=1).cpu().numpy())
    return np.concatenate(logits), np.concatenate(probabilities)


def _preflight_checkpoints(
    plan: dict[str, Any], checkpoints: dict[str, dict[str, Any]]
) -> None:
    """Load all model metadata before target data can be opened."""

    torch = _torch()
    device = torch.device("cpu")
    for unit in plan["units"]:
        entry = checkpoints[unit["unit_id"]]
        path = Path(entry["checkpoint_path"])
        if file_hash(path) != entry["checkpoint_sha256"]:
            raise ValueError(f"checkpoint changed after freeze: {unit['unit_id']}")
        model, _ = _load_checkpoint(path, unit, device)
        del model


def _shard_paths(unit_dir: Path, subject_key: str) -> tuple[Path, Path]:
    stem = _safe(subject_key)
    return (
        unit_dir / "participants" / f"{stem}.csv",
        unit_dir / "participant_summaries" / f"{stem}.json",
    )


def _completed_shard(
    unit: dict[str, Any], checkpoint_entry: dict[str, Any], unit_dir: Path, subject_key: str
) -> bool:
    prediction_path, summary_path = _shard_paths(unit_dir, subject_key)
    if not prediction_path.exists() and not summary_path.exists():
        return False
    if not prediction_path.is_file() or not summary_path.is_file():
        raise ValueError(f"partial participant shard exists: {unit['unit_id']} {subject_key}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "unit_id": unit["unit_id"],
        "subject_key": subject_key,
        "checkpoint_sha256": checkpoint_entry["checkpoint_sha256"],
        "inference_only": True,
        "labels_available_to_model": False,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError(f"participant shard metadata mismatch: {unit['unit_id']} {subject_key}")
    if summary.get("prediction_sha256") != file_hash(prediction_path):
        raise ValueError(f"participant shard hash mismatch: {unit['unit_id']} {subject_key}")
    return True


def _rows_after_outputs_fixed(
    entry: Any, logits: np.ndarray, probabilities: np.ndarray
) -> list[dict[str, Any]]:
    """Read labels only after a model's complete participant output is fixed."""

    directory = Path(entry.path)
    labels = np.load(directory / "labels.npy", mmap_mode="r", allow_pickle=False)
    sessions = np.load(directory / "sessions.npy", mmap_mode="r", allow_pickle=False)
    runs = np.load(directory / "runs.npy", mmap_mode="r", allow_pickle=False)
    if logits.shape != (entry.n_trials, 2) or probabilities.shape != (entry.n_trials, 2):
        raise ValueError(f"model output shape mismatch for {entry.subject_key}")
    rows = []
    for trial in range(entry.n_trials):
        raw_logit = float(logits[trial, 1] - logits[trial, 0])
        if not math.isfinite(raw_logit):
            raise ValueError("model emitted a non-finite raw logit")
        rows.append(
            {
                "subject_key": entry.subject_key,
                "dataset_id": entry.dataset_id,
                "session_id": str(sessions[trial]),
                "run_id": str(runs[trial]),
                "trial_index": trial,
                "trial_order": trial,
                "y_true": int(labels[trial]),
                "logit0": float(logits[trial, 0]),
                "logit1": float(logits[trial, 1]),
                "raw_logit": raw_logit,
                "p0": float(probabilities[trial, 0]),
                "p1": float(probabilities[trial, 1]),
                "y_pred": int(raw_logit >= 0.0),
            }
        )
    return rows


def _write_subject_shard(
    unit: dict[str, Any],
    checkpoint_entry: dict[str, Any],
    unit_dir: Path,
    entry: Any,
    logits: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    prediction_path, summary_path = _shard_paths(unit_dir, entry.subject_key)
    rows = _rows_after_outputs_fixed(entry, logits, probabilities)
    _atomic_csv(prediction_path, rows)
    _atomic_json(
        summary_path,
        {
            "schema_version": 1,
            "status": "participant_complete",
            "unit_id": unit["unit_id"],
            "subject_key": entry.subject_key,
            "n_trials": entry.n_trials,
            "checkpoint_sha256": checkpoint_entry["checkpoint_sha256"],
            "prediction_sha256": file_hash(prediction_path),
            "raw_logit_exported": True,
            "labels_available_to_model": False,
            "inference_only": True,
            "training_performed": False,
        },
    )


def _concatenate_shards(
    unit: dict[str, Any],
    checkpoint_entry: dict[str, Any],
    unit_dir: Path,
    entries: list[Any],
    cohort_audit_path: Path,
    elapsed: float,
) -> None:
    prediction_path = unit_dir / "predictions.csv"
    summary_path = unit_dir / "summary.json"
    expected_summary = {
        "unit_id": unit["unit_id"],
        "checkpoint_sha256": checkpoint_entry["checkpoint_sha256"],
        "inference_only": True,
        "target_dataset": "Stieger2021",
    }
    if prediction_path.is_file() and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if all(summary.get(key) == value for key, value in expected_summary.items()):
            if summary.get("prediction_sha256") == file_hash(prediction_path):
                return
        raise ValueError(f"completed unit metadata mismatch: {unit['unit_id']}")
    temporary = prediction_path.with_suffix(".csv.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shard_summaries = []
    total_trials = 0
    with temporary.open("w", newline="", encoding="utf-8") as output_handle:
        output_handle.write(",".join(PREDICTION_FIELDS) + "\n")
        for entry in entries:
            shard, shard_summary = _shard_paths(unit_dir, entry.subject_key)
            if not _completed_shard(unit, checkpoint_entry, unit_dir, entry.subject_key):
                raise ValueError(f"missing participant shard: {unit['unit_id']} {entry.subject_key}")
            with shard.open(encoding="utf-8") as input_handle:
                header = input_handle.readline().rstrip("\r\n").split(",")
                if header != PREDICTION_FIELDS:
                    raise ValueError(f"participant shard columns changed: {shard}")
                shutil.copyfileobj(input_handle, output_handle)
            payload = json.loads(shard_summary.read_text(encoding="utf-8"))
            total_trials += int(payload["n_trials"])
            shard_summaries.append(payload["prediction_sha256"])
    temporary.replace(prediction_path)
    _atomic_json(
        summary_path,
        {
            "schema_version": 2,
            "status": "complete",
            **expected_summary,
            "source_protocol": unit["source_protocol"],
            "backbone": unit["backbone"],
            "candidate": unit["candidate"],
            "seed": unit["seed"],
            "n_participants": len(entries),
            "n_trials": total_trials,
            "cohort_audit_sha256": file_hash(cohort_audit_path),
            "participant_shard_set_sha256": _stable_hash(sorted(shard_summaries)),
            "prediction_sha256": file_hash(prediction_path),
            "raw_logit_exported": True,
            "labels_available_to_model": False,
            "training_performed": False,
            "participant_major_shared_read": True,
            "seconds_current_invocation": elapsed,
        },
    )


def run(
    protocol_path: Path,
    plan_path: Path,
    checkpoint_manifest_path: Path,
    lock_path: Path,
    config_path: Path,
    run_root: Path,
) -> None:
    torch = _torch()
    validate_stieger_lock(
        lock_path,
        protocol_path,
        plan_path,
        checkpoint_manifest_path,
        project_root=PROJECT_ROOT,
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    manifest = validate_checkpoint_manifest(
        checkpoint_manifest_path, plan_path, check_files=True
    )
    checkpoints = {row["unit_id"]: row for row in manifest["checkpoints"]}
    _preflight_checkpoints(plan, checkpoints)
    if not torch.cuda.is_available():
        raise RuntimeError("participant-major formal inference requires a visible CUDA GPU")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config["training"].get("forbidden"):
        raise PermissionError("Stieger config must explicitly forbid training")
    entries = sorted(
        (
            entry
            for entry in load_manifest(config["data_root"])
            if entry.dataset_id == "Stieger2021"
        ),
        key=lambda entry: int(entry.subject_id),
    )
    if len(entries) != 62:
        raise ValueError(f"formal inference requires 62 exported participants, found {len(entries)}")
    cohort_audit_path = Path(config["data_root"]) / "export_cohort_audit.json"
    cohort_audit = json.loads(cohort_audit_path.read_text(encoding="utf-8"))
    if not (
        cohort_audit.get("full_export") is True
        and cohort_audit.get("all_subjects_accounted") is True
        and cohort_audit.get("allowed_reasons_only") is True
        and int(cohort_audit.get("expected_subject_count", 0)) == 62
    ):
        raise ValueError("Stieger cohort audit does not satisfy the frozen contract")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    device = torch.device("cuda")
    window = (
        float(config["model_window"]["tmin"]),
        float(config["model_window"]["tmax"]),
    )
    batch_size = int(config["training"]["batch_size"])
    started = time.perf_counter()
    for participant_index, entry in enumerate(entries, start=1):
        missing_units = [
            unit
            for unit in plan["units"]
            if not _completed_shard(
                unit,
                checkpoints[unit["unit_id"]],
                run_root / unit["output_relative_path"],
                entry.subject_key,
            )
        ]
        if missing_units:
            signals = _crop(entry, window)
            preprocessing_cache = {
                preprocessing: _prepare(signals, preprocessing)
                for preprocessing in sorted(
                    {str(unit["candidate"]["preprocessing"]) for unit in missing_units}
                )
            }
            fixed_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for unit in missing_units:
                checkpoint_entry = checkpoints[unit["unit_id"]]
                model, checkpoint = _load_checkpoint(
                    Path(checkpoint_entry["checkpoint_path"]), unit, device
                )
                preprocessing = str(checkpoint["preprocessing"])
                fixed_outputs[unit["unit_id"]] = _predict(
                    model, preprocessing_cache[preprocessing], device, batch_size
                )
                del model
                torch.cuda.empty_cache()
            # Only now can labels and acquisition metadata be read for output rows.
            for unit in missing_units:
                logits, probabilities = fixed_outputs[unit["unit_id"]]
                _write_subject_shard(
                    unit,
                    checkpoints[unit["unit_id"]],
                    run_root / unit["output_relative_path"],
                    entry,
                    logits,
                    probabilities,
                )
            del fixed_outputs, preprocessing_cache, signals
        _atomic_json(
            run_root / "participant-major-progress.json",
            {
                "schema_version": 1,
                "status": "in_progress",
                "completed_participants": participant_index,
                "expected_participants": len(entries),
                "unit_count": len(plan["units"]),
                "last_subject_key": entry.subject_key,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(
            f"[stieger-participant-major] complete {participant_index}/62 {entry.subject_key}",
            flush=True,
        )
    elapsed = time.perf_counter() - started
    for unit in plan["units"]:
        _concatenate_shards(
            unit,
            checkpoints[unit["unit_id"]],
            run_root / unit["output_relative_path"],
            entries,
            cohort_audit_path,
            elapsed,
        )
    _atomic_json(
        run_root / "participant-major-complete.json",
        {
            "schema_version": 1,
            "status": "complete",
            "participant_count": len(entries),
            "unit_count": len(plan["units"]),
            "inference_only": True,
            "participant_major_shared_read": True,
            "labels_loaded_after_all_model_outputs_per_participant": True,
            "training_performed": False,
            "seconds_current_invocation": elapsed,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.jobs != 1:
        raise ValueError("participant-major single-GPU inference requires --jobs 1")
    run(
        args.protocol,
        args.plan,
        args.checkpoint_manifest,
        args.lock,
        args.config,
        args.run_root,
    )


if __name__ == "__main__":
    main()
