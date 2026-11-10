from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aog.data.store import load_manifest  # noqa: E402
from aog.integrity import file_hash  # noqa: E402
from aog.models import build_model  # noqa: E402
from aog.normalization import fit_input_preprocessor, transform_input_signal  # noqa: E402
from acdd.stieger_protocol import (  # noqa: E402
    REQUIRED_PREDICTION_COLUMNS,
    validate_checkpoint_manifest,
    validate_stieger_lock,
)


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for frozen Stieger inference") from exc
    return torch


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows or not REQUIRED_PREDICTION_COLUMNS.issubset(rows[0]):
        raise ValueError("Stieger inference must export raw logits and probabilities")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_checkpoint(path: Path, unit: dict[str, Any], device: Any) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    checkpoint = torch.load(path, map_location="cpu")
    candidate = unit["candidate"]
    required = {
        "model", "model_name", "n_channels", "n_times", "normalization",
        "preprocessing", "seed", "stage",
    }
    if not required.issubset(checkpoint):
        raise ValueError(f"checkpoint lacks frozen inference metadata: {sorted(required - set(checkpoint))}")
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


def _predict(model: Any, signals: np.ndarray, device: Any, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    torch = _torch()
    logits, probabilities = [], []
    with torch.inference_mode():
        for start in range(0, len(signals), batch_size):
            batch = torch.from_numpy(signals[start : start + batch_size].copy()).to(device)
            batch_logits = model(batch)
            logits.append(batch_logits.cpu().numpy())
            probabilities.append(torch.softmax(batch_logits, dim=1).cpu().numpy())
    return np.concatenate(logits), np.concatenate(probabilities)


def run_unit(
    unit: dict[str, Any],
    checkpoint_entry: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
) -> Path:
    torch = _torch()
    if unit.get("training_allowed") is not False:
        raise PermissionError("Stieger runner accepts inference-only units")
    checkpoint_path = Path(checkpoint_entry["checkpoint_path"])
    if file_hash(checkpoint_path) != checkpoint_entry["checkpoint_sha256"]:
        raise ValueError("checkpoint changed after Stieger freeze")
    prediction_path = output_dir / "predictions.csv"
    summary_path = output_dir / "summary.json"
    expected = {
        "unit_id": unit["unit_id"],
        "checkpoint_sha256": checkpoint_entry["checkpoint_sha256"],
        "inference_only": True,
        "target_dataset": "Stieger2021",
    }
    if prediction_path.is_file() and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if all(summary.get(key) == value for key, value in expected.items()):
            return prediction_path
        raise ValueError("completed Stieger unit metadata mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = _load_checkpoint(checkpoint_path, unit, device)
    entries = [entry for entry in load_manifest(config["data_root"]) if entry.dataset_id == "Stieger2021"]
    if not entries:
        raise ValueError("Stieger store contains no participants")
    cohort_audit_path = Path(config["data_root"]) / "export_cohort_audit.json"
    if not cohort_audit_path.is_file():
        raise FileNotFoundError("Stieger export cohort audit is missing")
    cohort_audit = json.loads(cohort_audit_path.read_text(encoding="utf-8"))
    if (
        cohort_audit.get("full_export") is not True
        or cohort_audit.get("all_subjects_accounted") is not True
        or cohort_audit.get("allowed_reasons_only") is not True
        or int(cohort_audit.get("expected_subject_count", 0)) != 62
    ):
        raise ValueError("Stieger cohort audit does not satisfy the frozen 62-participant contract")
    window = (float(config["model_window"]["tmin"]), float(config["model_window"]["tmax"]))
    batch_size = int(config["training"]["batch_size"])
    rows = []
    started = time.perf_counter()
    for entry in sorted(entries, key=lambda item: item.subject_key):
        signals = _crop(entry, window)
        # Alignment/scaling is fit separately within each target participant
        # using signals only, preserving the declared full-unlabeled-signal access.
        fitted = fit_input_preprocessor(signals, checkpoint["preprocessing"])
        transformed = np.stack(
            [transform_input_signal(signal, checkpoint["preprocessing"], fitted) for signal in signals]
        ).astype(np.float32)
        logits, probabilities = _predict(model, transformed, device, batch_size)
        directory = Path(entry.path)
        # Labels and metadata are read only after all model outputs are fixed.
        labels = np.load(directory / "labels.npy", mmap_mode="r", allow_pickle=False)
        sessions = np.load(directory / "sessions.npy", mmap_mode="r", allow_pickle=False)
        runs = np.load(directory / "runs.npy", mmap_mode="r", allow_pickle=False)
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
    _write_predictions(prediction_path, rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "status": "complete",
        **expected,
        "source_protocol": unit["source_protocol"],
        "backbone": unit["backbone"],
        "candidate": unit["candidate"],
        "seed": unit["seed"],
        "n_participants": len(entries),
        "cohort_audit_sha256": file_hash(cohort_audit_path),
        "n_trials": len(rows),
        "prediction_sha256": file_hash(prediction_path),
        "raw_logit_exported": True,
        "labels_available_to_model": False,
        "training_performed": False,
        "seconds": time.perf_counter() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return prediction_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    validate_stieger_lock(
        args.lock,
        args.protocol,
        args.plan,
        args.checkpoint_manifest,
        project_root=PROJECT_ROOT,
    )
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    manifest = validate_checkpoint_manifest(args.checkpoint_manifest, args.plan, check_files=True)
    checkpoints = {row["unit_id"]: row for row in manifest["checkpoints"]}
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not config["training"].get("forbidden"):
        raise PermissionError("Stieger config must explicitly forbid training")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    failures = []
    # Multiple CPU workers are supported; on a single GPU use jobs=1 to avoid
    # competing model replicas. Multi-GPU launchers may shard the plan outside
    # this process while preserving the same frozen unit IDs.
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                run_unit,
                unit,
                checkpoints[unit["unit_id"]],
                config,
                args.run_root / unit["output_relative_path"],
            ): unit["unit_id"]
            for unit in plan["units"]
        }
        for future in as_completed(futures):
            unit_id = futures[future]
            try:
                print(f"[stieger] complete {unit_id}: {future.result()}", flush=True)
            except Exception as exc:
                failures.append((unit_id, repr(exc)))
                print(f"[stieger] failed {unit_id}: {exc!r}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} Stieger inference units failed: {failures[:5]}")


if __name__ == "__main__":
    main()
