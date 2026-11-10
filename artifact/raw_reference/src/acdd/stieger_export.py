from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np

from aog.data.store import MANIFEST_NAME, load_manifest, merge_manifest_entry, subject_directory, write_subject
from aog.integrity import stable_json_hash
from .stieger_protocol import validate_stieger_lock, validate_stieger_protocol


EXPECTED_RAW_MAT_FILES = 598
EXPECTED_RAW_SUBJECTS = 62
EXPECTED_RAW_MAT_BYTES = 376_894_087_292


def _append_exclusion(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _read_exclusions(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _coordinates(epochs: Any) -> np.ndarray:
    montage = epochs.get_montage()
    positions = montage.get_positions()["ch_pos"] if montage is not None else {}
    values = []
    for channel in epochs.ch_names:
        position = positions.get(channel, positions.get(channel.upper()))
        values.append(np.asarray(position if position is not None else [np.nan] * 3, dtype=float))
    result = np.stack(values)
    if not np.isfinite(result).all():
        raise ValueError(f"missing coordinates for frozen Stieger channels: {epochs.ch_names}")
    return result


def _validate_config(config: dict[str, Any], protocol: dict[str, Any]) -> None:
    preprocessing = protocol["preprocessing"]
    checks = {
        "dataset": (config["datasets"], ["Stieger2021"]),
        "events": (config["events"], protocol["dataset"]["task"]),
        "canonical_channels": (config["canonical_channels"], preprocessing["canonical_channels"]),
        "sampling_rate": (
            float(config["preprocessing"]["sfreq"]),
            float(preprocessing["sampling_rate_hz"]),
        ),
        "bandpass": (
            [float(config["preprocessing"]["fmin"]), float(config["preprocessing"]["fmax"])],
            [float(value) for value in preprocessing["bandpass_hz"]],
        ),
        "model_window": (
            [float(config["model_window"]["tmin"]), float(config["model_window"]["tmax"])],
            [float(value) for value in preprocessing["model_window_seconds"]],
        ),
    }
    mismatches = {key: value for key, value in checks.items() if value[0] != value[1]}
    if mismatches:
        raise ValueError(f"Stieger export config changed from the frozen protocol: {mismatches}")


def _validate_raw_prefetch_completion(
    config: dict[str, Any], project_root: Path, cache: Path
) -> dict[str, Any]:
    relative_path = config["confirmation"].get("raw_prefetch_completion")
    if not relative_path:
        raise PermissionError("Stieger export requires the frozen raw-prefetch completion proof")
    proof_path = (project_root / relative_path).resolve()
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    payload = dict(proof)
    recorded_hash = payload.pop("completion_payload_sha256", None)
    if recorded_hash != stable_json_hash(payload):
        raise ValueError("raw-prefetch completion proof payload hash mismatch")
    required = {
        "status": "all_raw_mat_files_prefetched_and_verified",
        "subject_count": EXPECTED_RAW_SUBJECTS,
        "mat_file_count": EXPECTED_RAW_MAT_FILES,
        "mat_total_bytes": EXPECTED_RAW_MAT_BYTES,
        "target_mat_files_opened": False,
        "target_signals_or_labels_inspected": False,
        "target_prediction_units_completed": 0,
        "training_allowed": False,
        "scientific_protocol_changed": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": proof.get(key)}
        for key, expected in required.items()
        if proof.get(key) != expected
    }
    verification = proof.get("verification", {})
    if verification.get("partial_file_count") != 0:
        mismatches["partial_file_count"] = verification.get("partial_file_count")
    if verification.get("md5_verified_files_in_integrity_audit") != EXPECTED_RAW_MAT_FILES:
        mismatches["md5_verified_files_in_integrity_audit"] = verification.get(
            "md5_verified_files_in_integrity_audit"
        )
    if mismatches:
        raise ValueError(f"raw-prefetch completion proof changed: {mismatches}")
    raw_directory = cache.resolve() / "MNE-Stieger2021-data"
    mat_files = [path for path in raw_directory.glob("*.mat") if path.is_file()]
    partials = [path for path in raw_directory.glob("*.part") if path.is_file()]
    actual_bytes = sum(path.stat().st_size for path in mat_files)
    if len(mat_files) != EXPECTED_RAW_MAT_FILES or partials or actual_bytes != EXPECTED_RAW_MAT_BYTES:
        raise ValueError(
            "raw Stieger cache no longer matches its completion proof: "
            f"mat_files={len(mat_files)}, partials={len(partials)}, bytes={actual_bytes}"
        )
    return proof


def _require_disk_reserve(path: Path, minimum_free_gb: float) -> None:
    free_bytes = shutil.disk_usage(path).free
    required_bytes = int(minimum_free_gb * 1024**3)
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Stieger export paused before target parsing: free_bytes={free_bytes} "
            f"is below the frozen reserve={required_bytes}"
        )


def _local_stieger_paths(
    raw_directory: Path, subject: int, sessions: list[int] | None = None
) -> list[str]:
    pattern = re.compile(rf"S{int(subject)}_Session_([1-9][0-9]*)\.mat")
    selected: list[tuple[int, Path]] = []
    for path in raw_directory.glob(f"S{int(subject)}_Session_*.mat"):
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected frozen Stieger filename: {path.name}")
        session = int(match.group(1))
        if sessions is None or session in sessions:
            selected.append((session, path.resolve()))
    selected.sort(key=lambda item: item[0])
    session_ids = [session for session, _ in selected]
    if not selected:
        raise FileNotFoundError(f"no frozen local Stieger files for subject {subject}")
    if len(session_ids) != len(set(session_ids)):
        raise ValueError(f"duplicate frozen Stieger sessions for subject {subject}")
    return [str(path) for _, path in selected]


def _session_from_stieger_path(path: str | Path) -> int:
    match = re.fullmatch(r"S[1-9][0-9]*_Session_([1-9][0-9]*)\.mat", Path(path).name)
    if match is None:
        raise ValueError(f"unexpected frozen Stieger path: {path}")
    return int(match.group(1))


def _build_local_stieger_dataset(
    cache: Path,
    canonical_channels: list[str],
    sessions: list[int] | None = None,
) -> Any:
    import mne
    from moabb.datasets import Stieger2021 as MOABBStieger2021

    frozen_raw_directory = cache.resolve() / "MNE-Stieger2021-data"
    frozen_channels = list(canonical_channels)

    class Stieger2021(MOABBStieger2021):
        """MOABB-compatible local, session-streaming Stieger loader."""

        def data_path(
            self,
            subject: int,
            path: str | None = None,
            force_update: bool = False,
            update_path: bool | None = None,
            verbose: object | None = None,
        ) -> list[str]:
            del path, update_path, verbose
            if force_update:
                raise PermissionError("force_update is forbidden for the frozen local archive")
            if subject not in self.subject_list:
                raise ValueError(f"invalid frozen Stieger subject: {subject}")
            return _local_stieger_paths(
                frozen_raw_directory,
                subject,
                self.sessions,
            )

        def _get_single_subject_data(self, subject: int) -> dict[str, dict[str, Any]]:
            subject_data: dict[str, dict[str, Any]] = {}
            montage = mne.channels.make_standard_montage("standard_1005")
            for file_path in self.data_path(subject):
                session = self._parse_session(file_path)
                container = self._load_container(file_path)
                sampling_rate = float(container.SRATE)
                all_channel_names = [
                    str(channel).replace("Z", "z").replace("FP", "Fp")
                    for channel in container.chaninfo.label.tolist()
                ]
                missing = sorted(set(frozen_channels) - set(all_channel_names))
                if missing:
                    raise ValueError(
                        f"missing frozen common channels in subject {subject} session "
                        f"{session}: {missing}"
                    )
                channel_indices = [all_channel_names.index(channel) for channel in frozen_channels]
                signal_segments: list[np.ndarray] = []
                stimulus_segments: list[np.ndarray] = []
                accepted_trial_lengths: list[float] = []
                for trial_index in range(container.data.shape[0]):
                    signal_segments.append(
                        np.asarray(container.data[trial_index][channel_indices, :])
                    )
                    trial = container.TrialData[trial_index]
                    stimulus = np.zeros_like(container.time[trial_index])
                    if trial.artifact == 0 and trial.triallength >= self.interval[1]:
                        cue_index = int(2 * sampling_rate)
                        if container.time[trial_index][cue_index] != 0:
                            raise ValueError("unexpected Stieger cue time-point")
                        stimulus[cue_index] = trial.targetnumber
                        accepted_trial_lengths.append(float(trial.triallength))
                    stimulus_segments.append(stimulus[None, :])
                signals = np.concatenate(signal_segments, axis=1)
                stimulus = np.concatenate(stimulus_segments, axis=1)
                data = np.concatenate([signals * 1e-6, stimulus], axis=0)
                info = mne.create_info(
                    ch_names=frozen_channels + ["stim"],
                    ch_types=["eeg"] * len(frozen_channels) + ["stim"],
                    sfreq=sampling_rate,
                )
                raw = mne.io.RawArray(data=data, info=info, verbose=False)
                raw.set_montage(montage)
                events = mne.find_events(raw, shortest_event=0, verbose=False)
                if len(events) > 0:
                    annotations = mne.annotations_from_events(
                        events,
                        raw.info["sfreq"],
                        {value: key for key, value in self.event_id.items()},
                        first_samp=raw.first_samp,
                        verbose=False,
                    )
                    if len(annotations) != len(accepted_trial_lengths):
                        raise ValueError(
                            "Stieger annotations and accepted trial lengths do not match"
                        )
                    annotations.extras = [
                        {"triallength": value} for value in accepted_trial_lengths
                    ]
                    raw.set_annotations(annotations)
                noise_channels = container.chaninfo.noisechan
                if isinstance(noise_channels, int):
                    bad_indices = [noise_channels]
                elif isinstance(noise_channels, np.ndarray):
                    bad_indices = noise_channels.tolist()
                else:
                    bad_indices = []
                for one_based_index in bad_indices:
                    bad_name = all_channel_names[int(one_based_index) - 1]
                    if bad_name in frozen_channels:
                        raw.info["bads"].append(bad_name)
                if raw.info["bads"] and self.fix_bads:
                    raw = raw.interpolate_bads()
                subject_data[str(session)] = {"0": raw}
                del container, signals, stimulus, data, signal_segments, stimulus_segments
                gc.collect()
            return subject_data

    return Stieger2021(sessions=sessions)


def export_stieger_from_config(
    config_path: str | Path,
    confirmation_lock: str | Path | None = None,
    subject_limit: int | None = None,
) -> Path:
    if confirmation_lock is None:
        raise PermissionError("Stieger2021 is locked until the V5 confirmation freeze exists")
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    confirmation = config["confirmation"]
    protocol_path = project_root / confirmation["protocol"]
    plan_path = project_root / confirmation["plan"]
    checkpoint_manifest_path = project_root / confirmation["checkpoint_manifest"]
    protocol = validate_stieger_protocol(protocol_path)
    _validate_config(config, protocol)
    validate_stieger_lock(
        confirmation_lock,
        protocol_path,
        plan_path,
        checkpoint_manifest_path,
        project_root=project_root,
    )
    root = Path(config["data_root"])
    cache = Path(config["mne_cache"])
    root.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    _validate_raw_prefetch_completion(config, project_root, cache)
    minimum_free_gb = float(config["storage"]["minimum_free_gb"])
    _require_disk_reserve(root, minimum_free_gb)
    try:
        from moabb.paradigms import MotorImagery
        from moabb.utils import set_download_dir
    except ImportError as exc:
        raise RuntimeError("MOABB is required for frozen Stieger export") from exc
    os.environ["MNE_DATA"] = str(cache.resolve())
    set_download_dir(str(cache))

    paradigm = MotorImagery(
        n_classes=2,
        events=config["events"],
        fmin=float(config["preprocessing"]["fmin"]),
        fmax=float(config["preprocessing"]["fmax"]),
        tmin=float(config["export_epoch"]["tmin"]),
        tmax=float(config["export_epoch"]["tmax"]),
        channels=config["canonical_channels"],
        resample=float(config["preprocessing"]["sfreq"]),
    )
    dataset = _build_local_stieger_dataset(cache, config["canonical_channels"])
    existing = (
        {entry.subject_key: entry for entry in load_manifest(root)}
        if (root / MANIFEST_NAME).exists()
        else {}
    )
    exclusion_path = root / "export_exclusions.jsonl"
    subjects = list(dataset.subject_list)
    expected_subjects = int(config["expected_subjects"]["Stieger2021"])
    if subject_limit is None and len(subjects) != expected_subjects:
        raise ValueError(
            f"frozen Stieger cohort expects {expected_subjects} participants, "
            f"but MOABB exposes {len(subjects)}"
        )
    if subject_limit is not None:
        subjects = subjects[:subject_limit]
    for subject in subjects:
        _require_disk_reserve(root, minimum_free_gb)
        key = f"Stieger2021::{subject}"
        directory = subject_directory(root, "Stieger2021", str(subject))
        if key in existing and (directory / "signals.npy").is_file():
            continue
        signal_chunks: list[np.ndarray] = []
        label_chunks: list[np.ndarray] = []
        session_chunks: list[np.ndarray] = []
        run_chunks: list[np.ndarray] = []
        channel_names: list[str] | None = None
        channel_coordinates: np.ndarray | None = None
        sampling_rate: float | None = None
        for session_path in _local_stieger_paths(
            cache.resolve() / "MNE-Stieger2021-data", subject
        ):
            session = _session_from_stieger_path(session_path)
            _require_disk_reserve(root, minimum_free_gb)
            session_dataset = _build_local_stieger_dataset(
                cache, config["canonical_channels"], [session]
            )
            try:
                epochs, raw_labels, metadata = paradigm.get_data(
                    dataset=session_dataset,
                    subjects=[subject],
                    return_epochs=True,
                )
            except (OSError, EOFError) as exc:
                _append_exclusion(
                    exclusion_path,
                    {
                        "dataset_id": "Stieger2021",
                        "subject_id": str(subject),
                        "session_id": str(session),
                        "reason": "corrupted_file",
                        "exception_type": type(exc).__name__,
                    },
                )
                continue
            missing = sorted(set(config["canonical_channels"]) - set(epochs.ch_names))
            if missing:
                _append_exclusion(
                    exclusion_path,
                    {
                        "dataset_id": "Stieger2021",
                        "subject_id": str(subject),
                        "session_id": str(session),
                        "reason": "missing_any_frozen_common_channel",
                        "missing_channels": missing,
                    },
                )
                continue
            labels = np.asarray(
                [config["label_map"][str(label)] for label in raw_labels], dtype=int
            )
            valid = len(labels) >= 30 and set(labels.tolist()) == {0, 1}
            if not valid:
                _append_exclusion(
                    exclusion_path,
                    {
                        "dataset_id": "Stieger2021",
                        "subject_id": str(subject),
                        "session_id": str(session),
                        "reason": "fewer_than_30_valid_left_right_trials_in_a_session",
                        "valid_left_right_trials": int(len(labels)),
                    },
                )
                continue
            if channel_names is None:
                channel_names = list(epochs.ch_names)
                channel_coordinates = _coordinates(epochs)
                sampling_rate = float(epochs.info["sfreq"])
            elif list(epochs.ch_names) != channel_names:
                raise ValueError("frozen channel order changed across Stieger sessions")
            signal_chunks.append(
                epochs.get_data(copy=True).astype(np.float32, copy=False)
            )
            label_chunks.append(labels)
            session_chunks.append(np.full(len(labels), str(session)))
            run_chunks.append(metadata["run"].astype(str).to_numpy())
            del epochs, raw_labels, metadata, labels, session_dataset
            gc.collect()
        if not signal_chunks:
            continue
        if channel_names is None or channel_coordinates is None or sampling_rate is None:
            raise AssertionError("Stieger session streaming metadata was not materialized")
        entry = write_subject(
            root=root,
            dataset_id="Stieger2021",
            subject_id=str(subject),
            signals=np.concatenate(signal_chunks, axis=0),
            labels=np.concatenate(label_chunks),
            sessions=np.concatenate(session_chunks),
            runs=np.concatenate(run_chunks),
            channel_names=channel_names,
            channel_coordinates=channel_coordinates,
            sampling_rate=sampling_rate,
            epoch_tmin=float(config["export_epoch"]["tmin"]),
            reference="dataset_native_then_moabb_preprocessing",
            synthetic=False,
        )
        existing[entry.subject_key] = entry
        merge_manifest_entry(root, entry)
    included_entries = load_manifest(root) if (root / MANIFEST_NAME).exists() else []
    included_ids = sorted(
        entry.subject_id
        for entry in included_entries
        if entry.dataset_id == "Stieger2021"
    )
    exclusions = _read_exclusions(exclusion_path)
    expected_ids = sorted(str(subject) for subject in subjects)
    accounted = set(included_ids)
    for subject in expected_ids:
        rows = [row for row in exclusions if str(row["subject_id"]) == subject]
        participant_level = any(row.get("session_id") is None for row in rows)
        excluded_sessions = {str(row["session_id"]) for row in rows if row.get("session_id") is not None}
        if participant_level or (subject not in included_ids and excluded_sessions):
            accounted.add(subject)
    audit = {
        "schema_version": 1,
        "dataset": "Stieger2021",
        "full_export": subject_limit is None,
        "expected_subject_count": expected_subjects if subject_limit is None else len(expected_ids),
        "expected_subject_ids": expected_ids,
        "included_subject_ids": included_ids,
        "participant_or_session_exclusions": exclusions,
        "accounted_subject_ids": sorted(accounted),
        "unaccounted_subject_ids": sorted(set(expected_ids) - accounted),
        "all_subjects_accounted": set(expected_ids) == accounted,
        "allowed_reasons_only": all(
            row.get("reason")
            in {
                "corrupted_file",
                "missing_any_frozen_common_channel",
                "fewer_than_30_valid_left_right_trials_in_a_session",
            }
            for row in exclusions
        ),
    }
    (root / "export_cohort_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    if not audit["all_subjects_accounted"] or not audit["allowed_reasons_only"]:
        raise RuntimeError(f"Stieger export cohort audit failed: {audit}")
    return root / MANIFEST_NAME


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--confirmation-lock", type=Path, required=True)
    parser.add_argument("--subject-limit", type=int)
    args = parser.parse_args()
    print(export_stieger_from_config(args.config, args.confirmation_lock, args.subject_limit))


if __name__ == "__main__":
    main()
