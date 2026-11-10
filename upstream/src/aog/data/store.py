from __future__ import annotations

from dataclasses import asdict
from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator

import numpy as np

from ..types import SubjectEntry


MANIFEST_NAME = "manifest.jsonl"


@contextmanager
def _manifest_lock(root: Path) -> Iterator[None]:
    """Serialize manifest updates across independent export workers."""
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".manifest.lock"
    handle: BinaryIO = lock_path.open("a+b")
    try:
        if os.name == "nt":  # pragma: no cover - server execution is Linux
            import msvcrt

            if lock_path.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if os.name == "nt":  # pragma: no cover - server execution is Linux
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _safe_component(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def subject_directory(root: Path, dataset_id: str, subject_id: str) -> Path:
    return root / _safe_component(dataset_id) / f"subject-{_safe_component(subject_id)}"


def write_subject(
    root: str | Path,
    dataset_id: str,
    subject_id: str,
    signals: np.ndarray,
    labels: np.ndarray,
    sessions: Iterable[str],
    runs: Iterable[str],
    channel_names: Iterable[str],
    channel_coordinates: np.ndarray,
    sampling_rate: float,
    epoch_tmin: float,
    reference: str,
    synthetic: bool = False,
) -> SubjectEntry:
    root = Path(root)
    directory = subject_directory(root, dataset_id, str(subject_id))
    directory.mkdir(parents=True, exist_ok=True)
    signals = np.asarray(signals, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    sessions = np.asarray(list(sessions), dtype=str)
    runs = np.asarray(list(runs), dtype=str)
    coordinates = np.asarray(channel_coordinates, dtype=np.float32)
    names = tuple(str(name) for name in channel_names)

    if signals.ndim != 3:
        raise ValueError("signals must have shape [trials, channels, time]")
    if labels.shape != (signals.shape[0],):
        raise ValueError("labels do not match trial count")
    if sessions.shape != labels.shape or runs.shape != labels.shape:
        raise ValueError("session/run metadata do not match trial count")
    if coordinates.shape != (signals.shape[1], 3):
        raise ValueError("channel coordinates do not match signals")
    if len(names) != signals.shape[1]:
        raise ValueError("channel names do not match signals")
    if set(np.unique(labels)) - {0, 1}:
        raise ValueError("only binary labels 0 and 1 are supported")
    if not np.isfinite(signals).all():
        raise ValueError("signals contain NaN or infinity")

    np.save(directory / "signals.npy", signals, allow_pickle=False)
    np.save(directory / "labels.npy", labels, allow_pickle=False)
    np.save(directory / "sessions.npy", sessions, allow_pickle=False)
    np.save(directory / "runs.npy", runs, allow_pickle=False)
    np.save(directory / "channel_coordinates.npy", coordinates, allow_pickle=False)
    metadata = {
        "dataset_id": dataset_id,
        "subject_id": str(subject_id),
        "channel_names": names,
        "sampling_rate": float(sampling_rate),
        "epoch_tmin": float(epoch_tmin),
        "reference": reference,
        "synthetic": bool(synthetic),
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return SubjectEntry(
        dataset_id=dataset_id,
        subject_id=str(subject_id),
        path=str(directory.resolve()),
        n_trials=int(signals.shape[0]),
        n_channels=int(signals.shape[1]),
        n_times=int(signals.shape[2]),
        sampling_rate=float(sampling_rate),
        epoch_tmin=float(epoch_tmin),
        synthetic=bool(synthetic),
    )


def write_manifest(root: str | Path, entries: Iterable[SubjectEntry]) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / MANIFEST_NAME
    rows = sorted(entries, key=lambda entry: (entry.dataset_id, entry.subject_id))
    text = "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
    return path


def merge_manifest_entry(root: str | Path, entry: SubjectEntry) -> Path:
    """Atomically merge one completed subject into the shared manifest."""
    root = Path(root)
    with _manifest_lock(root):
        path = root / MANIFEST_NAME
        entries: dict[str, SubjectEntry] = {}
        if path.exists():
            entries = {row.subject_key: row for row in load_manifest(root)}
        entries[entry.subject_key] = entry
        return write_manifest(root, entries.values())


def load_manifest(root: str | Path) -> tuple[SubjectEntry, ...]:
    path = Path(root) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(SubjectEntry(**json.loads(line)))
    return tuple(entries)


def read_subject(entry: SubjectEntry, mmap: bool = True) -> dict[str, object]:
    directory = Path(entry.path)
    mode = "r" if mmap else None
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    return {
        "signals": np.load(directory / "signals.npy", mmap_mode=mode, allow_pickle=False),
        "labels": np.load(directory / "labels.npy", mmap_mode=mode, allow_pickle=False),
        "sessions": np.load(directory / "sessions.npy", mmap_mode=mode, allow_pickle=False),
        "runs": np.load(directory / "runs.npy", mmap_mode=mode, allow_pickle=False),
        "channel_coordinates": np.load(
            directory / "channel_coordinates.npy", mmap_mode=mode, allow_pickle=False
        ),
        "metadata": metadata,
    }


def validate_store(root: str | Path) -> dict[str, object]:
    entries = load_manifest(root)
    if not entries:
        raise ValueError("manifest is empty")
    subject_keys: set[str] = set()
    datasets: dict[str, int] = {}
    total_trials = 0
    synthetic_flags: set[bool] = set()
    for entry in entries:
        if entry.subject_key in subject_keys:
            raise ValueError(f"duplicate subject: {entry.subject_key}")
        subject_keys.add(entry.subject_key)
        data = read_subject(entry, mmap=True)
        signals = data["signals"]
        labels = data["labels"]
        if signals.shape != (entry.n_trials, entry.n_channels, entry.n_times):
            raise ValueError(f"shape mismatch for {entry.subject_key}")
        if labels.shape != (entry.n_trials,):
            raise ValueError(f"label mismatch for {entry.subject_key}")
        datasets[entry.dataset_id] = datasets.get(entry.dataset_id, 0) + 1
        total_trials += entry.n_trials
        synthetic_flags.add(entry.synthetic)
    if len(synthetic_flags) > 1:
        raise ValueError("real and synthetic subjects must not share a manifest")
    return {
        "n_datasets": len(datasets),
        "datasets": datasets,
        "n_subjects": len(entries),
        "n_trials": total_trials,
        "synthetic": synthetic_flags.pop(),
    }
