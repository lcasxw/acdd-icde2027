from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrialRecord:
    """One epoched trial and all fields needed for leakage-safe evaluation."""

    signal: np.ndarray
    label: int
    subject_id: str
    session_id: str
    dataset_id: str
    channel_names: tuple[str, ...]
    channel_coordinates: np.ndarray
    sampling_rate: float
    reference: str
    cue_time: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        signal = np.asarray(self.signal)
        coords = np.asarray(self.channel_coordinates)
        if signal.ndim != 2:
            raise ValueError("signal must have shape [channels, time]")
        if coords.shape != (signal.shape[0], 3):
            raise ValueError("channel_coordinates must have shape [channels, 3]")
        if len(self.channel_names) != signal.shape[0]:
            raise ValueError("channel_names and signal channel count differ")
        if self.label not in (0, 1):
            raise ValueError("binary MI label must be 0 or 1")
        if not np.isfinite(signal).all():
            raise ValueError("signal contains NaN or infinity")
        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")

    @property
    def subject_key(self) -> str:
        return f"{self.dataset_id}::{self.subject_id}"

    def with_signal(self, signal: np.ndarray, **metadata: Any) -> "TrialRecord":
        merged = {**self.metadata, **metadata}
        return replace(self, signal=np.asarray(signal, dtype=np.float32), metadata=merged)


@dataclass(frozen=True)
class SubjectEntry:
    dataset_id: str
    subject_id: str
    path: str
    n_trials: int
    n_channels: int
    n_times: int
    sampling_rate: float
    epoch_tmin: float
    synthetic: bool = False

    @property
    def subject_key(self) -> str:
        return f"{self.dataset_id}::{self.subject_id}"

