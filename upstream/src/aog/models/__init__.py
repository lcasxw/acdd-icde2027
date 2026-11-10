from __future__ import annotations

from .eegnet import EEGNet
from .conformer import CompactEEGConformer


def build_model(
    name: str,
    n_channels: int,
    n_times: int,
    n_classes: int = 2,
    normalization: str = "batch_norm",
):
    normalized = name.lower().replace("-", "_")
    if normalized == "eegnet":
        return EEGNet(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            normalization=normalization,
        )
    if normalized in {"eeg_conformer", "conformer"}:
        return CompactEEGConformer(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            normalization=normalization,
        )
    raise KeyError(f"unknown model: {name}")


__all__ = ["EEGNet", "CompactEEGConformer", "build_model"]
