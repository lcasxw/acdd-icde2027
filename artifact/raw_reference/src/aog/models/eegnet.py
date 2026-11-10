from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised on research environment
    raise RuntimeError("PyTorch is required for aog.models") from exc


def _normalization(kind: str, channels: int) -> nn.Module:
    normalized = kind.lower().replace("-", "_")
    if normalized in {"batch", "batch_norm", "batchnorm"}:
        return nn.BatchNorm2d(channels)
    if normalized in {"group", "group_norm", "groupnorm"}:
        groups = min(4, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if normalized in {"layer", "layer_norm", "layernorm"}:
        # One group normalizes each example across all feature maps and spatial positions.
        return nn.GroupNorm(1, channels)
    if normalized in {"instance", "instance_norm", "instancenorm"}:
        return nn.InstanceNorm2d(channels, affine=True, track_running_stats=False)
    raise KeyError(f"unknown EEGNet normalization: {kind}")


class EEGNet(nn.Module):
    """Compact EEGNet-style encoder exposing a representation for audit losses."""

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int = 2,
        f1: int = 16,
        depth_multiplier: int = 2,
        f2: int = 32,
        dropout: float = 0.35,
        normalization: str = "batch_norm",
    ) -> None:
        super().__init__()
        temporal_kernel = min(125, max(15, n_times // 4))
        if temporal_kernel % 2 == 0:
            temporal_kernel += 1
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, (1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False),
            _normalization(normalization, f1),
            nn.Conv2d(f1, f1 * depth_multiplier, (n_channels, 1), groups=f1, bias=False),
            _normalization(normalization, f1 * depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(
                f1 * depth_multiplier,
                f1 * depth_multiplier,
                (1, 15),
                padding=(0, 7),
                groups=f1 * depth_multiplier,
                bias=False,
            ),
            nn.Conv2d(f1 * depth_multiplier, f2, (1, 1), bias=False),
            _normalization(normalization, f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 8)),
        )
        self.representation_dim = f2 * 8
        self.normalization = normalization
        self.classifier = nn.Linear(self.representation_dim, n_classes)

    def forward(self, x: torch.Tensor, return_representation: bool = False):
        if x.ndim != 3:
            raise ValueError("EEGNet input must have shape [batch, channels, time]")
        representation = self.features(x.unsqueeze(1)).flatten(1)
        logits = self.classifier(representation)
        return (logits, representation) if return_representation else logits
