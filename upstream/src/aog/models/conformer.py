from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for aog.models") from exc


def _token_normalization(kind: str, channels: int) -> nn.Module:
    normalized = kind.lower().replace("-", "_")
    if normalized in {"batch", "batch_norm", "batchnorm"}:
        return nn.BatchNorm2d(channels)
    if normalized in {"group", "group_norm", "groupnorm"}:
        groups = min(4, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError("Compact EEG-Conformer supports batch_norm or group_norm tokenizers")


class CompactEEGConformer(nn.Module):
    """Small convolutional Transformer suitable for a single 24GB 4090D."""

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int = 2,
        embedding_dim: int = 96,
        depth: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        normalization: str = "batch_norm",
    ) -> None:
        super().__init__()
        self.tokenizer = nn.Sequential(
            nn.Conv2d(1, 32, (1, 25), padding=(0, 12), bias=False),
            _token_normalization(normalization, 32),
            nn.Conv2d(32, embedding_dim, (n_channels, 1), bias=False),
            _token_normalization(normalization, embedding_dim),
            nn.ELU(),
            nn.AvgPool2d((1, 10), stride=(1, 5)),
            nn.Dropout(dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embedding_dim)
        self.representation_dim = embedding_dim
        self.normalization = normalization
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor, return_representation: bool = False):
        if x.ndim != 3:
            raise ValueError("Conformer input must have shape [batch, channels, time]")
        tokens = self.tokenizer(x.unsqueeze(1)).squeeze(2).transpose(1, 2)
        representation = self.norm(self.encoder(tokens).mean(dim=1))
        logits = self.classifier(representation)
        return (logits, representation) if return_representation else logits
