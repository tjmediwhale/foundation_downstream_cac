"""Downstream model heads for CAC prediction."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import warnings


warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True",
    category=UserWarning,
)


class TransformerHead(nn.Module):
    """Transformer encoder head operating on tokenized foundation features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        num_tokens: int = 4,
        dropout: float = 0.2,
        output_classes: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens

        self.input_proj = nn.Linear(input_dim, hidden_dim * num_tokens)
        self.positional_embed = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        try:
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )
        except TypeError:
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, output_classes)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = features.size(0)
        tokens = self.input_proj(features).reshape(batch_size, self.num_tokens, self.hidden_dim)
        tokens = tokens + self.positional_embed
        encoded = self.encoder(tokens)
        pooled = self.norm(encoded.mean(dim=1))
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        ranking_scores = torch.softmax(logits, dim=1)[:, 1]
        return logits, ranking_scores


class MLPHead(nn.Module):
    """MLP-based head for CAC classification and ranking."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.2,
        output_classes: int = 2,
    ) -> None:
        super().__init__()
        blocks = []
        dim = input_dim
        for _ in range(num_layers):
            blocks.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            dim = hidden_dim
        self.backbone = nn.Sequential(*blocks)
        self.classifier = nn.Linear(dim, output_classes)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        logits = self.classifier(hidden)
        ranking_scores = torch.softmax(logits, dim=1)[:, 1]
        return logits, ranking_scores


def build_downstream_head(cfg: Dict, input_dim: int) -> nn.Module:
    """Factory for downstream head from config."""
    downstream_cfg = cfg.get("model", {}).get("downstream", {})
    model_type = str(downstream_cfg.get("type", "transformer")).lower()

    common_kwargs = {
        "input_dim": input_dim,
        "hidden_dim": int(downstream_cfg.get("hidden_dim", 512)),
        "num_layers": int(downstream_cfg.get("num_layers", 3)),
        "dropout": float(downstream_cfg.get("dropout", 0.2)),
        "output_classes": int(downstream_cfg.get("output_classes", 2)),
    }

    if model_type == "transformer":
        return TransformerHead(
            **common_kwargs,
            num_heads=int(downstream_cfg.get("num_heads", 8)),
            num_tokens=int(downstream_cfg.get("num_tokens", 4)),
        )
    if model_type == "mlp":
        return MLPHead(**common_kwargs)

    raise ValueError(f"Unsupported downstream model type: {model_type}")

