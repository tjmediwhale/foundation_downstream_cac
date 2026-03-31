"""Foundation model feature extractor backed by local DINOv3 repository."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

ARCH_TO_FEATURE_DIM = {
    "dinov3_vits16": 384,
    "dinov3_vitb16": 768,
    "dinov3_vitl16": 1024,
}

_BACKBONE_PREFIXES = (
    "student.backbone.",
    "_fsdp_wrapped_module.student.backbone.",
    "module.student.backbone.",
    "_orig_mod.student.backbone.",
    "backbone.",
    "module.backbone.",
)


class FoundationExtractor(nn.Module):
    """DINOv3 backbone wrapper with checkpoint compatibility helpers."""

    def __init__(
        self,
        architecture: str,
        checkpoint_path: Optional[str],
        dinov3_repo: Optional[str] = None,
        feature_type: str = "cls",
        freeze_backbone: bool = True,
        freeze_layers: int = 0,
    ) -> None:
        super().__init__()
        project_root = Path(__file__).resolve().parents[3]
        default_repo = project_root / "third_party" / "dinov3"

        self.architecture = architecture
        self.feature_type = str(feature_type).lower()
        self.dinov3_repo = Path(dinov3_repo) if dinov3_repo else default_repo

        self._setup_import_path()
        self.backbone = self._build_backbone(architecture)

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        self.feature_dim = int(ARCH_TO_FEATURE_DIM.get(architecture, getattr(self.backbone, "embed_dim", 1024)))

        if freeze_backbone:
            self.freeze_all()
        else:
            self.unfreeze_all()
            if freeze_layers > 0:
                self.freeze_bottom_n_blocks(freeze_layers)

    def _setup_import_path(self) -> None:
        if not self.dinov3_repo.exists():
            raise FileNotFoundError(
                f"DINOv3 repo path not found: {self.dinov3_repo} "
                "(set model.foundation.dinov3_repo in config)"
            )
        if str(self.dinov3_repo) not in sys.path:
            sys.path.insert(0, str(self.dinov3_repo))

    def _build_backbone(self, architecture: str) -> nn.Module:
        from dinov3.hub import backbones

        if not hasattr(backbones, architecture):
            raise ValueError(f"Unsupported DINOv3 architecture: {architecture}")
        builder = getattr(backbones, architecture)
        return builder(pretrained=False)

    @staticmethod
    def _extract_state_dict(ckpt: Dict) -> Dict[str, torch.Tensor]:
        if not isinstance(ckpt, dict):
            raise ValueError("Checkpoint must be a dictionary")

        for key in ("model", "state_dict", "student"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                return value

        if all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt
        raise ValueError("Unable to locate state_dict in checkpoint")

    @staticmethod
    def _strip_backbone_prefix(key: str) -> Optional[str]:
        for prefix in _BACKBONE_PREFIXES:
            if key.startswith(prefix):
                return key[len(prefix) :]
        return None

    @staticmethod
    def _normalize_tensor(value: torch.Tensor) -> torch.Tensor:
        if hasattr(value, "full_tensor"):
            return value.full_tensor()
        return value

    def _filter_backbone_weights(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        filtered = {}
        passthrough_prefix = (
            "patch_embed.",
            "cls_token",
            "storage_tokens",
            "mask_token",
            "blocks.",
            "norm.",
            "cls_norm.",
            "local_cls_norm.",
            "rope_embed.",
            "head.",
        )
        for key, value in state_dict.items():
            if not isinstance(key, str):
                continue
            mapped_key = self._strip_backbone_prefix(key)
            if mapped_key is not None:
                filtered[mapped_key] = self._normalize_tensor(value)
                continue
            if key.startswith(passthrough_prefix):
                filtered[key] = self._normalize_tensor(value)
        return filtered

    def load_checkpoint(self, checkpoint_path: str, strict: bool = False) -> Tuple[list, list]:
        """Load foundation checkpoint with flexible key remapping."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        source_state = self._extract_state_dict(checkpoint)
        backbone_state = self._filter_backbone_weights(source_state)

        if not backbone_state:
            raise RuntimeError(
                "No compatible backbone keys were found in checkpoint. "
                f"path={checkpoint_path}"
            )

        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=strict)
        return list(missing), list(unexpected)

    def freeze_all(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_bottom_n_blocks(self, n_blocks: int) -> None:
        if not hasattr(self.backbone, "blocks"):
            return
        n_blocks = max(0, int(n_blocks))
        for idx, block in enumerate(self.backbone.blocks):
            requires_grad = idx >= n_blocks
            for param in block.parameters():
                param.requires_grad = requires_grad

    def is_frozen(self) -> bool:
        return not any(param.requires_grad for param in self.backbone.parameters())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(images)
        if isinstance(features, dict):
            if self.feature_type == "global_pool":
                return features["x_norm_patchtokens"].mean(dim=1)
            return features["x_norm_clstoken"]
        if isinstance(features, torch.Tensor):
            return features
        raise TypeError(f"Unsupported backbone output type: {type(features)!r}")

