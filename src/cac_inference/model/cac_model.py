"""End-to-end CAC model: foundation extractor + downstream head."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .downstream_model import build_downstream_head
from .foundation_extractor import FoundationExtractor


class FeatureCache:
    """Simple feature cache with memory + optional disk persistence."""

    def __init__(
        self,
        enabled: bool,
        max_items: int = 10000,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_items = int(max_items)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._memory: "OrderedDict[str, torch.Tensor]" = OrderedDict()

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(path: str) -> str:
        return hashlib.sha1(path.encode("utf-8")).hexdigest()

    def _disk_path(self, key: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{key}.pt"

    def get(self, image_path: str) -> Optional[torch.Tensor]:
        if not self.enabled:
            return None

        key = self._key(image_path)
        if key in self._memory:
            value = self._memory.pop(key)
            self._memory[key] = value
            return value.clone()

        disk_path = self._disk_path(key)
        if disk_path is not None and disk_path.exists():
            value = torch.load(disk_path, map_location="cpu")
            self._memory[key] = value
            self._evict_if_needed()
            return value.clone()

        return None

    def set(self, image_path: str, feature: torch.Tensor) -> None:
        if not self.enabled:
            return
        value = feature.detach().cpu().clone()
        key = self._key(image_path)
        self._memory[key] = value
        self._evict_if_needed()

        disk_path = self._disk_path(key)
        if disk_path is not None and not disk_path.exists():
            torch.save(value, disk_path)

    def _evict_if_needed(self) -> None:
        while len(self._memory) > self.max_items:
            self._memory.popitem(last=False)


class CACModel(nn.Module):
    """Full CAC model with foundation extractor + downstream head."""

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        foundation_cfg = cfg.get("model", {}).get("foundation", {})

        self.foundation_extractor = FoundationExtractor(
            architecture=foundation_cfg.get("architecture", "dinov3_vitl16"),
            checkpoint_path=foundation_cfg.get("checkpoint"),
            dinov3_repo=foundation_cfg.get("dinov3_repo"),
            feature_type=foundation_cfg.get("feature_type", "cls"),
            freeze_backbone=bool(foundation_cfg.get("freeze_backbone", True)),
            freeze_layers=int(foundation_cfg.get("freeze_layers", 0)),
        )
        self.downstream_head = build_downstream_head(cfg, input_dim=self.foundation_extractor.feature_dim)

        self.feature_cache = FeatureCache(
            enabled=bool(foundation_cfg.get("use_feature_cache", False)),
            max_items=int(foundation_cfg.get("feature_cache_size", 10000)),
            cache_dir=foundation_cfg.get("feature_cache_dir"),
        )

    def _extract_features_with_cache(self, images: torch.Tensor, image_paths: List[str]) -> torch.Tensor:
        device = images.device
        cached = [self.feature_cache.get(path) for path in image_paths]

        missing_indices = [i for i, feat in enumerate(cached) if feat is None]
        if missing_indices:
            missing_tensor = images[missing_indices]
            missing_features = self.foundation_extractor(missing_tensor)
            for local_idx, feature in zip(missing_indices, missing_features):
                self.feature_cache.set(image_paths[local_idx], feature)
                cached[local_idx] = feature.detach().cpu()

        output = torch.stack([feat for feat in cached], dim=0).to(device=device, dtype=images.dtype)
        return output

    def forward(self, images: torch.Tensor, image_paths: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        use_cache = image_paths is not None and self.feature_cache.enabled and self.foundation_extractor.is_frozen()

        if use_cache:
            features = self._extract_features_with_cache(images, image_paths)
        else:
            features = self.foundation_extractor(images)

        logits, ranking_scores = self.downstream_head(features)
        return {
            "logits": logits,
            "ranking_scores": ranking_scores,
            "features": features,
        }

