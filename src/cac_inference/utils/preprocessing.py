"""Image preprocessing utilities for CAC inference."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
from PIL import Image
from torchvision import transforms


def _resolve_drnoon_improc() -> Optional[object]:
    """Resolve drnoon-image-transform improc module from known locations."""
    project_root = Path(__file__).resolve().parents[3]
    env_dir = os.environ.get("DRNOON_TRANSFORM_DIR")
    candidates = [
        Path(env_dir) if env_dir else None,
        project_root / "third_party" / "drnoon-image-transform",
    ]

    for candidate in candidates:
        if candidate is not None and candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    try:
        from drnoon_image_transform.utils import improc

        return improc
    except ImportError:
        return None


def fundus_preprocess_drnoon(
    pil_image: Image.Image,
    precrop: Optional[float] = 0.4,
    circle_mask: bool = True,
) -> Image.Image:
    """Apply DrNoon retinal preprocessing."""
    improc = _resolve_drnoon_improc()
    if improc is None:
        return pil_image

    image_np = np.array(pil_image)
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)

    if precrop is not None and 0 < precrop <= 1:
        image_np = improc.center_crop_square(image_np, precrop)
    if circle_mask:
        image_np = improc.mask_center_circle(image_np)

    return Image.fromarray(image_np.astype(np.uint8))


def build_image_transform(cfg: Dict, is_train: bool = False) -> Callable[[Image.Image], object]:
    """Build torchvision transform for inference (or train-time aug if requested)."""
    preprocessing_cfg = cfg.get("data", {}).get("preprocessing", {})
    image_size = int(preprocessing_cfg.get("image_size", 1024))
    mean = preprocessing_cfg.get("normalize_mean", [0.485, 0.456, 0.406])
    std = preprocessing_cfg.get("normalize_std", [0.229, 0.224, 0.225])

    use_drnoon = bool(preprocessing_cfg.get("use_drnoon_preprocess", False))
    drnoon_precrop = preprocessing_cfg.get("drnoon_precrop", 0.4)
    drnoon_circle_mask = bool(preprocessing_cfg.get("drnoon_circle_mask", True))

    train_aug = preprocessing_cfg.get("train_augmentation", {})
    hflip_prob = float(train_aug.get("hflip_prob", 0.5))
    vflip_prob = float(train_aug.get("vflip_prob", 0.0))
    rotation_degree = float(train_aug.get("rotation_degree", 15))
    color_jitter_cfg = train_aug.get("color_jitter", {})

    tfms = []
    if use_drnoon:
        tfms.append(
            transforms.Lambda(
                lambda img: fundus_preprocess_drnoon(
                    img,
                    precrop=drnoon_precrop,
                    circle_mask=drnoon_circle_mask,
                )
            )
        )

    tfms.append(transforms.Resize((image_size, image_size), antialias=True))

    if is_train:
        tfms.extend(
            [
                transforms.RandomHorizontalFlip(p=hflip_prob),
                transforms.RandomVerticalFlip(p=vflip_prob),
                transforms.RandomRotation(degrees=rotation_degree),
                transforms.ColorJitter(
                    brightness=float(color_jitter_cfg.get("brightness", 0.15)),
                    contrast=float(color_jitter_cfg.get("contrast", 0.15)),
                    saturation=float(color_jitter_cfg.get("saturation", 0.1)),
                    hue=float(color_jitter_cfg.get("hue", 0.02)),
                ),
            ]
        )

    tfms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(tfms)

