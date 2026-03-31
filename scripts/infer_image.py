#!/usr/bin/env python3
"""Run CAC inference for a single fundus image."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-image CAC inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--config", default="configs/inference.yaml", help="Inference YAML config path")
    parser.add_argument("--downstream_checkpoint", default=None, help="Override downstream checkpoint path")
    parser.add_argument("--foundation_checkpoint", default=None, help="Override foundation checkpoint path")
    parser.add_argument("--dinov3_repo", default=None, help="Override DINOv3 repo path")
    parser.add_argument("--threshold", type=float, default=None, help="Override decision threshold")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--disable_drnoon", action="store_true", help="Force disable DrNoon preprocessing")
    parser.add_argument("--output_json", default=None, help="Optional path to save JSON result")
    return parser.parse_args()


def _resolve_path(path_str: str, base_dir: Path) -> str:
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _load_config(config_path: str) -> Dict:
    import yaml

    path = Path(config_path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_dir"] = str(path.parent)
    return cfg


def _select_device(device_arg: str) -> torch.device:
    import torch

    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_autocast_context(device: torch.device, mixed_precision: str):
    import torch

    mp = str(mixed_precision).lower()
    if device.type != "cuda" or mp == "fp32":
        return nullcontext()
    if mp == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def main() -> None:
    args = parse_args()

    import torch
    from PIL import Image
    from cac_inference.model import CACModel
    from cac_inference.utils.checkpoint import load_model_checkpoint
    from cac_inference.utils.preprocessing import build_image_transform

    cfg = _load_config(args.config)
    config_dir = Path(cfg["_config_dir"])

    if args.downstream_checkpoint is not None:
        cfg.setdefault("inference", {})["downstream_checkpoint"] = args.downstream_checkpoint
    if args.foundation_checkpoint is not None:
        cfg.setdefault("model", {}).setdefault("foundation", {})["checkpoint"] = args.foundation_checkpoint
    if args.dinov3_repo is not None:
        cfg.setdefault("model", {}).setdefault("foundation", {})["dinov3_repo"] = args.dinov3_repo
    if args.disable_drnoon:
        cfg.setdefault("data", {}).setdefault("preprocessing", {})["use_drnoon_preprocess"] = False

    downstream_ckpt = cfg.get("inference", {}).get("downstream_checkpoint")
    foundation_ckpt = cfg.get("model", {}).get("foundation", {}).get("checkpoint")
    dinov3_repo = cfg.get("model", {}).get("foundation", {}).get("dinov3_repo")

    if not downstream_ckpt:
        raise ValueError("Missing inference.downstream_checkpoint in config")
    if not foundation_ckpt:
        raise ValueError("Missing model.foundation.checkpoint in config")
    if not dinov3_repo:
        raise ValueError("Missing model.foundation.dinov3_repo in config")

    cfg["inference"]["downstream_checkpoint"] = _resolve_path(downstream_ckpt, config_dir)
    cfg["model"]["foundation"]["checkpoint"] = _resolve_path(foundation_ckpt, config_dir)
    cfg["model"]["foundation"]["dinov3_repo"] = _resolve_path(dinov3_repo, config_dir)

    image_path = Path(args.image).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = _select_device(args.device)
    mixed_precision = cfg.get("training", {}).get("mixed_precision", "bf16")

    model = CACModel(cfg).to(device)
    ckpt_info = load_model_checkpoint(model, cfg["inference"]["downstream_checkpoint"])
    model.eval()

    transform = build_image_transform(cfg, is_train=False)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        with _get_autocast_context(device, mixed_precision):
            outputs = model(image_tensor, image_paths=[str(image_path)])

    logits = outputs["logits"].detach().to(torch.float32)
    prob_positive = float(torch.softmax(logits, dim=1)[:, 1].item())
    ranking_score = float(outputs["ranking_scores"].detach().to(torch.float32).item())
    threshold = float(args.threshold if args.threshold is not None else cfg.get("inference", {}).get("threshold", 0.5))
    pred_binary = int(prob_positive >= threshold)

    result = {
        "image_path": str(image_path),
        "prob_positive": prob_positive,
        "ranking_score": ranking_score,
        "threshold": threshold,
        "pred_binary": pred_binary,
        "pred_label": "CAC>0" if pred_binary == 1 else "CAC=0",
        "device": str(device),
        "downstream_checkpoint": cfg["inference"]["downstream_checkpoint"],
        "foundation_checkpoint": cfg["model"]["foundation"]["checkpoint"],
        "num_missing_keys": len(ckpt_info.get("missing_keys", [])),
        "num_unexpected_keys": len(ckpt_info.get("unexpected_keys", [])),
    }

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)

    if args.output_json:
        output_path = Path(args.output_json)
        if not output_path.is_absolute():
            output_path = (PROJECT_ROOT / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
