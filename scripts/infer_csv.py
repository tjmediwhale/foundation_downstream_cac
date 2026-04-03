#!/usr/bin/env python3
"""Run CAC inference for all images listed in a CSV file."""

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
    parser = argparse.ArgumentParser(description="CSV batch CAC inference")
    parser.add_argument("--csv", required=True, help="Path to input CSV")
    parser.add_argument("--output_csv", default="outputs/predictions.csv", help="Path to save prediction CSV")
    parser.add_argument("--config", default="configs/inference.yaml", help="Inference YAML config path")
    parser.add_argument("--downstream_checkpoint", default=None, help="Override downstream checkpoint path")
    parser.add_argument("--foundation_checkpoint", default=None, help="Override foundation checkpoint path")
    parser.add_argument("--dinov3_repo", default=None, help="Override DINOv3 repo path")
    parser.add_argument("--threshold", type=float, default=None, help="Override decision threshold")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size for inference")
    parser.add_argument("--num_workers", type=int, default=None, help="Override DataLoader workers")
    parser.add_argument("--force_rescan", action="store_true", help="Filter rows whose image path does not exist")
    parser.add_argument("--target_column", default=None, help="Optional target column name for reference")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--disable_drnoon", action="store_true", help="Force disable DrNoon preprocessing")
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


def _select_device(device_arg: str):
    import torch

    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_autocast_context(device, mixed_precision: str):
    import torch

    mp = str(mixed_precision).lower()
    if device.type != "cuda" or mp == "fp32":
        return nullcontext()
    if mp == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def main() -> None:
    args = parse_args()

    import pandas as pd
    import torch
    from cac_inference.model import CACModel
    from cac_inference.utils.checkpoint import load_model_checkpoint
    from cac_inference.utils.csv_dataset import build_inference_dataloader

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
    if not dinov3_repo:
        raise ValueError("Missing model.foundation.dinov3_repo in config")

    cfg["inference"]["downstream_checkpoint"] = _resolve_path(downstream_ckpt, config_dir)
    if foundation_ckpt:
        cfg["model"]["foundation"]["checkpoint"] = _resolve_path(foundation_ckpt, config_dir)
    else:
        cfg["model"]["foundation"]["checkpoint"] = None
    cfg["model"]["foundation"]["dinov3_repo"] = _resolve_path(dinov3_repo, config_dir)

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = (PROJECT_ROOT / output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    device = _select_device(args.device)
    mixed_precision = cfg.get("training", {}).get("mixed_precision", "bf16")
    threshold = float(args.threshold if args.threshold is not None else cfg.get("inference", {}).get("threshold", 0.5))

    model = CACModel(cfg).to(device)
    ckpt_info = load_model_checkpoint(model, cfg["inference"]["downstream_checkpoint"])
    model.eval()

    loader, _, id_columns, has_target, target_column = build_inference_dataloader(
        cfg=cfg,
        csv_path=str(csv_path),
        batch_size_override=args.batch_size,
        num_workers_override=args.num_workers,
        force_rescan_override=(True if args.force_rescan else None),
        target_column_override=args.target_column,
    )

    records = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            image_paths = list(batch["image_path"])
            row_ids = batch["row_id"].tolist()

            with _get_autocast_context(device, mixed_precision):
                outputs = model(images, image_paths=image_paths)

            logits = outputs["logits"].detach().to(torch.float32)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            rank_scores = outputs["ranking_scores"].detach().to(torch.float32).cpu().numpy()

            for i, row_id in enumerate(row_ids):
                pred_binary = int(float(probs[i]) >= threshold)
                row = {
                    "row_id": int(row_id),
                    "image_path": image_paths[i],
                    "prob_positive": float(probs[i]),
                    "ranking_score": float(rank_scores[i]),
                    "threshold": threshold,
                    "pred_binary": pred_binary,
                    "pred_label": "CAC>0" if pred_binary == 1 else "CAC=0",
                }

                for col in id_columns:
                    values = batch.get(col)
                    if values is not None:
                        row[col] = values[i]

                if has_target and "target_score" in batch:
                    target_score = float(batch["target_score"][i].item())
                    row["target_score"] = target_score
                    row["target_binary"] = int(target_score > 0)

                records.append(row)

    pred_df = pd.DataFrame(records).sort_values("row_id").drop(columns=["row_id"])
    pred_df.to_csv(output_csv, index=False)

    summary = {
        "csv_path": str(csv_path),
        "output_csv": str(output_csv),
        "num_rows": int(len(pred_df)),
        "threshold": threshold,
        "device": str(device),
        "downstream_checkpoint": cfg["inference"]["downstream_checkpoint"],
        "foundation_checkpoint": cfg["model"]["foundation"].get("checkpoint"),
        "dinov3_repo": cfg["model"]["foundation"]["dinov3_repo"],
        "num_matched_keys": int(ckpt_info.get("matched_keys", 0)),
        "num_missing_keys": len(ckpt_info.get("missing_keys", [])),
        "num_unexpected_keys": len(ckpt_info.get("unexpected_keys", [])),
        "has_target_column": bool(has_target),
        "target_column": target_column if has_target else None,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
