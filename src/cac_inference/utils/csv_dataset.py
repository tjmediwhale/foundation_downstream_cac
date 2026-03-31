"""CSV dataset utilities for CAC batch inference."""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .preprocessing import build_image_transform

_DEFAULT_LOCAL_PREFIX = "/nas/mediwhale_processed_data/"


def replace_gs_path(path: str, local_prefix: Optional[str] = None) -> str:
    """Replace ``gs://`` prefix with local mounted path."""
    prefix = (local_prefix or _DEFAULT_LOCAL_PREFIX).rstrip("/") + "/"
    if isinstance(path, str) and path.startswith("gs://"):
        return path.replace("gs://", prefix)
    return path


def _read_csv_with_fallback(csv_path: str, usecols: List[str]) -> pd.DataFrame:
    """Read CSV robustly with fallback parser settings."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.DtypeWarning)
            return pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    except (pd.errors.ParserError, UnicodeDecodeError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.DtypeWarning)
            return pd.read_csv(
                csv_path,
                usecols=usecols,
                engine="python",
                encoding="utf-8",
                on_bad_lines="skip",
                low_memory=False,
            )


def _get_available_columns(csv_path: str) -> List[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.DtypeWarning)
        header = pd.read_csv(csv_path, nrows=0, low_memory=False)
    return list(header.columns)


def load_inference_dataframe(
    csv_path: str,
    image_column: str,
    id_columns: Optional[List[str]] = None,
    target_column: Optional[str] = None,
    local_prefix: Optional[str] = None,
    force_rescan: bool = False,
) -> Tuple[pd.DataFrame, bool]:
    """
    Load and sanitize CSV for inference.

    Returns:
        dataframe, has_target_column
    """
    id_columns = id_columns or []
    available_columns = _get_available_columns(csv_path)
    has_target = bool(target_column) and target_column in available_columns

    usecols = [image_column] + id_columns
    if has_target:
        usecols.append(target_column)  # type: ignore[arg-type]
    usecols = sorted(set(usecols))

    df = _read_csv_with_fallback(csv_path, usecols=usecols)
    if image_column not in df.columns:
        raise KeyError(f"image_column not found: {image_column}")

    df[image_column] = df[image_column].astype(str).map(lambda x: replace_gs_path(x, local_prefix))
    df = df[df[image_column].notna() & (df[image_column].astype(str).str.strip() != "")]

    if has_target and target_column:
        df[target_column] = pd.to_numeric(df[target_column], errors="coerce")

    if force_rescan:
        df = df[df[image_column].map(os.path.exists)]

    df = df.reset_index(drop=True)
    df["__row_id__"] = df.index.astype(int)
    return df, has_target


class CACInferenceDataset(Dataset):
    """PyTorch dataset for CSV-based CAC inference."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_column: str,
        transform,
        id_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        has_target: bool = False,
    ) -> None:
        self.df = dataframe
        self.image_column = image_column
        self.transform = transform
        self.id_columns = id_columns or []
        self.target_column = target_column
        self.has_target = has_target

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.df.iloc[index]
        image_path = str(row[self.image_column])
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)

        sample: Dict[str, object] = {
            "image": tensor,
            "image_path": image_path,
            "row_id": int(row["__row_id__"]),
        }
        for col in self.id_columns:
            sample[col] = "" if pd.isna(row[col]) else str(row[col])

        if self.has_target and self.target_column:
            target_value = row[self.target_column]
            if pd.notna(target_value):
                sample["target_score"] = torch.tensor(float(target_value), dtype=torch.float32)

        return sample


def build_inference_dataloader(
    cfg: Dict,
    csv_path: str,
    batch_size_override: Optional[int] = None,
    num_workers_override: Optional[int] = None,
    force_rescan_override: Optional[bool] = None,
    target_column_override: Optional[str] = None,
):
    """Build DataLoader for CSV inference with same preprocessing policy as training."""
    data_cfg = cfg.get("data", {})
    image_column = data_cfg.get("image_column", "jpg_h1024_path")
    id_columns = data_cfg.get("id_columns", [])
    target_column = target_column_override if target_column_override is not None else data_cfg.get("target_column")
    local_prefix = data_cfg.get("local_prefix", _DEFAULT_LOCAL_PREFIX)
    force_rescan = (
        bool(force_rescan_override)
        if force_rescan_override is not None
        else bool(data_cfg.get("force_rescan", False))
    )

    df, has_target = load_inference_dataframe(
        csv_path=csv_path,
        image_column=image_column,
        id_columns=id_columns,
        target_column=target_column,
        local_prefix=local_prefix,
        force_rescan=force_rescan,
    )

    dataset = CACInferenceDataset(
        dataframe=df,
        image_column=image_column,
        transform=build_image_transform(cfg, is_train=False),
        id_columns=id_columns,
        target_column=target_column,
        has_target=has_target,
    )

    default_batch = int(cfg.get("inference", {}).get("batch_size", cfg.get("training", {}).get("batch_size", 16)))
    batch_size = int(batch_size_override if batch_size_override is not None else default_batch)
    num_workers = int(num_workers_override if num_workers_override is not None else data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    persistent_workers = bool(data_cfg.get("persistent_workers", True) and num_workers > 0)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return loader, df, id_columns, bool(has_target and target_column), target_column

