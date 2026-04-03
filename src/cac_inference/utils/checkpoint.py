"""Checkpoint loading utilities for standalone inference."""

from __future__ import annotations

from typing import Dict

import torch


def _extract_state_dict(payload: Dict) -> Dict[str, torch.Tensor]:
    for key in ("model_state_dict", "model", "state_dict"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    if all(isinstance(k, str) for k in payload.keys()):
        return payload
    raise ValueError("Checkpoint does not contain a model state dict")


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    prefixes = ("module.", "_fsdp_wrapped_module.")
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def _count_matching_keys(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> int:
    model_keys = set(model.state_dict().keys())
    return sum(1 for key in state_dict.keys() if key in model_keys)


def _normalize_state_dict_keys(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    stripped = _strip_module_prefix(state_dict)
    if _count_matching_keys(model, stripped) >= _count_matching_keys(model, state_dict):
        return stripped
    return state_dict


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> Dict:
    """Load checkpoint into model and return metadata."""
    payload = torch.load(checkpoint_path, map_location="cpu")
    payload_dict = payload if isinstance(payload, dict) else {"state_dict": payload}
    state_dict = _extract_state_dict(payload_dict)
    state_dict = _normalize_state_dict_keys(model, state_dict)
    num_matched = _count_matching_keys(model, state_dict)
    if num_matched == 0:
        raise RuntimeError(
            "No checkpoint keys matched the inference model. "
            f"checkpoint={checkpoint_path}"
        )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    return {
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "matched_keys": num_matched,
    }
