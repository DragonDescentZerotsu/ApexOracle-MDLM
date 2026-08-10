"""Input-boundary validation shared by package and standalone Hub code."""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import torch


def normalize_attention_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Return the boolean, non-empty mask required by FlashAttention.

    ``transformers`` tokenizers normally return an integer attention mask.
    Indexing the upstream non-padding DiT with that integer tensor selects
    rows rather than filtering them. Converting at this public boundary is
    therefore part of the model's correctness contract, not just a dtype
    convenience.
    """

    if input_ids.ndim != 2:
        raise ValueError(
            f"input_ids must have shape [batch, sequence], got {input_ids.shape}."
        )
    if attention_mask is None:
        mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape as input_ids; "
                f"got {attention_mask.shape} and {input_ids.shape}."
            )
        mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
    if not torch.all(mask.any(dim=1)):
        raise ValueError("Every input row must contain at least one unmasked token.")
    return mask


def resolve_runtime_root(
    module_file: str | PathLike[str],
    runtime_root: str | PathLike[str] | None = None,
) -> Path:
    """Locate sibling runtime files without dereferencing Hub-cache symlinks."""

    if runtime_root is not None:
        return Path(runtime_root).resolve()
    module_parent = Path(module_file).absolute().parent
    if (module_parent / "models" / "dit.py").is_file():
        return module_parent
    return Path(module_file).resolve().parents[3]
