"""Side-effect-free helpers for historical ApexOracle checkpoint payloads."""

from __future__ import annotations

from collections import OrderedDict
from os import PathLike
from typing import Any, Mapping

import torch


def load_torch_file(
    path: str | PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
    weights_only: bool = False,
    mmap: bool | None = None,
) -> Any:
    """Load a torch payload across torch versions with explicit device placement."""

    kwargs: dict[str, Any] = {
        "map_location": map_location,
        "weights_only": weights_only,
    }
    if mmap is not None:
        kwargs["mmap"] = mmap
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        # Older torch versions do not accept ``weights_only`` or ``mmap``.
        return torch.load(path, map_location=map_location)


def extract_state_dict(
    payload: Mapping[str, Any],
    *,
    key: str = "state_dict",
) -> Mapping[str, torch.Tensor]:
    """Return a checkpoint state dict or raise a precise schema error."""

    if not isinstance(payload, Mapping):
        raise TypeError(
            f"Checkpoint payload must be a mapping, got {type(payload).__name__}."
        )
    if key not in payload:
        available = ", ".join(sorted(str(item) for item in payload.keys()))
        raise KeyError(
            f"Checkpoint does not contain {key!r}; available keys: {available or '<none>'}."
        )

    state_dict = payload[key]
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            f"Checkpoint field {key!r} must be a mapping, got {type(state_dict).__name__}."
        )
    return state_dict


def strip_state_dict_prefix(
    state_dict: Mapping[str, torch.Tensor],
    prefix: str = "backbone.",
) -> OrderedDict[str, torch.Tensor]:
    """Copy a state dict while removing ``prefix`` from keys that carry it.

    This preserves the historical ApexOracle behavior used by the repeated
    ``load_DIT`` implementations without mutating the checkpoint payload.
    """

    if not prefix:
        raise ValueError("prefix must be a non-empty string.")

    stripped: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        new_key = key[len(prefix) :] if key.startswith(prefix) else key
        if new_key in stripped:
            raise ValueError(
                f"Removing prefix {prefix!r} creates duplicate state-dict key {new_key!r}."
            )
        stripped[new_key] = value
    return stripped
