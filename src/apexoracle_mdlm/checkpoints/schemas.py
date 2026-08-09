"""Frozen checkpoint schemas used by ApexOracle guided generation.

The validators in this module inspect payload structure only. They do not
instantiate a model or move tensors to a GPU.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch


GENERATION_GUIDANCE_REQUIRED_KEYS = (
    "mdlm_model_state_dict",
    "re_head_state_dict",
    "co_cross_attn_genome",
    "co_cross_attn_text",
    "learnable_embedding_weight",
)

REGRESSION_HEAD_SHAPES = {
    "dense_1.weight": (3072, 12288),
    "dense_1.bias": (3072,),
    "dense_2.weight": (128, 3072),
    "dense_2.bias": (128,),
    "out_proj.weight": (1, 128),
    "out_proj.bias": (1,),
}


def _attention_shapes(condition_dim: int) -> dict[str, tuple[int, ...]]:
    return {
        "mol_to_genome_dim.weight": (condition_dim, 768),
        "mol_to_genome_dim.bias": (condition_dim,),
        "key_value_projection.weight": (condition_dim * 2, condition_dim),
        "key_value_projection.bias": (condition_dim * 2,),
        "mha.in_proj_weight": (condition_dim * 3, condition_dim),
        "mha.in_proj_bias": (condition_dim * 3,),
        "mha.out_proj.weight": (condition_dim, condition_dim),
        "mha.out_proj.bias": (condition_dim,),
        "attn_norm.weight": (condition_dim,),
        "attn_norm.bias": (condition_dim,),
        "norm1.weight": (condition_dim,),
        "norm1.bias": (condition_dim,),
        "ffn.0.weight": (condition_dim, condition_dim),
        "ffn.0.bias": (condition_dim,),
        "ffn.2.weight": (condition_dim, condition_dim),
        "ffn.2.bias": (condition_dim,),
        "norm2.weight": (condition_dim,),
        "norm2.bias": (condition_dim,),
    }


GENOME_ATTENTION_SHAPES = _attention_shapes(8192)
TEXT_ATTENTION_SHAPES = _attention_shapes(4096)

PEPTIDE_CLASSIFIER_HEAD_SHAPES = {
    "ClsHead.dense_1.weight": (384, 768),
    "ClsHead.dense_1.bias": (384,),
    "ClsHead.dense_2.weight": (128, 384),
    "ClsHead.dense_2.bias": (128,),
    "ClsHead.out_proj.weight": (1, 128),
    "ClsHead.out_proj.bias": (1,),
}


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in payload:
        raise KeyError(f"Checkpoint is missing required field {key!r}.")
    value = payload[key]
    if not isinstance(value, Mapping):
        raise TypeError(
            f"Checkpoint field {key!r} must be a mapping, got {type(value).__name__}."
        )
    return value


def _validate_shapes(
    state_dict: Mapping[str, Any],
    expected: Mapping[str, tuple[int, ...]],
    *,
    field: str,
) -> None:
    actual_keys = set(state_dict)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"Checkpoint field {field!r} has incompatible keys; "
            f"missing={missing}, extra={extra}."
        )

    for key, expected_shape in expected.items():
        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Checkpoint tensor {field}.{key} must be torch.Tensor, "
                f"got {type(tensor).__name__}."
            )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Checkpoint tensor {field}.{key} has shape {tuple(tensor.shape)}, "
                f"expected {expected_shape}."
            )


def validate_generation_dlm_checkpoint(
    payload: Mapping[str, Any],
) -> Mapping[str, torch.Tensor]:
    """Validate the Lightning state-dict contract consumed by Generation."""

    state_dict = _require_mapping(payload, "state_dict")
    required_keys = {
        "backbone.vocab_embed.embedding",
        "backbone.sigma_map.mlp.0.weight",
    }
    missing = required_keys - set(state_dict)
    if missing:
        raise ValueError(
            f"DLM state_dict is missing generation keys: {sorted(missing)}."
        )
    if not any(key.startswith("backbone.blocks.0.") for key in state_dict):
        raise ValueError(
            "DLM state_dict does not contain backbone.blocks.0 parameters."
        )
    return state_dict  # type: ignore[return-value]


def validate_generation_mic_guidance_checkpoint(
    payload: Mapping[str, Any],
) -> None:
    """Validate the fixed 768/8192/4096 MIC-guidance checkpoint profile."""

    missing = [key for key in GENERATION_GUIDANCE_REQUIRED_KEYS if key not in payload]
    if missing:
        raise KeyError(f"MIC-guidance checkpoint is missing fields: {missing}.")

    mdlm_state = _require_mapping(payload, "mdlm_model_state_dict")
    if not any(key.startswith("backbone.blocks.0.") for key in mdlm_state):
        raise ValueError("MIC-guidance checkpoint lacks backbone.blocks.0 parameters.")
    _validate_shapes(
        _require_mapping(payload, "re_head_state_dict"),
        REGRESSION_HEAD_SHAPES,
        field="re_head_state_dict",
    )
    _validate_shapes(
        _require_mapping(payload, "co_cross_attn_genome"),
        GENOME_ATTENTION_SHAPES,
        field="co_cross_attn_genome",
    )
    _validate_shapes(
        _require_mapping(payload, "co_cross_attn_text"),
        TEXT_ATTENTION_SHAPES,
        field="co_cross_attn_text",
    )
    learnable = payload["learnable_embedding_weight"]
    if not isinstance(learnable, torch.Tensor) or tuple(learnable.shape) != (1, 8192):
        actual = tuple(learnable.shape) if isinstance(learnable, torch.Tensor) else None
        raise ValueError(
            "learnable_embedding_weight must be a tensor with shape (1, 8192); "
            f"got {actual}."
        )


def validate_generation_peptide_classifier_checkpoint(
    payload: Mapping[str, Any],
) -> None:
    """Validate the Lightning v1 classifier keys remapped by Generation."""

    state_dict = _require_mapping(payload, "state_dict")
    if not any(key.startswith("backbone.backbone.blocks.0.") for key in state_dict):
        raise ValueError(
            "Peptide-classifier state_dict lacks backbone.backbone.blocks.0 parameters."
        )
    selected = {
        key: state_dict[key]
        for key in PEPTIDE_CLASSIFIER_HEAD_SHAPES
        if key in state_dict
    }
    _validate_shapes(
        selected,
        PEPTIDE_CLASSIFIER_HEAD_SHAPES,
        field="state_dict/ClsHead",
    )
