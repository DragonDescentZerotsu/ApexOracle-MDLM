"""Standalone Hugging Face wrapper for ApexOracle DLM hidden states.

This file is also copied verbatim into the model repository as
``DLM_emb_model.py``.  Keep it independent of the ``apexoracle_mdlm`` package:
the only local runtime files required beside it are ``models/dit.py`` and
``noise_schedule.py``.
"""

from __future__ import annotations

from contextlib import nullcontext
import importlib
from os import PathLike
from pathlib import Path
import sys
from typing import Any

from huggingface_hub import PyTorchModelHubMixin
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
from torch.nn import functional as F

try:
    from .masking import normalize_attention_mask
except ImportError:  # Standalone copy in the Hugging Face model repository.
    from masking import normalize_attention_mask


def _runtime_root(runtime_root: str | PathLike[str] | None) -> Path:
    if runtime_root is not None:
        return Path(runtime_root).resolve()
    module_parent = Path(__file__).resolve().parent
    if (module_parent / "models" / "dit.py").is_file():
        return module_parent
    return Path(__file__).resolve().parents[3]


def _load_runtime(runtime_root: str | PathLike[str] | None) -> tuple[Any, Any]:
    root = _runtime_root(runtime_root)
    required = (root / "models" / "dit.py", root / "noise_schedule.py")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The attributed MDLM runtime is incomplete; missing: " + ", ".join(missing)
        )

    for module_name in ("models", "noise_schedule"):
        loaded = sys.modules.get(module_name)
        loaded_path = getattr(loaded, "__file__", None)
        if loaded_path is not None and root not in Path(loaded_path).resolve().parents:
            raise RuntimeError(
                f"A conflicting top-level {module_name!r} module is loaded from "
                f"{loaded_path}; expected the runtime under {root}."
            )

    root_string = str(root)
    inserted = root_string not in sys.path
    if inserted:
        sys.path.insert(0, root_string)
    try:
        dit_module = importlib.import_module("models.dit")
        noise_module = importlib.import_module("noise_schedule")
    finally:
        if inserted:
            sys.path.remove(root_string)
    return dit_module, noise_module


class MolEmbDLM(nn.Module, PyTorchModelHubMixin):
    """Extract clean-input DLM token embeddings.

    The constructor fields intentionally match the existing Hub
    ``config.json`` contract.  ``token_type_ids`` is accepted and ignored so a
    complete tokenizer ``BatchEncoding`` can be passed as ``model(**batch)``.
    """

    def __init__(
        self,
        config: dict[str, Any] | DictConfig,
        vocab_size: int,
        ckpt_path: str | PathLike[str] | None = None,
        mask_index: int = 4,
        runtime_root: str | PathLike[str] | None = None,
    ) -> None:
        super().__init__()
        self.config = (
            config if isinstance(config, DictConfig) else OmegaConf.create(config)
        )
        self.vocab_size = int(vocab_size)
        self.mask_index = int(mask_index)
        self.parameterization = self.config.parameterization
        self.time_conditioning = self.config.time_conditioning
        dit_module, noise_module = _load_runtime(runtime_root)
        self.backbone = dit_module.DIT_non_pad(self.config, self.vocab_size)
        self.noise = noise_module.get_noise(self.config)
        if ckpt_path is not None:
            self._load_legacy_checkpoint(ckpt_path)

    def _load_legacy_checkpoint(self, checkpoint_path: str | PathLike[str]) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload)
        backbone_state = {
            key.removeprefix("backbone."): value
            for key, value in state.items()
            if key.startswith("backbone.")
        }
        incompatible = self.backbone.load_state_dict(backbone_state, strict=False)
        if incompatible.missing_keys:
            raise RuntimeError(
                "The legacy checkpoint is missing DLM backbone tensors: "
                + ", ".join(incompatible.missing_keys)
            )

    def _process_sigma(self, sigma: torch.Tensor | None) -> torch.Tensor | None:
        if sigma is None:
            if self.parameterization != "ar":
                raise AssertionError(
                    "sigma=None is only valid for AR parameterization."
                )
            return None
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        if sigma.ndim != 1:
            raise AssertionError(f"Expected one-dimensional sigma, got {sigma.shape}.")
        return sigma

    @staticmethod
    def _sample_t(n: int, device: torch.device) -> torch.Tensor:
        sampling_eps = 1e-3
        epsilon_t = torch.rand(n, device=device)
        t = (1 - sampling_eps) * epsilon_t + sampling_eps
        return t * 0

    def _forward(
        self,
        token_ids: torch.Tensor,
        sigma: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        processed_sigma = self._process_sigma(sigma)
        hidden = self.backbone.vocab_embed(token_ids)
        conditioning = F.silu(self.backbone.sigma_map(processed_sigma))
        rotary_cos_sin = self.backbone.rotary_emb(hidden)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if hidden.device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            for block in self.backbone.blocks:
                hidden = block(
                    hidden,
                    rotary_cos_sin,
                    conditioning,
                    seqlens=None,
                    attnmask=attention_mask,
                )
        return hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del token_type_ids
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected model inputs: {unknown}")
        mask = normalize_attention_mask(input_ids, attention_mask)
        t = self._sample_t(input_ids.shape[0], input_ids.device)
        sigma, _ = self.noise(t)
        move_chance = 1 - torch.exp(-sigma[:, None])
        move_indices = (
            torch.rand(*input_ids.shape, device=input_ids.device) < move_chance
        )
        clean_or_masked = torch.where(
            move_indices,
            torch.as_tensor(self.mask_index, device=input_ids.device),
            input_ids,
        )
        return self._forward(clean_or_masked, sigma[:, None], mask)
