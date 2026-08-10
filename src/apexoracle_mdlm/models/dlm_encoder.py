"""Downstream hidden-state adapter for the upstream MDLM DiT backbone."""

from __future__ import annotations

from contextlib import nullcontext
import importlib
from os import PathLike
from pathlib import Path
import sys
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F

from apexoracle_mdlm.checkpoints import (
    extract_state_dict,
    load_torch_file,
    strip_state_dict_prefix,
)


BackboneFactory = Callable[[Any, int], nn.Module]
NoiseFactory = Callable[[Any], nn.Module]


class DLMHiddenStateEncoder(nn.Module):
    """Expose the clean ``t=0`` hidden states used by ApexOracle scoring.

    Attribute names intentionally match the historical ``mol_emb_mdlm``
    wrapper so its ``mdlm_model_state_dict`` loads strictly. The adapter keeps
    the historical RNG-consuming zero-time sampler and bfloat16 DiT block
    execution while removing checkpoint and Hydra side effects.
    """

    def __init__(
        self,
        config: Any,
        vocab_size: int,
        *,
        backbone_factory: BackboneFactory,
        noise_factory: NoiseFactory,
        mask_index: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.mask_index = vocab_size - 1 if mask_index is None else mask_index
        self.parameterization = config.parameterization
        self.time_conditioning = config.time_conditioning
        self.backbone = backbone_factory(config, vocab_size)
        self.noise = noise_factory(config)

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

    def _forward(self, token_ids: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
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
                )
        return hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask
        t = self._sample_t(input_ids.shape[0], input_ids.device)
        sigma, _ = self.noise(t)
        move_chance = 1 - torch.exp(-sigma[:, None])
        # Preserve the historical clean-input path exactly: even at t=0 the
        # old wrappers sampled this mask (which is all false).  The random
        # draw matters when a caller deliberately keeps the backbone in train
        # mode and therefore uses dropout after this point.
        move_indices = (
            torch.rand(*input_ids.shape, device=input_ids.device) < move_chance
        )
        clean_or_masked = torch.where(
            move_indices,
            torch.as_tensor(self.mask_index, device=input_ids.device),
            input_ids,
        )
        return self._forward(clean_or_masked, sigma[:, None])

    def load_backbone_checkpoint(
        self,
        checkpoint_path: str | PathLike[str],
    ) -> tuple[list[str], list[str]]:
        """Load the historical Lightning ``backbone.*`` state into the DiT.

        Historical DLM+MTR checkpoints also carry output/regression modules
        that the hidden-state adapter intentionally does not instantiate.
        Their keys are reported as unexpected instead of silently changing
        the encoder architecture.
        """

        payload = load_torch_file(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = strip_state_dict_prefix(extract_state_dict(payload))
        incompatible = self.backbone.load_state_dict(state_dict, strict=False)
        return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def build_upstream_dlm_hidden_state_encoder(
    config: Any,
    vocab_size: int,
    *,
    runtime_root: str | PathLike[str] | None = None,
) -> DLMHiddenStateEncoder:
    """Build the adapter from this checkout's attributed upstream runtime."""

    root = (
        Path(runtime_root).resolve()
        if runtime_root is not None
        else Path(__file__).resolve().parents[3]
    )
    required = (root / "models" / "dit.py", root / "noise_schedule.py")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Attributed upstream MDLM runtime is incomplete; missing: "
            + ", ".join(missing)
        )
    for module_name in ("models", "noise_schedule"):
        loaded_module = sys.modules.get(module_name)
        loaded_path = getattr(loaded_module, "__file__", None)
        if loaded_path is not None and root not in Path(loaded_path).resolve().parents:
            raise RuntimeError(
                f"A conflicting top-level {module_name!r} module is already imported "
                f"from {loaded_path}; expected the attributed runtime under {root}."
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

    return DLMHiddenStateEncoder(
        config,
        vocab_size,
        backbone_factory=lambda cfg, size: dit_module.DIT(cfg, vocab_size=size),
        noise_factory=noise_module.get_noise,
    )
