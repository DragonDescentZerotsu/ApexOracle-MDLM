"""Checkpoint-compatible MIC-guidance training model and legacy profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from .heads import FirstTokenCrossAttention, RegressionHead


@dataclass(frozen=True)
class MICGuidanceProfile:
    """One scientifically distinct historical all-data guidance protocol."""

    name: str
    legacy_sources: tuple[str, ...]
    backbone_variant: str
    sampling: str
    preserve_padding: bool
    encoder_mode: str
    batch_size: int
    epochs: int
    backbone_checkpoint_label: str
    historical_output_label: str


MIC_GUIDANCE_PROFILES: Mapping[str, MICGuidanceProfile] = {
    "noisy_standard": MICGuidanceProfile(
        name="noisy_standard",
        legacy_sources=("guaidance_regressor_all_data.py",),
        backbone_variant="dit",
        sampling="random_time",
        preserve_padding=False,
        encoder_mode="train",
        batch_size=70,
        epochs=100,
        backbone_checkpoint_label="last_reg_v1.ckpt",
        historical_output_label="guidance_regressor",
    ),
    "noisy_padding_preserved": MICGuidanceProfile(
        name="noisy_padding_preserved",
        legacy_sources=("guaidance_regressor_all_data_pad_no_mask.py",),
        backbone_variant="dit",
        sampling="random_time",
        preserve_padding=True,
        encoder_mode="train",
        batch_size=70,
        epochs=100,
        backbone_checkpoint_label="last_reg_v1.ckpt",
        historical_output_label="guidance_regressor_pad_no_mask",
    ),
    "noisy_non_pad": MICGuidanceProfile(
        name="noisy_non_pad",
        legacy_sources=(
            "guaidance_regressor_all_data_non_pad.py",
            "guaidance_regressor_all_data_non_pad_cls.py",
        ),
        backbone_variant="dit_non_pad",
        sampling="random_time",
        preserve_padding=False,
        encoder_mode="train",
        batch_size=70,
        epochs=200,
        backbone_checkpoint_label="1-255000-fine-tune.ckpt",
        historical_output_label="guidance_regressor_non_pad",
    ),
    "clean_non_pad": MICGuidanceProfile(
        name="clean_non_pad",
        legacy_sources=("guaidance_regressor_all_data_non_pad_cls_clean.py",),
        backbone_variant="dit_non_pad",
        # The legacy code multiplied the random draw by zero before adding
        # sampling_eps, so this is fixed t=1e-3 rather than exact t=0.
        sampling="fixed_epsilon",
        preserve_padding=False,
        encoder_mode="train",
        batch_size=30,
        epochs=13,
        backbone_checkpoint_label="1-255000-fine-tune.ckpt",
        historical_output_label="guidance_regressor_non_pad_clean",
    ),
    "noisy_non_pad_eval": MICGuidanceProfile(
        name="noisy_non_pad_eval",
        legacy_sources=("guaidance_regressor_all_data_non_pad_cls_noise.py",),
        backbone_variant="dit_non_pad",
        sampling="random_time",
        preserve_padding=False,
        encoder_mode="eval",
        batch_size=70,
        epochs=200,
        backbone_checkpoint_label="1-255000-fine-tune.ckpt",
        historical_output_label="guidance_regressor_non_pad_noise",
    ),
}


def get_mic_guidance_profile(name: str) -> MICGuidanceProfile:
    try:
        return MIC_GUIDANCE_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(MIC_GUIDANCE_PROFILES))
        raise ValueError(
            f"Unknown MIC-guidance profile {name!r}; use {choices}."
        ) from exc


class MICGuidanceRegressor(nn.Module):
    """DLM plus genome/text attention used to produce guidance checkpoints.

    Attribute and checkpoint field names intentionally match the historical
    all-data trainers and ApexOracle-Generation consumers.
    """

    def __init__(
        self,
        mdlm_model: nn.Module,
        *,
        molecule_dim: int = 768,
        genome_dim: int = 8192,
        text_dim: int = 4096,
        num_heads: int = 4,
        attention_dropout: float = 0.1,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.mdlm_model = mdlm_model
        self.co_cross_attn_genome = FirstTokenCrossAttention(
            molecule_dim,
            genome_dim,
            num_heads,
            attention_dropout,
            legacy_squeeze=True,
        )
        self.co_cross_attn_text = FirstTokenCrossAttention(
            molecule_dim,
            text_dim,
            num_heads,
            attention_dropout,
            legacy_squeeze=True,
        )
        fused_dim = genome_dim + text_dim
        self.reg_head = RegressionHead(fused_dim, fused_dim // 4, 128, 1, head_dropout)
        # The historical trainers saved this head even though the active
        # small-molecule classification batches were commented out.
        self.cls_head = RegressionHead(fused_dim, fused_dim // 4, 128, 1, head_dropout)
        self.learnable_embedding_weight = nn.Parameter(torch.randn(1, genome_dim))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_valid_mask: torch.Tensor,
        genome_embeddings: torch.Tensor | None = None,
        genome_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.mdlm_model(input_ids, attention_mask)
        molecule_cls = hidden[:, 0, :]
        if genome_embeddings is None:
            genome_embeddings = self.learnable_embedding_weight[:, None, :].expand(
                input_ids.shape[0], -1, -1
            )
            genome_valid_mask = torch.ones(
                input_ids.shape[0], 1, dtype=torch.bool, device=input_ids.device
            )
        if genome_valid_mask is None:
            raise ValueError("genome_valid_mask is required with genome_embeddings.")
        genome_condition = self.co_cross_attn_genome(
            molecule_cls,
            genome_embeddings,
            ~genome_valid_mask.to(torch.bool),
        )
        text_condition = self.co_cross_attn_text(
            molecule_cls,
            text_embeddings,
            ~text_valid_mask.to(torch.bool),
        )
        fused = torch.cat(
            (
                genome_condition.reshape(-1, genome_embeddings.shape[-1]),
                text_condition.reshape(-1, text_embeddings.shape[-1]),
            ),
            dim=1,
        )
        return self.reg_head(fused), self.cls_head(fused)

    def load_apexoracle_state(self, payload: Mapping[str, Any]) -> None:
        fields = {
            "mdlm_model_state_dict": self.mdlm_model,
            "re_head_state_dict": self.reg_head,
            "cls_head_state_dict": self.cls_head,
            "co_cross_attn_genome": self.co_cross_attn_genome,
            "co_cross_attn_text": self.co_cross_attn_text,
        }
        missing = [
            key for key in (*fields, "learnable_embedding_weight") if key not in payload
        ]
        if missing:
            raise KeyError(f"MIC-guidance checkpoint is missing fields: {missing}.")
        for key, module in fields.items():
            module.load_state_dict(payload[key], strict=True)
        learnable = payload["learnable_embedding_weight"]
        if not isinstance(learnable, torch.Tensor):
            raise TypeError("learnable_embedding_weight must be a tensor.")
        if tuple(learnable.shape) != tuple(self.learnable_embedding_weight.shape):
            raise ValueError("learnable_embedding_weight has an incompatible shape.")
        with torch.no_grad():
            self.learnable_embedding_weight.copy_(learnable)

    def checkpoint_payload(
        self,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        r2: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mdlm_model_state_dict": self.mdlm_model.state_dict(),
            "re_head_state_dict": self.reg_head.state_dict(),
            "cls_head_state_dict": self.cls_head.state_dict(),
            "co_cross_attn_genome": self.co_cross_attn_genome.state_dict(),
            "co_cross_attn_text": self.co_cross_attn_text.state_dict(),
            "learnable_embedding_weight": self.learnable_embedding_weight.detach(),
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if r2 is not None:
            payload["R2"] = r2
        return payload
