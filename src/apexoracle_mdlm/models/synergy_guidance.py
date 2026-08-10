"""Checkpoint-compatible experimental synergy-guidance training model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from .heads import FirstTokenCrossAttention, RegressionHead


@dataclass(frozen=True)
class SynergyGuidanceProfile:
    """One scientifically distinct historical all-data training protocol."""

    name: str
    legacy_sources: tuple[str, ...]
    first_molecule_noisy: bool
    second_molecule_noisy: bool
    backbone_checkpoint_label: str
    historical_output_label: str
    batch_size: int = 70
    epochs: int = 40


SYNERGY_GUIDANCE_PROFILES: Mapping[str, SynergyGuidanceProfile] = {
    "asymmetric_partner_noise": SynergyGuidanceProfile(
        name="asymmetric_partner_noise",
        legacy_sources=(
            "synergy_Evo_train_new_reg_MDLM_one_base_model_all_data_classification.py",
            "synergy_Evo_train_new_reg_MDLM_one_base_model_all_data_classification_noise.py",
        ),
        first_molecule_noisy=False,
        second_molecule_noisy=True,
        backbone_checkpoint_label="1-255000-fine-tune.ckpt",
        historical_output_label="guidance_noise_synergy/cls",
    ),
    "clean_pair": SynergyGuidanceProfile(
        name="clean_pair",
        legacy_sources=(
            "synergy_Evo_train_new_reg_MDLM_one_base_model_all_data_classification_clean.py",
        ),
        first_molecule_noisy=False,
        second_molecule_noisy=False,
        backbone_checkpoint_label="last_reg_v1.ckpt",
        historical_output_label="synergy_judger/cls",
    ),
}


def get_synergy_guidance_profile(name: str) -> SynergyGuidanceProfile:
    try:
        return SYNERGY_GUIDANCE_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(SYNERGY_GUIDANCE_PROFILES))
        raise ValueError(
            f"Unknown synergy-guidance profile {name!r}; use {choices}."
        ) from exc


def build_lora_condition_attention(
    molecule_dim: int,
    condition_dim: int,
    *,
    num_heads: int = 4,
    attention_dropout: float = 0.1,
    lora_rank: int = 64,
    lora_alpha: int = 32,
) -> nn.Module:
    """Build the exact PEFT attention wrapper stored in guidance checkpoints."""

    from peft import LoraConfig, TaskType, get_peft_model

    base = FirstTokenCrossAttention(
        molecule_dim,
        condition_dim,
        num_heads,
        attention_dropout,
        legacy_squeeze=True,
    )
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "mol_to_genome_dim",
            "key_value_projection",
            "mha.out_proj",
            "ffn.0",
            "ffn.2",
        ],
        task_type=TaskType.FEATURE_EXTRACTION,
        lora_dropout=0.1,
        bias="none",
    )
    return get_peft_model(base, config)


def symmetric_pair_logits(
    head: nn.Module,
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    """Average both molecule orders used by the historical classifier."""

    first_order = head(torch.cat((first, second), dim=1))
    second_order = head(torch.cat((second, first), dim=1))
    return (first_order + second_order) / 2


class SynergyGuidanceClassifier(nn.Module):
    """Trainable DLM/condition classifier for interleaved molecule pairs.

    This model reproduces the experimental all-data producer used by
    ApexOracle-Generation.  It is intentionally separate from the paper's Core
    cross-validation synergy model.
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
        lora_rank: int = 64,
        lora_alpha: int = 32,
    ) -> None:
        super().__init__()
        self.mdlm_model = mdlm_model
        self.co_cross_attn_genome = build_lora_condition_attention(
            molecule_dim,
            genome_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        self.co_cross_attn_text = build_lora_condition_attention(
            molecule_dim,
            text_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        fused_dim = genome_dim + text_dim
        self.reg_head = RegressionHead(
            fused_dim * 2,
            fused_dim // 4,
            128,
            1,
            head_dropout,
        )
        self.learnable_embedding_weight = nn.Parameter(
            torch.randn(1, genome_dim), requires_grad=False
        )

    def initialize_conditions_from_mic_checkpoint(
        self, payload: Mapping[str, Any]
    ) -> None:
        """Load the frozen condition base used before LoRA adapter training."""

        required = (
            "co_cross_attn_genome",
            "co_cross_attn_text",
            "learnable_embedding_weight",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"Base MIC checkpoint is missing fields: {missing}.")
        self._load_pre_lora_attention(
            self.co_cross_attn_genome, payload["co_cross_attn_genome"]
        )
        self._load_pre_lora_attention(
            self.co_cross_attn_text, payload["co_cross_attn_text"]
        )
        learnable = payload["learnable_embedding_weight"]
        if not isinstance(learnable, torch.Tensor):
            raise TypeError("learnable_embedding_weight must be a tensor.")
        if tuple(learnable.shape) != tuple(self.learnable_embedding_weight.shape):
            raise ValueError("learnable_embedding_weight has an incompatible shape.")
        with torch.no_grad():
            self.learnable_embedding_weight.copy_(learnable)

    @staticmethod
    def _load_pre_lora_attention(
        wrapped: nn.Module, state_dict: Mapping[str, torch.Tensor]
    ) -> None:
        """Load a plain MIC attention state after PEFT has wrapped its linears."""

        target_prefixes = (
            "mol_to_genome_dim",
            "key_value_projection",
            "mha.out_proj",
            "ffn.0",
            "ffn.2",
        )
        translated: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            prefix, separator, parameter = str(key).rpartition(".")
            if (
                separator
                and prefix in target_prefixes
                and parameter in {"weight", "bias"}
            ):
                translated[f"{prefix}.base_layer.{parameter}"] = value
            else:
                translated[str(key)] = value
        incompatible = wrapped.base_model.model.load_state_dict(
            translated, strict=False
        )
        unexpected = list(incompatible.unexpected_keys)
        non_lora_missing = [
            key for key in incompatible.missing_keys if ".lora_" not in key
        ]
        if unexpected or non_lora_missing:
            raise RuntimeError(
                "MIC attention state is incompatible with the LoRA wrapper: "
                f"missing={non_lora_missing}, unexpected={unexpected}."
            )

    def _condition(
        self,
        molecule_cls: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_valid_mask: torch.Tensor,
        genome_embeddings: torch.Tensor | None,
        genome_valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if genome_embeddings is None:
            genome_embeddings = self.learnable_embedding_weight[:, None, :].expand(
                molecule_cls.shape[0], -1, -1
            )
            genome_valid_mask = torch.ones(
                molecule_cls.shape[0],
                1,
                dtype=torch.bool,
                device=molecule_cls.device,
            )
        if genome_valid_mask is None:
            raise ValueError("genome_valid_mask is required with genome_embeddings.")
        genome_condition = self.co_cross_attn_genome(
            mol_cls_emb=molecule_cls,
            condition_embeddings=genome_embeddings,
            key_padding_mask=~genome_valid_mask.to(torch.bool),
        )
        text_condition = self.co_cross_attn_text(
            mol_cls_emb=molecule_cls,
            condition_embeddings=text_embeddings,
            key_padding_mask=~text_valid_mask.to(torch.bool),
        )
        return torch.cat(
            (
                genome_condition.reshape(-1, genome_embeddings.shape[-1]),
                text_condition.reshape(-1, text_embeddings.shape[-1]),
            ),
            dim=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_valid_mask: torch.Tensor,
        genome_embeddings: torch.Tensor | None = None,
        genome_valid_mask: torch.Tensor | None = None,
        *,
        first_molecule_noisy: bool,
        second_molecule_noisy: bool,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[0] % 2:
            raise ValueError(
                "input_ids must contain interleaved pairs with shape 2B×L."
            )
        first_hidden = self.mdlm_model(input_ids[::2], apply_noise=first_molecule_noisy)
        second_hidden = self.mdlm_model(
            input_ids[1::2], apply_noise=second_molecule_noisy
        )
        first = self._condition(
            first_hidden[:, 0, :],
            text_embeddings[::2],
            text_valid_mask[::2],
            None if genome_embeddings is None else genome_embeddings[::2],
            None if genome_valid_mask is None else genome_valid_mask[::2],
        )
        second = self._condition(
            second_hidden[:, 0, :],
            text_embeddings[1::2],
            text_valid_mask[1::2],
            None if genome_embeddings is None else genome_embeddings[1::2],
            None if genome_valid_mask is None else genome_valid_mask[1::2],
        )
        return symmetric_pair_logits(self.reg_head, first, second)

    def load_apexoracle_state(self, payload: Mapping[str, Any]) -> None:
        fields = {
            "mdlm_model_state_dict": self.mdlm_model,
            "re_head_state_dict": self.reg_head,
            "co_cross_attn_genome": self.co_cross_attn_genome,
            "co_cross_attn_text": self.co_cross_attn_text,
        }
        missing = [
            key for key in (*fields, "learnable_embedding_weight") if key not in payload
        ]
        if missing:
            raise KeyError(f"Synergy-guidance checkpoint is missing fields: {missing}.")
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
        auroc: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mdlm_model_state_dict": self.mdlm_model.state_dict(),
            "re_head_state_dict": self.reg_head.state_dict(),
            "co_cross_attn_genome": self.co_cross_attn_genome.state_dict(),
            "co_cross_attn_text": self.co_cross_attn_text.state_dict(),
            "learnable_embedding_weight": self.learnable_embedding_weight.detach(),
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if auroc is not None:
            payload["AUROC"] = auroc
        return payload
