"""Condition-aware MIC scoring without Hydra, path, or plotting side effects."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import torch
from torch import nn

from apexoracle_mdlm.checkpoints import load_torch_file
from apexoracle_mdlm.embeddings import load_atcc_embeddings, load_text_embeddings
from apexoracle_mdlm.models import (
    FirstTokenCrossAttention,
    RegressionHead,
    build_upstream_dlm_hidden_state_encoder,
)


class Tokenizer(Protocol):
    pad_token_id: int

    def __call__(
        self, text: Sequence[str], **kwargs: Any
    ) -> Mapping[str, torch.Tensor]: ...


@dataclass(frozen=True)
class ConditionEmbeddingBanks:
    """Genome, matched ATCC text, and text-only condition tensors."""

    genomes: Mapping[str, torch.Tensor]
    atcc_text: Mapping[str, torch.Tensor]
    text_only: Mapping[str, torch.Tensor]


def load_condition_embedding_banks(
    *,
    genome_directory: str | PathLike[str],
    atcc_text_directory: str | PathLike[str],
    text_only_directory: str | PathLike[str],
    genome_scale: float = 1e14,
) -> ConditionEmbeddingBanks:
    """Load the three frozen embedding banks onto CPU."""

    return ConditionEmbeddingBanks(
        genomes=load_atcc_embeddings(genome_directory, scale=genome_scale),
        atcc_text=load_atcc_embeddings(atcc_text_directory),
        text_only=load_text_embeddings(text_only_directory),
    )


class CandidateMICRegressor(nn.Module):
    """Checkpoint-compatible DLM + genome/text condition MIC regressor."""

    def __init__(
        self,
        mdlm_model: nn.Module,
        condition_embeddings: ConditionEmbeddingBanks,
        *,
        molecule_dim: int = 768,
        genome_dim: int = 8192,
        text_dim: int = 4096,
        num_heads: int = 4,
        attention_dropout: float = 0.1,
        head_dropout: float = 0.2,
        legacy_squeeze: bool = True,
    ) -> None:
        super().__init__()
        self.mdlm_model = mdlm_model
        self.condition_embeddings = condition_embeddings
        self.co_cross_attn_genome = FirstTokenCrossAttention(
            molecule_dim,
            genome_dim,
            num_heads,
            attention_dropout,
            legacy_squeeze=legacy_squeeze,
        )
        self.co_cross_attn_text = FirstTokenCrossAttention(
            molecule_dim,
            text_dim,
            num_heads,
            attention_dropout,
            legacy_squeeze=legacy_squeeze,
        )
        fused_dim = genome_dim + text_dim
        self.reg_head = RegressionHead(
            fused_dim,
            fused_dim // 4,
            128,
            1,
            head_dropout,
        )
        self.learnable_embedding_weight = nn.Parameter(torch.randn(1, genome_dim))

    def load_apexoracle_state(self, payload: Mapping[str, Any]) -> None:
        """Strictly load the five fields used by the historical scorer."""

        required = (
            "mdlm_model_state_dict",
            "re_head_state_dict",
            "co_cross_attn_genome",
            "co_cross_attn_text",
            "learnable_embedding_weight",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"MIC checkpoint is missing fields: {missing}.")
        self.mdlm_model.load_state_dict(payload["mdlm_model_state_dict"], strict=True)
        self.reg_head.load_state_dict(payload["re_head_state_dict"], strict=True)
        self.co_cross_attn_genome.load_state_dict(
            payload["co_cross_attn_genome"], strict=True
        )
        self.co_cross_attn_text.load_state_dict(
            payload["co_cross_attn_text"], strict=True
        )
        learnable = payload["learnable_embedding_weight"]
        if not isinstance(learnable, torch.Tensor):
            raise TypeError("learnable_embedding_weight must be a torch.Tensor.")
        if tuple(learnable.shape) != tuple(self.learnable_embedding_weight.shape):
            raise ValueError(
                "learnable_embedding_weight shape mismatch: "
                f"expected {tuple(self.learnable_embedding_weight.shape)}, "
                f"got {tuple(learnable.shape)}."
            )
        with torch.no_grad():
            self.learnable_embedding_weight.copy_(learnable)

    def _conditions(
        self,
        strain: str,
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        banks = self.condition_embeddings
        if strain in banks.genomes:
            if strain not in banks.atcc_text:
                raise KeyError(
                    f"Strain {strain!r} has a genome but no matched ATCC text."
                )
            genome = banks.genomes[strain].to(device)
            text = banks.atcc_text[strain].to(device)
            genome = genome.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            if strain not in banks.text_only:
                raise KeyError(
                    f"No genome or text-only condition exists for {strain!r}."
                )
            genome = self.learnable_embedding_weight[:, None, :].expand(
                batch_size, -1, -1
            )
            text = banks.text_only[strain].to(device)
        text = text.unsqueeze(0).expand(batch_size, -1, -1)
        return genome, text

    def forward(self, input_ids: torch.Tensor, strain: str) -> torch.Tensor:
        molecule_hidden = self.mdlm_model(input_ids)
        molecule_cls = molecule_hidden[:, 0, :]
        genome, text = self._conditions(
            strain,
            batch_size=molecule_cls.shape[0],
            device=molecule_cls.device,
        )
        genome_mask = torch.zeros(
            genome.shape[:2], dtype=torch.bool, device=molecule_cls.device
        )
        text_mask = torch.zeros(
            text.shape[:2], dtype=torch.bool, device=molecule_cls.device
        )
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if molecule_cls.device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            genome_condition = self.co_cross_attn_genome(
                molecule_cls, genome, genome_mask
            )
            text_condition = self.co_cross_attn_text(molecule_cls, text, text_mask)
            fused = torch.cat(
                (
                    genome_condition.reshape(-1, genome.shape[-1]),
                    text_condition.reshape(-1, text.shape[-1]),
                ),
                dim=1,
            )
            return self.reg_head(fused)


def build_candidate_mic_regressor(
    config: Any,
    *,
    vocab_size: int,
    condition_embeddings: ConditionEmbeddingBanks,
) -> CandidateMICRegressor:
    """Build the formal 768/8192/4096 scorer from the upstream runtime."""

    encoder = build_upstream_dlm_hidden_state_encoder(config, vocab_size)
    return CandidateMICRegressor(encoder, condition_embeddings)


def load_candidate_mic_regressor(
    config: Any,
    *,
    vocab_size: int,
    condition_embeddings: ConditionEmbeddingBanks,
    checkpoint_path: str | PathLike[str],
    device: str | torch.device,
) -> CandidateMICRegressor:
    """Build and strictly load a candidate MIC scorer on an explicit device."""

    model = build_candidate_mic_regressor(
        config,
        vocab_size=vocab_size,
        condition_embeddings=condition_embeddings,
    )
    payload = load_torch_file(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(payload, Mapping):
        raise TypeError("MIC checkpoint payload must be a mapping.")
    model.load_apexoracle_state(payload)
    return model.to(device).eval()


def normalize_selfies_for_tokenizer(selfies: str) -> str:
    """Insert the spaces expected by the frozen SELFIES-TED tokenizer."""

    return selfies.replace("][", "] [")


def regression_logit_to_mic(logits: torch.Tensor) -> torch.Tensor:
    """Invert the historical target transform: MIC = 10**(-logit) * 10."""

    return torch.pow(10.0, -logits) * 10.0


@torch.inference_mode()
def score_selfies_strings(
    model: CandidateMICRegressor,
    tokenizer: Tokenizer,
    selfies_strings: Sequence[str],
    *,
    strain: str,
    device: str | torch.device,
) -> torch.Tensor:
    """Score SELFIES in the exact historical one-molecule batches."""

    if not selfies_strings:
        return torch.empty(0, dtype=torch.float32)
    encoded = tokenizer(
        [normalize_selfies_for_tokenizer(item) for item in selfies_strings],
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"]
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    predictions: list[torch.Tensor] = []
    target_device = torch.device(device)
    for row in input_ids:
        unpadded = row[row != tokenizer.pad_token_id].unsqueeze(0).to(target_device)
        mic = regression_logit_to_mic(model(unpadded, strain)).squeeze()
        predictions.append(mic.detach().cpu().to(torch.float32))
    return torch.stack(predictions)


def read_selfies_file(path: str | PathLike[str]) -> list[str]:
    """Read one SELFIES string per line without silently dropping empty rows."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return source.read_text(encoding="utf-8").splitlines()
