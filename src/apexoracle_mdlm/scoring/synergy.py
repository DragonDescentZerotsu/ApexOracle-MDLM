"""Experimental symmetric-pair synergy scoring for generated candidates."""

from __future__ import annotations

from contextlib import nullcontext
from os import PathLike
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from apexoracle_mdlm.checkpoints import (
    load_torch_file,
    validate_generation_synergy_guidance_checkpoint,
)
from apexoracle_mdlm.models import (
    RegressionHead,
    build_lora_condition_attention,
    build_upstream_dlm_hidden_state_encoder,
    symmetric_pair_logits,
)

from .mic import ConditionEmbeddingBanks, Tokenizer, normalize_selfies_for_tokenizer


def load_partner_embedding(
    path: str | PathLike[str],
    key: str | int,
) -> torch.Tensor:
    """Load one explicit partner key from the historical embedding mapping."""

    payload = load_torch_file(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, Mapping):
        raise TypeError("Partner embedding file must contain a mapping.")
    if key not in payload:
        raise KeyError(f"Partner embedding key {key!r} is absent.")
    embedding = payload[key]
    if not isinstance(embedding, torch.Tensor):
        raise TypeError(f"Partner embedding {key!r} must be a torch.Tensor.")
    return embedding


class CandidateSynergyClassifier(nn.Module):
    """DLM + strain condition classifier for one fixed partner molecule.

    This is the historical all-data/Generation guidance profile, not the Core
    cross-validation model used for the paper's synergy benchmark.
    """

    def __init__(
        self,
        mdlm_model: nn.Module,
        condition_embeddings: ConditionEmbeddingBanks,
        partner_embedding: torch.Tensor,
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
        normalized_partner = partner_embedding.reshape(-1, molecule_dim)
        if tuple(normalized_partner.shape) != (1, molecule_dim):
            raise ValueError(
                "partner_embedding must contain exactly one molecule vector with "
                f"dimension {molecule_dim}; got {tuple(partner_embedding.shape)}."
            )
        self.mdlm_model = mdlm_model
        self.condition_embeddings = condition_embeddings
        self.register_buffer(
            "partner_embedding", normalized_partner.detach().clone(), persistent=False
        )
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
        self.learnable_embedding_weight = nn.Parameter(torch.randn(1, genome_dim))

    def load_apexoracle_state(self, payload: Mapping[str, Any]) -> None:
        """Validate and strictly load the five historical checkpoint fields."""

        validate_generation_synergy_guidance_checkpoint(payload)
        self.mdlm_model.load_state_dict(payload["mdlm_model_state_dict"], strict=True)
        self.reg_head.load_state_dict(payload["re_head_state_dict"], strict=True)
        self.co_cross_attn_genome.load_state_dict(
            payload["co_cross_attn_genome"], strict=True
        )
        self.co_cross_attn_text.load_state_dict(
            payload["co_cross_attn_text"], strict=True
        )
        with torch.no_grad():
            self.learnable_embedding_weight.copy_(payload["learnable_embedding_weight"])

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

    def encode_molecules(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.mdlm_model(input_ids)[:, 0, :]

    def _condition_molecule(
        self,
        molecule: torch.Tensor,
        genome: torch.Tensor,
        text: torch.Tensor,
        genome_mask: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        genome_condition = self.co_cross_attn_genome(
            mol_cls_emb=molecule,
            condition_embeddings=genome,
            key_padding_mask=genome_mask,
        )
        text_condition = self.co_cross_attn_text(
            mol_cls_emb=molecule,
            condition_embeddings=text,
            key_padding_mask=text_mask,
        )
        return torch.cat(
            (
                genome_condition.reshape(-1, genome.shape[-1]),
                text_condition.reshape(-1, text.shape[-1]),
            ),
            dim=1,
        )

    def predict_from_cls_embedding(
        self,
        molecule_cls: torch.Tensor,
        strain: str,
    ) -> torch.Tensor:
        genome, text = self._conditions(
            strain,
            batch_size=molecule_cls.shape[0],
            device=molecule_cls.device,
        )
        partner = self.partner_embedding.expand_as(molecule_cls)
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
            candidate_condition = self._condition_molecule(
                molecule_cls, genome, text, genome_mask, text_mask
            )
            partner_condition = self._condition_molecule(
                partner, genome, text, genome_mask, text_mask
            )
            return symmetric_pair_logits(
                self.reg_head, candidate_condition, partner_condition
            )

    def forward(self, input_ids: torch.Tensor, strain: str) -> torch.Tensor:
        return self.predict_from_cls_embedding(self.encode_molecules(input_ids), strain)


def build_candidate_synergy_classifier(
    config: Any,
    *,
    vocab_size: int,
    condition_embeddings: ConditionEmbeddingBanks,
    partner_embedding: torch.Tensor,
    runtime_root: str | PathLike[str] | None = None,
) -> CandidateSynergyClassifier:
    encoder = build_upstream_dlm_hidden_state_encoder(
        config, vocab_size, runtime_root=runtime_root
    )
    return CandidateSynergyClassifier(encoder, condition_embeddings, partner_embedding)


def load_candidate_synergy_classifier(
    config: Any,
    *,
    vocab_size: int,
    condition_embeddings: ConditionEmbeddingBanks,
    partner_embedding: torch.Tensor,
    checkpoint_path: str | PathLike[str],
    device: str | torch.device,
    runtime_root: str | PathLike[str] | None = None,
) -> CandidateSynergyClassifier:
    model = build_candidate_synergy_classifier(
        config,
        vocab_size=vocab_size,
        condition_embeddings=condition_embeddings,
        partner_embedding=partner_embedding,
        runtime_root=runtime_root,
    )
    payload = load_torch_file(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    if not isinstance(payload, Mapping):
        raise TypeError("Synergy checkpoint payload must be a mapping.")
    model.load_apexoracle_state(payload)
    return model.to(device).eval()


@torch.inference_mode()
def score_selfies_synergy(
    model: CandidateSynergyClassifier,
    tokenizer: Tokenizer,
    selfies_strings: Sequence[str],
    *,
    strain: str,
    device: str | torch.device,
) -> torch.Tensor:
    """Return historical sigmoid probabilities in one-molecule batches."""

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
    target_device = torch.device(device)
    probabilities = []
    for row in input_ids:
        unpadded = row[row != tokenizer.pad_token_id].unsqueeze(0).to(target_device)
        probability = torch.sigmoid(model(unpadded, strain)).squeeze()
        probabilities.append(probability.detach().cpu().to(torch.float32))
    return torch.stack(probabilities)
