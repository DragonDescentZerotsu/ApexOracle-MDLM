"""Reusable peptide-classifier pooling and historical training profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from .heads import PeptideClassificationHead


@dataclass(frozen=True)
class PeptideClassifierProfile:
    """A named historical protocol without any author-machine paths."""

    name: str
    legacy_source: str
    dataset_label: str
    backbone_variant: str
    pooling: str
    preserve_padding: bool
    positive_rate: float
    default_devices: int
    known_limitation: str | None = None

    @property
    def positive_weight(self) -> float:
        return (1.0 - self.positive_rate) / self.positive_rate


PEPTIDE_CLASSIFIER_PROFILES: Mapping[str, PeptideClassifierProfile] = {
    "v1_noisy_cls": PeptideClassifierProfile(
        name="v1_noisy_cls",
        legacy_source="guaidance_classifier_all_data.py",
        dataset_label="hf_pep_SM_cls_1024",
        backbone_variant="dit",
        pooling="first_token",
        preserve_padding=False,
        positive_rate=0.125,
        default_devices=3,
    ),
    "v1_noisy_non_pad_mean": PeptideClassifierProfile(
        name="v1_noisy_non_pad_mean",
        legacy_source="guaidance_classifier_all_data_non_pad_mean.py",
        dataset_label="hf_pep_SM_cls_1024",
        backbone_variant="dit_non_pad",
        pooling="masked_mean",
        preserve_padding=False,
        positive_rate=0.125,
        default_devices=3,
        known_limitation=(
            "The snapshot validation_step omitted the required attention_mask; "
            "the clean trainer supplies it consistently."
        ),
    ),
    "v1_noisy_padding_preserved_cls": PeptideClassifierProfile(
        name="v1_noisy_padding_preserved_cls",
        legacy_source="guaidance_classifier_all_data_pad_no_mask.py",
        dataset_label="hf_pep_SM_cls_1024",
        backbone_variant="dit",
        pooling="first_token",
        preserve_padding=True,
        positive_rate=0.125,
        default_devices=4,
        known_limitation=(
            "The exact 2025-05 node002 producer was verified in the Core asset "
            "ledger but is not byte-for-byte present in this repository snapshot; "
            "the snapshot source retains the padding algorithm with v2 main settings."
        ),
    ),
    "v2_noisy_padding_preserved_cls": PeptideClassifierProfile(
        name="v2_noisy_padding_preserved_cls",
        legacy_source="guaidance_classifier_all_data_pad_no_mask.py",
        dataset_label="hf_pep_SM_cls_1024_v2",
        backbone_variant="dit",
        pooling="first_token",
        preserve_padding=True,
        positive_rate=0.0008861772425766144,
        default_devices=4,
    ),
}


def get_peptide_classifier_profile(name: str) -> PeptideClassifierProfile:
    try:
        return PEPTIDE_CLASSIFIER_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PEPTIDE_CLASSIFIER_PROFILES))
        raise ValueError(
            f"Unknown peptide-classifier profile {name!r}; use {choices}."
        ) from exc


def masked_mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool valid tokens with explicit empty-sequence validation."""

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, hidden].")
    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError(
            "attention_mask shape must match hidden_states batch/sequence axes."
        )
    mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
    counts = mask.sum(dim=1)
    if bool(counts.eq(0).any()):
        raise ValueError("Cannot mean-pool a sequence with no valid tokens.")
    return (hidden_states * mask).sum(dim=1) / counts


class FrozenEncoderPeptideClassifier(nn.Module):
    """Train a checkpoint-compatible head over an injected frozen encoder.

    The encoder remains injectable because the attributed MDLM runtime owns the
    DiT implementation. Attribute names intentionally remain ``backbone`` and
    ``ClsHead`` so Lightning checkpoints retain the deployed schema.
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        pooling: str = "first_token",
        head: PeptideClassificationHead | None = None,
    ) -> None:
        super().__init__()
        if pooling not in {"first_token", "masked_mean"}:
            raise ValueError("pooling must be 'first_token' or 'masked_mean'.")
        self.backbone = encoder
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.ClsHead = head or PeptideClassificationHead()
        self.pooling = pooling

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.backbone(input_ids, attention_mask)
        if self.pooling == "first_token":
            features = hidden[:, 0, :]
        else:
            if attention_mask is None:
                raise ValueError("masked_mean pooling requires attention_mask.")
            features = masked_mean_pool(hidden, attention_mask)
        return self.ClsHead(features)
