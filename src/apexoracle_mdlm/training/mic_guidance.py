"""Prepared-table data pipeline for downstream MIC-guidance training."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def parse_token_ids(value: str | Sequence[int] | torch.Tensor) -> torch.Tensor:
    """Parse one historical stringified token list into a one-dimensional tensor."""

    if isinstance(value, torch.Tensor):
        result = value.detach().to(dtype=torch.long, device="cpu")
    else:
        parsed: Any = ast.literal_eval(value) if isinstance(value, str) else value
        if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes)):
            raise ValueError("SMILES must contain a sequence of integer token IDs.")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in parsed):
            raise ValueError("Every token ID must be an integer.")
        result = torch.tensor(parsed, dtype=torch.long)
    if result.ndim != 1 or result.numel() == 0:
        raise ValueError("Token IDs must be a non-empty one-dimensional sequence.")
    return result


def mic_to_training_target(mic_umol: torch.Tensor) -> torch.Tensor:
    """Apply the exact historical target transform: ``-log10(MIC / 10)``."""

    values = torch.as_tensor(mic_umol, dtype=torch.float32)
    if not torch.isfinite(values).all() or torch.any(values <= 0):
        raise ValueError("MIC values must be finite and positive.")
    return -torch.log10(values / 10.0)


@dataclass(frozen=True)
class _GuidanceRecord:
    input_ids: torch.Tensor
    mic_umol: float
    strain_name: str


class GuidanceMICDataset(Dataset[dict[str, Any]]):
    """Prepared MIC rows joined to explicit text and optional genome tensors.

    Input tables must already use canonical strain keys. Historical strain-name
    cleanup and dataset-specific filtering remain documented provenance steps,
    rather than hidden machine-specific behavior inside the trainer.
    """

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        text_embeddings: Mapping[str, torch.Tensor],
        genome_embeddings: Mapping[str, torch.Tensor] | None = None,
        require_genome: bool,
        max_length: int = 1024,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive.")
        self.text_embeddings = text_embeddings
        self.genome_embeddings = genome_embeddings
        self.require_genome = require_genome
        self.records: list[_GuidanceRecord] = []
        for row_number, row in enumerate(rows, start=2):
            try:
                strain = str(row["strain_name"])
                token_ids = parse_token_ids(row["SMILES"])
                mic = float(row["MIC"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid prepared MIC row {row_number}: {exc}"
                ) from exc
            if token_ids.numel() > max_length:
                continue
            if not math.isfinite(mic) or mic <= 0:
                raise ValueError(
                    f"MIC must be finite and positive at row {row_number}."
                )
            if strain not in text_embeddings:
                raise KeyError(f"Missing text embedding for strain {strain!r}.")
            if require_genome and (
                genome_embeddings is None or strain not in genome_embeddings
            ):
                raise KeyError(f"Missing genome embedding for strain {strain!r}.")
            self.records.append(_GuidanceRecord(token_ids, mic, strain))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        item: dict[str, Any] = {
            "input_ids": record.input_ids,
            "mic_umol": record.mic_umol,
            "strain_name": record.strain_name,
            "text_embedding": self.text_embeddings[record.strain_name],
        }
        if self.require_genome:
            assert self.genome_embeddings is not None
            item["genome_embedding"] = self.genome_embeddings[record.strain_name]
        return item


def partition_guidance_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    text_keys: set[str],
    genome_keys: set[str],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Split prepared rows into genome+text and text-only training streams."""

    genome_text: list[Mapping[str, Any]] = []
    text_only: list[Mapping[str, Any]] = []
    for row_number, row in enumerate(rows, start=2):
        if "strain_name" not in row:
            raise ValueError(f"Prepared MIC row {row_number} lacks strain_name.")
        strain = str(row["strain_name"])
        if strain not in text_keys:
            raise KeyError(f"Missing text embedding for strain {strain!r}.")
        (genome_text if strain in genome_keys else text_only).append(row)
    return genome_text, text_only


def _pad_conditions(
    tensors: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not tensors:
        raise ValueError("Cannot collate an empty condition batch.")
    if any(tensor.ndim != 2 or tensor.shape[0] == 0 for tensor in tensors):
        raise ValueError("Condition tensors must be non-empty and rank-2.")
    widths = {tensor.shape[1] for tensor in tensors}
    if len(widths) != 1:
        raise ValueError("Condition tensors must be rank-2 with one shared width.")
    maximum = max(tensor.shape[0] for tensor in tensors)
    width = tensors[0].shape[1]
    padded = torch.zeros(len(tensors), maximum, width, dtype=torch.bfloat16)
    valid = torch.zeros(len(tensors), maximum, dtype=torch.bool)
    for index, tensor in enumerate(tensors):
        length = tensor.shape[0]
        padded[index, :length] = tensor.to(dtype=torch.bfloat16, device="cpu")
        valid[index, :length] = True
    return padded, valid


def collate_guidance_mic(
    batch: Sequence[Mapping[str, Any]],
    *,
    pad_token_id: int,
    max_length: int = 1024,
) -> dict[str, Any]:
    """Pad one genome+text or text-only batch using the legacy 1024 contract."""

    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    token_ids = [torch.as_tensor(item["input_ids"], dtype=torch.long) for item in batch]
    if any(item.numel() > max_length for item in token_ids):
        raise ValueError("A token sequence exceeds max_length.")
    padded_ids = pad_sequence(token_ids, batch_first=True, padding_value=pad_token_id)
    input_ids = torch.full((len(batch), max_length), pad_token_id, dtype=torch.long)
    input_ids[:, : padded_ids.shape[1]] = padded_ids
    attention_mask = torch.zeros(len(batch), max_length, dtype=torch.bool)
    for index, item in enumerate(token_ids):
        attention_mask[index, : item.numel()] = True
    text_embeddings, text_valid_mask = _pad_conditions(
        [torch.as_tensor(item["text_embedding"]) for item in batch]
    )
    has_genome = ["genome_embedding" in item for item in batch]
    if any(has_genome) and not all(has_genome):
        raise ValueError("Genome+text and text-only rows must use separate batches.")
    result: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": mic_to_training_target(
            torch.tensor([float(item["mic_umol"]) for item in batch])
        ),
        "text_embeddings": text_embeddings,
        "text_valid_mask": text_valid_mask,
        "strain_names": [str(item["strain_name"]) for item in batch],
    }
    if all(has_genome):
        genome_embeddings, genome_valid_mask = _pad_conditions(
            [torch.as_tensor(item["genome_embedding"]) for item in batch]
        )
        result["genome_embeddings"] = genome_embeddings
        result["genome_valid_mask"] = genome_valid_mask
    return result
