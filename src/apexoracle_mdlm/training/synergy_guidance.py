"""Prepared-table data contracts for experimental synergy-guidance training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .mic_guidance import pad_condition_embeddings, parse_token_ids


def fici_to_synergy_label(fici: float | torch.Tensor) -> torch.Tensor:
    """Map the historical strict ``FICI < 0.5`` rule to a binary label."""

    value = torch.as_tensor(fici, dtype=torch.float32)
    if not torch.isfinite(value).all():
        raise ValueError("FICI values must be finite.")
    return (value < 0.5).to(torch.float32)


@dataclass(frozen=True)
class _SynergyRecord:
    first_input_ids: torch.Tensor
    second_input_ids: torch.Tensor
    fici: float
    strain_name: str


class SynergyGuidanceDataset(Dataset[dict[str, Any]]):
    """Prepared molecule-pair rows joined to explicit condition tensors.

    The table must already contain canonical strain keys and stringified token
    lists in ``input_ids_1``/``input_ids_2``.  Raw SMILES conversion and the
    historical project-specific strain cleanup are deliberately outside this
    reusable training contract.
    """

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        text_embeddings: Mapping[str, torch.Tensor],
        genome_embeddings: Mapping[str, torch.Tensor] | None = None,
        require_genome: bool,
        max_molecule_length: int = 512,
    ) -> None:
        if max_molecule_length <= 0:
            raise ValueError("max_molecule_length must be positive.")
        self.text_embeddings = text_embeddings
        self.genome_embeddings = genome_embeddings
        self.require_genome = require_genome
        self.records: list[_SynergyRecord] = []
        for row_number, row in enumerate(rows, start=2):
            try:
                strain = str(row["strain_name"])
                first = parse_token_ids(row["input_ids_1"])
                second = parse_token_ids(row["input_ids_2"])
                fici = float(row["FICI"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid prepared synergy row {row_number}: {exc}"
                ) from exc
            if (
                first.numel() > max_molecule_length
                or second.numel() > max_molecule_length
            ):
                continue
            if not math.isfinite(fici):
                raise ValueError(f"FICI must be finite at row {row_number}.")
            if strain not in text_embeddings:
                raise KeyError(f"Missing text embedding for strain {strain!r}.")
            if require_genome and (
                genome_embeddings is None or strain not in genome_embeddings
            ):
                raise KeyError(f"Missing genome embedding for strain {strain!r}.")
            self.records.append(_SynergyRecord(first, second, fici, strain))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        item: dict[str, Any] = {
            "input_ids_1": record.first_input_ids,
            "input_ids_2": record.second_input_ids,
            "fici": record.fici,
            "strain_name": record.strain_name,
            "text_embedding": self.text_embeddings[record.strain_name],
        }
        if self.require_genome:
            assert self.genome_embeddings is not None
            item["genome_embedding"] = self.genome_embeddings[record.strain_name]
        return item


def partition_synergy_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    text_keys: set[str],
    genome_keys: set[str],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Split prepared rows into the two historical condition streams."""

    genome_text: list[Mapping[str, Any]] = []
    text_only: list[Mapping[str, Any]] = []
    for row_number, row in enumerate(rows, start=2):
        if "strain_name" not in row:
            raise ValueError(f"Prepared synergy row {row_number} lacks strain_name.")
        strain = str(row["strain_name"])
        if strain not in text_keys:
            raise KeyError(f"Missing text embedding for strain {strain!r}.")
        (genome_text if strain in genome_keys else text_only).append(row)
    return genome_text, text_only


def collate_synergy_guidance(
    batch: Sequence[Mapping[str, Any]],
    *,
    pad_token_id: int,
    sequence_length: int = 1024,
) -> dict[str, Any]:
    """Interleave molecule pairs and duplicate each strain condition exactly once."""

    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    token_ids: list[torch.Tensor] = []
    text: list[torch.Tensor] = []
    genome: list[torch.Tensor] = []
    for item in batch:
        token_ids.extend(
            (
                torch.as_tensor(item["input_ids_1"], dtype=torch.long),
                torch.as_tensor(item["input_ids_2"], dtype=torch.long),
            )
        )
        text_embedding = torch.as_tensor(item["text_embedding"])
        text.extend((text_embedding, text_embedding))
        if "genome_embedding" in item:
            genome_embedding = torch.as_tensor(item["genome_embedding"])
            genome.extend((genome_embedding, genome_embedding))
    if any(ids.numel() > sequence_length for ids in token_ids):
        raise ValueError("A token sequence exceeds sequence_length.")
    has_genome = ["genome_embedding" in item for item in batch]
    if any(has_genome) and not all(has_genome):
        raise ValueError("Genome+text and text-only rows must use separate batches.")

    padded = pad_sequence(token_ids, batch_first=True, padding_value=pad_token_id)
    input_ids = torch.full(
        (len(token_ids), sequence_length), pad_token_id, dtype=torch.long
    )
    input_ids[:, : padded.shape[1]] = padded
    text_embeddings, text_valid_mask = pad_condition_embeddings(text)
    result: dict[str, Any] = {
        "input_ids": input_ids,
        "labels": fici_to_synergy_label(
            torch.tensor([float(item["fici"]) for item in batch])
        ),
        "text_embeddings": text_embeddings,
        "text_valid_mask": text_valid_mask,
        "strain_names": [str(item["strain_name"]) for item in batch],
    }
    if all(has_genome):
        genome_embeddings, genome_valid_mask = pad_condition_embeddings(genome)
        result["genome_embeddings"] = genome_embeddings
        result["genome_valid_mask"] = genome_valid_mask
    return result
