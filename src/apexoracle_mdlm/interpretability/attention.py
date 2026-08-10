"""Genome-window and cross-attention contracts used by the paper case study."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any, Sequence

import torch

from apexoracle_mdlm.scoring.mic import (
    CandidateMICRegressor,
    MICAttentionOutput,
    Tokenizer,
    normalize_selfies_for_tokenizer,
)


@dataclass(frozen=True)
class GenomeWindow:
    fragment_index: int
    contig_index: int
    start: int
    end: int


@dataclass(frozen=True)
class VerifiedGenomeAssets:
    """Sequence-matched FASTA/GenBank records and saved-tensor windows."""

    fasta_records: tuple[Any, ...]
    genbank_records: tuple[Any, ...]
    windows: tuple[GenomeWindow, ...]
    embedding_shape: tuple[int, ...]


def build_saved_tensor_windows(
    record_lengths: Sequence[int],
    *,
    window_length: int = 11_000,
    step: int = 10_000,
) -> list[GenomeWindow]:
    """Reconstruct the historical saved-tensor indexing exactly.

    ``fragment_index`` intentionally does not reset between FASTA records.
    This is a compatibility contract for existing Evo-2 tensors, not a new
    recommendation for multi-contig embedding producers.
    """

    if window_length <= 0 or step <= 0:
        raise ValueError("window_length and step must be positive.")
    windows: list[GenomeWindow] = []
    fragment_index = 0
    for record_index, raw_length in enumerate(record_lengths):
        record_length = int(raw_length)
        if record_length < 0:
            raise ValueError("record lengths cannot be negative.")
        while fragment_index * step < record_length:
            start = fragment_index * step
            windows.append(
                GenomeWindow(
                    fragment_index=fragment_index,
                    contig_index=record_index,
                    start=start,
                    end=min(start + window_length, record_length),
                )
            )
            fragment_index += 1
    return windows


def load_verified_genome_assets(
    *,
    fasta_path: str | PathLike[str],
    genbank_path: str | PathLike[str],
    embedding_path: str | PathLike[str],
) -> VerifiedGenomeAssets:
    """Require exact FASTA/GenBank sequence order and saved tensor row count."""

    from Bio import SeqIO

    fasta_records = tuple(SeqIO.parse(fasta_path, "fasta"))
    genbank_records = tuple(SeqIO.parse(genbank_path, "genbank"))
    if not fasta_records:
        raise ValueError("Genome FASTA contains no records.")
    if len(fasta_records) != len(genbank_records):
        raise ValueError("FASTA and GenBank record counts differ.")
    for index, (fasta_record, genbank_record) in enumerate(
        zip(fasta_records, genbank_records)
    ):
        if str(fasta_record.seq).upper() != str(genbank_record.seq).upper():
            raise ValueError(f"FASTA/GenBank sequence mismatch at record {index}.")

    embedding = torch.load(
        embedding_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2:
        raise TypeError("Genome embedding must be a rank-two tensor.")
    windows = build_saved_tensor_windows([len(record.seq) for record in fasta_records])
    if len(windows) != int(embedding.shape[0]):
        raise ValueError(
            "Saved-tensor row count does not match reconstructed windows: "
            f"{embedding.shape[0]} != {len(windows)}."
        )
    return VerifiedGenomeAssets(
        fasta_records=fasta_records,
        genbank_records=genbank_records,
        windows=tuple(windows),
        embedding_shape=tuple(embedding.shape),
    )


def _one_attention_vector(attention: torch.Tensor) -> torch.Tensor:
    values = attention.detach().cpu().to(torch.float32)
    while values.ndim > 1 and values.shape[0] == 1:
        values = values.squeeze(0)
    if values.ndim != 1:
        raise ValueError(
            "Expected one batch/query attention vector; "
            f"got shape {tuple(attention.shape)}."
        )
    if not torch.isfinite(values).all():
        raise ValueError("Attention vector contains non-finite values.")
    return values


def attention_rows(
    attention: torch.Tensor,
    windows: Sequence[GenomeWindow],
    *,
    threshold: float = 0.05,
) -> list[dict[str, int | float | bool]]:
    """Join one averaged attention vector to its verified genome windows."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one.")
    values = _one_attention_vector(attention)
    if len(values) != len(windows):
        raise ValueError(
            f"Attention/window length mismatch: {len(values)} != {len(windows)}."
        )
    return [
        {
            "fragment_index": window.fragment_index,
            "attention_weight": float(values[index]),
            "selected": bool(values[index] > threshold),
            "contig_index": window.contig_index,
            "start": window.start,
            "end": window.end,
        }
        for index, window in enumerate(windows)
    ]


def indexed_attention_rows(
    attention: torch.Tensor,
    *,
    threshold: float = 0.05,
) -> list[dict[str, int | float | bool]]:
    """Return an index/weight table when no coordinate mapping is available."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one.")
    values = _one_attention_vector(attention)
    return [
        {
            "index": index,
            "attention_weight": float(value),
            "selected": bool(value > threshold),
        }
        for index, value in enumerate(values)
    ]


def annotate_selected_windows(
    rows: Sequence[dict[str, int | float | bool]],
    assets: VerifiedGenomeAssets,
) -> list[dict[str, int | float | bool | str]]:
    """Return every CDS overlapping a selected attention window."""

    annotations: list[dict[str, int | float | bool | str]] = []
    for row in rows:
        if not row["selected"]:
            continue
        record = assets.genbank_records[int(row["contig_index"])]
        window_start = int(row["start"])
        window_end = int(row["end"])
        for feature in record.features:
            if feature.type != "CDS":
                continue
            for part in feature.location.parts:
                feature_start = int(part.start)
                feature_end = int(part.end)
                if feature_start >= window_end or feature_end <= window_start:
                    continue
                annotations.append(
                    {
                        "fragment_index": int(row["fragment_index"]),
                        "attention_weight": float(row["attention_weight"]),
                        "contig_index": int(row["contig_index"]),
                        "record_id": str(record.id),
                        "window_start": window_start,
                        "window_end": window_end,
                        "feature_start": feature_start,
                        "feature_end": feature_end,
                        "fully_contained": bool(
                            feature_start >= window_start and feature_end <= window_end
                        ),
                        "gene": "|".join(feature.qualifiers.get("gene", [])),
                        "locus_tag": "|".join(feature.qualifiers.get("locus_tag", [])),
                        "product": "|".join(feature.qualifiers.get("product", [])),
                    }
                )
                break
    return annotations


@torch.inference_mode()
def score_single_selfies_attention(
    model: CandidateMICRegressor,
    tokenizer: Tokenizer,
    selfies: str,
    *,
    strain: str,
    device: str | torch.device,
) -> MICAttentionOutput:
    """Score one SELFIES after removing padding and return both attentions."""

    encoded = tokenizer(
        [normalize_selfies_for_tokenizer(selfies)],
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=True,
    )["input_ids"]
    if encoded.ndim == 1:
        encoded = encoded.unsqueeze(0)
    input_ids = encoded[0]
    input_ids = input_ids[input_ids != tokenizer.pad_token_id].unsqueeze(0)
    return model.forward_with_attention(input_ids.to(device), strain)
