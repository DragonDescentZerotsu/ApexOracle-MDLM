"""Reusable clean-input DLM molecule-embedding producers."""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
import re
from typing import Any, Callable, Hashable, Iterable, Mapping, Protocol

import torch
from torch import nn


LEGACY_POOLING_METHODS = (
    "cls_wo_pad",
    "cls_wo_pad_eval",
    "mean_w_pad",
    "cls_w_pad",
    "mean_wo_pad",
    "mean_wo_pad_eval",
)


class MoleculeTokenizer(Protocol):
    pad_token_id: int
    unk_token_id: int | None

    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, torch.Tensor]: ...


@dataclass(frozen=True)
class EmbeddingExportResult:
    embeddings: dict[Hashable, torch.Tensor]
    input_count: int
    output_count: int
    skipped_unknown: int
    skipped_too_long: int


def embedding_dictionary_schema(
    embeddings: Mapping[Hashable, torch.Tensor],
) -> dict[str, Any]:
    """Summarize key types and tensor contracts for an export manifest."""

    if not embeddings:
        raise ValueError("Embedding dictionary is empty.")
    for key, value in embeddings.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Embedding {key!r} must be a torch.Tensor, got {type(value).__name__}."
            )
    return {
        "entries": len(embeddings),
        "key_types": sorted({type(key).__name__ for key in embeddings}),
        "tensor_shapes": sorted({tuple(value.shape) for value in embeddings.values()}),
        "tensor_dtypes": sorted({str(value.dtype) for value in embeddings.values()}),
    }


def _coerce_identifier(value: str, kind: str) -> Hashable:
    value = value.strip()
    if kind == "string":
        return value
    if kind == "integer":
        if not re.fullmatch(r"[+-]?\d+", value):
            raise ValueError(f"Expected integer identifier, got {value!r}.")
        return int(value)
    raise ValueError(f"Unsupported identifier type {kind!r}.")


def load_token_id_csv(
    path: str | PathLike[str],
    *,
    id_column: str,
    token_column: str,
    id_type: str = "string",
) -> dict[Hashable, torch.Tensor]:
    """Load the first token-id row for each molecule, matching legacy deduplication."""

    result: dict[Hashable, torch.Tensor] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Token CSV has no header: {path}")
        missing = {id_column, token_column}.difference(reader.fieldnames)
        if missing:
            raise KeyError(f"Token CSV is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            key = _coerce_identifier(row[id_column], id_type)
            if key in result:
                continue
            try:
                values = ast.literal_eval(row[token_column])
                tensor = torch.as_tensor(values, dtype=torch.long)
            except (SyntaxError, ValueError, TypeError) as exc:
                raise ValueError(f"Invalid token list at {path}:{row_number}.") from exc
            if tensor.ndim != 1:
                raise ValueError(
                    f"Token list at {path}:{row_number} must be one-dimensional."
                )
            result[key] = tensor
    return result


def collect_pair_smiles_tokens(
    path: str | PathLike[str],
    *,
    tokenizer: MoleculeTokenizer,
    smiles_to_selfies: Callable[[str], str],
    first_id_column: str,
    second_id_column: str,
    first_smiles_column: str,
    second_smiles_column: str,
    first_id_type: str = "integer",
    second_id_type: str = "string",
    max_length: int = 1024,
) -> tuple[dict[Hashable, torch.Tensor], int, int]:
    """Stream a pair table and retain the first valid tokenization per ID.

    Explicit ID coercion replaces the fragile pandas dtype inference used by
    the old scripts.  It is required because the frozen public synergy cache
    intentionally contains integer peptide IDs and string partner IDs.
    """

    result: dict[Hashable, torch.Tensor] = {}
    skipped_unknown = 0
    skipped_too_long = 0
    specifications = (
        (first_id_column, first_smiles_column, first_id_type),
        (second_id_column, second_smiles_column, second_id_type),
    )
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Pair CSV has no header: {path}")
        required = {item for spec in specifications for item in spec[:2]}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise KeyError(f"Pair CSV is missing columns: {sorted(missing)}")
        for row in reader:
            for id_column, smiles_column, id_type in specifications:
                key = _coerce_identifier(row[id_column], id_type)
                if key in result:
                    continue
                selfies = smiles_to_selfies(row[smiles_column])
                input_ids = (
                    tokenizer(
                        selfies.replace("][", "] ["),
                        return_tensors="pt",
                        padding=False,
                        truncation=False,
                    )["input_ids"]
                    .squeeze(0)
                    .to(torch.long)
                )
                if tokenizer.unk_token_id is not None and bool(
                    input_ids.eq(tokenizer.unk_token_id).any()
                ):
                    skipped_unknown += 1
                    continue
                if input_ids.numel() > max_length:
                    skipped_too_long += 1
                    continue
                result[key] = input_ids
    return result, skipped_unknown, skipped_too_long


def pool_molecule_hidden_states(
    encoder: nn.Module,
    token_ids: torch.Tensor,
    *,
    pooling_method: str,
    pad_token_id: int,
    padded_length: int = 1024,
) -> torch.Tensor:
    """Apply one of the six historical pooling names without hidden side effects."""

    if pooling_method not in LEGACY_POOLING_METHODS:
        raise ValueError(
            f"Unknown pooling method {pooling_method!r}; expected one of {LEGACY_POOLING_METHODS}."
        )
    if token_ids.ndim != 1:
        raise ValueError(f"Expected one-dimensional token IDs, got {token_ids.shape}.")

    padded = pooling_method in {"mean_w_pad", "cls_w_pad"}
    if padded:
        if token_ids.numel() > padded_length:
            raise ValueError(
                f"Sequence length {token_ids.numel()} exceeds padded length {padded_length}."
            )
        model_input = torch.full(
            (1, padded_length),
            pad_token_id,
            dtype=torch.long,
            device=token_ids.device,
        )
        model_input[0, : token_ids.numel()] = token_ids
    else:
        model_input = token_ids.unsqueeze(0)

    hidden = encoder(model_input)
    if pooling_method.startswith("cls_"):
        return hidden[:, 0, :].detach().cpu()
    valid_hidden = hidden[0, : token_ids.numel(), :]
    return valid_hidden.detach().cpu().mean(dim=0)


def export_molecule_embeddings(
    encoder: nn.Module,
    token_ids_by_id: Mapping[Hashable, torch.Tensor],
    *,
    pooling_method: str,
    pad_token_id: int,
    device: str | torch.device,
    model_mode: str = "eval",
    padded_length: int = 1024,
    skipped_unknown: int = 0,
    skipped_too_long: int = 0,
    progress: (
        Callable[
            [Iterable[tuple[Hashable, torch.Tensor]]],
            Iterable[tuple[Hashable, torch.Tensor]],
        ]
        | None
    ) = None,
) -> EmbeddingExportResult:
    """Encode a deduplicated molecule mapping with an explicit model mode."""

    if model_mode == "eval":
        encoder.eval()
    elif model_mode == "train":
        encoder.train()
    else:
        raise ValueError("model_mode must be 'eval' or 'train'.")
    target_device = torch.device(device)
    encoder.to(target_device)
    items: Iterable[tuple[Hashable, torch.Tensor]] = token_ids_by_id.items()
    if progress is not None:
        items = progress(items)
    embeddings: dict[Hashable, torch.Tensor] = {}
    # DLMHiddenStateEncoder owns the historical bfloat16 block context.  Do
    # not add an outer fp16 context here: the legacy wrapper's attempted
    # float32 autocast disabled its caller context around vocab/sigma mapping.
    with torch.inference_mode():
        for key, token_ids in items:
            embeddings[key] = pool_molecule_hidden_states(
                encoder,
                token_ids.to(target_device),
                pooling_method=pooling_method,
                pad_token_id=pad_token_id,
                padded_length=padded_length,
            )
    return EmbeddingExportResult(
        embeddings=embeddings,
        input_count=len(token_ids_by_id),
        output_count=len(embeddings),
        skipped_unknown=skipped_unknown,
        skipped_too_long=skipped_too_long,
    )
