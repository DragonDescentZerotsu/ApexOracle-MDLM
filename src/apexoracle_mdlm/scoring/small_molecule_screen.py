"""Collection-level contracts for the paper small-molecule MIC screen."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
from torch import nn

from .mic import Tokenizer, read_selfies_file, score_selfies_strings


@dataclass(frozen=True)
class StrainInput:
    """One explicitly named strain and its one-SELFIES-per-line input."""

    strain: str
    path: Path


@dataclass(frozen=True)
class StrainScreen:
    """Raw row-level predictions and the legacy last-occurrence deduplication."""

    strain: str
    source_path: Path
    selfies_strings: tuple[str, ...]
    mic_values: torch.Tensor
    mic_by_selfies: Mapping[str, float]


def parse_strain_input(value: str) -> StrainInput:
    """Parse the repeatable CLI form ``STRAIN=PATH`` without guessing names."""

    strain, separator, path_text = value.partition("=")
    if not separator or not strain.strip() or not path_text.strip():
        raise ValueError("Input must use the form STRAIN=PATH.")
    return StrainInput(strain=strain.strip(), path=Path(path_text.strip()).expanduser())


def last_mic_by_selfies(
    selfies_strings: Sequence[str], mic_values: Sequence[float]
) -> dict[str, float]:
    """Apply the historical dict assignment rule to duplicate SELFIES rows."""

    if len(selfies_strings) != len(mic_values):
        raise ValueError("SELFIES and MIC value counts differ.")
    result: dict[str, float] = {}
    for selfies_string, mic_value in zip(selfies_strings, mic_values):
        result[selfies_string] = float(mic_value)
    return result


def score_small_molecule_inputs(
    model: nn.Module,
    tokenizer: Tokenizer,
    inputs: Sequence[StrainInput],
    *,
    device: str | torch.device,
) -> dict[str, StrainScreen]:
    """Score each input with the historical unpadded one-molecule protocol."""

    strains = [item.strain for item in inputs]
    if len(set(strains)) != len(strains):
        raise ValueError("Each strain may appear only once.")
    screens: dict[str, StrainScreen] = {}
    for item in inputs:
        selfies_strings = read_selfies_file(item.path)
        predictions = score_selfies_strings(
            model,
            tokenizer,
            selfies_strings,
            strain=item.strain,
            device=device,
        )
        screens[item.strain] = StrainScreen(
            strain=item.strain,
            source_path=item.path,
            selfies_strings=tuple(selfies_strings),
            mic_values=predictions,
            mic_by_selfies=last_mic_by_selfies(selfies_strings, predictions.tolist()),
        )
    return screens


def decoded_wide_rows(
    screens: Mapping[str, StrainScreen],
    *,
    decoder: Callable[[str], str] | None = None,
) -> list[dict[str, str | float | None]]:
    """Build deterministic legacy-compatible ``SMILES_Sequence`` wide rows.

    The legacy script iterated a Python set, so its CSV row order varied by
    process. Sorting by source SELFIES retains the same rows and values while
    making the released artifact reproducible. Missing strain values remain
    empty CSV cells, matching the historical wide-table contract.
    """

    if not screens:
        return []
    if decoder is None:
        import selfies

        decoder = selfies.decoder
    all_selfies = sorted(
        {
            selfies_string
            for screen in screens.values()
            for selfies_string in screen.mic_by_selfies
        }
    )
    rows: list[dict[str, str | float | None]] = []
    for selfies_string in all_selfies:
        smiles = decoder(selfies_string)
        if not isinstance(smiles, str) or not smiles:
            raise ValueError(
                f"SELFIES decoder returned no SMILES for {selfies_string!r}."
            )
        row: dict[str, str | float | None] = {"SMILES_Sequence": smiles}
        for strain, screen in screens.items():
            row[strain] = screen.mic_by_selfies.get(selfies_string)
        rows.append(row)
    return rows


def load_strain_inputs(values: Sequence[str]) -> list[StrainInput]:
    """Parse repeatable input specs and validate their files up front."""

    inputs = [parse_strain_input(value) for value in values]
    if not inputs:
        raise ValueError("At least one STRAIN=PATH input is required.")
    strains = [item.strain for item in inputs]
    if len(set(strains)) != len(strains):
        raise ValueError("Each strain may appear only once.")
    for item in inputs:
        if not item.path.is_file():
            raise FileNotFoundError(item.path)
    return inputs
