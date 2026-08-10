"""Collection-level contracts for the paper small-molecule MIC screen."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

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


@dataclass(frozen=True)
class ScreenPrediction:
    """One decoded structure and its predicted MIC."""

    smiles: str
    mic: float


@dataclass(frozen=True)
class StructureSetComparison:
    """Counts for two canonical-structure sets."""

    left_count: int
    right_count: int
    intersection_count: int

    @property
    def left_only_count(self) -> int:
        return self.left_count - self.intersection_count

    @property
    def right_only_count(self) -> int:
        return self.right_count - self.intersection_count

    @property
    def union_count(self) -> int:
        return self.left_count + self.right_count - self.intersection_count

    def to_dict(self) -> dict[str, int]:
        return {
            "left_count": self.left_count,
            "right_count": self.right_count,
            "intersection_count": self.intersection_count,
            "left_only_count": self.left_only_count,
            "right_only_count": self.right_only_count,
            "union_count": self.union_count,
        }


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


def load_screen_predictions(
    path: str | Path,
    *,
    strain: str,
    smiles_column: str = "SMILES_Sequence",
) -> list[ScreenPrediction]:
    """Load and validate one decoded small-molecule prediction table."""

    source = Path(path)
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = {smiles_column, strain} - columns
        if missing:
            raise ValueError(f"Missing columns in {source}: {sorted(missing)}")
        predictions: list[ScreenPrediction] = []
        for row_index, row in enumerate(reader, start=2):
            smiles = (row.get(smiles_column) or "").strip()
            if not smiles:
                raise ValueError(f"Empty SMILES at {source}:{row_index}")
            try:
                mic = float(row[strain])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid MIC at {source}:{row_index}: {row.get(strain)!r}"
                ) from error
            if not math.isfinite(mic) or mic <= 0:
                raise ValueError(
                    f"MIC must be finite and positive at {source}:{row_index}"
                )
            predictions.append(ScreenPrediction(smiles=smiles, mic=mic))
    return predictions


def filter_screen_predictions(
    predictions: Iterable[ScreenPrediction], *, cutoff: float
) -> list[ScreenPrediction]:
    """Retain predictions at or below an explicit MIC cutoff."""

    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("MIC cutoff must be finite and positive.")
    return [prediction for prediction in predictions if prediction.mic <= cutoff]


def canonicalize_smiles(smiles: str) -> str:
    """Return RDKit canonical isomeric SMILES or reject an invalid structure."""

    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return str(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))


def canonical_prediction_set(
    predictions: Iterable[ScreenPrediction],
) -> set[str]:
    """Canonicalize a prediction collection and collapse duplicate structures."""

    return {canonicalize_smiles(prediction.smiles) for prediction in predictions}


def load_active_reference_structures(
    path: str | Path,
    *,
    smiles_column: str,
    label_column: str,
    threshold: float,
) -> set[str]:
    """Load canonical structures whose numeric reference label exceeds a cutoff."""

    source = Path(path)
    active: set[str] = set()
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = {smiles_column, label_column} - columns
        if missing:
            raise ValueError(f"Missing columns in {source}: {sorted(missing)}")
        for row_index, row in enumerate(reader, start=2):
            try:
                label = float(row[label_column])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid label at {source}:{row_index}: {row.get(label_column)!r}"
                ) from error
            if label > threshold:
                active.add(canonicalize_smiles((row.get(smiles_column) or "").strip()))
    return active


def compare_structure_sets(left: set[str], right: set[str]) -> StructureSetComparison:
    """Summarize two already-canonicalized structure sets."""

    return StructureSetComparison(
        left_count=len(left),
        right_count=len(right),
        intersection_count=len(left & right),
    )
