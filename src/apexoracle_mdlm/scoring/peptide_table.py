"""Peptide-table conversion and MIC result assembly without path side effects."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, Sequence

from .mic import CandidateMICRegressor, Tokenizer, score_selfies_across_strains


STRUCTURE_COLUMNS = (
    "row_id",
    "Peptide",
    "Protein",
    "SMILES",
    "SELFIES",
    "conversion_status",
    "invalid_reason",
)


def load_peptide_table(
    path: str | PathLike[str],
    *,
    peptide_column: str,
    protein_column: str,
    limit: int | None = None,
) -> Any:
    """Load two named columns and assign stable zero-based source row IDs."""

    import numpy as np
    import pandas as pd

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative.")
    # Preserve empty cells as empty strings.  With pandas' default NA parsing,
    # ``astype(str)`` turns a blank peptide into the literal sequence ``"nan"``;
    # RDKit then accepts it as Asn-Ala-Asn instead of retaining an invalid row.
    frame = pd.read_csv(source, keep_default_na=False)
    missing = [
        column
        for column in (peptide_column, protein_column)
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns in {source}: {missing}")
    selected = frame[[peptide_column, protein_column]].copy()
    selected.columns = ["Peptide", "Protein"]
    selected["Peptide"] = selected["Peptide"].astype(str).str.strip()
    selected["Protein"] = selected["Protein"].astype(str).str.strip()
    if limit is not None:
        selected = selected.head(limit).copy()
    selected.insert(0, "row_id", np.arange(len(selected), dtype=np.int64))
    return selected


def convert_peptides_to_structures(peptide_frame: Any) -> Any:
    """Convert sequences through RDKit canonical SMILES and then SELFIES."""

    import selfies
    from rdkit import Chem

    required = {"row_id", "Peptide", "Protein"}
    missing = sorted(required.difference(peptide_frame.columns))
    if missing:
        raise ValueError(f"Peptide table is missing columns: {missing}")
    processed = peptide_frame.copy()
    processed["SMILES"] = ""
    processed["SELFIES"] = ""
    processed["conversion_status"] = "invalid"
    processed["invalid_reason"] = ""

    for row_index, peptide in zip(processed.index, processed["Peptide"]):
        if not peptide:
            processed.at[row_index, "invalid_reason"] = "empty_peptide"
            continue
        if "X" in peptide:
            processed.at[row_index, "invalid_reason"] = "contains_X"
            continue
        molecule = Chem.MolFromSequence(peptide)
        if molecule is None:
            processed.at[row_index, "invalid_reason"] = "rdkit_mol_from_sequence_failed"
            continue
        try:
            smiles = Chem.MolToSmiles(molecule, canonical=True)
        except Exception as error:  # RDKit exception types vary by release.
            processed.at[row_index, "invalid_reason"] = (
                f"rdkit_smiles_failed:{type(error).__name__}"
            )
            continue
        try:
            encoded_selfies = selfies.encoder(smiles)
        except Exception as error:  # SELFIES exception types vary by release.
            processed.at[row_index, "invalid_reason"] = (
                f"selfies_encode_failed:{type(error).__name__}"
            )
            continue
        processed.at[row_index, "SMILES"] = smiles
        processed.at[row_index, "SELFIES"] = encoded_selfies
        processed.at[row_index, "conversion_status"] = "valid"
        processed.at[row_index, "invalid_reason"] = ""
    return processed.loc[:, list(STRUCTURE_COLUMNS)]


def add_mic_predictions(
    structure_frame: Any,
    model: CandidateMICRegressor,
    tokenizer: Tokenizer,
    *,
    strains: Sequence[str],
    batch_size: int,
    device: str,
) -> Any:
    """Append one float32 MIC column per strain while retaining invalid rows."""

    import numpy as np

    missing = sorted(set(STRUCTURE_COLUMNS).difference(structure_frame.columns))
    if missing:
        raise ValueError(f"Structure table is missing columns: {missing}")
    predictions = structure_frame.copy()
    valid_mask = predictions["conversion_status"].eq("valid")
    valid_positions = np.flatnonzero(valid_mask.to_numpy())
    valid_selfies = predictions.loc[valid_mask, "SELFIES"].tolist()
    scored = score_selfies_across_strains(
        model,
        tokenizer,
        valid_selfies,
        strains=strains,
        batch_size=batch_size,
        device=device,
    )
    for strain in strains:
        values = np.full(len(predictions), np.nan, dtype=np.float32)
        values[valid_positions] = scored[strain].numpy()
        predictions[strain] = values
    return predictions


def conversion_summary(structure_frame: Any) -> dict[str, Any]:
    """Return compact counts suitable for a run manifest."""

    valid = structure_frame["conversion_status"].eq("valid")
    reasons = structure_frame.loc[~valid, "invalid_reason"].value_counts().sort_index()
    return {
        "total_rows": int(len(structure_frame)),
        "valid_rows": int(valid.sum()),
        "invalid_rows": int((~valid).sum()),
        "invalid_reason_counts": {
            str(key): int(value) for key, value in reasons.items()
        },
    }
