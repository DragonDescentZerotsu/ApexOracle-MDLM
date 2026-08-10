"""Portable table-level SMILES/SELFIES conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Callable


def convert_smiles_table_to_selfies(
    input_path: str | Path,
    output_path: str | Path,
    *,
    smiles_column: str = "SMILES",
    encoder: Callable[[str], str] | None = None,
) -> int:
    """Replace one SMILES column with SELFIES while preserving all other cells.

    The complete input is read before the output is opened, so an explicitly
    requested in-place conversion is safe. The return value is the row count.
    """

    import pandas as pd

    source = Path(input_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    table = pd.read_csv(source)
    if smiles_column not in table.columns:
        raise ValueError(f"Missing SMILES column {smiles_column!r} in {source}")
    if encoder is None:
        import selfies

        encoder = selfies.encoder

    converted: list[str] = []
    for row_number, value in enumerate(table[smiles_column], start=2):
        if not isinstance(value, str) or not value:
            raise ValueError(f"Invalid SMILES at {source}:{row_number}: {value!r}")
        converted_value = encoder(value)
        if not isinstance(converted_value, str) or not converted_value:
            raise ValueError(
                f"SELFIES encoder returned no value at {source}:{row_number}"
            )
        converted.append(converted_value)
    table[smiles_column] = converted
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    return len(table)
