#!/usr/bin/env python
"""Compatibility bridge for the historical peptide-structure parser import."""

from __future__ import annotations

import sys
from pathlib import Path


try:
    from apexoracle_mdlm.chemistry import smiles_to_peptide_sequence
except ModuleNotFoundError:
    source_root = Path(__file__).resolve().parent / "src"
    if source_root.is_dir():
        sys.path.insert(0, str(source_root))
    from apexoracle_mdlm.chemistry import smiles_to_peptide_sequence


def smiles_to_pepseq(structure: str) -> tuple[str, str | None]:
    """Delegate the legacy function name to the canonical parser."""

    return smiles_to_peptide_sequence(structure)


__all__ = ["smiles_to_pepseq", "smiles_to_peptide_sequence"]
