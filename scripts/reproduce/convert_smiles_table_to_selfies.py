#!/usr/bin/env python
"""Convert one explicit table column from SMILES to SELFIES."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from apexoracle_mdlm.chemistry import convert_smiles_table_to_selfies


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smiles-column", default="SMILES")
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_hash = _sha256(args.input)
    rows = convert_smiles_table_to_selfies(
        args.input,
        args.output,
        smiles_column=args.smiles_column,
    )
    manifest = {
        "schema_version": 1,
        "operation": "SMILES column to SELFIES",
        "input": str(args.input),
        "input_sha256": input_hash,
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "smiles_column": args.smiles_column,
        "rows": rows,
    }
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
