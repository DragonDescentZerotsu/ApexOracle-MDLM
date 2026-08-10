#!/usr/bin/env python
"""Convert a raw molecule-pair table into the canonical token-ID contract."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize two SMILES columns for synergy-guidance training."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-smiles-column", default="AMP_smiles")
    parser.add_argument("--second-smiles-column", default="antibiotic_smiles")
    parser.add_argument("--strain-column", default="strain_name")
    parser.add_argument("--fici-column", default="FICI")
    parser.add_argument("--tokenizer", default="ibm-research/materials.selfies-ted")
    parser.add_argument("--max-molecule-length", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import selfies as sf
    from transformers import AutoTokenizer

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.max_molecule_length <= 0:
        raise ValueError("max-molecule-length must be positive.")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            args.first_smiles_column,
            args.second_smiles_column,
            args.strain_column,
            args.fici_column,
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Input table is missing columns: {sorted(missing)}")
        rows = list(reader)

    prepared: list[dict[str, object]] = []
    excluded_long = 0
    for row_number, row in enumerate(rows, start=2):
        try:
            first_selfies = sf.encoder(row[args.first_smiles_column]).replace(
                "][", "] ["
            )
            second_selfies = sf.encoder(row[args.second_smiles_column]).replace(
                "][", "] ["
            )
            first = tokenizer(first_selfies, truncation=False)["input_ids"]
            second = tokenizer(second_selfies, truncation=False)["input_ids"]
            fici = float(row[args.fici_column])
        except Exception as exc:
            raise ValueError(
                f"Failed to prepare input row {row_number}: {exc}"
            ) from exc
        if (
            len(first) > args.max_molecule_length
            or len(second) > args.max_molecule_length
        ):
            excluded_long += 1
            continue
        prepared.append(
            {
                "input_ids_1": json.dumps(first),
                "input_ids_2": json.dumps(second),
                "strain_name": row[args.strain_column],
                "FICI": fici,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("input_ids_1", "input_ids_2", "strain_name", "FICI"),
        )
        writer.writeheader()
        writer.writerows(prepared)
    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "output_rows": len(prepared),
                "excluded_long": excluded_long,
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
