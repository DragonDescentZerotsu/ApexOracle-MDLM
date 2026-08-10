#!/usr/bin/env python
"""Verify the frozen 44,608-entry small-molecule screen input/output lineage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import selfies

from apexoracle_mdlm.scoring import load_strain_inputs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_mapping(value: str) -> tuple[str, Path]:
    strain, separator, path_text = value.partition("=")
    if not separator or not strain.strip() or not path_text.strip():
        raise ValueError("Output must use the form STRAIN=PATH.")
    return strain.strip(), Path(path_text.strip()).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", action="append", required=True, metavar="STRAIN=PATH"
    )
    parser.add_argument(
        "--legacy-output", action="append", required=True, metavar="STRAIN=PATH"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def verify(args: argparse.Namespace) -> dict:
    inputs = load_strain_inputs(args.input)
    output_items = [parse_mapping(value) for value in args.legacy_output]
    if len({strain for strain, _ in output_items}) != len(output_items):
        raise ValueError("Each legacy-output strain may appear only once.")
    outputs = dict(output_items)
    if set(outputs) != {item.strain for item in inputs}:
        raise ValueError("Input and legacy-output strain sets differ.")

    strain_results = {}
    for item in inputs:
        source_rows = item.path.read_text(encoding="utf-8").splitlines()
        unique_selfies = set(source_rows)
        decoded_smiles = [selfies.decoder(value) for value in unique_selfies]
        output_path = outputs[item.strain]
        with output_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            expected_columns = ["SMILES_Sequence", item.strain]
            if reader.fieldnames != expected_columns:
                raise AssertionError(
                    f"{output_path} columns {reader.fieldnames} != {expected_columns}."
                )
            output_rows = list(reader)
        output_smiles = [row["SMILES_Sequence"] for row in output_rows]
        mic_values = [float(row[item.strain]) for row in output_rows]
        checks = {
            "no_empty_source_rows": all(source_rows),
            "no_empty_decoded_smiles": all(decoded_smiles),
            "decoded_smiles_are_unique": len(set(decoded_smiles))
            == len(decoded_smiles),
            "no_empty_output_smiles": all(output_smiles),
            "output_smiles_are_unique": len(set(output_smiles)) == len(output_smiles),
            "decoded_smiles_set_equals_output": set(decoded_smiles)
            == set(output_smiles),
            "all_mic_values_finite_positive": all(
                math.isfinite(value) and value > 0 for value in mic_values
            ),
        }
        if not all(checks.values()):
            raise AssertionError(f"Lineage checks failed for {item.strain}: {checks}")
        strain_results[item.strain] = {
            "input": {
                "path": str(item.path),
                "bytes": item.path.stat().st_size,
                "sha256": sha256(item.path),
                "rows": len(source_rows),
                "unique_selfies": len(unique_selfies),
            },
            "legacy_output": {
                "path": str(output_path),
                "bytes": output_path.stat().st_size,
                "sha256": sha256(output_path),
                "rows": len(output_rows),
                "columns": expected_columns,
            },
            "checks": checks,
        }
    result = {
        "schema_version": 1,
        "scope": "frozen paper small-molecule screen input/output lineage",
        "protocol": {
            "raw_rows_scored_individually": True,
            "csv_duplicate_rule": "last_prediction_wins_per_source_selfies",
            "legacy_csv_row_order": "python_set_iteration_not_reproducible",
            "canonical_csv_row_order": "lexicographic_source_selfies",
        },
        "strains": strain_results,
        "status": "passed",
    }
    hashes = {value["input"]["sha256"] for value in strain_results.values()}
    result["all_strains_share_input_sha256"] = len(hashes) == 1
    return result


def main() -> None:
    args = parse_args()
    result = verify(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
