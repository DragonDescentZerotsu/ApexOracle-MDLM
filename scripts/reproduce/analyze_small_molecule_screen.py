#!/usr/bin/env python
"""Filter and compare decoded small-molecule MIC screening tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from apexoracle_mdlm.scoring import (
    canonical_prediction_set,
    compare_structure_sets,
    filter_screen_predictions,
    load_active_reference_structures,
    load_screen_predictions,
    parse_strain_input,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a MIC cutoff to decoded screening tables and compare canonical "
            "structure sets."
        )
    )
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="STRAIN=PATH",
        help="Repeat for each table; PATH must contain SMILES_Sequence and STRAIN.",
    )
    parser.add_argument("--mic-cutoff", type=float, default=15.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-smiles-column", default="SMILES")
    parser.add_argument("--reference-label-column", default="MIC")
    parser.add_argument("--reference-threshold", type=float, default=0.5)
    return parser.parse_args()


def write_predictions(path: Path, strain: str, predictions) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["SMILES_Sequence", strain],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {"SMILES_Sequence": prediction.smiles, strain: prediction.mic}
            for prediction in predictions
        )


def main() -> None:
    args = parse_args()
    specifications = [parse_strain_input(value) for value in args.prediction]
    strains = [item.strain for item in specifications]
    if len(strains) != len(set(strains)):
        raise ValueError("Each prediction strain may appear only once.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    structure_sets: dict[str, set[str]] = {}
    summary: dict = {
        "schema_version": 1,
        "mic_cutoff_umol": args.mic_cutoff,
        "cutoff_rule": "predicted_mic <= cutoff",
        "canonicalization": "RDKit canonical isomeric SMILES",
        "predictions": {},
        "comparisons": {},
    }
    for item in specifications:
        predictions = load_screen_predictions(item.path, strain=item.strain)
        active = filter_screen_predictions(predictions, cutoff=args.mic_cutoff)
        structures = canonical_prediction_set(active)
        structure_sets[item.strain] = structures
        output = args.output_dir / f"{item.strain}_mic_at_or_below_cutoff.csv"
        write_predictions(output, item.strain, active)
        summary["predictions"][item.strain] = {
            "input": str(item.path),
            "input_sha256": sha256(item.path),
            "input_rows": len(predictions),
            "active_rows": len(active),
            "active_canonical_structures": len(structures),
            "filtered_output": str(output),
            "filtered_output_sha256": sha256(output),
        }

    if len(structure_sets) > 1:
        names = list(structure_sets)
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                key = f"{left_name}__vs__{right_name}"
                summary["comparisons"][key] = compare_structure_sets(
                    structure_sets[left_name], structure_sets[right_name]
                ).to_dict()

    if args.reference is not None:
        reference = load_active_reference_structures(
            args.reference,
            smiles_column=args.reference_smiles_column,
            label_column=args.reference_label_column,
            threshold=args.reference_threshold,
        )
        summary["reference"] = {
            "path": str(args.reference),
            "sha256": sha256(args.reference),
            "smiles_column": args.reference_smiles_column,
            "label_column": args.reference_label_column,
            "active_rule": f"label > {args.reference_threshold}",
            "active_canonical_structures": len(reference),
        }
        for strain, structures in structure_sets.items():
            summary["comparisons"][f"{strain}__vs__reference"] = compare_structure_sets(
                structures, reference
            ).to_dict()

    union = set().union(*structure_sets.values()) if structure_sets else set()
    summary["all_predictions_union_canonical_structures"] = len(union)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
