#!/usr/bin/env python
"""Verify the useful behavior of the legacy small-molecule debug scripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from apexoracle_mdlm.scoring import (
    canonical_prediction_set,
    compare_structure_sets,
    filter_screen_predictions,
    load_active_reference_structures,
    load_screen_predictions,
    parse_strain_input,
)

LEGACY_PATHS = (
    "debug_temp_SMs_MIC_analysis.py",
    "debug_temp_SMs_MIC_analysis_2.py",
    "debug_temp_SMs_MIC_analysis_3.py",
    "debug_temp_SMs_MIC_analysis_4.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_mapping(values: list[str]) -> dict[str, Path]:
    items = [parse_strain_input(value) for value in values]
    result = {item.strain: item.path for item in items}
    if len(result) != len(items):
        raise ValueError("Each strain may appear only once.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction", action="append", required=True, metavar="STRAIN=PATH"
    )
    parser.add_argument(
        "--legacy-filtered", action="append", default=[], metavar="STRAIN=PATH"
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-smiles-column", default="SMILES")
    parser.add_argument("--reference-label-column", default="MIC")
    parser.add_argument("--reference-threshold", type=float, default=0.5)
    parser.add_argument("--mic-cutoff", type=float, default=15.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    predictions = parse_mapping(args.prediction)
    legacy_filtered = parse_mapping(args.legacy_filtered)
    unknown = set(legacy_filtered) - set(predictions)
    if unknown:
        raise ValueError(
            f"Legacy filtered tables lack prediction inputs: {sorted(unknown)}"
        )

    sets: dict[str, set[str]] = {}
    strain_results = {}
    for strain, path in predictions.items():
        records = load_screen_predictions(path, strain=strain)
        active = filter_screen_predictions(records, cutoff=args.mic_cutoff)
        structures = canonical_prediction_set(active)
        sets[strain] = structures
        result = {
            "prediction_file": path.name,
            "prediction_sha256": sha256(path),
            "prediction_rows": len(records),
            "active_rows": len(active),
            "active_canonical_structures": len(structures),
        }
        if strain in legacy_filtered:
            legacy_path = legacy_filtered[strain]
            historical = load_screen_predictions(legacy_path, strain=strain)
            expected_pairs = [(record.smiles, record.mic) for record in active]
            actual_pairs = [(record.smiles, record.mic) for record in historical]
            if actual_pairs != expected_pairs:
                raise AssertionError(f"Legacy filtered content mismatch for {strain}")
            result["legacy_filtered"] = {
                "file": legacy_path.name,
                "sha256": sha256(legacy_path),
                "rows": len(historical),
                "content_and_order_match": True,
                "ignored_extra_column": "Unnamed: 0",
            }
        strain_results[strain] = result

    reference = load_active_reference_structures(
        args.reference,
        smiles_column=args.reference_smiles_column,
        label_column=args.reference_label_column,
        threshold=args.reference_threshold,
    )
    comparisons = {}
    names = list(sets)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            comparisons[f"{left_name}__vs__{right_name}"] = compare_structure_sets(
                sets[left_name], sets[right_name]
            ).to_dict()
    for strain, structures in sets.items():
        comparisons[f"{strain}__vs__reference"] = compare_structure_sets(
            structures, reference
        ).to_dict()

    result = {
        "schema_version": 1,
        "scope": "legacy small-molecule debug-script behavior",
        "protocol": {
            "mic_cutoff_umol": args.mic_cutoff,
            "cutoff_rule": "predicted_mic <= cutoff",
            "canonicalization": "RDKit canonical isomeric SMILES",
            "reference_active_rule": f"{args.reference_label_column} > {args.reference_threshold}",
        },
        "legacy_sources": {
            path: {
                "snapshot_ref": "legacy-code-snapshot-2026-08-09",
                "sha256": hashlib.sha256(
                    subprocess.check_output(
                        [
                            "git",
                            "show",
                            f"legacy-code-snapshot-2026-08-09:{path}",
                        ],
                        cwd=repository_root,
                    )
                ).hexdigest(),
                "active_tree_removed": not (repository_root / path).exists(),
            }
            for path in LEGACY_PATHS
        },
        "strains": strain_results,
        "reference": {
            "file": args.reference.name,
            "sha256": sha256(args.reference),
            "active_canonical_structures": len(reference),
        },
        "comparisons": comparisons,
        "all_prediction_union_canonical_structures": len(set().union(*sets.values())),
        "status": "passed",
    }
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
