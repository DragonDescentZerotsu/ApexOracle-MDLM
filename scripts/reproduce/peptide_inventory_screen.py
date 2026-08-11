#!/usr/bin/env python
"""Prepare peptide inventories and summarize canonical MIC table predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from apexoracle_mdlm.scoring import (
    cutoff_slug,
    prediction_token_lengths,
    prepare_peptide_inventory,
    summarize_peptide_inventory,
)


DEFAULT_TOKENIZER_REVISION = "55e83392264cb998f7aa5014847df29868aefeb8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sheet")
    parser.add_argument("--sequence-column", required=True)
    parser.add_argument("--identifier-column", required=True)
    parser.add_argument("--residue-count-column")
    parser.add_argument("--n-terminus-column")
    parser.add_argument("--c-terminus-column")
    parser.add_argument("--cyclic-column")
    parser.add_argument("--modification-column", action="append", default=[])
    parser.add_argument("--free-terminus-value", default="Free")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")


def add_summarize_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--strain", required=True)
    parser.add_argument("--target-label")
    parser.add_argument("--stock-column")
    parser.add_argument("--stock-unit")
    parser.add_argument("--mic-cutoff", type=float, default=15.0)
    parser.add_argument("--max-token-length", type=int)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument("--expected-batch-size", type=int, default=32)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_prepare_arguments(subparsers.add_parser("prepare"))
    add_summarize_arguments(subparsers.add_parser("summarize"))
    return parser.parse_args()


def read_inventory(path: Path, sheet: str | None) -> pd.DataFrame:
    suffix = path.suffix.casefold()
    if suffix in {".xlsx", ".xlsm"}:
        if sheet is None:
            raise ValueError("--sheet is required for Excel inputs")
        return pd.read_excel(path, sheet_name=sheet)
    if suffix in {".csv", ".tsv"}:
        if sheet is not None:
            raise ValueError("--sheet is only valid for Excel inputs")
        separator = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=separator, keep_default_na=False)
    raise ValueError(f"Unsupported inventory format: {path.suffix}")


def ensure_outputs_available(outputs: dict[str, Path], overwrite: bool) -> None:
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: "
            + ", ".join(str(path) for path in existing)
        )


def prepare(args: argparse.Namespace) -> dict:
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_directory = args.output_directory.resolve()
    outputs = {
        "screen_input": output_directory / "screen_input.csv",
        "inventory_rows": output_directory / "inventory_rows.csv",
        "manifest": output_directory / "preparation_manifest.json",
    }
    ensure_outputs_available(outputs, args.overwrite)
    frame = read_inventory(input_path, args.sheet)
    screen_input, inventory, summary = prepare_peptide_inventory(
        frame,
        sequence_column=args.sequence_column,
        identifier_column=args.identifier_column,
        residue_count_column=args.residue_count_column,
        n_terminus_column=args.n_terminus_column,
        c_terminus_column=args.c_terminus_column,
        cyclic_column=args.cyclic_column,
        modification_columns=args.modification_column,
        free_terminus_value=args.free_terminus_value,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    screen_input.to_csv(outputs["screen_input"], index=False)
    inventory.to_csv(outputs["inventory_rows"], index=False)
    payload = {
        "schema_version": 1,
        "input": {
            "path": str(input_path),
            "sha256": sha256(input_path),
            "sheet": args.sheet,
        },
        "columns": {
            "sequence": args.sequence_column,
            "identifier": args.identifier_column,
            "residue_count": args.residue_count_column,
            "n_terminus": args.n_terminus_column,
            "c_terminus": args.c_terminus_column,
            "cyclic": args.cyclic_column,
            "modification": args.modification_column,
        },
        "grain": "one source inventory row; source order and duplicates retained",
        "screen_protocol": {
            "sequence_normalization": "strip then uppercase",
            "structure_interpretation": (
                "sequence-only RDKit MolFromSequence; declared chemistry is "
                "retained for audit but is not encoded"
            ),
            "missing_sequence_policy": "retain row with X sentinel for explicit rejection",
            "deduplication": False,
        },
        "quality_summary": summary,
        "outputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in outputs.items()
            if name != "manifest"
        },
    }
    outputs["manifest"].write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def summarize(args: argparse.Namespace) -> dict:
    for path in (args.inventory, args.predictions, args.model_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    model_manifest = json.loads(args.model_manifest.read_text(encoding="utf-8"))
    batch_size = int(model_manifest.get("batch_size", 0))
    if batch_size != args.expected_batch_size:
        raise ValueError(
            "Model manifest batch size disagrees with the expected protocol: "
            f"{batch_size} != {args.expected_batch_size}"
        )
    if args.strain not in model_manifest.get("strains", []):
        raise ValueError(f"Strain {args.strain!r} is absent from the model manifest")
    manifest_model_limit = model_manifest.get("tokenization", {}).get(
        "resolved_model_max_length"
    )
    if args.max_token_length is None and manifest_model_limit is None:
        raise ValueError(
            "The model manifest does not record its resolved model length; "
            "pass --max-token-length explicitly"
        )
    max_token_length = int(
        args.max_token_length
        if args.max_token_length is not None
        else manifest_model_limit
    )
    if manifest_model_limit is not None and max_token_length != int(
        manifest_model_limit
    ):
        raise ValueError(
            "--max-token-length disagrees with the model manifest: "
            f"{max_token_length} != {manifest_model_limit}"
        )
    tokenizer_revision = model_manifest.get(
        "tokenizer_revision", args.tokenizer_revision
    )
    inventory = pd.read_csv(args.inventory, keep_default_na=False)
    predictions = pd.read_csv(args.predictions, keep_default_na=False)
    token_lengths = prediction_token_lengths(
        predictions,
        tokenizer_name=model_manifest["tokenizer"],
        tokenizer_revision=tokenizer_revision,
        model_max_length=max_token_length,
    )
    joined, all_hits, exact_hits, exact_in_stock_hits, summary = (
        summarize_peptide_inventory(
            inventory,
            predictions,
            strain=args.strain,
            mic_cutoff=args.mic_cutoff,
            token_lengths=token_lengths,
            max_token_length=max_token_length,
            stock_column=args.stock_column,
        )
    )
    output_directory = args.output_directory.resolve()
    slug = cutoff_slug(args.mic_cutoff)
    outputs = {
        "all_rows": output_directory / "screened_inventory.csv",
        "all_hits": output_directory
        / f"mic_le_{slug}_all_sequence_interpretations.csv",
        "exact_hits": output_directory / f"mic_le_{slug}_exact_unmodified.csv",
        "exact_in_stock_hits": output_directory
        / f"mic_le_{slug}_exact_unmodified_in_stock.csv",
        "summary": output_directory / "summary.json",
    }
    ensure_outputs_available(outputs, args.overwrite)
    output_directory.mkdir(parents=True, exist_ok=True)
    joined.to_csv(outputs["all_rows"], index=False)
    all_hits.to_csv(outputs["all_hits"], index=False)
    exact_hits.to_csv(outputs["exact_hits"], index=False)
    exact_in_stock_hits.to_csv(outputs["exact_in_stock_hits"], index=False)
    payload = {
        "schema_version": 1,
        "target": args.target_label or args.strain,
        "protocol": {
            "model": "canonical MDLM peptide-table MIC scorer",
            "strain_key": args.strain,
            "batch_size": batch_size,
            "max_token_length": max_token_length,
            "tokenizer": model_manifest["tokenizer"],
            "tokenizer_revision": tokenizer_revision,
            "mic_cutoff_umol": args.mic_cutoff,
            "stock_column": args.stock_column,
            "stock_unit": args.stock_unit,
            "ranking": "ascending predicted MIC; stable source_row_id tie-break",
            "activity_status": "model prioritization only; no wet-lab activity claim",
        },
        "inputs": {
            "inventory": {
                "path": str(args.inventory),
                "sha256": sha256(args.inventory),
            },
            "predictions": {
                "path": str(args.predictions),
                "sha256": sha256(args.predictions),
            },
            "model_manifest": {
                "path": str(args.model_manifest),
                "sha256": sha256(args.model_manifest),
            },
        },
        "summary": summary,
        "outputs": {
            key: {"path": str(path), "sha256": sha256(path), "rows": int(rows)}
            for key, path, rows in (
                ("all_rows", outputs["all_rows"], len(joined)),
                ("all_hits", outputs["all_hits"], len(all_hits)),
                ("exact_hits", outputs["exact_hits"], len(exact_hits)),
                (
                    "exact_in_stock_hits",
                    outputs["exact_in_stock_hits"],
                    len(exact_in_stock_hits),
                ),
            )
        },
    }
    outputs["summary"].write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    args = parse_args()
    payload = prepare(args) if args.command == "prepare" else summarize(args)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
