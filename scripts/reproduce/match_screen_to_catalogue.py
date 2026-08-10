#!/usr/bin/env python
"""Match scored structures to a tabular supplier catalogue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from apexoracle_mdlm.chemistry import (
    CATALOG_MATCH_COLUMNS,
    catalog_match_rows,
    load_catalog_queries,
    match_catalogue_files,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prediction(value: str) -> tuple[str, Path]:
    strain, separator, path = value.partition("=")
    if not separator or not strain.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Prediction must use STRAIN=PATH.")
    return strain.strip(), Path(path.strip()).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        type=_prediction,
        metavar="STRAIN=PATH",
    )
    parser.add_argument("--catalogue-dir", type=Path, required=True)
    parser.add_argument("--catalogue-pattern", default="*.txt")
    parser.add_argument("--catalogue-smiles-column", required=True)
    parser.add_argument("--catalogue-id-column", required=True)
    parser.add_argument("--query-smiles-column", required=True)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = dict(args.prediction)
    if len(predictions) != len(args.prediction):
        raise ValueError("Each strain may appear only once.")
    catalogue_files = sorted(args.catalogue_dir.glob(args.catalogue_pattern))
    if not catalogue_files:
        raise FileNotFoundError(
            f"No catalogue files match {args.catalogue_dir / args.catalogue_pattern}"
        )
    query_index, query_counts = load_catalog_queries(
        predictions,
        smiles_column=args.query_smiles_column,
    )
    result = match_catalogue_files(
        query_index,
        catalogue_files,
        smiles_column=args.catalogue_smiles_column,
        id_column=args.catalogue_id_column,
        chunk_size=args.chunk_size,
        workers=args.workers,
    )
    rows = catalog_match_rows(result.matches)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=CATALOG_MATCH_COLUMNS).to_csv(args.output, index=False)

    unique_ids = {match.entry.identifier for match in result.matches}
    unique_structures = {match.entry.canonical_smiles for match in result.matches}
    manifest = {
        "schema_version": 1,
        "match_rule": "RDKit canonical isomeric SMILES exact equality",
        "predictions": {
            strain: {"path": str(path), "sha256": _sha256(path)}
            for strain, path in predictions.items()
        },
        "query_counts": query_counts,
        "catalogue": {
            "directory": str(args.catalogue_dir),
            "pattern": args.catalogue_pattern,
            "files": [
                {"path": str(path), "sha256": _sha256(path)} for path in catalogue_files
            ],
            "rows": result.catalogue_rows,
            "valid_rows": result.valid_catalogue_rows,
            "invalid_rows": result.invalid_catalogue_rows,
            "matched_rows": result.matched_catalogue_rows,
        },
        "output": {
            "path": str(args.output),
            "sha256": _sha256(args.output),
            "rows": len(rows),
            "unique_catalogue_ids": len(unique_ids),
            "unique_canonical_structures": len(unique_structures),
        },
    }
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
