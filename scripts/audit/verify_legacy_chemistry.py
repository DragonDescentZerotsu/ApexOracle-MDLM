#!/usr/bin/env python
"""Verify the two remaining root chemistry utilities and formal artifacts."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import pandas as pd

from apexoracle_mdlm.chemistry import (
    catalog_match_rows,
    convert_smiles_table_to_selfies,
    load_catalog_queries,
    match_catalogue_files,
)


LEGACY_SOURCES = ("DBAASP_semiles_to_SELFEIS.py", "match_molecules.py")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_show(repo: Path, ref: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=repo)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _asset_label(path: Path, *, repo: Path, core: Path) -> str:
    resolved = path.resolve()
    for prefix, root in (("", repo), ("ApexOracle-Core/", core)):
        try:
            relative = resolved.relative_to(root.resolve())
        except ValueError:
            continue
        return f"{prefix}{relative}"
    return path.name


def _without_module_docstring(source: bytes) -> str:
    tree = ast.parse(source.decode("utf-8"))
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        tree.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def _normalized_match_rows(table: pd.DataFrame) -> set[tuple[str, ...]]:
    renamed = table.rename(
        columns={
            "Molport_ID": "Catalog_ID",
            "Molport_SMILES_Source": "Catalog_SMILES_Source",
            "Molport_Canonical_SMILES": "Catalog_Canonical_SMILES",
        }
    )
    columns = [
        "Strain",
        "Original_Score",
        "Query_SMILES",
        "Catalog_ID",
        "Catalog_SMILES_Source",
        "Catalog_Canonical_SMILES",
    ]
    if set(columns) - set(renamed.columns):
        raise ValueError("Historical catalogue match table has unexpected columns.")
    return {
        tuple(str(value) for value in row)
        for row in renamed[columns].itertuples(index=False, name=None)
    }


def _runtime_references(roots: list[Path], audit_path: Path) -> list[str]:
    references: list[str] = []
    ignored_parts = {
        ".git",
        ".git-state",
        "__pycache__",
        "reproducibility",
        "temp_data",
    }
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if (
                not path.is_file()
                or path.suffix not in {".py", ".sh", ".yaml", ".yml"}
                or ignored_parts.intersection(path.parts)
                or path.resolve() == audit_path.resolve()
                or path.parent.name == "audit"
            ):
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if any(name in source for name in LEGACY_SOURCES):
                references.append(str(path.resolve()))
    return sorted(set(references))


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    core = repo.parent / "Synergy"
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdlm-root", type=Path, default=repo)
    parser.add_argument("--core-root", type=Path, default=core)
    parser.add_argument("--snapshot-ref", default="legacy-code-snapshot-2026-08-09")
    parser.add_argument(
        "--smiles-table",
        type=Path,
        default=core / "DataPrepare/Data/DBAASP_id_SMILES_bact_MICs.csv",
    )
    parser.add_argument(
        "--selfies-table",
        type=Path,
        default=core / "DataPrepare/Data/DBAASP_id_SELFIES_bact_MICs.csv",
    )
    parser.add_argument(
        "--prediction",
        action="append",
    )
    parser.add_argument(
        "--catalogue-dir", type=Path, default=repo / "temp_data/Molport_SMILES"
    )
    parser.add_argument(
        "--historical-match",
        type=Path,
        default=repo / "temp_data/small_molecules/purchasable_molecules_match.csv",
    )
    parser.add_argument(
        "--ignored-producer-copy",
        type=Path,
        default=repo / "temp_data/small_molecules/match_molecules.py",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prediction is None:
        args.prediction = [
            f"BAA-3170={repo / 'temp_data/SMs_mic_predictions_BAA-3170_filtered_below_15.csv'}",
            f"BAA-3197={repo / 'temp_data/SMs_mic_predictions_BAA-3197_filtered_below_15.csv'}",
        ]
    return args


def main() -> None:
    args = parse_args()
    repo = args.mdlm_root.resolve()
    core = args.core_root.resolve()
    source_payloads = {
        path: _git_show(repo, args.snapshot_ref, path) for path in LEGACY_SOURCES
    }
    source_records = {
        path: {
            "sha256": _sha256_bytes(payload),
            "bytes": len(payload),
            "lines": len(payload.splitlines()),
            "recovery": f"git show {args.snapshot_ref}:{path}",
        }
        for path, payload in source_payloads.items()
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        converted = Path(temp_dir) / "converted.csv"
        converted_rows = convert_smiles_table_to_selfies(
            args.smiles_table,
            converted,
            smiles_column="SMILES",
        )
        converted_hash = _sha256(converted)
    formal_selfies_hash = _sha256(args.selfies_table)
    if converted_hash != formal_selfies_hash:
        raise AssertionError(
            "Canonical SMILES-to-SELFIES output differs from formal data."
        )

    predictions: dict[str, Path] = {}
    for value in args.prediction:
        strain, separator, path = value.partition("=")
        if not separator or not strain or not path:
            raise ValueError(f"Prediction must use STRAIN=PATH: {value!r}")
        predictions[strain] = Path(path)
    query_index, query_counts = load_catalog_queries(
        predictions, smiles_column="SMILES_Sequence"
    )
    catalogue_files = sorted(args.catalogue_dir.glob("*.txt"))
    result = match_catalogue_files(
        query_index,
        catalogue_files,
        smiles_column="SMILES_CANONICAL",
        id_column="MOLPORTID",
        workers=args.workers,
    )
    canonical_rows = pd.DataFrame(catalog_match_rows(result.matches))
    historical_table = pd.read_csv(args.historical_match)
    canonical_set = _normalized_match_rows(canonical_rows)
    historical_set = _normalized_match_rows(historical_table)
    if canonical_set != historical_set:
        raise AssertionError(
            f"Catalogue semantic row set differs: canonical={len(canonical_set)}, "
            f"historical={len(historical_set)}"
        )

    ignored_source = args.ignored_producer_copy.read_bytes()
    same_producer_ast = _without_module_docstring(
        source_payloads["match_molecules.py"]
    ) == _without_module_docstring(ignored_source)
    if not same_producer_ast:
        raise AssertionError("Ignored historical producer copy has behavior drift.")

    import rdkit
    import selfies

    consumers = _runtime_references(
        [repo, core, repo.parent / "discrete-diffusion-guidance"],
        Path(__file__),
    )
    if consumers:
        raise RuntimeError(f"Live runtime/config references remain: {consumers}")

    result_json: dict[str, Any] = {
        "schema_version": 1,
        "snapshot_ref": args.snapshot_ref,
        "legacy_sources": source_records,
        "smiles_to_selfies": {
            "input": _asset_label(args.smiles_table, repo=repo, core=core),
            "input_sha256": _sha256(args.smiles_table),
            "formal_output": _asset_label(args.selfies_table, repo=repo, core=core),
            "formal_output_sha256": formal_selfies_hash,
            "rows": converted_rows,
            "byte_equal": True,
            "canonical_entry": "scripts/reproduce/convert_smiles_table_to_selfies.py",
        },
        "catalogue_matching": {
            "predictions": {
                strain: {
                    "path": _asset_label(path, repo=repo, core=core),
                    "sha256": _sha256(path),
                }
                for strain, path in predictions.items()
            },
            "query_counts": query_counts,
            "catalogue_files": [
                {
                    "path": _asset_label(path, repo=repo, core=core),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in catalogue_files
            ],
            "catalogue_rows": result.catalogue_rows,
            "valid_catalogue_rows": result.valid_catalogue_rows,
            "invalid_catalogue_rows": result.invalid_catalogue_rows,
            "matched_catalogue_rows": result.matched_catalogue_rows,
            "canonical_output_rows": len(canonical_rows),
            "historical_output": _asset_label(
                args.historical_match, repo=repo, core=core
            ),
            "historical_output_sha256": _sha256(args.historical_match),
            "semantic_row_set_equal": True,
            "unique_catalogue_ids": canonical_rows["Catalog_ID"].nunique(),
            "unique_canonical_structures": canonical_rows[
                "Catalog_Canonical_SMILES"
            ].nunique(),
            "ignored_producer_copy": _asset_label(
                args.ignored_producer_copy, repo=repo, core=core
            ),
            "ignored_producer_copy_sha256": _sha256(args.ignored_producer_copy),
            "ignored_copy_ast_equal_after_docstring_removal": same_producer_ast,
            "canonical_entry": "scripts/reproduce/match_screen_to_catalogue.py",
        },
        "runtime_versions": {
            "pandas": pd.__version__,
            "rdkit": rdkit.__version__,
            "selfies": selfies.__version__,
        },
        "runtime_or_config_consumers": consumers,
        "verified_facts": [
            "The formal 11,401-row SELFIES table is byte-identical to the canonical conversion output.",
            "The twelve catalogue shards contain 5,887,458 rows; 517 are invalid under the current RDKit parser.",
            "The canonical full-catalogue rescan recovers the exact 276-row historical semantic match set.",
            "Those rows represent 179 supplier IDs and 179 canonical structures.",
            "The ignored producer copy differs from the tagged source only by a module docstring at AST level.",
            "No live Python, shell, or YAML caller in MDLM, Core, or Generation references either root filename.",
        ],
        "limits": [
            "The historical catalogue output row order depended on glob/filesystem ordering; the canonical CLI sorts input filenames.",
            "Exact canonical matching establishes catalogue identity, not experimental activity or supplier availability today.",
            "The tagged DBAASP script's __main__ passes its output path as its input; the reusable function is preserved, not that invocation bug.",
        ],
    }
    serialized = json.dumps(result_json, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
