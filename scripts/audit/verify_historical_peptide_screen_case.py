#!/usr/bin/env python
"""Verify one historical peptide-screen case without making it a public API."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import selfies
from PIL import Image

from apexoracle_mdlm.chemistry import smiles_to_peptide_sequence
from apexoracle_mdlm.figures import render_annotated_candidate


IMAGE_NAME = re.compile(r"^mol_(?P<row>\d+)_mic_(?P<mic>\d+(?:\.\d+)?)\.png$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_manifest(path: Path) -> dict[str, object]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    total_bytes = 0
    for item in files:
        relative = item.relative_to(path).as_posix()
        size = item.stat().st_size
        item_hash = sha256(item)
        digest.update(f"{relative}\0{size}\0{item_hash}\n".encode())
        total_bytes += size
    return {
        "path": str(path),
        "files": len(files),
        "bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("legacy_smiles_to_peptide", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import legacy parser from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--legacy-ref", default="legacy-code-snapshot-2026-08-09")
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--qualified-directory", type=Path, required=True)
    parser.add_argument("--image-directory", type=Path, required=True)
    parser.add_argument(
        "--strains", nargs="+", default=["BAA-999", "15700", "15697", "23272", "4356"]
    )
    parser.add_argument("--mic-threshold", type=float, default=15.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def verify(args: argparse.Namespace, legacy_parser) -> dict:
    input_files = sorted(args.input_directory.glob("strain_*.txt"))
    if not input_files:
        raise FileNotFoundError(f"No strain_*.txt files in {args.input_directory}.")
    input_records = []
    source_rows = None
    for path in input_files:
        rows = path.read_text(encoding="utf-8").splitlines()
        record = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "rows": len(rows),
        }
        input_records.append(record)
        if source_rows is None:
            source_rows = rows
        elif rows != source_rows:
            raise AssertionError(f"Historical strain input differs: {path}.")
    assert source_rows is not None

    strain_records = {}
    parser_comparisons = 0
    raster_candidate = None
    for strain in args.strains:
        qualified_path = args.qualified_directory / f"strain_{strain}.txt"
        image_path = args.image_directory / f"strain_{strain}"
        qualified_rows = qualified_path.read_text(encoding="utf-8").splitlines()
        images = []
        for path in image_path.glob("*.png"):
            match = IMAGE_NAME.fullmatch(path.name)
            if match is None:
                raise AssertionError(f"Unexpected historical image name: {path}.")
            images.append((int(match.group("row")), float(match.group("mic")), path))
        images.sort(key=lambda item: item[0])
        if len({row for row, _, _ in images}) != len(images):
            raise AssertionError(f"Duplicate image row index for {strain}.")
        if len(images) != len(qualified_rows):
            raise AssertionError(
                f"Qualified/image count mismatch for {strain}: "
                f"{len(qualified_rows)} != {len(images)}."
            )

        reconstructed = []
        sequences = []
        for row_index, mic, image in images:
            if row_index >= len(source_rows):
                raise AssertionError(f"Image row is outside source pool: {image}.")
            if not 0 < mic <= args.mic_threshold:
                raise AssertionError(f"Image MIC is outside threshold: {image}.")
            smiles = selfies.decoder(source_rows[row_index])
            legacy_result = legacy_parser.smiles_to_pepseq(smiles)
            canonical_result = smiles_to_peptide_sequence(smiles)
            parser_comparisons += 1
            if legacy_result != canonical_result:
                raise AssertionError(
                    f"Parser parity failed for {strain} row {row_index}."
                )
            _, sequence = canonical_result
            if sequence is None or "X" in sequence:
                raise AssertionError(
                    f"Historical qualified row no longer parses: {strain} {row_index}."
                )
            reconstructed.append(selfies.encoder(smiles))
            sequences.append(sequence)
            if raster_candidate is None:
                raster_candidate = (smiles, mic, sequence, image)
        if reconstructed != qualified_rows:
            raise AssertionError(f"Qualified SELFIES provenance failed for {strain}.")
        strain_records[strain] = {
            "qualified_selfies": {
                "path": str(qualified_path),
                "bytes": qualified_path.stat().st_size,
                "sha256": sha256(qualified_path),
                "rows": len(qualified_rows),
            },
            "images": tree_manifest(image_path),
            "row_index_range": [images[0][0], images[-1][0]] if images else None,
            "rounded_mic_range_umol": (
                [
                    min(item[1] for item in images),
                    max(item[1] for item in images),
                ]
                if images
                else None
            ),
        }

    if raster_candidate is None:
        raise AssertionError("No historical candidate image was available for parity.")
    smiles, mic, sequence, historical_image = raster_candidate
    canonical_image = render_annotated_candidate(
        smiles, predicted_mic_umol=mic, peptide_sequence=sequence
    )
    historical_pixels = np.asarray(Image.open(historical_image).convert("RGB"))
    canonical_pixels = np.asarray(canonical_image.convert("RGB"))
    difference = np.abs(
        historical_pixels.astype(np.int16) - canonical_pixels.astype(np.int16)
    )
    raster_parity = {
        "historical_path": str(historical_image),
        "shape": list(historical_pixels.shape),
        "exact_pixel_fraction": float(np.mean(difference == 0)),
        "mean_absolute_channel_difference": float(difference.mean()),
        "max_channel_difference": int(difference.max()),
    }
    if raster_parity["exact_pixel_fraction"] != 1.0:
        raise AssertionError(f"Historical raster parity failed: {raster_parity}")

    hashes = {record["sha256"] for record in input_records}
    return {
        "schema_version": 1,
        "scope": "historical external-project peptide screening case",
        "input_copies": input_records,
        "shared_candidate_pool": {
            "all_input_files_identical": len(hashes) == 1,
            "rows": len(source_rows),
            "sha256": next(iter(hashes)) if len(hashes) == 1 else None,
        },
        "protocol": {
            "mic_threshold_umol": args.mic_threshold,
            "qualification": "MIC <= threshold, parser returned sequence, uppercase X absent",
            "qualified_selfies": "SELFIES re-encoding of decoded source structure",
        },
        "strains": strain_records,
        "parser_parity": {
            "legacy_source": f"git:{args.legacy_ref}:smiles_to_peptide.py",
            "comparisons": parser_comparisons,
            "all_equal": True,
        },
        "raster_parity": raster_parity,
        "evidence_boundary": {
            "verified": "Input copies, qualified rows, image row/MIC naming, parser outputs, and one exact raster replay.",
            "not_verified": "No timestamped source revision or complete historical MIC table proves exact excluded-row replay.",
        },
        "status": "passed",
    }


def main() -> None:
    args = parse_args()
    source = subprocess.check_output(
        ["git", "show", f"{args.legacy_ref}:smiles_to_peptide.py"],
        cwd=args.repo_root,
    )
    with tempfile.TemporaryDirectory(
        prefix="apexoracle_legacy_peptide_parser_"
    ) as temp:
        path = Path(temp) / "smiles_to_peptide.py"
        path.write_bytes(source)
        result = verify(args, load_module(path))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
