#!/usr/bin/env python
"""Score one structure-encoded peptide pool across explicit strain conditions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from transformers import AutoTokenizer

from apexoracle_mdlm.figures import render_annotated_candidate
from apexoracle_mdlm.scoring import (
    load_candidate_mic_regressor,
    load_condition_embedding_banks,
    qualification_summary,
    qualify_peptide_candidates,
    read_selfies_file,
    score_selfies_strings,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_manifest(path: Path) -> dict[str, object]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    total_bytes = 0
    for item in files:
        relative = item.relative_to(path).as_posix()
        file_hash = sha256(item)
        size = item.stat().st_size
        digest.update(f"{relative}\0{size}\0{file_hash}\n".encode())
        total_bytes += size
    return {
        "path": str(path),
        "files": len(files),
        "bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, default=repository_root)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--strains", nargs="+", required=True)
    parser.add_argument("--mic-threshold", type=float, default=15.0)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--config-name", default="config")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--genome-embeddings", type=Path, required=True)
    parser.add_argument("--atcc-text-embeddings", type=Path, required=True)
    parser.add_argument("--text-only-embeddings", type=Path, required=True)
    parser.add_argument("--tokenizer", default="ibm-research/materials.selfies-ted")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--draw-qualified", action="store_true")
    parser.add_argument("--hash-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.strains)) != len(args.strains):
        raise ValueError("--strains must not contain duplicates.")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative.")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_directory}."
        )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "The attributed upstream DiT runtime requires an available CUDA device."
        )

    selfies_strings = read_selfies_file(args.input)
    if args.limit is not None:
        selfies_strings = selfies_strings[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    with initialize_config_dir(
        config_dir=str(args.config_dir.resolve()), version_base=None
    ):
        config = compose(config_name=args.config_name)
    banks = load_condition_embedding_banks(
        genome_directory=args.genome_embeddings,
        atcc_text_directory=args.atcc_text_embeddings,
        text_only_directory=args.text_only_embeddings,
    )
    model = load_candidate_mic_regressor(
        config,
        vocab_size=len(tokenizer.get_vocab()),
        condition_embeddings=banks,
        checkpoint_path=args.checkpoint,
        device=device,
        runtime_root=args.runtime_root.resolve(),
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    screen_path = args.output_directory / "candidate_screen.csv"
    qualified_directory = args.output_directory / "qualified_selfies"
    qualified_directory.mkdir()
    image_root = args.output_directory / "molecule_images"
    if args.draw_qualified:
        image_root.mkdir()
    fieldnames = [
        "strain",
        "row_index",
        "source_selfies",
        "smiles",
        "predicted_mic_umol",
        "peptide_sequence",
        "output_selfies",
        "qualification_status",
        "invalid_reason",
    ]
    strain_outputs = {}
    with screen_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for strain in args.strains:
            mic_values = score_selfies_strings(
                model,
                tokenizer,
                selfies_strings,
                strain=strain,
                device=device,
            )
            results = qualify_peptide_candidates(
                selfies_strings,
                mic_values.tolist(),
                mic_threshold=args.mic_threshold,
            )
            for result in results:
                writer.writerow({"strain": strain, **result.as_row()})
            qualified = [
                result
                for result in results
                if result.qualification_status == "qualified"
            ]
            qualified_path = qualified_directory / f"strain_{strain}.txt"
            qualified_path.write_text(
                "\n".join(result.output_selfies for result in qualified),
                encoding="utf-8",
            )
            images = None
            if args.draw_qualified:
                image_directory = image_root / f"strain_{strain}"
                image_directory.mkdir()
                for result in qualified:
                    image = render_annotated_candidate(
                        result.smiles,
                        predicted_mic_umol=result.predicted_mic_umol,
                        peptide_sequence=result.peptide_sequence,
                    )
                    image.save(
                        image_directory
                        / (
                            f"row_{result.row_index}_mic_"
                            f"{result.predicted_mic_umol:.2f}.png"
                        )
                    )
                images = directory_manifest(image_directory)
            strain_outputs[strain] = {
                "qualification": qualification_summary(results),
                "qualified_selfies": {
                    "path": str(qualified_path),
                    "sha256": sha256(qualified_path),
                    "rows": len(qualified),
                },
                "images": images,
            }

    manifest = {
        "schema_version": 1,
        "protocol": {
            "input_format": "one_selfies_per_line",
            "molecule_batch_size": 1,
            "padding_removed_before_model": True,
            "mic_threshold_umol": args.mic_threshold,
            "peptide_parser": "apexoracle_mdlm.chemistry.smiles_to_peptide_sequence",
            "unknown_residue_policy": "exclude_sequence_containing_uppercase_X",
            "drawing_is_qualification_gating": False,
        },
        "input": {
            "path": str(args.input),
            "sha256": sha256(args.input),
            "source_rows": len(read_selfies_file(args.input)),
            "scored_rows": len(selfies_strings),
        },
        "strains": args.strains,
        "runtime_root": str(args.runtime_root.resolve()),
        "tokenizer": args.tokenizer,
        "checkpoint": {
            "path": str(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
            "sha256": sha256(args.checkpoint) if args.hash_checkpoint else None,
        },
        "condition_directories": {
            "genome": str(args.genome_embeddings),
            "atcc_text": str(args.atcc_text_embeddings),
            "text_only": str(args.text_only_embeddings),
        },
        "outputs": {
            "screen": {
                "path": str(screen_path),
                "sha256": sha256(screen_path),
                "rows": len(selfies_strings) * len(args.strains),
            },
            "by_strain": strain_outputs,
        },
    }
    manifest_path = args.output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
