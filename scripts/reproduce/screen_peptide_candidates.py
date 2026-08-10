#!/usr/bin/env python
"""Screen explicit structure-encoded peptide pool/strain jobs."""

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
    load_peptide_screen_jobs,
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
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=Path)
    input_group.add_argument("--job-manifest", type=Path)
    parser.add_argument("--strains", nargs="+")
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
    if args.input is not None:
        if not args.strains:
            raise ValueError("--strains is required with --input.")
        if len(set(args.strains)) != len(args.strains):
            raise ValueError("--strains must not contain duplicates.")
        jobs = [
            {
                "job_id": f"strain_{strain}",
                "strain": strain,
                "input_path": args.input.resolve(),
            }
            for strain in args.strains
        ]
    else:
        if args.strains:
            raise ValueError("--strains cannot be combined with --job-manifest.")
        jobs = [
            {
                "job_id": job.job_id,
                "strain": job.strain,
                "input_path": job.input_path,
            }
            for job in load_peptide_screen_jobs(args.job_manifest)
        ]
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
        "job_id",
        "input_path",
        "row_index",
        "source_selfies",
        "smiles",
        "predicted_mic_umol",
        "peptide_sequence",
        "output_selfies",
        "qualification_status",
        "invalid_reason",
    ]
    job_outputs = {}
    input_cache = {}
    total_scored_rows = 0
    with screen_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for job in jobs:
            input_path = job["input_path"]
            if input_path not in input_cache:
                input_cache[input_path] = read_selfies_file(input_path)
            source_selfies = input_cache[input_path]
            selfies_strings = (
                source_selfies if args.limit is None else source_selfies[: args.limit]
            )
            total_scored_rows += len(selfies_strings)
            mic_values = score_selfies_strings(
                model,
                tokenizer,
                selfies_strings,
                strain=job["strain"],
                device=device,
            )
            results = qualify_peptide_candidates(
                selfies_strings,
                mic_values.tolist(),
                mic_threshold=args.mic_threshold,
            )
            for result in results:
                writer.writerow(
                    {
                        "strain": job["strain"],
                        "job_id": job["job_id"],
                        "input_path": str(input_path),
                        **result.as_row(),
                    }
                )
            qualified = [
                result
                for result in results
                if result.qualification_status == "qualified"
            ]
            qualified_path = qualified_directory / f"{job['job_id']}.txt"
            qualified_path.write_text(
                "\n".join(result.output_selfies for result in qualified),
                encoding="utf-8",
            )
            images = None
            if args.draw_qualified:
                image_directory = image_root / job["job_id"]
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
            job_outputs[job["job_id"]] = {
                "strain": job["strain"],
                "input": {
                    "path": str(input_path),
                    "sha256": sha256(input_path),
                    "source_rows": len(source_selfies),
                    "scored_rows": len(selfies_strings),
                },
                "qualification": qualification_summary(results),
                "qualified_selfies": {
                    "path": str(qualified_path),
                    "sha256": sha256(qualified_path),
                    "rows": len(qualified),
                },
                "images": images,
            }

    manifest = {
        "schema_version": 2,
        "protocol": {
            "input_format": "one_selfies_per_line",
            "molecule_batch_size": 1,
            "padding_removed_before_model": True,
            "mic_threshold_umol": args.mic_threshold,
            "peptide_parser": "apexoracle_mdlm.chemistry.smiles_to_peptide_sequence",
            "unknown_residue_policy": "exclude_sequence_containing_uppercase_X",
            "drawing_is_qualification_gating": False,
        },
        "input_mode": "shared_pool" if args.input is not None else "job_manifest",
        "job_manifest": (
            {
                "path": str(args.job_manifest),
                "sha256": sha256(args.job_manifest),
            }
            if args.job_manifest is not None
            else None
        ),
        "jobs": [
            {
                "job_id": job["job_id"],
                "strain": job["strain"],
                "input_path": str(job["input_path"]),
            }
            for job in jobs
        ],
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
                "rows": total_scored_rows,
            },
            "by_job": job_outputs,
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
