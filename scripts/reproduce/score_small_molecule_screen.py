#!/usr/bin/env python
"""Run the paper small-molecule MIC screen from explicit strain inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from transformers import AutoTokenizer

from apexoracle_mdlm.figures import plot_mic_distribution
from apexoracle_mdlm.scoring import (
    decoded_wide_rows,
    load_candidate_mic_regressor,
    load_condition_embedding_banks,
    load_strain_inputs,
    score_small_molecule_inputs,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, default=repository_root)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--config-name", default="config")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--genome-embeddings", type=Path, required=True)
    parser.add_argument("--atcc-text-embeddings", type=Path, required=True)
    parser.add_argument("--text-only-embeddings", type=Path, required=True)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="STRAIN=PATH",
        help="Repeat once per target strain; strain names are never inferred.",
    )
    parser.add_argument("--tokenizer", default="ibm-research/materials.selfies-ted")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument(
        "--hash-checkpoint",
        action="store_true",
        help="Read and SHA-256 the large checkpoint for the output manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    inputs = load_strain_inputs(args.input)

    with initialize_config_dir(
        config_dir=str(args.config_dir.resolve()), version_base=None
    ):
        config = compose(config_name=args.config_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
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
    screens = score_small_molecule_inputs(model, tokenizer, inputs, device=device)
    rows = decoded_wide_rows(screens)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["SMILES_Sequence", *screens]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    figures = {}
    if args.figure_dir is not None:
        args.figure_dir.mkdir(parents=True, exist_ok=True)
        for strain, screen in screens.items():
            figure, _ = plot_mic_distribution(screen.mic_values.tolist(), strain=strain)
            path = args.figure_dir / f"strain_{strain}_MIC_distribution.pdf"
            figure.savefig(path, format="pdf", bbox_inches="tight", dpi=300)
            figures[strain] = {"path": str(path), "sha256": sha256(path)}

    manifest = {
        "schema_version": 1,
        "protocol": {
            "molecule_batch_size": 1,
            "padding_removed_before_model": True,
            "duplicate_selfies_rule": "last_prediction_wins_per_strain",
            "wide_row_order": "lexicographic_source_selfies",
            "structure_column": "SMILES_Sequence",
        },
        "tokenizer": args.tokenizer,
        "config_name": args.config_name,
        "config_dir": str(args.config_dir),
        "runtime_root": str(args.runtime_root.resolve()),
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
        "inputs": {
            strain: {
                "path": str(screen.source_path),
                "sha256": sha256(screen.source_path),
                "rows": len(screen.selfies_strings),
                "unique_selfies": len(screen.mic_by_selfies),
            }
            for strain, screen in screens.items()
        },
        "output": {
            "path": str(args.output),
            "sha256": sha256(args.output),
            "rows": len(rows),
            "columns": fieldnames,
        },
        "figures": figures,
    }
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
