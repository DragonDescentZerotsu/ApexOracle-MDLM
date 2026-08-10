#!/usr/bin/env python
"""Score one generated SELFIES file with the canonical candidate MIC model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from transformers import AutoTokenizer

from apexoracle_mdlm.scoring import (
    load_candidate_mic_regressor,
    load_condition_embedding_banks,
    read_selfies_file,
    score_selfies_strings,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--config-name", default="config")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--genome-embeddings", type=Path, required=True)
    parser.add_argument("--atcc-text-embeddings", type=Path, required=True)
    parser.add_argument("--text-only-embeddings", type=Path, required=True)
    parser.add_argument("--generation-file", type=Path, required=True)
    parser.add_argument("--strain", required=True)
    parser.add_argument("--tokenizer", default="ibm-research/materials.selfies-ted")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
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

    with initialize_config_dir(
        config_dir=str(args.config_dir.resolve()),
        version_base=None,
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
    )
    selfies_strings = read_selfies_file(args.generation_file)
    predictions = score_selfies_strings(
        model,
        tokenizer,
        selfies_strings,
        strain=args.strain,
        device=device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("row_index", "strain", "selfies", "predicted_mic_umol"),
            lineterminator="\n",
        )
        writer.writeheader()
        for index, (selfies, prediction) in enumerate(
            zip(selfies_strings, predictions.tolist())
        ):
            writer.writerow(
                {
                    "row_index": index,
                    "strain": args.strain,
                    "selfies": selfies,
                    "predicted_mic_umol": format(prediction, ".10g"),
                }
            )

    manifest = {
        "schema_version": 1,
        "strain": args.strain,
        "tokenizer": args.tokenizer,
        "config_name": args.config_name,
        "config_dir": str(args.config_dir),
        "checkpoint": {
            "path": str(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
            "sha256": sha256(args.checkpoint) if args.hash_checkpoint else None,
        },
        "generation_input": {
            "path": str(args.generation_file),
            "sha256": sha256(args.generation_file),
            "rows": len(selfies_strings),
        },
        "condition_directories": {
            "genome": str(args.genome_embeddings),
            "atcc_text": str(args.atcc_text_embeddings),
            "text_only": str(args.text_only_embeddings),
        },
        "output": {
            "path": str(args.output),
            "sha256": sha256(args.output),
            "rows": len(predictions),
        },
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
