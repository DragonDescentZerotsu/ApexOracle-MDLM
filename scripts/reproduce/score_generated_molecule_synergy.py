#!/usr/bin/env python
"""Score generated SELFIES with the experimental symmetric-pair classifier."""

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
    load_candidate_synergy_classifier,
    load_condition_embedding_banks,
    load_partner_embedding,
    read_selfies_file,
    score_selfies_synergy,
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
    parser.add_argument("--partner-embeddings", type=Path, required=True)
    parser.add_argument("--partner-key", required=True)
    parser.add_argument(
        "--partner-key-type",
        choices=("string", "integer"),
        default="string",
    )
    parser.add_argument("--genome-embeddings", type=Path, required=True)
    parser.add_argument("--atcc-text-embeddings", type=Path, required=True)
    parser.add_argument("--text-only-embeddings", type=Path, required=True)
    parser.add_argument("--generation-file", type=Path, required=True)
    parser.add_argument("--strain", required=True)
    parser.add_argument("--tokenizer", default="ibm-research/materials.selfies-ted")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--hash-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    partner_key: str | int = args.partner_key
    if args.partner_key_type == "integer":
        try:
            partner_key = int(args.partner_key)
        except ValueError as error:
            raise ValueError("--partner-key must be an integer.") from error

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
    partner_embedding = load_partner_embedding(args.partner_embeddings, partner_key)
    model = load_candidate_synergy_classifier(
        config,
        vocab_size=len(tokenizer.get_vocab()),
        condition_embeddings=banks,
        partner_embedding=partner_embedding,
        checkpoint_path=args.checkpoint,
        device=device,
        runtime_root=args.runtime_root.resolve(),
    )
    selfies_strings = read_selfies_file(args.generation_file)
    probabilities = score_selfies_synergy(
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
            fieldnames=(
                "row_index",
                "strain",
                "partner_key",
                "partner_key_type",
                "selfies",
                "synergy_probability",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for index, (selfies, probability) in enumerate(
            zip(selfies_strings, probabilities.tolist())
        ):
            writer.writerow(
                {
                    "row_index": index,
                    "strain": args.strain,
                    "partner_key": partner_key,
                    "partner_key_type": args.partner_key_type,
                    "selfies": selfies,
                    "synergy_probability": format(probability, ".10g"),
                }
            )

    manifest = {
        "schema_version": 1,
        "model_profile": "experimental_all_data_symmetric_pair_classifier",
        "paper_core_synergy_cv_model": False,
        "strain": args.strain,
        "partner_key": {"value": partner_key, "type": args.partner_key_type},
        "tokenizer": args.tokenizer,
        "runtime_root": str(args.runtime_root.resolve()),
        "checkpoint": {
            "path": str(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
            "sha256": sha256(args.checkpoint) if args.hash_checkpoint else None,
        },
        "partner_embeddings": {
            "path": str(args.partner_embeddings),
            "bytes": args.partner_embeddings.stat().st_size,
            "sha256": sha256(args.partner_embeddings),
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
        "protocol": {
            "molecule_batch_size": 1,
            "padding_removed_before_model": True,
            "pair_order": "mean(head(candidate,partner),head(partner,candidate))",
            "output": "sigmoid_probability",
        },
        "output": {
            "path": str(args.output),
            "sha256": sha256(args.output),
            "rows": len(probabilities),
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
