#!/usr/bin/env python
"""Export clean-input DLM molecule embeddings from a portable data adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from tqdm import tqdm
from transformers import AutoTokenizer

from apexoracle_mdlm.embeddings import (
    LEGACY_POOLING_METHODS,
    collect_pair_smiles_tokens,
    embedding_dictionary_schema,
    export_molecule_embeddings,
    load_token_id_csv,
)
from apexoracle_mdlm.models import build_upstream_dlm_hidden_state_encoder


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, default=root)
    parser.add_argument("--config-dir", type=Path, default=root / "configs")
    parser.add_argument("--config-name", default="config")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", default="ibm-research/materials.selfies-ted")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--pooling-method", choices=LEGACY_POOLING_METHODS, required=True
    )
    parser.add_argument("--model-mode", choices=("eval", "train"), default="eval")
    parser.add_argument("--padded-length", type=int, default=1024)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--hash-checkpoint", action="store_true")
    parser.add_argument("--hash-input", action="store_true")

    subparsers = parser.add_subparsers(dest="adapter", required=True)
    token = subparsers.add_parser("token-csv")
    token.add_argument("--id-column", default="DBAASP_id")
    token.add_argument("--token-column", default="SMILES")
    token.add_argument("--id-type", choices=("string", "integer"), default="string")

    pair = subparsers.add_parser("pair-smiles-csv")
    pair.add_argument("--first-id-column", default="DBAASP_id")
    pair.add_argument("--second-id-column", default="antibio_id_or_name")
    pair.add_argument("--first-smiles-column", default="AMP_smiles")
    pair.add_argument("--second-smiles-column", default="antibiotic_smiles")
    pair.add_argument(
        "--first-id-type", choices=("string", "integer"), default="integer"
    )
    pair.add_argument(
        "--second-id-type", choices=("string", "integer"), default="string"
    )
    pair.add_argument("--max-length", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.pooling_method.endswith("_eval") and args.model_mode != "eval":
        raise ValueError("A '*_eval' pooling alias requires --model-mode eval.")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.adapter == "token-csv":
        token_ids = load_token_id_csv(
            args.input,
            id_column=args.id_column,
            token_column=args.token_column,
            id_type=args.id_type,
        )
        skipped_unknown = skipped_too_long = 0
        adapter_config = {
            "id_column": args.id_column,
            "token_column": args.token_column,
            "id_type": args.id_type,
        }
    else:
        import selfies

        token_ids, skipped_unknown, skipped_too_long = collect_pair_smiles_tokens(
            args.input,
            tokenizer=tokenizer,
            smiles_to_selfies=selfies.encoder,
            first_id_column=args.first_id_column,
            second_id_column=args.second_id_column,
            first_smiles_column=args.first_smiles_column,
            second_smiles_column=args.second_smiles_column,
            first_id_type=args.first_id_type,
            second_id_type=args.second_id_type,
            max_length=args.max_length,
        )
        adapter_config = {
            "first_id_column": args.first_id_column,
            "second_id_column": args.second_id_column,
            "first_smiles_column": args.first_smiles_column,
            "second_smiles_column": args.second_smiles_column,
            "first_id_type": args.first_id_type,
            "second_id_type": args.second_id_type,
            "max_length": args.max_length,
        }

    with initialize_config_dir(
        config_dir=str(args.config_dir.resolve()),
        version_base=None,
    ):
        config = compose(config_name=args.config_name)
    encoder = build_upstream_dlm_hidden_state_encoder(
        config,
        len(tokenizer.get_vocab()),
        runtime_root=args.runtime_root,
    )
    encoder.mask_index = tokenizer.mask_token_id
    missing_keys, unexpected_keys = encoder.load_backbone_checkpoint(args.checkpoint)
    result = export_molecule_embeddings(
        encoder,
        token_ids,
        pooling_method=args.pooling_method,
        pad_token_id=tokenizer.pad_token_id,
        device=device,
        model_mode=args.model_mode,
        padded_length=args.padded_length,
        skipped_unknown=skipped_unknown,
        skipped_too_long=skipped_too_long,
        progress=lambda items: tqdm(
            items, total=len(token_ids), desc="embedding molecules"
        ),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.embeddings, args.output)
    manifest = {
        "schema_version": 1,
        "adapter": args.adapter,
        "adapter_config": adapter_config,
        "tokenizer": args.tokenizer,
        "config_name": args.config_name,
        "runtime_root": str(args.runtime_root.resolve()),
        "pooling_method": args.pooling_method,
        "model_mode": args.model_mode,
        "padded_length": args.padded_length,
        "input": {
            "path": str(args.input),
            "bytes": args.input.stat().st_size,
            "sha256": sha256(args.input) if args.hash_input else None,
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
            "sha256": sha256(args.checkpoint) if args.hash_checkpoint else None,
            "missing_backbone_keys": missing_keys,
            "unexpected_checkpoint_keys": unexpected_keys,
        },
        "counts": {
            "deduplicated_inputs": result.input_count,
            "embeddings": result.output_count,
            "skipped_unknown": result.skipped_unknown,
            "skipped_too_long": result.skipped_too_long,
        },
        "output": {
            "path": str(args.output),
            "bytes": args.output.stat().st_size,
            "sha256": sha256(args.output),
            "schema": embedding_dictionary_schema(result.embeddings),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
