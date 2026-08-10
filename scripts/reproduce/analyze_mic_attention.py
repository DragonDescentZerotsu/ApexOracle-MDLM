#!/usr/bin/env python
"""Export MIC prediction attention with verified genome-window annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import selfies as sf
import torch
from hydra import compose, initialize_config_dir
from transformers import AutoTokenizer

from apexoracle_mdlm.interpretability import (
    annotate_selected_windows,
    attention_rows,
    indexed_attention_rows,
    load_verified_genome_assets,
    score_single_selfies_attention,
)
from apexoracle_mdlm.scoring import (
    load_candidate_mic_regressor,
    load_condition_embedding_banks,
    regression_logit_to_mic,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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
    parser.add_argument("--genome-fasta", type=Path, required=True)
    parser.add_argument("--genome-genbank", type=Path, required=True)
    parser.add_argument("--genome-embedding", type=Path, required=True)
    parser.add_argument("--molecule-file", type=Path, required=True)
    parser.add_argument(
        "--molecule-format", choices=("smiles", "selfies"), required=True
    )
    parser.add_argument("--strain", required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--tokenizer", default="ibm-research/materials.selfies-ted")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hash-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    molecule = args.molecule_file.read_text(encoding="utf-8").strip()
    if not molecule or "\n" in molecule:
        raise ValueError("--molecule-file must contain exactly one non-empty row.")
    selfies = sf.encoder(molecule) if args.molecule_format == "smiles" else molecule

    assets = load_verified_genome_assets(
        fasta_path=args.genome_fasta,
        genbank_path=args.genome_genbank,
        embedding_path=args.genome_embedding,
    )
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
    if args.strain not in banks.genomes:
        raise KeyError(
            f"Interpretability requires a genome-backed strain: {args.strain}."
        )
    if tuple(banks.genomes[args.strain].shape) != assets.embedding_shape:
        raise ValueError(
            "Selected strain embedding does not match the explicit genome asset: "
            f"{tuple(banks.genomes[args.strain].shape)} != {assets.embedding_shape}."
        )
    model = load_candidate_mic_regressor(
        config,
        vocab_size=len(tokenizer.get_vocab()),
        condition_embeddings=banks,
        checkpoint_path=args.checkpoint,
        device=device,
        runtime_root=args.runtime_root.resolve(),
    )
    result = score_single_selfies_attention(
        model,
        tokenizer,
        selfies,
        strain=args.strain,
        device=device,
    )
    genome_rows = attention_rows(
        result.genome_attention, assets.windows, threshold=args.threshold
    )
    annotation_rows = annotate_selected_windows(genome_rows, assets)
    text_rows = indexed_attention_rows(result.text_attention, threshold=args.threshold)
    outputs = {
        "genome_attention": args.output_dir / "genome_attention.csv",
        "genome_annotations": args.output_dir / "genome_annotations.csv",
        "text_attention": args.output_dir / "text_attention.csv",
    }
    write_csv(outputs["genome_attention"], genome_rows, tuple(genome_rows[0]))
    write_csv(
        outputs["genome_annotations"],
        annotation_rows,
        (
            "fragment_index",
            "attention_weight",
            "contig_index",
            "record_id",
            "window_start",
            "window_end",
            "feature_start",
            "feature_end",
            "fully_contained",
            "gene",
            "locus_tag",
            "product",
        ),
    )
    write_csv(outputs["text_attention"], text_rows, tuple(text_rows[0]))

    logit = float(result.logits.detach().cpu().to(torch.float32).squeeze())
    mic = float(
        regression_logit_to_mic(result.logits)
        .detach()
        .cpu()
        .to(torch.float32)
        .squeeze()
    )
    input_paths = {
        "molecule": args.molecule_file,
        "genome_fasta": args.genome_fasta,
        "genome_genbank": args.genome_genbank,
        "genome_embedding": args.genome_embedding,
    }
    manifest = {
        "schema_version": 1,
        "strain": args.strain,
        "molecule_format": args.molecule_format,
        "tokenizer": args.tokenizer,
        "checkpoint": {
            "path": str(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
            "sha256": sha256(args.checkpoint) if args.hash_checkpoint else None,
        },
        "inputs": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in input_paths.items()
        },
        "verified_genome_contract": {
            "record_count": len(assets.fasta_records),
            "embedding_shape": list(assets.embedding_shape),
            "window_count": len(assets.windows),
            "window_length": 11_000,
            "step": 10_000,
            "saved_tensor_global_fragment_index": True,
        },
        "prediction": {"regression_logit": logit, "predicted_mic_umol": mic},
        "attention": {
            "semantics": "torch MultiheadAttention weights averaged over heads",
            "selection": f"weight > {args.threshold}",
            "genome_selected_indices": [
                row["fragment_index"] for row in genome_rows if row["selected"]
            ],
            "genome_selected_weights": [
                row["attention_weight"] for row in genome_rows if row["selected"]
            ],
            "genome_annotation_rows": len(annotation_rows),
        },
        "outputs": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in outputs.items()
        },
        "interpretation_boundary": (
            "Attention is descriptive and hypothesis-generating; it is not causal "
            "single-gene attribution or proof of strain uniqueness."
        ),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
