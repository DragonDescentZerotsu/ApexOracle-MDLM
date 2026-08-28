#!/usr/bin/env python
"""Convert a peptide CSV and score it across one or more strain conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from apexoracle_mdlm.embeddings import (
    embedding_key_from_atcc_filename,
    embedding_key_from_text_filename,
)
from apexoracle_mdlm.figures import plot_mic_distribution
from apexoracle_mdlm.scoring import (
    add_mic_predictions,
    conversion_summary,
    convert_peptides_to_structures,
    load_candidate_mic_regressor,
    load_condition_embedding_banks,
    load_peptide_table,
    selfies_token_lengths,
)


DEFAULT_TOKENIZER_REVISION = "55e83392264cb998f7aa5014847df29868aefeb8"
DEFAULT_GENOME_SCALE = 1e14


def register_upstream_config_resolvers() -> None:
    """Register the resolvers used by the attributed upstream Hydra config."""

    resolvers = {
        "cwd": os.getcwd,
        "device_count": torch.cuda.device_count,
        "eval": eval,
        "div_up": lambda x, y: (x + y - 1) // y,
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)


def resolved_config_yaml(config) -> str:
    """Resolve the full config for provenance without importing training main."""

    register_upstream_config_resolvers()
    return OmegaConf.to_yaml(config, resolve=True, sort_keys=True)


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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--peptide-column", default="Peptide")
    parser.add_argument("--protein-column", default="Protein")
    parser.add_argument("--strains", nargs="+", required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--config-name", default="config")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--genome-embeddings", type=Path, required=True)
    parser.add_argument("--genome-scale", type=float, default=DEFAULT_GENOME_SCALE)
    parser.add_argument("--atcc-text-embeddings", type=Path, required=True)
    parser.add_argument("--text-only-embeddings", type=Path, required=True)
    parser.add_argument("--tokenizer", default="ibm-research/materials.selfies-ted")
    parser.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--structures-name", default="peptide_structures.csv")
    parser.add_argument("--predictions-name", default="peptide_mic_predictions.csv")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--hash-checkpoint", action="store_true")
    return parser.parse_args()


def _tensor_file_index(directory: Path, key_parser) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(directory.resolve().glob("*.pt")):
        key = key_parser(path.name)
        if key in index:
            raise ValueError(
                f"Multiple embedding files resolve to key {key!r} in {directory}."
            )
        index[key] = path
    return index


def condition_embedding_provenance(
    *,
    strains: list[str],
    banks,
    genome_directory: Path,
    atcc_text_directory: Path,
    text_only_directory: Path,
) -> dict[str, dict[str, object]]:
    genome_files = _tensor_file_index(
        genome_directory, embedding_key_from_atcc_filename
    )
    atcc_text_files = _tensor_file_index(
        atcc_text_directory, embedding_key_from_atcc_filename
    )
    text_only_files = _tensor_file_index(
        text_only_directory, embedding_key_from_text_filename
    )
    provenance: dict[str, dict[str, object]] = {}
    for strain in strains:
        if strain in banks.genomes:
            if strain not in genome_files or strain not in atcc_text_files:
                raise KeyError(f"Cannot resolve condition files for strain {strain!r}.")
            files = {
                "genome": (genome_files[strain], banks.genomes[strain]),
                "text": (atcc_text_files[strain], banks.atcc_text[strain]),
            }
            mode = "genome_and_atcc_text"
        else:
            if strain not in text_only_files:
                raise KeyError(f"Cannot resolve text-only file for strain {strain!r}.")
            files = {"text": (text_only_files[strain], banks.text_only[strain])}
            mode = "text_only_with_learnable_genome"
        provenance[strain] = {
            "mode": mode,
            "files": {
                name: {
                    "path": str(path),
                    "sha256": sha256(path),
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                }
                for name, (path, tensor) in files.items()
            },
        }
    return provenance


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if not math.isfinite(args.genome_scale) or args.genome_scale <= 0:
        raise ValueError("--genome-scale must be finite and positive.")
    if len(set(args.strains)) != len(args.strains):
        raise ValueError("--strains must not contain duplicates.")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "The attributed upstream DiT runtime requires an available CUDA device."
        )
    peptide_frame = load_peptide_table(
        args.input,
        peptide_column=args.peptide_column,
        protein_column=args.protein_column,
        limit=args.limit,
    )
    structure_frame = convert_peptides_to_structures(peptide_frame)
    with initialize_config_dir(
        config_dir=str(args.config_dir.resolve()),
        version_base=None,
    ):
        config = compose(config_name=args.config_name)
    model_max_length = int(config.model.length)
    if model_max_length < 1:
        raise ValueError("Resolved config.model.length must be positive.")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
    )
    tokenizer_declared_max_length = int(tokenizer.model_max_length)
    # The tokenizer repository declares 512, but this scorer's DLM owns the
    # position contract and was configured/trained at 1024.  Align warnings and
    # validation with the resolved downstream model rather than silently using
    # the tokenizer metadata as a model limit.
    tokenizer.model_max_length = model_max_length
    valid_selfies = structure_frame.loc[
        structure_frame["conversion_status"].eq("valid"), "SELFIES"
    ].tolist()
    token_lengths = selfies_token_lengths(tokenizer, valid_selfies)
    over_limit = [length for length in token_lengths if length > model_max_length]
    if over_limit:
        raise ValueError(
            f"{len(over_limit)} valid molecules exceed resolved model length "
            f"{model_max_length}; observed maximum {max(over_limit)}."
        )
    resolved_config = resolved_config_yaml(config)
    banks = load_condition_embedding_banks(
        genome_directory=args.genome_embeddings,
        atcc_text_directory=args.atcc_text_embeddings,
        text_only_directory=args.text_only_embeddings,
        genome_scale=args.genome_scale,
    )
    condition_provenance = condition_embedding_provenance(
        strains=args.strains,
        banks=banks,
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
    prediction_frame = add_mic_predictions(
        structure_frame,
        model,
        tokenizer,
        strains=args.strains,
        batch_size=args.batch_size,
        device=str(device),
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    structures_path = args.output_directory / args.structures_name
    predictions_path = args.output_directory / args.predictions_name
    structure_frame.to_csv(structures_path, index=False)
    prediction_frame.to_csv(predictions_path, index=False)
    figures: list[dict[str, object]] = []
    if args.plot:
        figure_directory = args.output_directory / "violin_figures"
        figure_directory.mkdir(parents=True, exist_ok=True)
        for strain in args.strains:
            if not prediction_frame[strain].gt(0).any():
                continue
            output = figure_directory / f"strain_{strain}_MIC_distribution.pdf"
            figure, _ = plot_mic_distribution(
                prediction_frame[strain].to_numpy(),
                strain=strain,
            )
            figure.savefig(output, format="pdf", bbox_inches="tight", dpi=300)
            plt.close(figure)
            figures.append(
                {"strain": strain, "path": str(output), "sha256": sha256(output)}
            )
    manifest = {
        "schema_version": 1,
        "input": {
            "path": str(args.input),
            "sha256": sha256(args.input),
        },
        "columns": {
            "peptide": args.peptide_column,
            "protein": args.protein_column,
        },
        "conversion": conversion_summary(structure_frame),
        "strains": args.strains,
        "batch_size": args.batch_size,
        "runtime_root": str(args.runtime_root.resolve()),
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "tokenization": {
            "tokenizer_declared_model_max_length": tokenizer_declared_max_length,
            "resolved_model_max_length": model_max_length,
            "valid_rows": len(token_lengths),
            "minimum_length": min(token_lengths) if token_lengths else None,
            "maximum_length": max(token_lengths) if token_lengths else None,
            "over_model_limit_rows": len(over_limit),
        },
        "config": {
            "directory": str(args.config_dir.resolve()),
            "name": args.config_name,
            "model_name": str(config.model.name),
            "resolved_yaml_sha256": hashlib.sha256(
                resolved_config.encode("utf-8")
            ).hexdigest(),
        },
        "condition_embeddings": condition_provenance,
        "condition_embedding_scales": {
            "genome": args.genome_scale,
            "atcc_text": 1.0,
            "text_only": 1.0,
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
            "sha256": sha256(args.checkpoint) if args.hash_checkpoint else None,
        },
        "outputs": {
            "structures": {
                "path": str(structures_path),
                "sha256": sha256(structures_path),
            },
            "predictions": {
                "path": str(predictions_path),
                "sha256": sha256(predictions_path),
            },
            "figures": figures,
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
