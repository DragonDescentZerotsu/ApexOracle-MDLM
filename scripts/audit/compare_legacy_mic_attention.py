#!/usr/bin/env python
"""Compare snapshot interpretability forward with the canonical attention API."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import selfies as sf
import torch
from transformers import AutoTokenizer

from apexoracle_mdlm.scoring import (
    ConditionEmbeddingBanks,
    load_candidate_mic_regressor,
    normalize_selfies_for_tokenizer,
    regression_logit_to_mic,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdlm-root", type=Path, default=root)
    parser.add_argument("--legacy-ref", default="legacy-code-snapshot-2026-08-09")
    parser.add_argument("--legacy-path-in-ref", default="visualize_attn.py")
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--molecule-file", type=Path, required=True)
    parser.add_argument(
        "--molecule-format", choices=("smiles", "selfies"), default="smiles"
    )
    parser.add_argument("--strain", action="append", required=True)
    parser.add_argument("--tokenizer", default="ibm-research/materials.selfies-ted")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("legacy_mic_attention", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load legacy module from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def token_ids(tokenizer, selfies: str, device: torch.device) -> torch.Tensor:
    encoded = tokenizer(
        [normalize_selfies_for_tokenizer(selfies)],
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=True,
    )["input_ids"]
    if encoded.ndim == 1:
        encoded = encoded.unsqueeze(0)
    row = encoded[0]
    return row[row != tokenizer.pad_token_id].unsqueeze(0).to(device)


def comparison(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> dict[str, object]:
    return {
        "torch_equal": bool(torch.equal(tensor_a, tensor_b)),
        "max_abs_difference": float((tensor_a - tensor_b).abs().max()),
        "shape": list(tensor_a.shape),
    }


def compare(args: argparse.Namespace, legacy_path: Path) -> dict[str, object]:
    if len(set(args.strain)) != len(args.strain):
        raise ValueError("--strain values must be unique.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    root = args.mdlm_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    legacy = load_module(legacy_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    legacy.tokenizer = tokenizer
    legacy.device = device
    legacy.current_directory = args.core_root.resolve()
    legacy_model = (
        legacy.MIC_regressor(legacy.config, args.checkpoint.resolve(), device)
        .to(device)
        .eval()
    )
    banks = ConditionEmbeddingBanks(
        genomes=legacy_model.ATCC_genome_emb_dict,
        atcc_text=legacy_model.ATCC_text_emb_dict,
        text_only=legacy_model.text_only_emb_dict,
    )
    canonical = load_candidate_mic_regressor(
        legacy.config,
        vocab_size=len(tokenizer.get_vocab()),
        condition_embeddings=banks,
        checkpoint_path=args.checkpoint,
        device=device,
        runtime_root=root,
    )
    molecule = args.molecule_file.read_text(encoding="utf-8").strip()
    if not molecule or "\n" in molecule:
        raise ValueError("--molecule-file must contain exactly one row.")
    selfies = sf.encoder(molecule) if args.molecule_format == "smiles" else molecule
    input_ids = token_ids(tokenizer, selfies, device)

    results = []
    for index, strain in enumerate(args.strain):
        torch.manual_seed(20260810 + index)
        with torch.inference_mode():
            legacy_logit, _, legacy_genome, legacy_text = legacy_model(
                input_ids, strain
            )
        torch.manual_seed(20260810 + index)
        with torch.inference_mode():
            canonical_result = canonical.forward_with_attention(input_ids, strain)
        tensors = {
            "logit": comparison(legacy_logit, canonical_result.logits),
            "mic": comparison(
                regression_logit_to_mic(legacy_logit),
                regression_logit_to_mic(canonical_result.logits),
            ),
            "genome_attention": comparison(
                legacy_genome, canonical_result.genome_attention
            ),
            "text_attention": comparison(legacy_text, canonical_result.text_attention),
        }
        if not all(item["torch_equal"] for item in tensors.values()):
            raise AssertionError(f"MIC attention parity failed for {strain}: {tensors}")
        genome_vector = canonical_result.genome_attention.squeeze().detach().cpu()
        results.append(
            {
                "strain": strain,
                "input_tokens": int(input_ids.shape[1]),
                "predicted_mic_umol": float(
                    regression_logit_to_mic(canonical_result.logits)
                    .detach()
                    .cpu()
                    .to(torch.float32)
                    .squeeze()
                ),
                "selected_genome_indices_weight_gt_0_05": torch.where(
                    genome_vector > 0.05
                )[0].tolist(),
                "selected_genome_weights": genome_vector[genome_vector > 0.05]
                .to(torch.float32)
                .tolist(),
                "comparisons": tensors,
            }
        )
    return {
        "schema_version": 1,
        "status": "passed",
        "legacy_source": f"git:{args.legacy_ref}:{args.legacy_path_in_ref}",
        "checkpoint": str(args.checkpoint),
        "molecule_file": str(args.molecule_file),
        "molecule_format": args.molecule_format,
        "attention_semantics": "weights averaged over attention heads",
        "strains": results,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="apexoracle_legacy_attention_") as temp:
        temp_root = Path(temp)
        (temp_root / "configs").symlink_to(
            args.mdlm_root.resolve() / "configs", target_is_directory=True
        )
        source = subprocess.check_output(
            [
                "git",
                "show",
                f"{args.legacy_ref}:{args.legacy_path_in_ref}",
            ],
            cwd=args.mdlm_root,
        )
        legacy_path = temp_root / Path(args.legacy_path_in_ref).name
        legacy_path.write_bytes(source)
        result = compare(args, legacy_path)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
