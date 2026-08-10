#!/usr/bin/env python
"""Compare the frozen synergy producer/judge forward with canonical scoring."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoTokenizer

from apexoracle_mdlm.scoring import (
    ConditionEmbeddingBanks,
    load_candidate_synergy_classifier,
    normalize_selfies_for_tokenizer,
    read_selfies_file,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdlm-root", type=Path, default=root)
    parser.add_argument("--legacy-ref", default="legacy-code-snapshot-2026-08-09")
    parser.add_argument(
        "--judge-path-in-ref", default="judge_generated_mols_synergy.py"
    )
    parser.add_argument(
        "--producer-path-in-ref",
        default="synergy_Evo_train_new_reg_MDLM_one_base_model_all_data_classification.py",
    )
    parser.add_argument(
        "--judge-dependency-path-in-ref",
        default="guaidance_regressor_all_data.py",
        help="Snapshot-only module imported by the legacy judge at import time.",
    )
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--partner-embeddings", type=Path, required=True)
    parser.add_argument("--partner-key", required=True)
    parser.add_argument("--generation-file", type=Path, required=True)
    parser.add_argument("--strain", required=True)
    parser.add_argument("--tokenizer", default="ibm-research/materials.selfies-ted")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def token_ids(tokenizer, selfies_strings, device):
    encoded = tokenizer(
        [normalize_selfies_for_tokenizer(item) for item in selfies_strings],
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=True,
    )["input_ids"]
    if encoded.ndim == 1:
        encoded = encoded.unsqueeze(0)
    return [
        row[row != tokenizer.pad_token_id].unsqueeze(0).to(device) for row in encoded
    ]


def compare(args: argparse.Namespace, judge_path: Path, producer_path: Path) -> dict:
    if args.limit < 1:
        raise ValueError("--limit must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    root = args.mdlm_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    temp_root = judge_path.parent.resolve()
    if str(temp_root) not in sys.path:
        sys.path.insert(0, str(temp_root))

    producer = load_module("legacy_synergy_producer", producer_path)
    judge = load_module("legacy_synergy_judge", judge_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    judge.tokenizer = tokenizer
    judge.device = device
    judge.current_directory = args.core_root.resolve()
    # The judge imported the tuple-returning MIC attention by mistake. The
    # checkpoint producer's tensor-returning class is the executable reference.
    judge.FirstTokenAttention_genome = producer.FirstTokenAttention_genome
    judge.RegressionHead = producer.RegressionHead
    reference = (
        judge.synergy_classifier(
            judge.config,
            args.checkpoint.resolve(),
            device,
            args.partner_key,
            args.partner_embeddings.resolve(),
        )
        .to(device)
        .eval()
    )
    banks = ConditionEmbeddingBanks(
        genomes=reference.ATCC_genome_emb_dict,
        atcc_text=reference.ATCC_text_emb_dict,
        text_only=reference.text_only_emb_dict,
    )
    canonical = load_candidate_synergy_classifier(
        judge.config,
        vocab_size=len(tokenizer.get_vocab()),
        condition_embeddings=banks,
        partner_embedding=reference.synergy_mol_emb,
        checkpoint_path=args.checkpoint,
        device=device,
        runtime_root=root,
    )
    inputs = token_ids(
        tokenizer,
        read_selfies_file(args.generation_file)[: args.limit],
        device,
    )
    comparisons = []
    for index, input_ids in enumerate(inputs):
        torch.manual_seed(20260810 + index)
        with torch.inference_mode():
            reference_logit = reference(input_ids, args.strain)
        torch.manual_seed(20260810 + index)
        with torch.inference_mode():
            canonical_logit = canonical(input_ids, args.strain)
        reference_probability = torch.sigmoid(reference_logit)
        canonical_probability = torch.sigmoid(canonical_logit)
        comparisons.append(
            {
                "row_index": index,
                "input_tokens": int(input_ids.shape[1]),
                "logit_equal": bool(torch.equal(reference_logit, canonical_logit)),
                "logit_max_abs_difference": float(
                    (reference_logit - canonical_logit).abs().max()
                ),
                "probability_equal": bool(
                    torch.equal(reference_probability, canonical_probability)
                ),
                "probability_max_abs_difference": float(
                    (reference_probability - canonical_probability).abs().max()
                ),
            }
        )
    if not all(
        item["logit_equal"] and item["probability_equal"] for item in comparisons
    ):
        raise AssertionError(f"Candidate synergy parity failed: {comparisons}")
    return {
        "schema_version": 1,
        "status": "passed",
        "reference": "snapshot checkpoint-producer attention/head plus judge symmetric-pair forward",
        "active_judge_direct_replay": "fails because imported MIC attention returns a tuple",
        "checkpoint": str(args.checkpoint),
        "partner_embeddings": str(args.partner_embeddings),
        "partner_key": args.partner_key,
        "generation_file": str(args.generation_file),
        "strain": args.strain,
        "comparisons": comparisons,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="apexoracle_legacy_synergy_") as temp:
        temp_root = Path(temp)
        (temp_root / "configs").symlink_to(
            args.mdlm_root.resolve() / "configs", target_is_directory=True
        )
        extracted = {}
        for label, path_in_ref in (
            ("judge", args.judge_path_in_ref),
            ("producer", args.producer_path_in_ref),
            ("judge_dependency", args.judge_dependency_path_in_ref),
        ):
            source = subprocess.check_output(
                ["git", "show", f"{args.legacy_ref}:{path_in_ref}"],
                cwd=args.mdlm_root,
            )
            path = temp_root / Path(path_in_ref).name
            path.write_bytes(source)
            extracted[label] = path
        result = compare(args, extracted["judge"], extracted["producer"])
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
