#!/usr/bin/env python
"""Compare the legacy and canonical candidate MIC scorers on formal assets."""

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
    load_candidate_mic_regressor,
    normalize_selfies_for_tokenizer,
    read_selfies_file,
    regression_logit_to_mic,
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdlm-root", type=Path, default=repo_root)
    parser.add_argument("--legacy-source", type=Path)
    parser.add_argument("--legacy-ref", default="legacy-code-snapshot-2026-08-09")
    parser.add_argument(
        "--legacy-path-in-ref",
        default="judge_generated_mols_MIC.py",
        help="Tracked scorer source to extract from --legacy-ref.",
    )
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--generation-file", type=Path, required=True)
    parser.add_argument("--strain", required=True)
    parser.add_argument("--tokenizer", default="ibm-research/materials.selfies-ted")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument(
        "--legacy-forward-calls",
        type=int,
        default=1,
        help="Replay accidental repeated legacy forwards when characterizing a driver.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_legacy_module(path: Path):
    spec = importlib.util.spec_from_file_location("legacy_generated_mic", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load legacy scorer from {path}.")
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


def compare(args: argparse.Namespace, legacy_source: Path, legacy_locator: str) -> dict:
    if args.limit < 1:
        raise ValueError("--limit must be positive.")
    if args.legacy_forward_calls < 1:
        raise ValueError("--legacy-forward-calls must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    mdlm_root = args.mdlm_root.resolve()
    if str(mdlm_root) not in sys.path:
        sys.path.insert(0, str(mdlm_root))

    legacy = load_legacy_module(legacy_source.resolve())
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    legacy.tokenizer = tokenizer
    legacy.device = device
    legacy.current_directory = args.core_root.resolve()
    legacy_model = (
        legacy.MIC_regressor(
            legacy.config,
            str(args.checkpoint.resolve()),
            device,
        )
        .to(device)
        .eval()
    )

    banks = ConditionEmbeddingBanks(
        genomes=legacy_model.ATCC_genome_emb_dict,
        atcc_text=legacy_model.ATCC_text_emb_dict,
        text_only=legacy_model.text_only_emb_dict,
    )
    canonical_model = load_candidate_mic_regressor(
        legacy.config,
        vocab_size=len(tokenizer.get_vocab()),
        condition_embeddings=banks,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    selfies_strings = read_selfies_file(args.generation_file)[: args.limit]
    inputs = token_ids(tokenizer, selfies_strings, device)
    comparisons = []
    for index, input_ids in enumerate(inputs):
        torch.manual_seed(20260809 + index)
        with torch.inference_mode():
            for _ in range(args.legacy_forward_calls):
                legacy_logit = legacy_model(input_ids, args.strain)
        torch.manual_seed(20260809 + index)
        with torch.inference_mode():
            canonical_logit = canonical_model(input_ids, args.strain)
        legacy_mic = regression_logit_to_mic(legacy_logit)
        canonical_mic = regression_logit_to_mic(canonical_logit)
        comparisons.append(
            {
                "row_index": index,
                "input_tokens": int(input_ids.shape[1]),
                "logit_equal": bool(torch.equal(legacy_logit, canonical_logit)),
                "logit_max_abs_difference": float(
                    (legacy_logit - canonical_logit).abs().max().item()
                ),
                "mic_equal": bool(torch.equal(legacy_mic, canonical_mic)),
                "mic_max_abs_difference": float(
                    (legacy_mic - canonical_mic).abs().max().item()
                ),
            }
        )
    batch_comparison = None
    if len(inputs) > 1 and len({item.shape[1] for item in inputs}) == 1:
        batched_input_ids = torch.cat(inputs, dim=0)
        torch.manual_seed(20260819)
        with torch.inference_mode():
            for _ in range(args.legacy_forward_calls):
                legacy_batch_logits = legacy_model(batched_input_ids, args.strain)
        torch.manual_seed(20260819)
        with torch.inference_mode():
            canonical_batch_logits = canonical_model(batched_input_ids, args.strain)
        legacy_batch_mic = regression_logit_to_mic(legacy_batch_logits)
        canonical_batch_mic = regression_logit_to_mic(canonical_batch_logits)
        batch_comparison = {
            "batch_size": len(inputs),
            "input_tokens": int(batched_input_ids.shape[1]),
            "logit_equal": bool(
                torch.equal(legacy_batch_logits, canonical_batch_logits)
            ),
            "logit_max_abs_difference": float(
                (legacy_batch_logits - canonical_batch_logits).abs().max().item()
            ),
            "mic_equal": bool(torch.equal(legacy_batch_mic, canonical_batch_mic)),
            "mic_max_abs_difference": float(
                (legacy_batch_mic - canonical_batch_mic).abs().max().item()
            ),
        }
    passed = all(
        item["logit_equal"] and item["mic_equal"] for item in comparisons
    ) and (
        batch_comparison is None
        or (batch_comparison["logit_equal"] and batch_comparison["mic_equal"])
    )
    if not passed:
        raise AssertionError(f"Candidate MIC parity failed: {comparisons}")
    return {
        "schema_version": 1,
        "legacy_source": legacy_locator,
        "checkpoint": str(args.checkpoint),
        "generation_file": str(args.generation_file),
        "strain": args.strain,
        "device": str(device),
        "legacy_forward_calls": args.legacy_forward_calls,
        "comparisons": comparisons,
        "batch_comparison": batch_comparison,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "status": "passed",
    }


def main() -> None:
    args = parse_args()
    if args.legacy_source is not None:
        result = compare(
            args,
            args.legacy_source,
            str(args.legacy_source),
        )
    else:
        with tempfile.TemporaryDirectory(
            prefix="apexoracle_legacy_candidate_mic_"
        ) as temp_dir:
            source = subprocess.check_output(
                [
                    "git",
                    "show",
                    f"{args.legacy_ref}:{args.legacy_path_in_ref}",
                ],
                cwd=args.mdlm_root,
            )
            temp_root = Path(temp_dir)
            (temp_root / "configs").symlink_to(
                args.mdlm_root.resolve() / "configs",
                target_is_directory=True,
            )
            legacy_path = temp_root / Path(args.legacy_path_in_ref).name
            legacy_path.write_bytes(source)
            result = compare(
                args,
                legacy_path,
                f"git:{args.legacy_ref}:{args.legacy_path_in_ref}",
            )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
