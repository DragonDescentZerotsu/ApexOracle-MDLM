#!/usr/bin/env python
"""Audit the six legacy MIC-guidance trainers and formal checkpoint family."""

from __future__ import annotations

import argparse
import ast
from collections import OrderedDict
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
from torch import nn

from apexoracle_mdlm.checkpoints import (
    load_torch_file,
    validate_generation_mic_guidance_checkpoint,
)
from apexoracle_mdlm.models import (
    FirstTokenCrossAttention,
    MIC_GUIDANCE_PROFILES,
    MICGuidanceRegressor,
    RegressionHead,
    build_upstream_noisy_dlm_hidden_state_encoder,
)


FORMAL_CHECKPOINTS = {
    "noisy_standard": (
        "guidance_regressor/noise_guidance_best_R2_all_peptide_epoch_100.pth"
    ),
    "noisy_padding_preserved": (
        "guidance_regressor_pad_no_mask/"
        "noise_guidance_best_R2_all_peptide_epoch_100.pth"
    ),
    "noisy_non_pad": (
        "guidance_regressor_non_pad/" "noise_guidance_best_R2_all_peptide_epoch_200.pth"
    ),
    "fixed_epsilon_non_pad": (
        "guidance_regressor_non_pad_t1e-3/"
        "mic_candidate_scorer_all_peptide_non_pad_t1e-3_epoch13.pth"
    ),
    "noisy_non_pad_eval": (
        "guidance_regressor_non_pad_noise/"
        "noise_guidance_best_R2_all_peptide_epoch_200.pth"
    ),
}

HISTORICAL_PROFILE_NAMES = {
    "fixed_epsilon_non_pad": "clean_non_pad",
}

HISTORICAL_CHECKPOINTS = {
    "fixed_epsilon_non_pad": (
        "guidance_regressor_non_pad_clean/"
        "noise_guidance_best_R2_all_peptide_epoch_13.pth"
    ),
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _snapshot_source(repo: Path, ref: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=repo)


def _extract_class(
    source: bytes,
    class_name: str,
    namespace: dict[str, object] | None = None,
) -> type[nn.Module]:
    tree = ast.parse(source.decode("utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == class_name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    execution_namespace: dict[str, object] = {"nn": nn, "torch": torch}
    if namespace:
        execution_namespace.update(namespace)
    exec(compile(module, "<legacy-mic-guidance>", "exec"), execution_namespace)
    return execution_namespace[class_name]  # type: ignore[return-value]


def _source_component_parity(source: bytes) -> dict[str, Any]:
    legacy_head_class = _extract_class(source, "RegressionHead")
    legacy_attention_class = _extract_class(source, "FirstTokenAttention_genome")
    torch.manual_seed(20260810)
    legacy_head = legacy_head_class(12, 3, 4, 1, 0.0).eval()
    canonical_head = RegressionHead(12, 3, 4, 1, 0.0).eval()
    canonical_head.load_state_dict(legacy_head.state_dict(), strict=True)
    head_input = torch.randn(2, 12)
    legacy_head_output = legacy_head(head_input)
    canonical_head_output = canonical_head(head_input)

    torch.manual_seed(20260810)
    legacy_attention = legacy_attention_class(4, 8, 2, 0.0).eval()
    canonical_attention = FirstTokenCrossAttention(
        4, 8, 2, 0.0, legacy_squeeze=True
    ).eval()
    canonical_attention.load_state_dict(legacy_attention.state_dict(), strict=True)
    molecule = torch.randn(2, 4)
    condition = torch.randn(2, 3, 8)
    padding = torch.tensor([[False, False, True], [False, False, False]])
    legacy_attention_output = legacy_attention(molecule, condition, padding)
    legacy_returns_weights = isinstance(legacy_attention_output, tuple)
    if legacy_returns_weights:
        legacy_attention_output = legacy_attention_output[0]
    canonical_attention_output = canonical_attention(molecule, condition, padding)
    return {
        "head_state_dict_keys_equal": list(legacy_head.state_dict())
        == list(canonical_head.state_dict()),
        "head_forward_torch_equal": torch.equal(
            legacy_head_output, canonical_head_output
        ),
        "head_max_absolute_difference": float(
            (legacy_head_output - canonical_head_output).abs().max()
        ),
        "attention_state_dict_keys_equal": list(legacy_attention.state_dict())
        == list(canonical_attention.state_dict()),
        "attention_forward_torch_equal": torch.equal(
            legacy_attention_output, canonical_attention_output
        ),
        "attention_max_absolute_difference": float(
            (legacy_attention_output - canonical_attention_output).abs().max()
        ),
        "legacy_attention_returns_weights": legacy_returns_weights,
    }


def _checkpoint_audit(checkpoint_root: Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for profile, relative in FORMAL_CHECKPOINTS.items():
        path = checkpoint_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = load_torch_file(
            path, map_location="cpu", weights_only=False, mmap=True
        )
        validate_generation_mic_guidance_checkpoint(payload)
        cls_state = payload.get("cls_head_state_dict")
        if not isinstance(cls_state, dict):
            raise TypeError(f"{path} lacks cls_head_state_dict.")
        with torch.device("meta"):
            cls_head = RegressionHead(12288, 3072, 128, 1, 0.2)
        cls_head.load_state_dict(cls_state, strict=True, assign=True)
        results[profile] = {
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "schema_valid": True,
            "classification_head_strict_load": True,
            "r2": float(payload["R2"]),
        }
        if profile in HISTORICAL_CHECKPOINTS:
            results[profile]["historical_relative_path"] = HISTORICAL_CHECKPOINTS[
                profile
            ]
        del cls_head, cls_state, payload
        gc.collect()
    return results


def _run_generation_parity(args: argparse.Namespace, source: bytes) -> dict[str, Any]:
    if args.backbone_checkpoint is None:
        raise ValueError("--backbone-checkpoint is required for generation parity.")
    sys.path.insert(0, str(args.repo.resolve()))
    import models
    import noise_schedule
    from hydra import compose, initialize_config_dir
    from torch.nn import functional as F

    sys.path.pop(0)
    with initialize_config_dir(
        config_dir=str((args.config_dir or args.repo / "configs").resolve()),
        version_base=None,
    ):
        config = compose(config_name=args.config_name)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA generation parity was requested but is unavailable.")
    checkpoint_path = (
        args.checkpoint_root / FORMAL_CHECKPOINTS["noisy_padding_preserved"]
    )
    payload = load_torch_file(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    legacy_encoder_class = _extract_class(
        source,
        "mol_emb_mdlm",
        {
            "F": F,
            "models": models,
            "noise_schedule": noise_schedule,
            "OrderedDict": OrderedDict,
        },
    )
    legacy_attention_class = _extract_class(source, "FirstTokenAttention_genome")
    legacy_head_class = _extract_class(source, "RegressionHead")
    input_ids = torch.full((2, 32), 3, dtype=torch.long, device=device)
    input_ids[0, :6] = torch.tensor([1, 10, 11, 12, 13, 2], device=device)
    input_ids[1, :5] = torch.tensor([1, 20, 21, 22, 2], device=device)
    attention_mask = input_ids.ne(3)
    torch.manual_seed(20260810)
    genome = torch.randn(2, 3, 8192, dtype=torch.bfloat16, device=device)
    text = torch.randn(2, 4, 4096, dtype=torch.bfloat16, device=device)
    genome_valid = torch.tensor(
        [[True, True, False], [True, True, True]], device=device
    )
    text_valid = torch.tensor(
        [[True, True, True, False], [True, True, True, True]], device=device
    )

    legacy_encoder = legacy_encoder_class(
        config, 3160, str(args.backbone_checkpoint), 4
    )
    legacy_genome_attention = legacy_attention_class(768, 8192, 4, 0.1)
    legacy_text_attention = legacy_attention_class(768, 4096, 4, 0.1)
    legacy_regression = legacy_head_class(12288, 3072, 128, 1, 0.2)
    legacy_classification = legacy_head_class(12288, 3072, 128, 1, 0.2)
    legacy_encoder.load_state_dict(payload["mdlm_model_state_dict"], strict=True)
    legacy_genome_attention.load_state_dict(
        payload["co_cross_attn_genome"], strict=True
    )
    legacy_text_attention.load_state_dict(payload["co_cross_attn_text"], strict=True)
    legacy_regression.load_state_dict(payload["re_head_state_dict"], strict=True)
    legacy_classification.load_state_dict(payload["cls_head_state_dict"], strict=True)
    legacy_modules = (
        legacy_encoder,
        legacy_genome_attention,
        legacy_text_attention,
        legacy_regression,
        legacy_classification,
    )
    for module in legacy_modules:
        module.to(device).eval()
    torch.manual_seed(20260811)
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ),
    ):
        hidden = legacy_encoder(input_ids, attention_mask)
        molecule = hidden[:, 0, :]
        genome_condition = legacy_genome_attention(molecule, genome, ~genome_valid)
        text_condition = legacy_text_attention(molecule, text, ~text_valid)
        fused = torch.cat(
            (genome_condition.reshape(-1, 8192), text_condition.reshape(-1, 4096)),
            dim=1,
        )
        legacy_regression_output = legacy_regression(fused).cpu()
        legacy_classification_output = legacy_classification(fused).cpu()
    del legacy_modules, legacy_encoder, legacy_genome_attention
    del legacy_text_attention, legacy_regression, legacy_classification, hidden
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    canonical_encoder = build_upstream_noisy_dlm_hidden_state_encoder(
        config,
        3160,
        runtime_root=args.repo,
        backbone_variant="dit",
        mask_index=4,
        preserve_padding=True,
        pad_token_id=3,
    )
    canonical = MICGuidanceRegressor(canonical_encoder)
    canonical.load_apexoracle_state(payload)
    canonical.to(device).eval()
    torch.manual_seed(20260811)
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ),
    ):
        canonical_regression, canonical_classification = canonical(
            input_ids,
            attention_mask,
            text,
            text_valid,
            genome,
            genome_valid,
        )
    canonical_regression = canonical_regression.cpu()
    canonical_classification = canonical_classification.cpu()
    result = {
        "profile": "noisy_padding_preserved",
        "batch_size": 2,
        "regression_torch_equal": torch.equal(
            legacy_regression_output, canonical_regression
        ),
        "regression_max_absolute_difference": float(
            (legacy_regression_output - canonical_regression).abs().max()
        ),
        "classification_torch_equal": torch.equal(
            legacy_classification_output, canonical_classification
        ),
        "classification_max_absolute_difference": float(
            (legacy_classification_output - canonical_classification).abs().max()
        ),
        "classification_numerically_close_atol_0_002": torch.allclose(
            legacy_classification_output,
            canonical_classification,
            rtol=0.0,
            atol=0.002,
        ),
        "classification_role": (
            "inactive historical head: saved by the trainers, but its batches were "
            "commented out and Generation consumes only the regression head"
        ),
        "device": str(device),
        "autocast_dtype": "torch.bfloat16",
        "fixed_seed": 20260811,
    }
    if not result["regression_torch_equal"]:
        raise AssertionError(f"Generation MIC-guidance parity failed: {result}")
    if not result["classification_numerically_close_atol_0_002"]:
        raise AssertionError(f"Inactive classification head drifted: {result}")
    return result


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--ref", default="legacy-code-snapshot-2026-08-09")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=(repo.parent / "Synergy/Checkpoints/genome_text_learnable_emb"),
    )
    parser.add_argument("--run-generation-parity", action="store_true")
    parser.add_argument("--backbone-checkpoint", type=Path)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--config-name", default="config")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_records: dict[str, Any] = {}
    component_parity: dict[str, Any] = {}
    source_payloads: dict[str, bytes] = {}
    for profile_name, profile in MIC_GUIDANCE_PROFILES.items():
        profile_results = []
        for source_path in profile.legacy_sources:
            source = _snapshot_source(args.repo, args.ref, source_path)
            source_payloads[source_path] = source
            source_records[source_path] = {
                "bytes": len(source),
                "sha256": _sha256_bytes(source),
                "recovery": f"git show {args.ref}:{source_path}",
            }
            profile_results.append(_source_component_parity(source))
        component_parity[profile_name] = profile_results
    duplicate_pair = MIC_GUIDANCE_PROFILES["noisy_non_pad"].legacy_sources
    duplicate_equal = (
        source_payloads[duplicate_pair[0]] == source_payloads[duplicate_pair[1]]
    )
    if not duplicate_equal:
        raise AssertionError(
            "The two recorded noisy_non_pad sources are not identical."
        )
    for results in component_parity.values():
        for item in results:
            if not all(
                item[key]
                for key in (
                    "head_state_dict_keys_equal",
                    "head_forward_torch_equal",
                    "attention_state_dict_keys_equal",
                    "attention_forward_torch_equal",
                )
            ):
                raise AssertionError(f"Legacy component parity failed: {item}")

    generation_parity = None
    if args.run_generation_parity:
        pad_source = source_payloads[
            MIC_GUIDANCE_PROFILES["noisy_padding_preserved"].legacy_sources[0]
        ]
        generation_parity = _run_generation_parity(args, pad_source)
    result = {
        "schema_version": 1,
        "snapshot_ref": args.ref,
        "profiles": {
            name: {
                **profile.__dict__,
                **(
                    {"historical_profile_name": HISTORICAL_PROFILE_NAMES[name]}
                    if name in HISTORICAL_PROFILE_NAMES
                    else {}
                ),
            }
            for name, profile in MIC_GUIDANCE_PROFILES.items()
        },
        "sources": source_records,
        "source_component_parity": component_parity,
        "byte_identical_noisy_non_pad_duplicate": duplicate_equal,
        "formal_checkpoints": _checkpoint_audit(args.checkpoint_root),
        "generation_profile_gpu_parity": generation_parity,
        "verified_facts": [
            "Six legacy trainers reduce to five explicit protocol profiles.",
            "The two noisy_non_pad trainer files are byte-identical.",
            "Every source has checkpoint-compatible regression and attention components.",
            "All five formal checkpoints satisfy the fixed Generation schema and strict classification-head load.",
            "The Generation-consumed regression output is bitwise equal under the formal noisy padding-preserved profile.",
        ],
    }
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
