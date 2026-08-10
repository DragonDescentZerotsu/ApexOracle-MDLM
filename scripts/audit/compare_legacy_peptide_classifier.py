#!/usr/bin/env python
"""Verify clean peptide-classifier components against the source snapshot."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from collections import OrderedDict
import sys

import torch
from torch import nn

from apexoracle_mdlm.checkpoints import (
    load_torch_file,
    validate_generation_peptide_classifier_checkpoint,
)
from apexoracle_mdlm.models import (
    PEPTIDE_CLASSIFIER_PROFILES,
    PeptideClassificationHead,
    build_upstream_noisy_dlm_hidden_state_encoder,
    load_peptide_classifier_head,
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    return sha256_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_source(repo: Path, ref: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=repo)


def extract_class(
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
    execution_namespace = {"nn": nn, "torch": torch}
    if namespace:
        execution_namespace.update(namespace)
    exec(compile(module, "<legacy-snapshot>", "exec"), execution_namespace)
    return execution_namespace[class_name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--ref", default="legacy-code-snapshot-2026-08-09")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--backbone-checkpoint", type=Path)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--config-name", default="config")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = {}
    parity = {}
    for name, profile in PEPTIDE_CLASSIFIER_PROFILES.items():
        source = snapshot_source(args.repo, args.ref, profile.legacy_source)
        sources[profile.legacy_source] = {
            "sha256": sha256_bytes(source),
            "bytes": len(source),
            "recovery": f"git show {args.ref}:{profile.legacy_source}",
        }
        legacy_class = extract_class(source, "ClsHead")
        torch.manual_seed(20260810)
        legacy = legacy_class(12, 8, 4, 1, 0.0).eval()
        canonical = PeptideClassificationHead(12, 8, 4, 1, 0.0).eval()
        canonical.load_state_dict(legacy.state_dict(), strict=True)
        features = torch.randn(5, 12)
        legacy_output = legacy(features)
        canonical_output = canonical(features)
        parity[name] = {
            "state_dict_keys_equal": list(legacy.state_dict())
            == list(canonical.state_dict()),
            "forward_torch_equal": torch.equal(legacy_output, canonical_output),
            "max_absolute_difference": float(
                (legacy_output - canonical_output).abs().max().item()
            ),
        }

    checkpoint_result = None
    if args.checkpoint is not None:
        payload = load_torch_file(
            args.checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        validate_generation_peptide_classifier_checkpoint(payload)
        head = load_peptide_classifier_head(str(args.checkpoint), mmap=True).eval()
        features = torch.linspace(-1.0, 1.0, steps=3 * 768).reshape(3, 768)
        with torch.inference_mode():
            logits = head(features)
        hyper_parameters = payload.get("hyper_parameters", {})
        checkpoint_result = {
            "path": str(args.checkpoint.resolve()),
            "bytes": args.checkpoint.stat().st_size,
            "sha256": sha256_file(args.checkpoint),
            "schema_valid": True,
            "strict_head_load": True,
            "fixed_logits_shape": list(logits.shape),
            "fixed_logits_sha256": sha256_tensor(logits),
            "positive_weight": float(hyper_parameters["pos_weight"]),
            "global_step": int(payload["global_step"]),
            "epoch": int(payload["epoch"]),
        }

    encoder_parity = None
    if args.backbone_checkpoint is not None:
        sys.path.insert(0, str(args.repo.resolve()))
        import models
        import noise_schedule
        from hydra import compose, initialize_config_dir
        from torch.nn import functional as F

        sys.path.pop(0)

        config_dir = args.config_dir or args.repo / "configs"
        with initialize_config_dir(
            config_dir=str(config_dir.resolve()), version_base=None
        ):
            config = compose(config_name=args.config_name)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA encoder parity was requested but is unavailable.")
        input_ids = torch.full((2, 32), 3, dtype=torch.long, device=device)
        input_ids[0, :6] = torch.tensor([1, 10, 11, 12, 13, 2], device=device)
        input_ids[1, :5] = torch.tensor([1, 20, 21, 22, 2], device=device)
        attention_mask = input_ids.ne(3)
        encoder_parity = {}
        for name, profile in PEPTIDE_CLASSIFIER_PROFILES.items():
            source = snapshot_source(args.repo, args.ref, profile.legacy_source)
            legacy_class = extract_class(
                source,
                "mol_emb_mdlm",
                {
                    "F": F,
                    "models": models,
                    "noise_schedule": noise_schedule,
                    "OrderedDict": OrderedDict,
                },
            )
            legacy = legacy_class(
                config,
                3160,
                str(args.backbone_checkpoint),
                4,
            ).to(device)
            canonical = build_upstream_noisy_dlm_hidden_state_encoder(
                config,
                3160,
                runtime_root=args.repo,
                backbone_variant=profile.backbone_variant,
                mask_index=4,
                preserve_padding=profile.preserve_padding,
                pad_token_id=3,
            ).to(device)
            canonical.load_backbone_checkpoint(args.backbone_checkpoint)
            legacy.train()
            canonical.train()
            torch.manual_seed(20260810)
            legacy_output = legacy(input_ids, attention_mask)
            torch.manual_seed(20260810)
            canonical_output = canonical(input_ids, attention_mask)
            encoder_parity[name] = {
                "shape": list(canonical_output.shape),
                "dtype": str(canonical_output.dtype),
                "torch_equal": torch.equal(legacy_output, canonical_output),
                "max_absolute_difference": float(
                    (legacy_output - canonical_output).abs().max().item()
                ),
                "mode": "train",
                "fixed_seed": 20260810,
            }
            del legacy, canonical, legacy_output, canonical_output
            if device.type == "cuda":
                torch.cuda.empty_cache()

    result = {
        "schema_version": 1,
        "snapshot_ref": args.ref,
        "sources": sources,
        "profiles": {
            name: item.__dict__ for name, item in PEPTIDE_CLASSIFIER_PROFILES.items()
        },
        "head_parity": parity,
        "deployed_v1_checkpoint": checkpoint_result,
        "noisy_encoder_parity": encoder_parity,
        "external_exact_producer_record": {
            "source": "ApexOracle-Core/experiments/peptide_classifier/README.md",
            "historical_path": (
                "node002:/data1/tianang/Projects/mdlm/"
                "guaidance_classifier_all_data_pad_no_mask.py"
            ),
            "status": (
                "Previously verified as the 2025-05 v1 producer; the exact source "
                "is not currently accessible from this MDLM checkout, so no Git "
                "blob or source hash is asserted here."
            ),
        },
        "verified_facts": [
            "All three snapshot sources define an identical ClsHead contract.",
            "The clean head has identical keys and fixed-input outputs for every source.",
            "The three sources differ in noisy encoder, pooling, padding, and dataset profile.",
        ],
        "inference": (
            "The checkpoint metadata and Core's frozen node002 record support the "
            "v1_noisy_padding_preserved_cls protocol. The active MDLM snapshot does "
            "not contain that exact 2025-05 source blob, so this repository does not "
            "claim byte-for-byte training-source reproduction."
        ),
    }
    if not all(item["forward_torch_equal"] for item in parity.values()):
        raise AssertionError("Legacy/canonical peptide-classifier head parity failed.")
    if encoder_parity is not None and not all(
        item["torch_equal"] for item in encoder_parity.values()
    ):
        raise AssertionError("Legacy/canonical noisy-encoder parity failed.")
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
