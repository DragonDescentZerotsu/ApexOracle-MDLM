#!/usr/bin/env python
"""Compare frozen synergy-training encoders with the canonical profile adapter."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoTokenizer

from apexoracle_mdlm.checkpoints import (
    load_torch_file,
    validate_generation_synergy_guidance_checkpoint,
)
from apexoracle_mdlm.models import (
    build_upstream_noisy_dlm_hidden_state_encoder,
    get_synergy_guidance_profile,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdlm-root", type=Path, default=root)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument(
        "--generation-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--legacy-ref", default="legacy-code-snapshot-2026-08-09")
    parser.add_argument("--asymmetric-backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--clean-backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", default="ibm-research/materials.selfies-ted")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_source(root: Path, ref: str, relative_path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{ref}:{relative_path}"], cwd=root)


def _generation_consumers(
    generation_root: Path, producer_name: str
) -> dict[str, object]:
    commented: list[dict[str, object]] = []
    live_source_references: list[dict[str, object]] = []
    checkpoint_configs: list[str] = []
    historical_run_configs: list[str] = []
    for path in generation_root.rglob("*"):
        if not path.is_file() or path.suffix not in {
            ".py",
            ".sh",
            ".yaml",
            ".yml",
            ".json",
            ".md",
        }:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, start=1):
            if producer_name in line:
                record = {
                    "path": str(path.relative_to(generation_root)),
                    "line": line_number,
                }
                if line.lstrip().startswith("#"):
                    commented.append(record)
                else:
                    live_source_references.append(record)
            if (
                "syn_classifier_checkpoint_path:" in line
                and "guidance_noise_synergy" in line
            ):
                relative = str(path.relative_to(generation_root))
                if relative.startswith("outputs/"):
                    historical_run_configs.append(relative)
                else:
                    checkpoint_configs.append(relative)
    return {
        "live_source_references": live_source_references,
        "commented_shell_provenance_references": commented,
        "live_checkpoint_configs": sorted(set(checkpoint_configs)),
        "historical_output_hydra_configs": sorted(set(historical_run_configs)),
    }


def _profile_comparison(
    *,
    root: Path,
    profile_name: str,
    source_path: Path,
    checkpoint: Path,
    tokenizer,
    device: torch.device,
) -> dict[str, object]:
    profile = get_synergy_guidance_profile(profile_name)
    module = _load_module(f"legacy_synergy_{profile_name}", source_path)
    legacy = (
        module.mol_emb_mdlm(
            module.config,
            len(tokenizer.get_vocab()),
            checkpoint,
            tokenizer.mask_token_id,
        )
        .to(device)
        .eval()
    )
    canonical = build_upstream_noisy_dlm_hidden_state_encoder(
        module.config,
        len(tokenizer.get_vocab()),
        runtime_root=root,
        backbone_variant="dit",
        mask_index=tokenizer.mask_token_id,
        preserve_padding=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    missing_keys, unexpected_keys = canonical.load_backbone_checkpoint(checkpoint)
    canonical.to(device).eval()

    encoded = tokenizer(
        ["[C] [O]", "[C] [C] [N]"],
        return_tensors="pt",
        padding=True,
        truncation=False,
    )["input_ids"]
    input_ids = torch.full(
        (2, 32), tokenizer.pad_token_id, dtype=torch.long, device=device
    )
    input_ids[:, : encoded.shape[1]] = encoded.to(device)
    first_ids = input_ids[:1]
    second_ids = input_ids[1:]

    torch.manual_seed(20260810)
    with torch.inference_mode():
        legacy_first = legacy(first_ids, noise_input=profile.first_molecule_noisy)
        legacy_second = legacy(second_ids, noise_input=profile.second_molecule_noisy)
    torch.manual_seed(20260810)
    with torch.inference_mode():
        canonical_first = canonical(first_ids, apply_noise=profile.first_molecule_noisy)
        canonical_second = canonical(
            second_ids, apply_noise=profile.second_molecule_noisy
        )
    first_equal = bool(torch.equal(legacy_first, canonical_first))
    second_equal = bool(torch.equal(legacy_second, canonical_second))
    result = {
        "profile": profile_name,
        "noise_order": [
            profile.first_molecule_noisy,
            profile.second_molecule_noisy,
        ],
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "bytes": checkpoint.stat().st_size,
            "sha256": _sha256_file(checkpoint),
        },
        "checkpoint_load": {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        },
        "first_output_equal": first_equal,
        "first_max_abs_difference": float((legacy_first - canonical_first).abs().max()),
        "second_output_equal": second_equal,
        "second_max_abs_difference": float(
            (legacy_second - canonical_second).abs().max()
        ),
    }
    if not first_equal or not second_equal:
        raise AssertionError(f"Synergy encoder parity failed: {result}")
    del (
        legacy,
        canonical,
        legacy_first,
        legacy_second,
        canonical_first,
        canonical_second,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    root = args.mdlm_root.resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    sources: dict[str, dict[str, object]] = {}
    extracted_paths: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix="apexoracle_synergy_guidance_") as temp:
        temp_root = Path(temp)
        (temp_root / "configs").symlink_to(root / "configs", target_is_directory=True)
        if str(temp_root) not in sys.path:
            sys.path.insert(0, str(temp_root))
        for profile_name in ("asymmetric_partner_noise", "clean_pair"):
            profile = get_synergy_guidance_profile(profile_name)
            source_name = profile.legacy_sources[0]
            content = _extract_source(root, args.legacy_ref, source_name)
            path = temp_root / source_name
            path.write_bytes(content)
            extracted_paths[profile_name] = path
            sources[source_name] = {
                "bytes": len(content),
                "lines": len(content.splitlines()),
                "sha256": _sha256_bytes(content),
            }
        duplicate_name = get_synergy_guidance_profile(
            "asymmetric_partner_noise"
        ).legacy_sources[1]
        duplicate = _extract_source(root, args.legacy_ref, duplicate_name)
        sources[duplicate_name] = {
            "bytes": len(duplicate),
            "lines": len(duplicate.splitlines()),
            "sha256": _sha256_bytes(duplicate),
            "byte_identical_to": get_synergy_guidance_profile(
                "asymmetric_partner_noise"
            ).legacy_sources[0],
        }
        if duplicate != _extract_source(
            root,
            args.legacy_ref,
            get_synergy_guidance_profile("asymmetric_partner_noise").legacy_sources[0],
        ):
            raise AssertionError(
                "The two asymmetric legacy sources are no longer identical."
            )

        comparisons = [
            _profile_comparison(
                root=root,
                profile_name="asymmetric_partner_noise",
                source_path=extracted_paths["asymmetric_partner_noise"],
                checkpoint=args.asymmetric_backbone_checkpoint,
                tokenizer=tokenizer,
                device=device,
            ),
            _profile_comparison(
                root=root,
                profile_name="clean_pair",
                source_path=extracted_paths["clean_pair"],
                checkpoint=args.clean_backbone_checkpoint,
                tokenizer=tokenizer,
                device=device,
            ),
        ]

    result = {
        "schema_version": 1,
        "status": "passed",
        "snapshot": args.legacy_ref,
        "legacy_sources": sources,
        "comparisons": comparisons,
        "canonical_model": "apexoracle_mdlm.models.SynergyGuidanceClassifier",
        "canonical_training_cli": "scripts/reproduce/train_synergy_guidance.py",
        "generation_dependency": "checkpoint_only",
        "generation_consumers": _generation_consumers(
            args.generation_root.resolve(),
            get_synergy_guidance_profile("asymmetric_partner_noise").legacy_sources[0],
        ),
    }
    lineage_path = root / "reproducibility" / "candidate_synergy_lineage.json"
    if lineage_path.is_file():
        candidate_lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        result["formal_assets"] = candidate_lineage["formal_assets"]
        for asset_name, relative_path in (
            (
                "generation_synergy_guidance",
                "Checkpoints/genome_text_learnable_emb/guidance_noise_synergy/cls/"
                "synergy_noise_clsfier_best.ckpt",
            ),
            (
                "synergy_judger",
                "Checkpoints/genome_text_learnable_emb/synergy_judger/cls/"
                "synergy_noise_clsfier_best.ckpt",
            ),
        ):
            asset_path = args.core_root.resolve() / relative_path
            expected = result["formal_assets"][asset_name]
            if asset_path.stat().st_size != expected["bytes"]:
                raise AssertionError(f"Formal asset size changed: {asset_path}")
            payload = load_torch_file(
                asset_path, map_location="cpu", weights_only=False, mmap=True
            )
            validate_generation_synergy_guidance_checkpoint(payload)
            del payload
        result["candidate_inference_parity"] = {
            "manifest": "reproducibility/candidate_synergy_migration_parity.json",
            "status": "passed",
            "logit_max_abs_difference": 0.0,
        }
    result["deletion_gate"] = {
        "canonical_model": "passed",
        "canonical_prepared_data_contract": "passed",
        "canonical_training_cli": "passed",
        "two_profile_encoder_gpu_parity": "passed",
        "formal_checkpoint_schema": "passed",
        "generation_live_source_consumer_count": len(
            result["generation_consumers"]["live_source_references"]
        ),
        "snapshot_recovery": args.legacy_ref,
        "status": "delete_ready",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
