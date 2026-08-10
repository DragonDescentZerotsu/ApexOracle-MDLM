#!/usr/bin/env python
"""Audit source-level contracts among ApexOracle Core, MDLM, and Generation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdlm-root", type=Path, default=repository_root)
    parser.add_argument("--synergy-root", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repository_root / "reproducibility" / "cross_repo_contracts.json",
    )
    parser.add_argument(
        "--check-assets",
        action="store_true",
        help="Mmap and validate the trusted formal local checkpoints in the manifest.",
    )
    parser.add_argument(
        "--check-gpu-head-parity",
        action="store_true",
        help="Compare Generation and canonical MIC heads with formal weights on one visible GPU.",
    )
    return parser.parse_args()


def class_node(path: Path, class_name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise ValueError(f"Class {class_name!r} was not found in {path}.")


def normalized_class_digest(path: Path, class_name: str) -> str:
    payload = ast.dump(class_node(path, class_name), include_attributes=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assigned_self_attributes(node: ast.ClassDef) -> set[str]:
    attributes: set[str] = set()
    for child in ast.walk(node):
        targets: list[ast.expr] = []
        if isinstance(child, ast.Assign):
            targets.extend(child.targets)
        elif isinstance(child, ast.AnnAssign):
            targets.append(child.target)
        for target in targets:
            for item in ast.walk(target):
                if (
                    isinstance(item, ast.Attribute)
                    and isinstance(item.value, ast.Name)
                    and item.value.id == "self"
                ):
                    attributes.add(item.attr)
    return attributes


def check_formal_assets(
    manifest: dict[str, Any], roots: dict[str, Path]
) -> list[dict[str, Any]]:
    import torch

    from apexoracle_mdlm.checkpoints import (
        validate_generation_dlm_checkpoint,
        validate_generation_mic_guidance_checkpoint,
        validate_generation_peptide_classifier_checkpoint,
        validate_generation_synergy_guidance_checkpoint,
    )
    from apexoracle_mdlm.models import FirstTokenCrossAttention, RegressionHead

    contracts = {item["id"]: item for item in manifest["artifact_contracts"]}

    def load(asset_id: str) -> tuple[Path, dict[str, Any]]:
        contract = contracts[asset_id]
        path = roots[contract["owner"]] / contract["relative_path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing formal asset {asset_id}: {path}")
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        return path, payload

    results: list[dict[str, Any]] = []
    path, payload = load("generation_dlm_v1")
    validate_generation_dlm_checkpoint(payload)
    results.append({"id": "generation_dlm_v1_schema", "status": "passed"})
    del payload

    path, payload = load("generation_peptide_classifier_v1")
    validate_generation_peptide_classifier_checkpoint(payload)
    classifier_state = {
        key.removeprefix("ClsHead."): value
        for key, value in payload["state_dict"].items()
        if key.startswith("ClsHead.")
    }
    with torch.device("meta"):
        classifier_head = RegressionHead(768, 384, 128, 1, 0.2)
    classifier_head.load_state_dict(classifier_state, strict=True, assign=True)
    results.append(
        {"id": "generation_peptide_classifier_v1_strict_head", "status": "passed"}
    )
    del payload, classifier_head

    for asset_id in (
        "generation_noisy_mic_guidance",
        "candidate_clean_mic_scorer",
    ):
        path, payload = load(asset_id)
        validate_generation_mic_guidance_checkpoint(payload)
        with torch.device("meta"):
            regression_head = RegressionHead(12288, 3072, 128, 1, 0.2)
            genome_attention = FirstTokenCrossAttention(
                768, 8192, 4, 0.1, return_attention=False
            )
            text_attention = FirstTokenCrossAttention(
                768, 4096, 4, 0.1, return_attention=False
            )
        regression_head.load_state_dict(
            payload["re_head_state_dict"], strict=True, assign=True
        )
        genome_attention.load_state_dict(
            payload["co_cross_attn_genome"], strict=True, assign=True
        )
        text_attention.load_state_dict(
            payload["co_cross_attn_text"], strict=True, assign=True
        )
        results.append({"id": f"{asset_id}_strict_heads", "status": "passed"})
        del payload, regression_head, genome_attention, text_attention

    path, payload = load("generation_synergy_guidance")
    validate_generation_synergy_guidance_checkpoint(payload)
    results.append({"id": "generation_synergy_guidance_schema", "status": "passed"})
    del payload

    path, payload = load("synergy_partner_embeddings")
    required_partners = ("Gentamicin", 447, 37)
    for key in required_partners:
        if key not in payload or tuple(payload[key].shape) != (1, 768):
            raise ValueError(
                f"Partner embedding {key!r} is missing or has an incompatible shape."
            )
    results.append({"id": "synergy_partner_embedding_keys", "status": "passed"})
    del payload

    return results


def check_saved_tensor_window_contract(roots: dict[str, Path]) -> dict[str, Any]:
    """Compare MDLM and Core window coordinates on edge-case contig lengths."""

    core_source_root = str(roots["synergy"] / "src")
    if core_source_root not in sys.path:
        sys.path.insert(0, core_source_root)
    from apexoracle.evaluation.genome_condition_reviewer import (
        build_saved_tensor_windows as core_windows,
    )
    from apexoracle_mdlm.interpretability import (
        build_saved_tensor_windows as mdlm_windows,
    )

    lengths = [21_500, 10_000, 35_000]
    expected = core_windows(lengths)
    actual = [
        {
            "fragment_index": row.fragment_index,
            "contig_index": row.contig_index,
            "start": row.start,
            "end": row.end,
        }
        for row in mdlm_windows(lengths)
    ]
    if actual != expected:
        raise RuntimeError(
            f"Core/MDLM saved-tensor window contract differs: {actual} != {expected}."
        )
    return {
        "id": "core_mdlm_saved_tensor_window_coordinates",
        "status": "passed",
        "edge_case_windows": actual,
    }


def check_gpu_head_parity(
    manifest: dict[str, Any], roots: dict[str, Path]
) -> dict[str, Any]:
    import importlib.util

    import torch

    from apexoracle_mdlm.models import FirstTokenCrossAttention, RegressionHead

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expose exactly one GPU with CUDA_VISIBLE_DEVICES for head parity."
        )

    source = roots["generation"] / "models" / "antibiotic_classifier.py"
    spec = importlib.util.spec_from_file_location(
        "generation_antibiotic_classifier", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Generation head module from {source}.")
    legacy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_module)

    contracts = {item["id"]: item for item in manifest["artifact_contracts"]}
    contract = contracts["generation_noisy_mic_guidance"]
    checkpoint_path = roots[contract["owner"]] / contract["relative_path"]
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )

    with torch.device("meta"):
        legacy_genome = legacy_module.FirstTokenAttention_genome(768, 8192, 4, 0.1)
        canonical_genome = FirstTokenCrossAttention(
            768, 8192, 4, 0.1, return_attention=False, legacy_squeeze=True
        )
        legacy_text = legacy_module.FirstTokenAttention_genome(768, 4096, 4, 0.1)
        canonical_text = FirstTokenCrossAttention(
            768, 4096, 4, 0.1, return_attention=False, legacy_squeeze=True
        )
        legacy_regression = legacy_module.RegressionHead(12288, 3072, 128, 1, 0.2)
        canonical_regression = RegressionHead(12288, 3072, 128, 1, 0.2)

    model_states = (
        (legacy_genome, checkpoint["co_cross_attn_genome"]),
        (canonical_genome, checkpoint["co_cross_attn_genome"]),
        (legacy_text, checkpoint["co_cross_attn_text"]),
        (canonical_text, checkpoint["co_cross_attn_text"]),
        (legacy_regression, checkpoint["re_head_state_dict"]),
        (canonical_regression, checkpoint["re_head_state_dict"]),
    )
    for model, state_dict in model_states:
        model.load_state_dict(state_dict, strict=True, assign=True)
        model.to("cuda").eval()

    torch.manual_seed(20260809)
    torch.cuda.reset_peak_memory_stats()
    molecule = torch.randn(2, 768, device="cuda") * 1e-3
    genome = torch.randn(2, 2, 8192, device="cuda") * 1e-3
    text = torch.randn(2, 3, 4096, device="cuda") * 1e-3
    genome_mask = torch.tensor([[False, False], [False, True]], device="cuda")
    text_mask = torch.tensor(
        [[False, False, False], [False, False, True]], device="cuda"
    )

    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        legacy_genome_output = legacy_genome(molecule, genome, genome_mask)
        canonical_genome_output = canonical_genome(molecule, genome, genome_mask)
        legacy_text_output = legacy_text(molecule, text, text_mask)
        canonical_text_output = canonical_text(molecule, text, text_mask)
        legacy_fused = torch.cat(
            (
                legacy_genome_output.reshape(-1, 8192),
                legacy_text_output.reshape(-1, 4096),
            ),
            dim=1,
        )
        canonical_fused = torch.cat(
            (
                canonical_genome_output.reshape(-1, 8192),
                canonical_text_output.reshape(-1, 4096),
            ),
            dim=1,
        )
        legacy_prediction = legacy_regression(legacy_fused)
        canonical_prediction = canonical_regression(canonical_fused)

    equal = {
        "genome": torch.equal(legacy_genome_output, canonical_genome_output),
        "text": torch.equal(legacy_text_output, canonical_text_output),
        "regression": torch.equal(legacy_prediction, canonical_prediction),
    }
    max_abs_diff = float((legacy_prediction - canonical_prediction).abs().max().float())
    if not all(equal.values()):
        raise RuntimeError(
            f"Generation/canonical head parity failed: equal={equal}, "
            f"max_abs_diff={max_abs_diff}."
        )
    return {
        "id": "generation_formal_bfloat16_gpu_head_parity",
        "status": "passed",
        "torch_equal": equal,
        "max_abs_diff": max_abs_diff,
        "output_shape": list(canonical_prediction.shape),
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }


def main() -> None:
    args = parse_args()
    roots = {
        "mdlm": args.mdlm_root.resolve(),
        "synergy": args.synergy_root.resolve(),
        "generation": args.generation_root.resolve(),
    }
    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    results.append(check_saved_tensor_window_contract(roots))

    for check in manifest["static_checks"]:
        path = roots[check["repository"]] / check["path"]
        source = path.read_text(encoding="utf-8")
        missing = [
            value for value in check["required_substrings"] if value not in source
        ]
        if missing:
            raise RuntimeError(f"{check['id']} failed for {path}; missing={missing}")
        results.append({"id": check["id"], "status": "passed"})

    for check in manifest["ast_equivalence_checks"]:
        left = check["left"]
        right = check["right"]
        left_digest = normalized_class_digest(
            roots[left["repository"]] / left["path"], left["class"]
        )
        right_digest = normalized_class_digest(
            roots[right["repository"]] / right["path"], right["class"]
        )
        if left_digest != right_digest:
            raise RuntimeError(
                f"{check['id']} failed: {left_digest} != {right_digest}."
            )
        results.append({"id": check["id"], "status": "passed", "sha256": left_digest})

    for check in manifest["class_module_checks"]:
        path = roots[check["repository"]] / check["path"]
        attributes = assigned_self_attributes(class_node(path, check["class"]))
        missing = sorted(set(check["required_self_attributes"]) - attributes)
        if missing:
            raise RuntimeError(f"{check['id']} failed for {path}; missing={missing}")
        results.append({"id": check["id"], "status": "passed"})

    if args.check_assets:
        results.extend(check_formal_assets(manifest, roots))
    if args.check_gpu_head_parity:
        results.append(check_gpu_head_parity(manifest, roots))

    print(
        json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "status": "passed",
                "checks": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
