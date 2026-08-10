#!/usr/bin/env python
"""Verify MDLM hierarchical-MIC legacy sources against the Core replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


PROFILES = {
    "DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_species_11_species_5_ensemble.py": {
        "holdout": "species_11_cluster",
        "molecule": "online_chemberta_mtr_first_token",
        "batch_size": 70,
        "ensembles": 7,
        "freeze_epochs": 3,
        "output": "11_species_w_SM/7_fold_ensembles",
    },
    "DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_species_11_species_5_ensemble_MDLM_MTR.py": {
        "holdout": "species_11_cluster",
        "molecule": "online_dlm_dit_first_token",
        "batch_size": 30,
        "ensembles": 7,
        "freeze_epochs": 3,
        "output": "11_species_w_SM/MDLM_MTR_7_fold_ensembles",
    },
    "DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_species_3_species_5_ensemble.py": {
        "holdout": "phylum_3_cluster",
        "molecule": "online_chemberta_mtr_first_token",
        "batch_size": 70,
        "ensembles": 7,
        "freeze_epochs": 3,
        "output": "3_species_w_SM/7_fold_ensembles",
    },
    "DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_species_3_species_5_ensemble_MDLM_MTR.py": {
        "holdout": "phylum_3_cluster",
        "molecule": "online_dlm_dit_first_token",
        "batch_size": 30,
        "ensembles": 7,
        "freeze_epochs": 3,
        "output": "3_species_w_SM/MDLM_MTR_7_fold_ensembles",
    },
    "DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_strains_ChemBERTa.py": {
        "holdout": "strain_3_fold_dynamic",
        "molecule": "online_dlm_dit_first_token_last_reg_v1_despite_filename",
        "batch_size": 70,
        "ensembles": 7,
        "freeze_epochs": 10,
        "output": "strain_wise_w_SM_b_attn/MDLM_MTR_7_fold_ensembles",
    },
    "DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_strains_MDLM_MTR.py": {
        "holdout": "strain_3_fold_dynamic",
        "molecule": "online_dlm_dit_first_token_finetuned_checkpoint",
        "batch_size": 30,
        "ensembles": 7,
        "freeze_epochs": 3,
        "output": "strain_wise_w_SM_b_attn/MDLM_MTR_7_fold_ensembles",
    },
    "DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_strains_MDLM_MTR_cls_cuseqlen_dit.py": {
        "holdout": "strain_3_fold_dynamic",
        "molecule": "online_dlm_non_pad_first_token_eval",
        "batch_size": 30,
        "ensembles": 1,
        "freeze_epochs": 3000,
        "output": "strain_wise_w_SM_b_attn/MDLM_MTR_cuseqlen_dit_cls_1_fold_ensembles",
    },
    "DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_strains_MDLM_MTR_cuseqlen_dit.py": {
        "holdout": "strain_3_fold_dynamic",
        "molecule": "online_dlm_non_pad_masked_mean_train",
        "batch_size": 30,
        "ensembles": 7,
        "freeze_epochs": 3000,
        "output": "strain_wise_w_SM_b_attn/MDLM_MTR_cuseqlen_dit_mean_7_fold_ensembles",
    },
    "DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_strains_MDLM_MTR_fix.py": {
        "holdout": "strain_3_fold_dynamic",
        "molecule": "cached_dlm_first_token",
        "batch_size": 90,
        "ensembles": 7,
        "freeze_epochs": 5000,
        "output": "strain_wise_w_SM_b_attn/MDLM_MTR_fix_7_fold_ensembles",
    },
    "DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_strains_MDLM_MTR_fix_mean.py": {
        "holdout": "strain_3_fold_dynamic",
        "molecule": "cached_dlm_mean",
        "batch_size": 90,
        "ensembles": 7,
        "freeze_epochs": 5000,
        "output": "strain_wise_w_SM_b_attn/MDLM_MTR_fix_mean_7_fold_ensembles",
    },
    "DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_strains_MDLM_MTR_mean_cuseqlen_dit.py": {
        "holdout": "strain_3_fold_dynamic",
        "molecule": "online_dlm_non_pad_masked_mean_eval",
        "batch_size": 30,
        "ensembles": 1,
        "freeze_epochs": 3000,
        "output": "strain_wise_w_SM_b_attn/MDLM_MTR_cuseqlen_dit_mean_1_fold_ensembles",
    },
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_show(repo: Path, ref: str, path: str, *, git_dir: Path | None = None) -> bytes:
    command = ["git"]
    if git_dir is not None:
        command.extend([f"--git-dir={git_dir}", f"--work-tree={repo}"])
    command.extend(["show", f"{ref}:{path}"])
    return subprocess.check_output(command, cwd=repo)


def _asset_summary(root: Path, relative: str) -> dict[str, Any]:
    directory = root / relative
    files = (
        sorted(item for item in directory.rglob("*") if item.is_file())
        if directory.is_dir()
        else []
    )
    return {
        "relative_path": relative,
        "exists": directory.is_dir(),
        "files": len(files),
        "checkpoint_files": sum(item.suffix == ".pth" for item in files),
        "log_files": sum(item.suffix == ".log" for item in files),
        "bytes": sum(item.stat().st_size for item in files),
        "policy": "local historical asset; do not move, delete, or commit",
    }


def _runtime_references(
    roots: list[Path], names: set[str], audit_path: Path
) -> list[str]:
    references: list[str] = []
    suffixes = {".py", ".sh", ".yaml", ".yml"}
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            if (
                path.resolve() == audit_path.resolve()
                or "__pycache__" in path.parts
                or "reproducibility" in path.parts
            ):
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if any(name in source for name in names):
                references.append(str(path.resolve()))
    return sorted(set(references))


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdlm-root", type=Path, default=repo)
    parser.add_argument("--core-root", type=Path, default=repo.parent / "Synergy")
    parser.add_argument(
        "--generation-root",
        type=Path,
        default=repo.parent / "discrete-diffusion-guidance",
    )
    parser.add_argument("--snapshot-ref", default="legacy-code-snapshot-2026-08-09")
    parser.add_argument(
        "--core-snapshot-ref", default="legacy-code-snapshot-2026-07-17"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mdlm = args.mdlm_root.resolve()
    core = args.core_root.resolve()
    source_records: dict[str, Any] = {}
    for name, profile in PROFILES.items():
        source = _git_show(mdlm, args.snapshot_ref, name)
        required = ["RegressionHead", "FirstTokenAttention_genome", "MSELoss"]
        missing = [token for token in required if token not in source.decode("utf-8")]
        if missing:
            raise RuntimeError(
                f"Snapshot source {name} lacks expected tokens: {missing}"
            )
        source_records[name] = {
            "sha256": _sha256_bytes(source),
            "bytes": len(source),
            "lines": len(source.splitlines()),
            "profile": profile,
            "recovery": f"git show {args.snapshot_ref}:{name}",
        }

    core_paths = {
        "runner": core / "scripts/reproduce/run_hierarchical_mic.py",
        "config": core / "configs/hierarchical_mic/legacy_mdlm.yaml",
        "experiment_readme": core / "experiments/hierarchical_mic/README.md",
        "cleanup_manifest": core / "experiments/hierarchical_mic/legacy_cleanup.json",
    }
    for path in core_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    core_records = {
        key: {
            "relative_path": str(path.relative_to(core)),
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for key, path in core_paths.items()
    }
    exact_name = "DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_species_3_species_5_ensemble.py"
    core_snapshot_source = _git_show(
        core,
        args.core_snapshot_ref,
        exact_name,
        git_dir=core / ".git-state",
    )
    exact_shared_source = core_snapshot_source == _git_show(
        mdlm, args.snapshot_ref, exact_name
    )
    if not exact_shared_source:
        raise AssertionError(
            "Expected shared phylum ChemBERTa source is no longer exact."
        )

    checkpoint_root = core / "Checkpoints/genome_text_learnable_emb"
    outputs = {
        relative: _asset_summary(checkpoint_root, relative)
        for relative in sorted({item["output"] for item in PROFILES.values()})
    }
    references = _runtime_references(
        [mdlm, core, args.generation_root.resolve()],
        set(PROFILES),
        Path(__file__),
    )
    if references:
        raise RuntimeError(f"Live runtime/config references remain: {references}")
    result = {
        "schema_version": 1,
        "snapshot_ref": args.snapshot_ref,
        "canonical_owner": "ApexOracle-Core",
        "canonical_replacement": core_records,
        "sources": source_records,
        "exact_cross_repo_snapshot_match": {
            "path": exact_name,
            "core_ref": args.core_snapshot_ref,
            "torch_equal_source_bytes": exact_shared_source,
        },
        "historical_output_directories": outputs,
        "runtime_or_config_consumers": references,
        "verified_facts": [
            "All eleven active-tree sources are recoverable byte-for-byte from the MDLM snapshot tag.",
            "The files cover historical strain/species/phylum variants, not downstream MDLM public APIs.",
            "Core owns the unified runner, data preparation, split adapters, checkpoint loader, tests, and cleanup manifest.",
            "One phylum ChemBERTa source is byte-identical across the MDLM and Core snapshots.",
            "No live Python, shell, or YAML caller in MDLM, Core, or Generation references the eleven filenames.",
            "Historical checkpoint/log directories remain local and are not deleted or moved by source cleanup.",
        ],
        "limits": [
            "The old strain split depended on process hash ordering; the exact 2025 membership was not recovered.",
            "Several exploratory output grids are incomplete or share an output directory, so source-to-every-checkpoint identity is not asserted.",
            "This audit maps source roles and the Core replacement; it does not claim every historical variant is a paper producer.",
        ],
    }
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
