#!/usr/bin/env python
"""Verify the frozen ApexOracle main Fig. 3a producer and asset lineage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    mdlm_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdlm-root", type=Path, default=mdlm_root)
    parser.add_argument("--core-root", type=Path, default=mdlm_root.parent / "Synergy")
    parser.add_argument(
        "--generation-root",
        type=Path,
        default=mdlm_root.parent / "discrete-diffusion-guidance",
    )
    parser.add_argument(
        "--manuscript-root",
        type=Path,
        default=mdlm_root.parent
        / "ApexOracle_cleaned"
        / "docs"
        / "ApexOracle_Nat_Biotech",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=mdlm_root / "reproducibility" / "paper_figure_lineage.json",
    )
    parser.add_argument(
        "--plotted-data",
        type=Path,
        default=mdlm_root / "reproducibility" / "paper_fig3a_plotted_data.csv",
    )
    parser.add_argument(
        "--write-plotted-data",
        action="store_true",
        help="Write the exact cache-backed rows used by the violin plot.",
    )
    parser.add_argument(
        "--include-large-assets",
        action="store_true",
        help="Also SHA-256 the 9.17 GB formal checkpoint.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob(root: Path, ref: str, relative_path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{ref}:{relative_path}"], cwd=root)


def resolve_asset(asset: dict[str, Any], roots: dict[str, Path]) -> Path:
    return roots[asset["owner"]] / asset["relative_path"]


def assert_close(actual: float, expected: float, *, tolerance: float = 1e-10) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"Expected {expected!r}, found {actual!r}")


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    roots = {
        "mdlm": args.mdlm_root.resolve(),
        "core": args.core_root.resolve(),
        "generation": args.generation_root.resolve(),
        "manuscript": args.manuscript_root.resolve(),
    }
    results: list[dict[str, Any]] = []

    producer = manifest["producer"]
    snapshot_payload = git_blob(
        roots["mdlm"], producer["snapshot_ref"], producer["relative_path"]
    )
    assert len(snapshot_payload) == producer["snapshot_file_bytes"]
    assert (
        hashlib.sha256(snapshot_payload).hexdigest() == producer["snapshot_file_sha256"]
    )
    current_producer = roots["mdlm"] / producer["relative_path"]
    assert sha256(current_producer) == producer["current_file_sha256_at_audit"]
    results.append({"check": "producer_snapshot_and_current_hash", "status": "passed"})

    cache_assets: dict[tuple[str, str], tuple[dict[str, Any], Path]] = {}
    for asset in manifest["assets"]:
        path = resolve_asset(asset, roots)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != asset["bytes"]:
            raise AssertionError(f"Size mismatch for {path}")
        if not asset["large_asset"] or args.include_large_assets:
            actual_hash = sha256(path)
            if actual_hash != asset["sha256"]:
                raise AssertionError(f"SHA-256 mismatch for {path}")
            hash_status = "passed"
        else:
            hash_status = "skipped_large_asset"
        results.append({"check": f"asset:{asset['id']}", "status": hash_status})
        if asset["id"].endswith("_scoring_cache"):
            parts = asset["id"].split("_")
            strain = "BAA-3170" if parts[0] == "baa3170" else "BAA-3197"
            group = "Unconditional" if parts[1] == "unconditional" else "Guided"
            cache_assets[(strain, group)] = (asset, path)

    for condition in manifest["condition_assets"]:
        path = resolve_asset(condition, roots)
        if not path.is_dir():
            raise FileNotFoundError(path)
        count = sum(item.is_file() for item in path.iterdir())
        if count != condition["verified_file_count"]:
            raise AssertionError(f"Condition asset count mismatch for {path}")
    results.append({"check": "condition_asset_directory_counts", "status": "passed"})

    manuscript = roots["manuscript"] / manifest["consumer"]["relative_path"]
    manuscript_text = manuscript.read_text(encoding="utf-8")
    if "\\includegraphics[width=\\textwidth]{Fig4.pdf}" not in manuscript_text:
        raise AssertionError(
            "Manuscript no longer includes the frozen assembled figure"
        )
    results.append({"check": "manuscript_consumer", "status": "passed"})

    import numpy as np
    import torch
    from scipy.stats import mannwhitneyu

    expected_by_strain = {
        item["strain"]: item for item in manifest["frozen_statistics"]
    }
    rows: list[dict[str, Any]] = []
    values_by_strain: dict[str, dict[str, Any]] = {}
    for (strain, group), (asset, path) in sorted(cache_assets.items()):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        values = payload["mics"].detach().cpu().to(torch.float32).numpy()
        target_mic = "1000" if group == "Unconditional" else "1"
        if payload["target_MIC"] != target_mic:
            raise AssertionError(f"Cache label mismatch for {path}")
        if Path(payload["source_file"]).name not in {
            item["relative_path"].rsplit("/", 1)[-1]
            for item in manifest["assets"]
            if "generation" in item["id"]
        }:
            raise AssertionError(f"Cache source is absent from manifest: {path}")
        values_by_strain.setdefault(strain, {})[group] = values
        for index, value in enumerate(values):
            rows.append(
                {
                    "figure_id": manifest["figure_id"],
                    "strain": strain,
                    "display_name": expected_by_strain[strain]["display_name"],
                    "group": group,
                    "target_mic_operational_label": target_mic,
                    "target_length": expected_by_strain[strain]["target_length"],
                    "guidance_method": "noise",
                    "row_index": index,
                    "predicted_mic_umol": format(float(value), ".10g"),
                    "log2_predicted_mic": format(float(np.log2(value)), ".10g"),
                    "source_cache_id": asset["id"],
                }
            )

    for strain, groups in values_by_strain.items():
        expected = expected_by_strain[strain]
        unconditional = groups["Unconditional"]
        guided = groups["Guided"]
        assert len(unconditional) == expected["unconditional_n"]
        assert len(guided) == expected["guided_n"]
        assert_close(
            float(np.mean(unconditional)),
            expected["unconditional_mean_mic"],
            tolerance=1e-6,
        )
        assert_close(
            float(np.mean(guided)), expected["guided_mean_mic"], tolerance=1e-6
        )
        assert_close(
            float(np.percentile(unconditional, 50)),
            expected["unconditional_plotted_median_mic"],
        )
        assert_close(
            float(np.percentile(guided, 50)), expected["guided_plotted_median_mic"]
        )
        p_value = mannwhitneyu(
            np.log2(unconditional), np.log2(guided), alternative="two-sided"
        ).pvalue
        assert_close(float(p_value), expected["mann_whitney_two_sided_p"])
        if f"{p_value:.4f}" != expected["displayed_p"]:
            raise AssertionError(f"Displayed p-value mismatch for {strain}")
    results.append(
        {"check": "cache_metadata_statistics_and_p_values", "status": "passed"}
    )

    fields = list(rows[0])
    if args.write_plotted_data:
        with args.plotted_data.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    elif args.plotted_data.is_file():
        with args.plotted_data.open(encoding="utf-8", newline="") as handle:
            frozen_rows = list(csv.DictReader(handle))
        expected_rows = [
            {key: str(value) for key, value in row.items()} for row in rows
        ]
        if frozen_rows != expected_rows:
            raise AssertionError("Frozen exact plotted-data CSV is stale")
        results.append({"check": "frozen_exact_plotted_data", "status": "passed"})
    else:
        raise FileNotFoundError(
            f"Missing {args.plotted_data}; run once with --write-plotted-data"
        )

    print(json.dumps({"figure_id": manifest["figure_id"], "checks": results}, indent=2))


if __name__ == "__main__":
    main()
