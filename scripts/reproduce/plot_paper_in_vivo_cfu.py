#!/usr/bin/env python
"""Recreate the paper Fig. 5b CFU panel when its raw CSVs are available."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

from apexoracle_mdlm.figures.in_vivo_cfu import (
    CFUDay,
    ReportedComparison,
    load_cfu_groups,
    plot_paper_in_vivo_cfu,
)

GROUP_ORDER = ("Control", "ApexOracle-23", "Polymyxin B")
GROUP_STYLE = {
    "Control": {
        "facecolor": "#000000",
        "edgecolor": "#000000",
        "pointcolor": "#000000",
    },
    "ApexOracle-23": {
        "facecolor": "#F279AB",
        "edgecolor": "#F279AB",
        "pointcolor": "#F279AB",
    },
    "Polymyxin B": {
        "facecolor": "#929292",
        "edgecolor": "#929292",
        "pointcolor": "#929292",
    },
}
REPORTED_COMPARISONS = {
    "Day 1": (
        ReportedComparison(0, 1, 8.95, 0.16, 6.05, "p = 0.1032", 9.08),
        ReportedComparison(0, 2, 10.35, 0.18, 4.62, "p = 0.0002", 10.48),
    ),
    "Day 2": (
        ReportedComparison(0, 1, 9.85, 0.18, 6.18, "p = 0.0463", 9.98),
        ReportedComparison(0, 2, 11.20, 0.18, 5.55, "p = 0.0007", 11.33),
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the Fig. 5b CFU panel. Reported p-value labels are annotations; "
            "this command does not perform the unpublished statistical test."
        )
    )
    parser.add_argument("--day-1", type=Path, required=True)
    parser.add_argument("--day-2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="paper_fig5b_in_vivo_cfu")
    parser.add_argument("--legend-position", choices=("right", "left"), default="right")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = (("Day 1", args.day_1), ("Day 2", args.day_2))
    days = [
        CFUDay(
            label=label,
            groups=load_cfu_groups(path, group_order=GROUP_ORDER),
            reported_comparisons=REPORTED_COMPARISONS[label],
        )
        for label, path in inputs
    ]
    figure, _ = plot_paper_in_vivo_cfu(
        days,
        group_order=GROUP_ORDER,
        group_style=GROUP_STYLE,
        legend_position=args.legend_position,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix, options in (("pdf", {}), ("png", {"dpi": 300})):
        output = args.output_dir / f"{args.stem}.{suffix}"
        figure.savefig(
            output, format=suffix, transparent=False, bbox_inches="tight", **options
        )
        outputs[suffix] = {"path": str(output), "sha256": sha256(output)}
    plt.close(figure)
    manifest = {
        "schema_version": 1,
        "role": "paper Fig. 5b display producer",
        "inputs": {
            label: {"path": str(path), "sha256": sha256(path)} for label, path in inputs
        },
        "groups": list(GROUP_ORDER),
        "replicates_per_group": {
            day.label: {group: len(values) for group, values in day.groups.items()}
            for day in days
        },
        "reported_p_value_labels": {
            label: [item.label for item in REPORTED_COMPARISONS[label]]
            for label, _ in inputs
        },
        "statistical_test_computed_by_this_entry": False,
        "outputs": outputs,
    }
    manifest_path = args.output_dir / f"{args.stem}.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
