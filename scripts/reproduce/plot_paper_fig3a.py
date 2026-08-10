#!/usr/bin/env python
"""Reproduce the generated-molecule MIC distribution source panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from apexoracle_mdlm.figures import (
    load_generated_mic_records,
    plot_generated_mic_distributions,
    summarize_generated_mic_records,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "reproducibility" / "paper_fig3a_plotted_data.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_generated_mic_records(args.input)
    statistics = summarize_generated_mic_records(records)
    figure, _, p_values = plot_generated_mic_distributions(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")

    summary = {
        "schema_version": 1,
        "figure_id": records[0].figure_id,
        "input": {
            "path": str(args.input),
            "sha256": sha256(args.input),
            "rows": len(records),
        },
        "statistics": [item.to_dict() for item in statistics],
        "p_values": p_values,
        "output": {
            "path": str(args.output),
            "sha256": sha256(args.output),
            "bytes": args.output.stat().st_size,
        },
    }
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
