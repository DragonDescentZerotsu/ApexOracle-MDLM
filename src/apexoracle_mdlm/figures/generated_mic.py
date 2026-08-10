"""Canonical producer for the generated-molecule MIC distribution panel."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Sequence


GROUP_ORDER = ("Unconditional", "Guided")
GROUP_OFFSETS = (-0.15, 0.15)
GROUP_STYLE = {
    "Unconditional": {"facecolor": "#000000", "edgecolor": "#000000"},
    "Guided": {"facecolor": "#F279AB", "edgecolor": "#F279AB"},
}
STRAIN_ORDER = ("BAA-3170", "BAA-3197")
STRAIN_LABELS = {
    "BAA-3170": r"$\it{E.\ coli}$ AR-0349",
    "BAA-3197": r"$\it{P.\ aeruginosa}$ PA5257",
}
STRAIN_CENTERS = {"BAA-3170": 0.0, "BAA-3197": 0.8}


@dataclass(frozen=True)
class GeneratedMICRecord:
    figure_id: str
    strain: str
    display_name: str
    group: str
    target_mic_operational_label: str
    target_length: int
    guidance_method: str
    row_index: int
    predicted_mic_umol: float
    log2_predicted_mic: float
    source_cache_id: str


@dataclass(frozen=True)
class GeneratedMICStatistics:
    strain: str
    group: str
    count: int
    mean_mic_umol: float
    median_mic_umol: float
    mann_whitney_two_sided_p: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_generated_mic_records(
    path: str | PathLike[str],
) -> list[GeneratedMICRecord]:
    """Load and validate exact plotted rows exported by a scoring capsule."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    records: list[GeneratedMICRecord] = []
    identities: set[tuple[str, str, int]] = set()
    with source.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            record = GeneratedMICRecord(
                figure_id=row["figure_id"],
                strain=row["strain"],
                display_name=row["display_name"],
                group=row["group"],
                target_mic_operational_label=row["target_mic_operational_label"],
                target_length=int(row["target_length"]),
                guidance_method=row["guidance_method"],
                row_index=int(row["row_index"]),
                predicted_mic_umol=float(row["predicted_mic_umol"]),
                log2_predicted_mic=float(row["log2_predicted_mic"]),
                source_cache_id=row["source_cache_id"],
            )
            if record.group not in GROUP_ORDER:
                raise ValueError(f"Unsupported MIC figure group: {record.group!r}.")
            if (
                not math.isfinite(record.predicted_mic_umol)
                or record.predicted_mic_umol <= 0
            ):
                raise ValueError("predicted_mic_umol must be positive and finite.")
            expected_log2 = math.log2(record.predicted_mic_umol)
            if not math.isclose(
                record.log2_predicted_mic,
                expected_log2,
                rel_tol=0,
                # The frozen CSV records the legacy float32 ``torch.log2``.
                abs_tol=5e-7,
            ):
                raise ValueError(
                    f"Stored log2 MIC is inconsistent at {record.strain}/{record.group}/"
                    f"{record.row_index}."
                )
            identity = (record.strain, record.group, record.row_index)
            if identity in identities:
                raise ValueError(f"Duplicate plotted-row identity: {identity}.")
            identities.add(identity)
            records.append(record)
    if not records:
        raise ValueError(f"No generated MIC records were found in {source}.")
    return records


def _group_values(
    records: Iterable[GeneratedMICRecord],
) -> dict[tuple[str, str], list[float]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        grouped[(record.strain, record.group)].append(record.predicted_mic_umol)
    return dict(grouped)


def _group_log2_values(
    records: Iterable[GeneratedMICRecord],
) -> dict[tuple[str, str], list[float]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        grouped[(record.strain, record.group)].append(record.log2_predicted_mic)
    return dict(grouped)


def summarize_generated_mic_records(
    records: Sequence[GeneratedMICRecord],
) -> list[GeneratedMICStatistics]:
    """Compute the frozen per-group statistics and within-strain p-values."""

    import numpy as np
    from scipy.stats import mannwhitneyu

    grouped = _group_values(records)
    results: list[GeneratedMICStatistics] = []
    for strain in STRAIN_ORDER:
        missing = [group for group in GROUP_ORDER if (strain, group) not in grouped]
        if missing:
            raise ValueError(f"Strain {strain} is missing groups: {missing}.")
        unconditional = np.asarray(grouped[(strain, "Unconditional")], dtype=np.float32)
        guided = np.asarray(grouped[(strain, "Guided")], dtype=np.float32)
        p_value = float(
            mannwhitneyu(
                np.log2(unconditional),
                np.log2(guided),
                alternative="two-sided",
            ).pvalue
        )
        for group, values in (
            ("Unconditional", unconditional),
            ("Guided", guided),
        ):
            results.append(
                GeneratedMICStatistics(
                    strain=strain,
                    group=group,
                    count=len(values),
                    mean_mic_umol=float(np.mean(values)),
                    median_mic_umol=float(np.percentile(values, 50)),
                    mann_whitney_two_sided_p=p_value,
                )
            )
    return results


def format_p_value(p_value: float) -> str:
    if p_value < 1e-4:
        return "p < 1e-4"
    return f"p = {p_value:.4f}"


def _draw_significance(
    axis: Any,
    x1: float,
    x2: float,
    y_top: float,
    drop: float,
    text: str,
    text_y: float,
) -> None:
    axis.plot(
        [x1, x1, x2, x2],
        [y_top - drop, y_top, y_top, y_top - drop],
        color="#222222",
        linewidth=1.3,
        solid_capstyle="butt",
        clip_on=False,
        zorder=4,
    )
    axis.text(
        (x1 + x2) / 2,
        text_y,
        text,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="black",
    )


def _add_p_value_annotation(
    axis: Any,
    x1: float,
    x2: float,
    left_values: Any,
    right_values: Any,
) -> float:
    import numpy as np
    from scipy.stats import mannwhitneyu

    p_value = float(
        mannwhitneyu(left_values, right_values, alternative="two-sided").pvalue
    )
    combined = np.concatenate((left_values, right_values))
    y_min = float(np.min(combined))
    y_max = float(np.max(combined))
    y_span = max(y_max - y_min, 1.0)
    y_top = y_max + 0.08 * y_span
    drop = 0.04 * y_span
    text_y = y_top + 0.015 * y_span
    _draw_significance(
        axis,
        x1,
        x2,
        y_top,
        drop,
        format_p_value(p_value),
        text_y,
    )
    current_bottom, current_top = axis.get_ylim()
    axis.set_ylim(current_bottom, max(current_top, text_y + 0.08 * y_span))
    return p_value


def _style_violin_body(body: Any, label: str) -> None:
    import matplotlib.colors as colors

    style = GROUP_STYLE[label]
    body.set_facecolor(colors.to_rgba(style["facecolor"], alpha=0.55))
    body.set_edgecolor("none")
    body.set_linewidth(0.0)
    body.set_antialiased(False)


def _add_violin_outline(axis: Any, body: Any, label: str) -> None:
    from matplotlib.patches import PathPatch

    outline = PathPatch(
        body.get_paths()[0],
        facecolor="none",
        edgecolor=GROUP_STYLE[label]["edgecolor"],
        linewidth=1.4,
        joinstyle="round",
        capstyle="round",
        antialiased=True,
        zorder=body.get_zorder() + 0.2,
    )
    axis.add_patch(outline)


def _violin_span_at_y(body: Any, y_value: float) -> tuple[float, float]:
    import numpy as np

    vertices = body.get_paths()[0].vertices
    intersections: list[float] = []
    for (x1, y1), (x2, y2) in zip(vertices[:-1], vertices[1:]):
        if y1 == y2:
            if y_value == y1:
                intersections.extend([x1, x2])
            continue
        if min(y1, y2) <= y_value <= max(y1, y2):
            ratio = (y_value - y1) / (y2 - y1)
            intersections.append(x1 + ratio * (x2 - x1))
    if len(intersections) < 2:
        center = float(np.mean(vertices[:, 0]))
        return center, center
    return min(intersections), max(intersections)


def _add_distribution_summary(axis: Any, body: Any, values: Any, label: str) -> None:
    import numpy as np

    median = float(np.percentile(values, 50))
    x_min, x_max = _violin_span_at_y(body, median)
    axis.hlines(
        median,
        x_min,
        x_max,
        colors=GROUP_STYLE[label]["edgecolor"],
        linestyles=(0, (1, 1)),
        linewidth=1.2,
        zorder=3,
    )


def plot_generated_mic_distributions(
    records: Sequence[GeneratedMICRecord],
) -> tuple[Any, Any, dict[str, float]]:
    """Render the frozen Fig. 3a source-panel design from exact plotted rows."""

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    grouped = _group_log2_values(records)
    figure, axis = plt.subplots(figsize=(5, 5))
    p_values: dict[str, float] = {}
    for strain in STRAIN_ORDER:
        center = STRAIN_CENTERS[strain]
        x_positions = {
            group: center + offset for group, offset in zip(GROUP_ORDER, GROUP_OFFSETS)
        }
        plotted: dict[str, Any] = {}
        for group in GROUP_ORDER:
            key = (strain, group)
            if key not in grouped:
                raise ValueError(f"Missing plotted values for {strain}/{group}.")
            # Consume the frozen legacy float32 values directly; recomputing
            # log2 in float64 creates avoidable sub-pixel geometry drift.
            values = np.asarray(grouped[key], dtype=np.float32)
            plotted[group] = values
            parts = axis.violinplot(
                [values],
                positions=[x_positions[group]],
                widths=0.22,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                bw_method=0.35,
                points=300,
            )
            body = parts["bodies"][0]
            _style_violin_body(body, group)
            _add_violin_outline(axis, body, group)
            _add_distribution_summary(axis, body, values, group)
        p_values[strain] = _add_p_value_annotation(
            axis,
            x_positions["Unconditional"],
            x_positions["Guided"],
            plotted["Unconditional"],
            plotted["Guided"],
        )

    axis.grid(axis="y", linestyle="--", alpha=0.35, linewidth=1.3)
    axis.set_axisbelow(True)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: int(2**value)))
    centers = [STRAIN_CENTERS[strain] for strain in STRAIN_ORDER]
    axis.set_xlim(
        min(centers) + min(GROUP_OFFSETS) - 0.18,
        max(centers) + max(GROUP_OFFSETS) + 0.18,
    )
    axis.set_xticks(centers)
    axis.set_xticklabels(
        [STRAIN_LABELS[strain] for strain in STRAIN_ORDER], fontsize=13
    )
    axis.set_xlabel("")
    y_label = axis.set_ylabel("log 2 scale MIC value (µmol)")
    y_label.set_fontsize(14)
    axis.set_title("Generated Molecule MIC Distribution", fontsize=14)
    handles = [
        Patch(
            facecolor=GROUP_STYLE[group]["facecolor"],
            edgecolor=GROUP_STYLE[group]["edgecolor"],
            linewidth=2.8,
            alpha=0.65,
            label=group,
        )
        for group in GROUP_ORDER
    ]
    axis.legend(
        handles=handles,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.5, 0.1),
        fontsize=13,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.6)
    axis.spines["bottom"].set_linewidth(1.6)
    axis.tick_params(axis="y", labelsize=10, width=1.6, length=6)
    axis.tick_params(axis="x", labelsize=13, width=1.6, length=6)
    figure.tight_layout()
    return figure, axis, p_values
