"""Validated plotting primitives for the paper murine CFU panel."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


@dataclass(frozen=True)
class ReportedComparison:
    """A figure annotation reported elsewhere, not a computed test result."""

    group_index_1: int
    group_index_2: int
    y_top: float
    left_drop: float
    right_bottom: float
    label: str
    text_y: float


@dataclass(frozen=True)
class CFUDay:
    """One day of positive raw CFU measurements grouped by treatment."""

    label: str
    groups: Mapping[str, tuple[float, ...]]
    reported_comparisons: tuple[ReportedComparison, ...] = ()


def load_cfu_groups(
    path: str | Path, *, group_order: Sequence[str]
) -> dict[str, tuple[float, ...]]:
    """Load the historical two-row wide CSV and validate positive CFU values."""

    source = Path(path)
    with source.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if len(rows) != 2 or len(rows[0]) != len(rows[1]) or len(rows[0]) < 2:
        raise ValueError(
            f"{source} must contain one header row and one equally wide value row."
        )
    headers = rows[0][1:]
    grouped: dict[str, list[float]] = {group: [] for group in group_order}
    for column, (header, value_text) in enumerate(zip(headers, rows[1][1:]), start=2):
        group = header.split(".")[0]
        if group not in grouped:
            raise ValueError(f"Unknown group {group!r} at {source} column {column}.")
        try:
            value = float(value_text)
        except ValueError as error:
            raise ValueError(
                f"Invalid CFU value {value_text!r} at {source} column {column}."
            ) from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"CFU values must be finite and positive in {source}.")
        grouped[group].append(value)
    if any(not values for values in grouped.values()):
        raise ValueError(f"Every configured group must have measurements in {source}.")
    return {group: tuple(values) for group, values in grouped.items()}


def median(values: Sequence[float]) -> float:
    """Return the median of a non-empty sequence without a NumPy dependency."""

    if not values:
        raise ValueError("Cannot compute a median for an empty sequence.")
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return float((ordered[midpoint - 1] + ordered[midpoint]) / 2)


def _draw_reported_comparison(ax, x1: float, x2: float, item: ReportedComparison):
    ax.plot(
        [x1, x1, x2, x2],
        [item.y_top - item.left_drop, item.y_top, item.y_top, item.right_bottom],
        color="#222222",
        linewidth=1.3,
        solid_capstyle="butt",
        clip_on=False,
        zorder=4,
    )
    ax.text(
        (x1 + x2) / 2,
        item.text_y,
        item.label,
        ha="center",
        va="bottom",
        fontsize=22,
        fontweight="bold",
        color="black",
    )


def plot_paper_in_vivo_cfu(
    days: Sequence[CFUDay],
    *,
    group_order: Sequence[str],
    group_style: Mapping[str, Mapping[str, str]],
    legend_position: str = "right",
):
    """Render the paper CFU panel from raw counts and reported annotations.

    The function intentionally does not calculate p-values. Values shown in
    ``reported_comparisons`` must remain identified as externally reported
    annotations until the statistical test and source data are both frozen.
    """

    if len(days) != 2:
        raise ValueError("The frozen paper layout requires exactly two days.")
    if legend_position not in {"left", "right"}:
        raise ValueError("legend_position must be 'left' or 'right'.")
    offsets = [-0.15, 0.0, 0.15]
    if len(group_order) != len(offsets):
        raise ValueError("The frozen paper layout requires exactly three groups.")
    centers = [0.0, 0.5]
    jitters = [-0.014, -0.006, 0.0, 0.006, 0.014, 0.0]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(10, 8.2), dpi=72, constrained_layout=False)
    if legend_position == "left":
        figure.subplots_adjust(left=0.17, right=0.97, bottom=0.16, top=0.88)
    else:
        figure.subplots_adjust(left=0.10, right=0.84, bottom=0.16, top=0.95)

    for day, center in zip(days, centers):
        positions = [center + offset for offset in offsets]
        for group, position in zip(group_order, positions):
            raw_values = day.groups[group]
            values = [math.log10(value) for value in raw_values]
            if len(values) > len(jitters):
                raise ValueError(
                    "The frozen paper jitter layout supports at most six replicates."
                )
            parts = axis.violinplot(
                [values],
                positions=[position],
                widths=0.1,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                bw_method=0.4,
                points=300,
            )
            body = parts["bodies"][0]
            body.set_facecolor(group_style[group]["facecolor"])
            body.set_edgecolor(group_style[group]["edgecolor"])
            body.set_linewidth(2.8)
            body.set_alpha(0.55)
            axis.hlines(
                median(values),
                position - 0.085,
                position + 0.085,
                colors=group_style[group]["edgecolor"],
                linestyles=(0, (1, 1)),
                linewidth=1.2,
                zorder=3,
            )
            axis.scatter(
                [position + jitter for jitter in jitters[: len(values)]],
                values,
                s=58,
                color=group_style[group]["pointcolor"],
                zorder=5,
            )
        for item in day.reported_comparisons:
            _draw_reported_comparison(
                axis,
                positions[item.group_index_1],
                positions[item.group_index_2],
                item,
            )

    axis.set_xlim(-0.27, 0.77)
    axis.set_ylim(2, 12.2)
    axis.set_xticks(centers)
    axis.set_xticklabels([day.label for day in days], fontsize=22, fontweight="bold")
    axis.set_yticks([2, 4, 6, 8, 10])
    axis.tick_params(axis="y", labelsize=22, width=1.6, length=6)
    axis.tick_params(axis="x", width=1.6, length=6)
    for label in axis.get_yticklabels():
        label.set_fontweight("bold")
    axis.set_ylabel(
        r"log$_{10}$ CFU g$^{-1}$", fontsize=28, fontweight="bold", labelpad=10
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.6)
    axis.spines["bottom"].set_linewidth(1.6)
    axis.vlines(
        0.25,
        2,
        10.0,
        colors="#222222",
        linewidth=1.2,
        linestyles=(0, (1.2, 1.2)),
        zorder=2,
    )
    handles = [
        Patch(
            facecolor=group_style[group]["facecolor"],
            edgecolor=group_style[group]["edgecolor"],
            linewidth=1.8,
            alpha=0.68,
            label=group,
        )
        for group in group_order
    ]
    if legend_position == "left":
        legend = axis.legend(
            handles=handles,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(-0.01, 1.13),
            fontsize=20,
            handlelength=0.55,
            handleheight=1.5,
            handletextpad=0.35,
            borderaxespad=0.0,
            labelspacing=0.18,
            ncol=3,
            columnspacing=0.9,
        )
    else:
        legend = axis.legend(
            handles=handles,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.005, 0.54),
            fontsize=22,
            handlelength=0.55,
            handleheight=1.6,
            handletextpad=0.35,
            borderaxespad=0.6,
            labelspacing=0.22,
        )
    for label in legend.get_texts():
        label.set_fontweight("bold")
    return figure, axis
