import argparse
import csv
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

GROUP_ORDER = ["Control", "ApexOracle-23", "Polymyxin B"]
GROUP_OFFSETS = [-0.15, 0.0, 0.15]
JITTERS = [-0.014, -0.006, 0.0, 0.006, 0.014, 0.0]
Y_LIM = (2, 12.2)
X_PADDING = 0.12
LEGEND_POSITIONS = ("right", "left")

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

DAY_CONFIGS = [
    {
        "csv_name": "CFU - Skin Scarification - A. baumannii ATCC19606 - Day 1.csv",
        "label": "Day 1",
        "center": 0.0,
        "annotations": [
            {
                "group_index_1": 0,
                "group_index_2": 1,
                "y_top": 8.95,
                "left_drop": 0.16,
                "right_bottom": 6.05,
                "text": "p = 0.1032",
                "text_y": 9.08,
            },
            {
                "group_index_1": 0,
                "group_index_2": 2,
                "y_top": 10.35,
                "left_drop": 0.18,
                "right_bottom": 4.62,
                "text": "p = 0.0002",
                "text_y": 10.48,
            },
        ],
    },
    {
        "csv_name": "CFU - Skin Scarification - A. baumannii ATCC19606 - Day 2.csv",
        "label": "Day 2",
        "center": 0.5,
        "annotations": [
            {
                "group_index_1": 0,
                "group_index_2": 1,
                "y_top": 9.85,
                "left_drop": 0.18,
                "right_bottom": 6.18,
                "text": "p = 0.0463",
                "text_y": 9.98,
            },
            {
                "group_index_1": 0,
                "group_index_2": 2,
                "y_top": 11.20,
                "left_drop": 0.18,
                "right_bottom": 5.55,
                "text": "p = 0.0007",
                "text_y": 11.33,
            },
        ],
    },
]


def load_log10_values(csv_path: Path) -> dict[str, list[float]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    headers = rows[0][1:]
    values = [float(value) for value in rows[1][1:]]
    grouped: dict[str, list[float]] = {group: [] for group in GROUP_ORDER}
    for header, value in zip(headers, values):
        grouped[header.split(".")[0]].append(math.log10(value))
    return grouped


def draw_significance(
    ax,
    x1: float,
    x2: float,
    y_top: float,
    left_drop: float,
    right_bottom: float,
    text: str,
    text_y: float,
) -> None:
    ax.plot(
        [x1, x1, x2, x2],
        [y_top - left_drop, y_top, y_top, right_bottom],
        color="#222222",
        linewidth=1.3,
        solid_capstyle="butt",
        clip_on=False,
        zorder=4,
    )
    ax.text(
        (x1 + x2) / 2,
        text_y,
        text,
        ha="center",
        va="bottom",
        fontsize=22,
        fontweight="bold",
        color="black",
    )


def median_value(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def draw_day(ax, day_config: dict) -> None:
    grouped = load_log10_values(ROOT / day_config["csv_name"])
    x_positions = [day_config["center"] + offset for offset in GROUP_OFFSETS]

    for group, xpos in zip(GROUP_ORDER, x_positions):
        values = grouped[group]
        parts = ax.violinplot(
            [values],
            positions=[xpos],
            widths=0.1,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.4,
            points=300,
        )
        body = parts["bodies"][0]
        body.set_facecolor(GROUP_STYLE[group]["facecolor"])
        body.set_edgecolor(GROUP_STYLE[group]["edgecolor"])
        body.set_linewidth(2.8)
        body.set_alpha(0.55)

        ax.hlines(
            median_value(values),
            xpos - 0.085,
            xpos + 0.085,
            colors=GROUP_STYLE[group]["edgecolor"],
            linestyles=(0, (1, 1)),
            linewidth=1.2,
            zorder=3,
        )

        ax.scatter(
            [xpos + jitter for jitter in JITTERS[: len(values)]],
            values,
            s=58,
            color=GROUP_STYLE[group]["pointcolor"],
            zorder=5,
        )

    for annotation in day_config["annotations"]:
        draw_significance(
            ax,
            x_positions[annotation["group_index_1"]],
            x_positions[annotation["group_index_2"]],
            annotation["y_top"],
            annotation["left_drop"],
            annotation["right_bottom"],
            annotation["text"],
            annotation["text_y"],
        )


def style_axes(ax) -> None:
    centers = [config["center"] for config in DAY_CONFIGS]
    ax.set_xlim(min(centers) + min(GROUP_OFFSETS) - X_PADDING, max(centers) + max(GROUP_OFFSETS) + X_PADDING)
    ax.set_ylim(*Y_LIM)
    ax.set_xticks(centers)
    ax.set_xticklabels([config["label"] for config in DAY_CONFIGS], fontsize=22, fontweight="bold")
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.tick_params(axis="y", labelsize=22, width=1.6, length=6)
    ax.tick_params(axis="x", width=1.6, length=6)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_ylabel(r"log$_{10}$ CFU g$^{-1}$", fontsize=28, fontweight="bold", labelpad=10)
    ax.set_xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)


def draw_separator(ax) -> None:
    separator_x = sum(config["center"] for config in DAY_CONFIGS) / len(DAY_CONFIGS)
    ax.vlines(
        separator_x,
        Y_LIM[0],
        10.0,
        colors="#222222",
        linewidth=1.2,
        linestyles=(0, (1.2, 1.2)),
        zorder=2,
    )


def add_legend(ax, legend_position: str) -> None:
    handles = [
        Patch(
            facecolor=GROUP_STYLE[group]["facecolor"],
            edgecolor=GROUP_STYLE[group]["edgecolor"],
            linewidth=1.8,
            alpha=0.68,
            label=group,
        )
        for group in GROUP_ORDER
    ]

    if legend_position == "left":
        legend_kwargs = {
            "loc": "upper left",
            "bbox_to_anchor": (-0.01, 1.13),
            "fontsize": 20,
            "handlelength": 0.55,
            "handleheight": 1.5,
            "handletextpad": 0.35,
            "borderaxespad": 0.0,
            "labelspacing": 0.18,
            "ncol": 3,
            "columnspacing": 0.9,
        }
    else:
        legend_kwargs = {
            "loc": "center left",
            "bbox_to_anchor": (1.005, 0.54),
            "fontsize": 22,
            "handlelength": 0.55,
            "handleheight": 1.6,
            "handletextpad": 0.35,
            "borderaxespad": 0.6,
            "labelspacing": 0.22,
        }

    legend = ax.legend(
        handles=handles,
        frameon=False,
        **legend_kwargs,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a two-day CFU violin plot.")
    parser.add_argument(
        "--legend-position",
        choices=LEGEND_POSITIONS,
        default="right",
        help="Place the legend on the right side or near the upper-left y-axis edge.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 8.2), dpi=72, constrained_layout=False)
    if args.legend_position == "left":
        fig.subplots_adjust(left=0.17, right=0.97, bottom=0.16, top=0.88)
    else:
        fig.subplots_adjust(left=0.10, right=0.84, bottom=0.16, top=0.95)

    for day_config in DAY_CONFIGS:
        draw_day(ax, day_config)

    style_axes(ax)
    draw_separator(ax)
    add_legend(ax, args.legend_position)

    stem = "CFU - Skin Scarification - A. baumannii ATCC19606 - Day 1 and Day 2"
    fig.savefig(ROOT / f"{stem} - recreated.png", dpi=300, transparent=False, bbox_inches="tight")
    fig.savefig(ROOT / f"{stem} - recreated.pdf", transparent=False, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
