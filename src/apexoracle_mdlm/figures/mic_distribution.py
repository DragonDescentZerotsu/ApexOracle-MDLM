"""Generic single-strain MIC distribution figure for scored peptide tables."""

from __future__ import annotations

from typing import Any, Sequence


def plot_mic_distribution(
    mic_values: Sequence[float],
    *,
    strain: str,
) -> tuple[Any, Any]:
    """Plot finite positive MIC values with the historical log2 violin design."""

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter

    values = np.asarray(mic_values, dtype=np.float32)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        raise ValueError(f"No finite positive MIC values are available for {strain!r}.")
    log2_values = np.log2(values)
    figure, axis = plt.subplots(figsize=(5, 5))
    parts = axis.violinplot(
        [log2_values],
        positions=[0],
        showmeans=False,
        showmedians=True,
        widths=0.5,
    )
    for body in parts["bodies"]:
        body.set_facecolor("#B49EDE")
        body.set_alpha(0.7)
        body.set_edgecolor("none")
    for key in ("cmedians", "cbars", "cmaxes", "cmins"):
        if key in parts:
            parts[key].set_edgecolor("#6B4FA8")
            parts[key].set_linewidth(2 if key == "cmedians" else 1.5)
    axis.grid(axis="y", linestyle="--", alpha=0.35, linewidth=1.6)
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{int(round(2**value))}")
    )
    axis.set_axisbelow(True)
    axis.set_title(f"Molecule MIC distribution\nagainst {strain}", fontsize=14)
    axis.set_xlabel("")
    axis.set_ylabel("log 2 scale MIC value (µmol)", fontsize=11)
    axis.set_xticks([])
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["bottom"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.tick_params(axis="both", which="both", length=0)
    figure.tight_layout()
    return figure, axis
