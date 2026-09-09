#!/usr/bin/env python3
"""Generate trajectory-quality and clearance panels from manuscript tables."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from plot_common import (
    METHODS,
    METHOD_LABELS,
    SCENARIOS,
    save_figure,
)


# Source: paper/mdpi/main.tex, Tables "Trajectory quality" and "Safety margins".
# Values are copied verbatim as (mean, sample standard deviation); None denotes N/A.
DATA = {
    "static_clutter": {
        "ippo": {"efficiency": (0.733, 0.007), "turn": (0.980, 0.055), "rr": (0.661, 0.023), "rp": None},
        "mappo": {"efficiency": (0.723, 0.010), "turn": (1.030, 0.075), "rr": (0.648, 0.033), "rp": None},
        "maddpg": {"efficiency": (0.736, 0.006), "turn": (0.945, 0.050), "rr": (0.692, 0.022), "rp": None},
        "mo_gat_mappo": {"efficiency": (0.741, 0.006), "turn": (0.920, 0.045), "rr": (0.723, 0.020), "rp": None},
    },
    "pedestrian_dynamic": {
        "ippo": {"efficiency": (0.940, 0.0016), "turn": (0.830, 0.055), "rr": (0.665, 0.028), "rp": (0.853, 0.035)},
        "mappo": {"efficiency": (0.938, 0.006), "turn": (0.790, 0.060), "rr": (0.676, 0.029), "rp": (0.872, 0.032)},
        "maddpg": {"efficiency": (0.927, 0.010), "turn": (0.865, 0.075), "rr": (0.644, 0.036), "rp": (0.826, 0.052)},
        "mo_gat_mappo": {"efficiency": (0.938, 0.006), "turn": (0.815, 0.050), "rr": (0.700, 0.022), "rp": (0.923, 0.032)},
    },
    "mixed_complex": {
        "ippo": {"efficiency": (0.780, 0.008), "turn": (0.735, 0.045), "rr": (0.825, 0.030), "rp": (0.780, 0.035)},
        "mappo": {"efficiency": (0.777, 0.009), "turn": (0.770, 0.050), "rr": (0.843, 0.025), "rp": (0.790, 0.021)},
        "maddpg": {"efficiency": (0.779, 0.009), "turn": (0.740, 0.060), "rr": (0.817, 0.021), "rp": (0.764, 0.043)},
        "mo_gat_mappo": {"efficiency": (0.793, 0.006), "turn": (0.700, 0.040), "rr": (0.851, 0.023), "rp": (0.812, 0.020)},
    },
}


def main() -> None:
    metrics = [
        ("efficiency", "Path efficiency ↑", True),
        ("turn", "Turning per metre\n(rad/m) ↓", False),
        ("rr", "Minimum RR distance\n(m) ↑", True),
        ("rp", "Minimum RP distance\n(m) ↑", True),
    ]
    scenario_short = ["SC", "DP", "MC"]
    columns = [(metric, scenario) for metric, _, _ in metrics for scenario in SCENARIOS]
    means = np.full((len(METHODS), len(columns)), np.nan, dtype=float)
    scores = np.full_like(means, np.nan)

    for column_index, (metric, scenario) in enumerate(columns):
        available = []
        for method_index, method in enumerate(METHODS):
            value = DATA[scenario][method][metric]
            if value is not None:
                means[method_index, column_index] = value[0]
                available.append(value[0])
        if not available:
            continue
        low, high = min(available), max(available)
        higher_is_better = next(item[2] for item in metrics if item[0] == metric)
        for method_index in range(len(METHODS)):
            value = means[method_index, column_index]
            if np.isnan(value):
                continue
            if np.isclose(high, low):
                score = 0.5
            else:
                score = (value - low) / (high - low)
            scores[method_index, column_index] = score if higher_is_better else 1.0 - score

    palette = LinearSegmentedColormap.from_list(
        "favorable_score",
        ["#F2C7A5", "#FFFDF8", "#BFD8E6"],
    )
    palette.set_bad("#ECEFF1")
    fig, (axis, bar_axis) = plt.subplots(
        2,
        1,
        figsize=(12.2, 4.15),
        gridspec_kw={"height_ratios": [5.0, 0.45]},
        constrained_layout=True,
    )
    axis.pcolormesh(
        np.arange(len(columns) + 1),
        np.arange(len(METHODS) + 1),
        np.ma.masked_invalid(scores),
        cmap=palette,
        vmin=0.0,
        vmax=1.0,
        shading="flat",
        edgecolors="#FFFFFF",
        linewidth=1.2,
        rasterized=False,
    )
    axis.set_xlim(0, len(columns))
    axis.set_ylim(len(METHODS), 0)

    for row in range(len(METHODS)):
        for column in range(len(columns)):
            value = means[row, column]
            text = "N/A" if np.isnan(value) else f"{value:.3f}"
            axis.text(column + 0.5, row + 0.5, text, ha="center", va="center", fontsize=12.4, color="#24313A")

    axis.set_yticks(np.arange(len(METHODS)) + 0.5, [METHOD_LABELS[method] for method in METHODS])
    axis.set_xticks(np.arange(len(columns)) + 0.5, scenario_short * len(metrics))
    axis.xaxis.tick_top()
    axis.tick_params(axis="x", length=0, pad=3, labelsize=12.2)
    axis.tick_params(axis="y", length=0, labelsize=12.2)
    for boundary in [3, 6, 9]:
        axis.axvline(boundary, color="#FFFFFF", linewidth=4.0)
    for center, (_, title, _) in zip([1.5, 4.5, 7.5, 10.5], metrics):
        axis.text(
            center,
            1.19,
            title,
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=12.7,
            fontweight="bold",
            color="#24313A",
        )
    for spine in axis.spines.values():
        spine.set_visible(False)
    segments = 64
    for index in range(segments):
        bar_axis.add_patch(
            Rectangle(
                (index / segments, 0.0),
                1.0 / segments,
                1.0,
                facecolor=palette((index + 0.5) / segments),
                edgecolor="none",
            )
        )
    bar_axis.set_xlim(0.0, 1.0)
    bar_axis.set_ylim(0.0, 1.0)
    bar_axis.set_yticks([])
    bar_axis.set_xticks([0.0, 0.5, 1.0], ["less favorable", "middle", "more favorable"])
    bar_axis.set_xlabel("Within-column favorable score", labelpad=2, fontsize=12.2)
    bar_axis.tick_params(axis="x", length=0, pad=2, labelsize=12.2)
    for spine in bar_axis.spines.values():
        spine.set_visible(False)
    fig.text(
        0.5,
        -0.08,
        "SC = static clutter;  DP = dynamic pedestrians;  MC = mixed complex;  cell text = mean value",
        ha="center",
        va="bottom",
        fontsize=12.2,
        color="#64727D",
    )
    save_figure(fig, "trajectory_clearance_summary")


if __name__ == "__main__":
    main()
