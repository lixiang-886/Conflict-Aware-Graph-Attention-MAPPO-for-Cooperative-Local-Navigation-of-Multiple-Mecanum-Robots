#!/usr/bin/env python3
"""Generate the safety-filter intervention audit from manuscript values."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from plot_common import (
    METHOD_COLORS,
    METHOD_LABELS,
    SCENARIOS,
    save_figure,
    style_axis,
)


# Source: paper/mdpi/main.tex, Table "Safety-filter intervention".
# Values are copied verbatim as (mean, sample standard deviation).
METHODS = ["orca", "ippo", "mappo", "maddpg", "mo_gat_mappo"]
DATA = {
    "static_clutter": {
        "ippo": {"ratio": (0.292, 0.026), "mean": (0.115, 0.002), "maximum": (0.999, 0.000)},
        "mappo": {"ratio": (0.280, 0.028), "mean": (0.127, 0.011), "maximum": (0.999, 0.000)},
        "maddpg": {"ratio": (0.269, 0.003), "mean": (0.120, 0.001), "maximum": (0.995, 0.004)},
        "mo_gat_mappo": {"ratio": (0.208, 0.005), "mean": (0.146, 0.002), "maximum": (0.953, 0.081)},
        "orca": {"ratio": (0.272, 0.008), "mean": (0.123, 0.008), "maximum": (0.999, 0.000)},
    },
    "pedestrian_dynamic": {
        "ippo": {"ratio": (0.303, 0.062), "mean": (0.148, 0.016), "maximum": (0.852, 0.000)},
        "mappo": {"ratio": (0.266, 0.006), "mean": (0.142, 0.001), "maximum": (0.852, 0.000)},
        "maddpg": {"ratio": (0.276, 0.029), "mean": (0.160, 0.016), "maximum": (0.880, 0.046)},
        "mo_gat_mappo": {"ratio": (0.268, 0.023), "mean": (0.129, 0.002), "maximum": (0.632, 0.004)},
        "orca": {"ratio": (0.258, 0.012), "mean": (0.170, 0.004), "maximum": (0.879, 0.042)},
    },
    "mixed_complex": {
        "ippo": {"ratio": (0.313, 0.010), "mean": (0.160, 0.001), "maximum": (0.842, 0.002)},
        "mappo": {"ratio": (0.320, 0.006), "mean": (0.162, 0.002), "maximum": (0.841, 0.001)},
        "maddpg": {"ratio": (0.325, 0.012), "mean": (0.164, 0.004), "maximum": (0.842, 0.006)},
        "mo_gat_mappo": {"ratio": (0.289, 0.007), "mean": (0.156, 0.002), "maximum": (0.599, 0.001)},
        "orca": {"ratio": (0.319, 0.002), "mean": (0.160, 0.001), "maximum": (0.840, 0.001)},
    },
}


FIGURE_SIZE = (11.5, 4.7)
TITLE_FONT_SIZE = 15.0
AXIS_LABEL_FONT_SIZE = 13.5
TICK_FONT_SIZE = 13.0
LEGEND_FONT_SIZE = 13.0
SCENARIO_ABBREVIATIONS = {
    "static_clutter": "SC",
    "pedestrian_dynamic": "DP",
    "mixed_complex": "MC",
}


def main() -> None:
    metrics = [
        ("ratio", "(a) Intervention ratio", "Ratio"),
        ("mean", "(b) Mean normalized\ndifference", "Mean normalized Δu"),
        ("maximum", "(c) Maximum normalized\ndifference", "Maximum normalized Δu"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE, constrained_layout=True)
    x = np.arange(len(SCENARIOS), dtype=float)
    width = 0.155
    scenario_labels = [SCENARIO_ABBREVIATIONS[name] for name in SCENARIOS]
    for axis, (metric, title, ylabel) in zip(axes, metrics):
        for method_index, method in enumerate(METHODS):
            means = [DATA[scenario][method][metric][0] for scenario in SCENARIOS]
            stds = [DATA[scenario][method][metric][1] for scenario in SCENARIOS]
            proposed = method == "mo_gat_mappo"
            offset = (method_index - (len(METHODS) - 1) / 2) * width
            axis.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                color=METHOD_COLORS[method],
                edgecolor="#6F4C78" if proposed else "white",
                linewidth=1.0 if proposed else 0.45,
                capsize=2.0,
                error_kw={"elinewidth": 0.8, "capthick": 0.8},
                label=METHOD_LABELS[method],
                zorder=4 if proposed else 3,
            )
        axis.set_title(title, fontsize=TITLE_FONT_SIZE, weight="bold", pad=10)
        axis.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE, labelpad=6)
        axis.set_xticks(x, scenario_labels, fontsize=TICK_FONT_SIZE, ha="center")
        axis.tick_params(axis="x", labelsize=TICK_FONT_SIZE, pad=6)
        axis.tick_params(axis="y", labelsize=TICK_FONT_SIZE, pad=4)
        style_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=3,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.7,
        handletextpad=0.5,
        columnspacing=1.1,
    )
    save_figure(fig, "safety_filter_intervention")


if __name__ == "__main__":
    main()
