#!/usr/bin/env python3
"""Generate the main efficiency comparison from fixed manuscript values."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from plot_common import (
    METHODS,
    METHOD_COLORS,
    METHOD_LABELS,
    SCENARIOS,
    SCENARIO_LABELS,
    save_figure,
    style_axis,
)


# Source: paper/mdpi/main.tex, Table "Main performance".
# Values are copied verbatim as (mean, sample standard deviation).
DATA = {
    "static_clutter": {
        "ippo": {"completion": (35.66, 0.34), "makespan": (38.47, 0.31), "waiting": (1.89, 0.05)},
        "mappo": {"completion": (37.44, 2.98), "makespan": (41.40, 5.22), "waiting": (2.82, 1.55)},
        "maddpg": {"completion": (35.28, 0.31), "makespan": (37.97, 0.32), "waiting": (1.39, 0.17)},
        "mo_gat_mappo": {"completion": (33.58, 0.11), "makespan": (36.70, 0.17), "waiting": (0.50, 0.07)},
    },
    "pedestrian_dynamic": {
        "ippo": {"completion": (32.12, 0.30), "makespan": (35.57, 0.50), "waiting": (1.67, 0.48)},
        "mappo": {"completion": (32.30, 0.39), "makespan": (35.53, 0.31), "waiting": (1.98, 0.65)},
        "maddpg": {"completion": (33.66, 1.69), "makespan": (39.90, 5.37), "waiting": (2.45, 0.39)},
        "mo_gat_mappo": {"completion": (31.70, 0.18), "makespan": (33.30, 0.40), "waiting": (1.03, 0.24)},
    },
    "mixed_complex": {
        "ippo": {"completion": (47.42, 0.39), "makespan": (50.33, 0.64), "waiting": (0.99, 0.08)},
        "mappo": {"completion": (47.61, 0.75), "makespan": (50.53, 0.76), "waiting": (1.08, 0.03)},
        "maddpg": {"completion": (47.73, 0.60), "makespan": (50.90, 1.22), "waiting": (0.99, 0.12)},
        "mo_gat_mappo": {"completion": (45.90, 0.64), "makespan": (48.65, 0.74), "waiting": (0.71, 0.09)},
    },
}


def main() -> None:
    # This figure is included at the manuscript's full text width.  Keeping the
    # source canvas close to that physical width prevents LaTeX from shrinking
    # otherwise reasonable Matplotlib fonts to roughly 5 pt in the final PDF.
    title_size = 10.5
    axis_label_size = 9.5
    tick_size = 9.0
    legend_size = 9.0
    metrics = [
        ("completion", "(a) Average completion time", "Time (s)"),
        ("makespan", "(b) Makespan", "Time (s)"),
        ("waiting", "(c) Average waiting time", "Time (s)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.75), constrained_layout=True)
    x = np.arange(len(SCENARIOS), dtype=float)
    markers = {"ippo": "o", "mappo": "s", "maddpg": "D", "mo_gat_mappo": "*"}
    for axis, (metric, title, ylabel) in zip(axes, metrics):
        for method in METHODS:
            means = [DATA[scenario][method][metric][0] for scenario in SCENARIOS]
            stds = [DATA[scenario][method][metric][1] for scenario in SCENARIOS]
            proposed = method == "mo_gat_mappo"
            axis.errorbar(
                x,
                means,
                yerr=stds,
                color=METHOD_COLORS[method],
                marker=markers[method],
                markersize=7.0 if proposed else 4.8,
                markeredgecolor="white" if proposed else METHOD_COLORS[method],
                markeredgewidth=0.7,
                linewidth=2.0 if proposed else 1.2,
                capsize=2.5,
                elinewidth=1.0,
                label=METHOD_LABELS[method],
                zorder=5 if proposed else 3,
            )
        axis.set_title(title, fontsize=title_size, weight="bold")
        axis.set_ylabel(ylabel, fontsize=axis_label_size)
        axis.set_xticks(
            x,
            [SCENARIO_LABELS[name] for name in SCENARIOS],
            rotation=18,
            ha="right",
            fontsize=tick_size,
        )
        axis.tick_params(axis="y", labelsize=tick_size)
        axis.margins(x=0.08, y=0.12)
        style_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=4,
        frameon=False,
        fontsize=legend_size,
        handlelength=1.6,
        columnspacing=1.2,
        handletextpad=0.45,
    )
    save_figure(fig, "main_efficiency_comparison")


if __name__ == "__main__":
    main()
