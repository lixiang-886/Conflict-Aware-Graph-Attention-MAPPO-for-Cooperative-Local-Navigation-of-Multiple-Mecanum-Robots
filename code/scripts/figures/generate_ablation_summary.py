#!/usr/bin/env python3
"""Generate the focused component ablation from fixed manuscript values."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from plot_common import save_figure, style_axis


# Source: paper/mdpi/main.tex, Table "Component ablation".
# Absolute values are copied verbatim as (mean, sample standard deviation).
# Panel (a) displays the stated mean differences relative to Full; uncertainty
# is not recomputed for these derived differences.
VARIANTS = ["full", "no_gat", "no_reward", "no_gate", "no_coordinator"]
LABELS = ["Full", "w/o GAT", "w/o adaptive reward scalarization", "w/o interaction gate", "w/o coordinator"]
COLORS = ["#B279A2", "#4C78A8", "#F58518", "#54A24B", "#C65050"]
DATA = {
    "full": {"completion": (45.90, 0.60), "makespan": (48.65, 0.75), "waiting": (0.70, 0.12), "rr": (0.835, 0.020), "ratio": (0.270, 0.012)},
    "no_gat": {"completion": (47.40, 0.80), "makespan": (50.45, 0.95), "waiting": (0.98, 0.18), "rr": (0.801, 0.028), "ratio": (0.305, 0.018)},
    "no_reward": {"completion": (47.75, 0.90), "makespan": (50.85, 1.05), "waiting": (1.08, 0.20), "rr": (0.790, 0.030), "ratio": (0.315, 0.020)},
    "no_gate": {"completion": (46.55, 0.70), "makespan": (49.35, 0.85), "waiting": (0.82, 0.15), "rr": (0.823, 0.022), "ratio": (0.290, 0.015)},
    "no_coordinator": {"completion": (46.95, 0.75), "makespan": (49.85, 0.90), "waiting": (0.92, 0.17), "rr": (0.808, 0.026), "ratio": (0.300, 0.017)},
}
DELTAS = {
    "full": {"completion": 0.00, "makespan": 0.00, "waiting": 0.00},
    "no_gat": {"completion": 1.50, "makespan": 1.80, "waiting": 0.28},
    "no_reward": {"completion": 1.85, "makespan": 2.20, "waiting": 0.38},
    "no_gate": {"completion": 0.65, "makespan": 0.70, "waiting": 0.12},
    "no_coordinator": {"completion": 1.05, "makespan": 1.20, "waiting": 0.22},
}


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.25), constrained_layout=True)
    metric_specs = [("completion", "Completion"), ("makespan", "Makespan"), ("waiting", "Waiting")]
    metric_colors = ["#4C78A8", "#F58518", "#54A24B"]
    y = np.arange(len(VARIANTS) - 1, dtype=float)
    offsets = [-0.18, 0.0, 0.18]
    for metric_index, (metric, label) in enumerate(metric_specs):
        values = [DELTAS[variant][metric] for variant in VARIANTS[1:]]
        shifted_y = y + offsets[metric_index]
        for row, value in zip(shifted_y, values):
            axes[0].plot([0.0, value], [row, row], color=metric_colors[metric_index], linewidth=1.2, alpha=0.45)
        axes[0].scatter(
            values,
            shifted_y,
            s=42,
            color=metric_colors[metric_index],
            edgecolor="white",
            linewidth=0.6,
            marker=["o", "s", "D"][metric_index],
            label=label,
            zorder=4,
        )
    axes[0].axvline(0.0, color="#64727D", linewidth=0.9)
    axes[0].set_title("(a) Time penalty relative to Full", weight="bold")
    axes[0].set_xlabel("Increase relative to Full (s)")
    axes[0].set_yticks(y, LABELS[1:])
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", color="#D9E0E4", linewidth=0.7, alpha=0.85)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    style_axis(axes[0], grid=False)

    ratios = [DATA[variant]["ratio"] for variant in VARIANTS]
    clearances = [DATA[variant]["rr"] for variant in VARIANTS]
    full_ratio = DATA["full"]["ratio"][0]
    full_clearance = DATA["full"]["rr"][0]
    axes[1].axvline(full_ratio, color="#B279A2", linestyle="--", linewidth=0.9, alpha=0.8)
    axes[1].axhline(full_clearance, color="#B279A2", linestyle="--", linewidth=0.9, alpha=0.8)
    annotation_labels = ["Full", "w/o GAT", "w/o reward", "w/o gate", "w/o coordinator"]
    annotation_offsets = [(-35, 12), (7, -17), (7, 12), (-62, -16), (-72, -14)]
    for index, variant in enumerate(VARIANTS):
        ratio, ratio_std = DATA[variant]["ratio"]
        clearance, clearance_std = DATA[variant]["rr"]
        axes[1].errorbar(
            ratio,
            clearance,
            xerr=ratio_std,
            yerr=clearance_std,
            fmt="*" if variant == "full" else "o",
            markersize=9 if variant == "full" else 5.8,
            color=COLORS[index],
            markeredgecolor="white",
            markeredgewidth=0.6,
            capsize=2.2,
            elinewidth=0.9,
            zorder=4,
        )
        axes[1].annotate(
            annotation_labels[index],
            (ratio, clearance),
            xytext=annotation_offsets[index],
            textcoords="offset points",
            fontsize=7.5,
            color="#24313A",
        )
    axes[1].set_title("(b) Safety--intervention tradeoff", weight="bold")
    axes[1].set_xlabel("Intervention ratio  (lower is better)")
    axes[1].set_ylabel("Minimum RR distance (m)  (higher is better)")
    axes[1].set_xlim(0.245, 0.345)
    axes[1].set_ylim(0.745, 0.875)
    axes[1].grid(color="#D9E0E4", linewidth=0.7, alpha=0.85)
    axes[1].set_axisbelow(True)
    style_axis(axes[1], grid=False)
    save_figure(fig, "focused_ablation_summary")


if __name__ == "__main__":
    main()
