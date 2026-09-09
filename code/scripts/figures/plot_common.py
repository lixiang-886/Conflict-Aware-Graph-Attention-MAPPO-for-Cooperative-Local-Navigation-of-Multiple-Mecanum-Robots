"""Shared Matplotlib configuration for editable manuscript data figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from common import EDITABLE_DIR


mpl.rcParams.update(
    {
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

SCENARIOS = ["static_clutter", "pedestrian_dynamic", "mixed_complex"]
SCENARIO_LABELS = {
    "static_clutter": "Static clutter",
    "pedestrian_dynamic": "Dynamic pedestrians",
    "mixed_complex": "Mixed complex",
}
METHODS = ["ippo", "mappo", "maddpg", "mo_gat_mappo"]
METHOD_LABELS = {
    "ippo": "IPPO",
    "mappo": "MAPPO",
    "maddpg": "MADDPG-style",
    "mo_gat_mappo": "MO-GAT-MAPPO",
    "orca": "Classical local controller",
}
METHOD_COLORS = {
    "ippo": "#4C78A8",
    "mappo": "#F58518",
    "maddpg": "#54A24B",
    "mo_gat_mappo": "#B279A2",
    "orca": "#8C8C8C",
}


def style_axis(axis: plt.Axes, *, grid: bool = True) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid:
        axis.grid(axis="y", color="#D9E0E4", linewidth=0.7, alpha=0.85)
        axis.set_axisbelow(True)


def save_figure(fig: plt.Figure, name: str) -> Path:
    EDITABLE_DIR.mkdir(parents=True, exist_ok=True)
    output = EDITABLE_DIR / f"{name}.svg"
    fig.savefig(output, format="svg", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    lines = output.read_text(encoding="utf-8").splitlines()
    output.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    return output
