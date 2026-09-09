#!/usr/bin/env python3
"""Generate traceable Sensors result figures from processed CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO_ROOT = ROOT / "src/mrpp_experiments/config/scenarios"
METHODS = (
    ("orca", "Classical", "#5B5B5B", "o"),
    ("matched_mappo", "Matched MAPPO", "#3D78B4", "s"),
    ("gat_mappo", "GAT-MAPPO", "#2E9D74", "D"),
    ("sensors_mo_gat_mappo", "MO-GAT-MAPPO", "#C84D4D", "^"),
)
ROBUSTNESS_METHODS = tuple(item for item in METHODS if item[0] != "gat_mappo")
SCENARIOS = (
    ("static_clutter", "Static"),
    ("pedestrian_dynamic", "Dynamic"),
    ("mixed_complex", "Mixed"),
)
PROFILES = (
    ("nominal", "Nominal"),
    ("mild", "Mild"),
    ("moderate", "Moderate"),
    ("severe", "Severe"),
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _scenario_horizons(root: Path) -> dict[str, float]:
    horizons = {}
    for scenario, _label in SCENARIOS:
        path = root / f"{scenario}.yaml"
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        horizons[scenario] = float(config["max_steps"]) * float(config["dt"])
    return horizons


def _episode_metric(
    row: dict[str, str],
    metric: str,
    horizons: dict[str, float],
) -> float:
    if metric == "penalized_average_completion_time":
        success = float(row["success_rate"])
        return (
            success * float(row["average_completion_time"])
            + (1.0 - success) * horizons[row["scenario"]]
        )
    if metric == "penalized_makespan":
        return (
            float(row["makespan"])
            if float(row["success_rate"]) >= 1.0 - 1e-12
            else horizons[row["scenario"]]
        )
    return float(row[metric])


def _save(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def main_distribution_figure(
    episodes: list[dict[str, str]],
    seeds: list[dict[str, str]],
    horizons: dict[str, float],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True)
    metrics = (
        ("penalized_average_completion_time", "Penalized average time (s)"),
        ("penalized_makespan", "Penalized makespan (s)"),
        ("intervention_time_ratio", "Intervention ratio"),
    )
    offsets = np.linspace(-0.24, 0.24, len(METHODS))
    for axis, (metric, ylabel) in zip(axes, metrics):
        for method_index, (method, label, color, marker) in enumerate(METHODS):
            for scenario_index, (scenario, _scenario_label) in enumerate(SCENARIOS):
                episode_values = [
                    _episode_metric(row, metric, horizons)
                    for row in episodes
                    if row["algorithm"] == method and row["scenario"] == scenario
                ]
                seed_values = [
                    float(row[metric])
                    for row in seeds
                    if row["algorithm"] == method and row["scenario"] == scenario
                ]
                x = scenario_index + offsets[method_index]
                if episode_values:
                    jitter = np.linspace(-0.035, 0.035, len(episode_values))
                    axis.scatter(
                        x + jitter,
                        episode_values,
                        s=10,
                        color=color,
                        alpha=0.18,
                        linewidths=0,
                    )
                if seed_values:
                    axis.scatter(
                        [x] * len(seed_values),
                        seed_values,
                        s=42,
                        color=color,
                        marker=marker,
                        edgecolors="white",
                        linewidths=0.6,
                        label=label if axis is axes[0] and scenario_index == 0 else None,
                        zorder=3,
                    )
        axis.set_xticks(range(len(SCENARIOS)))
        axis.set_xticklabels([label for _name, label in SCENARIOS])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    _save(fig, output)


def sensor_robustness_figure(
    rows: list[dict[str, str]],
    output: Path,
) -> None:
    indexed = {
        (row["scenario"], row["sensor_profile"], row["algorithm"]): row
        for row in rows
    }
    fig, axes = plt.subplots(3, 2, figsize=(9.2, 8.0), constrained_layout=True)
    x = np.arange(len(PROFILES))
    for scenario_index, (scenario, scenario_label) in enumerate(SCENARIOS):
        for method, label, color, marker in ROBUSTNESS_METHODS:
            selected = [
                indexed[(scenario, profile, method)]
                for profile, _profile_label in PROFILES
            ]
            success_mean = [100.0 * float(row["success_rate_mean"]) for row in selected]
            success_std = [100.0 * float(row["success_rate_std"]) for row in selected]
            time_mean = [
                float(row["penalized_average_completion_time_mean"])
                for row in selected
            ]
            time_std = [
                float(row["penalized_average_completion_time_std"])
                for row in selected
            ]
            axes[scenario_index, 0].errorbar(
                x,
                success_mean,
                yerr=success_std,
                color=color,
                marker=marker,
                linewidth=1.3,
                markersize=4.5,
                capsize=2.0,
                label=label if scenario_index == 0 else None,
            )
            axes[scenario_index, 1].errorbar(
                x,
                time_mean,
                yerr=time_std,
                color=color,
                marker=marker,
                linewidth=1.3,
                markersize=4.5,
                capsize=2.0,
            )
        axes[scenario_index, 0].set_ylabel(f"{scenario_label}\nSuccess (%)")
        axes[scenario_index, 1].set_ylabel("Penalized time (s)")
        for axis in axes[scenario_index]:
            axis.set_xticks(x)
            axis.set_xticklabels([label for _name, label in PROFILES])
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
    _save(fig, output)


def safety_module_figure(
    rows: list[dict[str, str]],
    output: Path,
) -> None:
    metrics = (
        ("intervention_time_ratio", "Total"),
        ("lidar_intervention_ratio", "LiDAR"),
        ("robot_robot_filter_intervention_ratio", "RR"),
        ("robot_pedestrian_filter_intervention_ratio", "RP"),
        ("stale_sensor_stop_ratio", "Stale"),
    )
    indexed = {(row["scenario"], row["algorithm"]): row for row in rows}
    values = []
    labels = []
    for scenario, scenario_label in SCENARIOS:
        for method, method_label, _color, _marker in METHODS:
            row = indexed[(scenario, method)]
            values.append(
                [
                    np.nan
                    if scenario == "static_clutter" and metric == "robot_pedestrian_filter_intervention_ratio"
                    else float(row[f"{metric}_mean"])
                    for metric, _label in metrics
                ]
            )
            labels.append(f"{scenario_label} | {method_label}")
    matrix = np.asarray(values, dtype=float)
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#F2F2F2")
    fig, axis = plt.subplots(figsize=(8.3, 6.2), constrained_layout=True)
    image = axis.imshow(masked, aspect="auto", cmap=cmap, vmin=0.0)
    axis.set_xticks(range(len(metrics)))
    axis.set_xticklabels([label for _metric, label in metrics])
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels, fontsize=8)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            text = "--" if np.isnan(value) else f"{value:.3f}"
            axis.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=7,
                color=(
                    "white"
                    if np.isfinite(value) and value > 0.55 * np.nanmax(matrix)
                    else "black"
                ),
            )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    colorbar.set_label("Intervention ratio")
    _save(fig, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--robustness-summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scenario-root",
        type=Path,
        default=DEFAULT_SCENARIO_ROOT,
    )
    args = parser.parse_args()
    main_distribution_figure(
        _read(args.episodes),
        _read(args.seeds),
        _scenario_horizons(args.scenario_root),
        args.output,
    )
    if args.robustness_summary is not None:
        sensor_robustness_figure(
            _read(args.robustness_summary),
            args.output.with_name("sensor_robustness_profiles.pdf"),
        )
    if args.summary is not None:
        safety_module_figure(
            _read(args.summary),
            args.output.with_name("safety_module_interventions.pdf"),
        )
    print(f"Generated Sensors figures in {args.output.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
