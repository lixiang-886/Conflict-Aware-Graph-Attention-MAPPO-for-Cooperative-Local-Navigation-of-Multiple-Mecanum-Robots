# Copyright 2026 lixiang-886
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import csv
import os
import warnings
from collections import defaultdict
from pathlib import Path

from mrpp_experiments.scenario import Scenario, load_scenario


TrajectoryPoint = dict[str, float | str | int]
PedestrianPath = dict[str, list[tuple[float, float]]]


def read_trajectory_csv(path: Path) -> dict[str, list[TrajectoryPoint]]:
    grouped: dict[str, list[TrajectoryPoint]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            robot_name = row["robot_name"]
            grouped[robot_name].append(
                {
                    "t": float(row["t"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "v": float(row["v"]),
                    "omega": float(row["omega"]),
                    "success": int(float(row.get("success", "0") or 0)),
                }
            )
    for points in grouped.values():
        points.sort(key=lambda item: float(item["t"]))
    return dict(sorted(grouped.items()))


def plot_trajectories(
    trajectory_csv: Path,
    scenario_path: Path,
    output_path: Path,
    world_path: Path | None = None,
    title: str | None = None,
    legend: str = "inside",
    style: str = "default",
) -> None:
    _configure_plot_cache()
    try:
        warnings.filterwarnings(
            "ignore",
            message="Unable to import Axes3D.*",
            category=UserWarning,
        )
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import patches
        from matplotlib.transforms import Affine2D
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to generate trajectory figures"
        ) from exc

    scenario = load_scenario(scenario_path)
    trajectories = read_trajectory_csv(trajectory_csv)
    pedestrian_paths = _pedestrian_paths_for_world(scenario.world)
    resolved_world_path = world_path or _default_world_path(scenario)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if style == "paper":
        plt.rcParams.update(
            {
                "font.size": 9,
                "axes.titlesize": 10,
                "axes.labelsize": 9,
                "legend.fontsize": 8,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
            }
        )
        fig, ax = plt.subplots(figsize=(5.8, 5.2), dpi=300)
        colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    else:
        fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=180)
        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    _draw_obstacles(ax, resolved_world_path, patches, Affine2D)
    _draw_pedestrian_paths(ax, pedestrian_paths, style=style)
    _draw_starts_and_goals(ax, scenario, colors)

    for index, (robot_name, points) in enumerate(trajectories.items()):
        if not points:
            continue
        color = colors[index % len(colors)] if colors else None
        xs = [float(point["x"]) for point in points]
        ys = [float(point["y"]) for point in points]
        success = bool(points[-1].get("success", 0))
        linestyle = "-" if success else "--"
        if style == "paper":
            label = f"R{index + 1}"
            linewidth = 2.2
        else:
            label = f"{robot_name} {'success' if success else 'unfinished'}"
            linewidth = 2.0
        ax.plot(xs, ys, linestyle=linestyle, linewidth=linewidth, color=color, label=label)
        ax.scatter(
            xs[-1],
            ys[-1],
            marker="o",
            s=22 if style == "paper" else 24,
            color=color,
            edgecolors="white",
            linewidths=0.7,
            zorder=9,
        )

    if style == "paper":
        if title:
            ax.set_title(title)
    else:
        ax.set_title(title or f"{scenario.name} trajectories")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    _draw_legend(ax, legend)
    _set_axis_limits(ax, scenario, trajectories, pedestrian_paths)
    fig.tight_layout()
    save_kwargs = {"bbox_inches": "tight"} if legend == "outside" else {}
    fig.savefig(output_path, **save_kwargs)
    plt.close(fig)


def _configure_plot_cache() -> None:
    for env_name, default_path in (
        ("MPLCONFIGDIR", "/tmp/mrpp_matplotlib"),
        ("XDG_CACHE_HOME", "/tmp/mrpp_plot_cache"),
    ):
        path = Path(os.environ.setdefault(env_name, default_path))
        path.mkdir(parents=True, exist_ok=True)


def _default_world_path(scenario: Scenario) -> Path | None:
    world_file = f"{scenario.world}.sdf"
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        candidate = parent / "src" / "mrpp_gazebo" / "worlds" / world_file
        if candidate.exists():
            return candidate
    try:
        from ament_index_python.packages import get_package_share_directory
    except Exception:
        return None
    candidate = Path(get_package_share_directory("mrpp_gazebo")) / "worlds" / world_file
    return candidate if candidate.exists() else None


def _draw_starts_and_goals(ax, scenario: Scenario, colors: list[str]) -> None:
    for index, task in enumerate(scenario.robots):
        color = colors[index % len(colors)] if colors else None
        ax.scatter(
            task.start.x,
            task.start.y,
            marker="o",
            s=72,
            color=color,
            edgecolors="black",
            linewidths=1.0,
            zorder=7,
        )
        ax.scatter(
            task.goal.x,
            task.goal.y,
            marker="*",
            s=140,
            color=color,
            edgecolors="black",
            linewidths=1.0,
            zorder=8,
        )


def _draw_obstacles(ax, world_path, patches, affine_cls) -> None:
    if world_path is None:
        return
    try:
        from mrpp_rl.waypoints import parse_static_obstacles
    except Exception:
        return
    for obstacle in parse_static_obstacles(world_path):
        if hasattr(obstacle, "sx") and hasattr(obstacle, "sy"):
            rect = patches.Rectangle(
                (
                    float(obstacle.x) - float(obstacle.sx) / 2.0,
                    float(obstacle.y) - float(obstacle.sy) / 2.0,
                ),
                float(obstacle.sx),
                float(obstacle.sy),
                facecolor="0.72",
                edgecolor="0.30",
                linewidth=0.8,
                alpha=0.75,
                zorder=1,
            )
            transform = (
                affine_cls()
                .rotate_around(float(obstacle.x), float(obstacle.y), float(obstacle.yaw))
                + ax.transData
            )
            rect.set_transform(transform)
            ax.add_patch(rect)
        elif hasattr(obstacle, "radius"):
            circle = patches.Circle(
                (float(obstacle.x), float(obstacle.y)),
                float(obstacle.radius),
                facecolor="0.72",
                edgecolor="0.30",
                linewidth=0.8,
                alpha=0.75,
                zorder=1,
            )
            ax.add_patch(circle)


def _pedestrian_paths_for_world(world_name: str) -> PedestrianPath:
    try:
        from mrpp_gazebo.pedestrian_controller import (
            get_pedestrian_configs,
            pedestrian_pose_at,
        )
    except Exception:
        return {}

    paths: PedestrianPath = {}
    duration = 40.0
    sample_count = 81
    for cfg in get_pedestrian_configs(world_name):
        points: list[tuple[float, float]] = []
        for index in range(sample_count):
            t = duration * index / (sample_count - 1)
            x, y, _yaw, _vx, _vy = pedestrian_pose_at(cfg, t)
            points.append((x, y))
        paths[cfg.name] = points
    return paths


def _draw_pedestrian_paths(
    ax,
    pedestrian_paths: PedestrianPath,
    style: str = "default",
) -> None:
    if not pedestrian_paths:
        return
    color = "#8c2d8f"
    for index, (name, points) in enumerate(pedestrian_paths.items()):
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        label = "Pedestrian paths" if style == "paper" and index == 0 else None
        if style != "paper" and index == 0:
            label = "pedestrian path"
        ax.plot(
            xs,
            ys,
            linestyle=":",
            linewidth=1.8 if style == "paper" else 2.0,
            color=color,
            alpha=0.9,
            label=label,
            zorder=3,
        )
        ax.scatter(
            xs[0],
            ys[0],
            marker="s",
            s=40,
            color=color,
            edgecolors="black",
            zorder=6,
        )
        ax.scatter(
            xs[len(xs) // 2],
            ys[len(ys) // 2],
            marker="^",
            s=48,
            color=color,
            edgecolors="black",
            zorder=6,
        )
        if len(points) > 4:
            start = points[1]
            end = points[4]
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={
                    "arrowstyle": "->",
                    "color": color,
                    "linewidth": 1.6,
                    "alpha": 0.9,
                },
                zorder=7,
            )


def _draw_legend(ax, legend: str) -> None:
    if legend == "none":
        return
    if legend == "outside":
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            frameon=True,
            borderaxespad=0.0,
        )
        return
    ax.legend(loc="best", fontsize=8, frameon=True)


def _set_axis_limits(
    ax,
    scenario: Scenario,
    trajectories: dict[str, list[TrajectoryPoint]],
    pedestrian_paths: PedestrianPath | None = None,
) -> None:
    xs = [task.start.x for task in scenario.robots] + [
        task.goal.x for task in scenario.robots
    ]
    ys = [task.start.y for task in scenario.robots] + [
        task.goal.y for task in scenario.robots
    ]
    for points in trajectories.values():
        xs.extend(float(point["x"]) for point in points)
        ys.extend(float(point["y"]) for point in points)
    for points in (pedestrian_paths or {}).values():
        xs.extend(point[0] for point in points)
        ys.extend(point[1] for point in points)
    if not xs or not ys:
        return
    margin = 0.8
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot robot trajectories.")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--world", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--legend",
        choices=("inside", "outside", "none"),
        default="inside",
        help="Legend placement for generated figures.",
    )
    parser.add_argument(
        "--style",
        choices=("default", "paper"),
        default="default",
        help="Use paper for compact publication figures without code-style titles.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plot_trajectories(
        trajectory_csv=Path(args.trajectory),
        scenario_path=Path(args.scenario),
        output_path=Path(args.output),
        world_path=Path(args.world) if args.world else None,
        title=args.title,
        legend=args.legend,
        style=args.style,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
