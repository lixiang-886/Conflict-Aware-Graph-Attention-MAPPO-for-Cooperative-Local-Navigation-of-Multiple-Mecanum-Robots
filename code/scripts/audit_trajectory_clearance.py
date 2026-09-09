#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def audit_trajectory_csv(
    trajectory_csv: Path,
    scenario_name: str | None = None,
    robot_robot_near_miss_threshold: float = 0.55,
    robot_pedestrian_near_miss_threshold: float = 0.7,
) -> dict[str, Any]:
    rows = _read_rows(trajectory_csv)
    if not rows:
        raise ValueError(f"empty trajectory CSV: {trajectory_csv}")

    scenario = scenario_name or str(rows[0].get("scenario", ""))
    by_robot: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    by_time: dict[float, list[tuple[str, float, float]]] = defaultdict(list)
    for row in rows:
        name = str(row["robot_name"])
        t = round(float(row["t"]), 6)
        x = float(row["x"])
        y = float(row["y"])
        by_robot[name].append((t, x, y))
        by_time[t].append((name, x, y))

    segment_speeds: list[float] = []
    path_lengths: list[float] = []
    net_displacements: list[float] = []
    durations: list[float] = []
    for points in by_robot.values():
        points.sort()
        length = 0.0
        for prev, curr in zip(points, points[1:]):
            dt = curr[0] - prev[0]
            dist = math.hypot(curr[1] - prev[1], curr[2] - prev[2])
            length += dist
            if dt > 1e-9:
                segment_speeds.append(dist / dt)
        path_lengths.append(length)
        if points:
            durations.append(points[-1][0] - points[0][0])
            net_displacements.append(
                math.hypot(points[-1][1] - points[0][1], points[-1][2] - points[0][2])
            )

    min_robot_robot = math.inf
    robot_robot_near_misses = 0
    for points in by_time.values():
        for i, (_name, x, y) in enumerate(points):
            for _other_name, other_x, other_y in points[i + 1:]:
                distance = math.hypot(x - other_x, y - other_y)
                min_robot_robot = min(min_robot_robot, distance)
                if distance < robot_robot_near_miss_threshold:
                    robot_robot_near_misses += 1

    min_robot_pedestrian = math.inf
    robot_pedestrian_near_misses = 0
    pedestrians = _load_pedestrian_configs(scenario)
    if pedestrians:
        for t, points in by_time.items():
            for ped in pedestrians:
                ped_x, ped_y, _yaw, _vx, _vy = ped.pose_at(t)
                for _robot_name, x, y in points:
                    distance = math.hypot(x - ped_x, y - ped_y)
                    min_robot_pedestrian = min(min_robot_pedestrian, distance)
                    if distance < robot_pedestrian_near_miss_threshold:
                        robot_pedestrian_near_misses += 1

    return {
        "trajectory": str(trajectory_csv),
        "scenario": scenario,
        "algorithm": str(rows[0].get("algorithm", "")),
        "seed": _parse_int(rows[0].get("seed", "")),
        "robot_count": len(by_robot),
        "sample_count": len(rows),
        "timestep_count": len(by_time),
        "duration_s": _mean(durations),
        "mean_segment_speed_mps": _mean(segment_speeds),
        "p95_segment_speed_mps": _percentile(segment_speeds, 0.95),
        "max_segment_speed_mps": max(segment_speeds, default=0.0),
        "average_path_length_m": _mean(path_lengths),
        "average_net_displacement_m": _mean(net_displacements),
        "average_path_efficiency": _safe_ratio(
            _mean(net_displacements),
            _mean(path_lengths),
        ),
        "min_robot_robot_distance_m": _finite_or_zero(min_robot_robot),
        "robot_robot_near_miss_count": robot_robot_near_misses,
        "min_robot_pedestrian_distance_m": _finite_or_zero(min_robot_pedestrian),
        "robot_pedestrian_near_miss_count": robot_pedestrian_near_misses,
        "robot_robot_near_miss_threshold_m": robot_robot_near_miss_threshold,
        "robot_pedestrian_near_miss_threshold_m": robot_pedestrian_near_miss_threshold,
    }


class _PedestrianWrapper:
    def __init__(self, cfg: object, pose_at_func: object) -> None:
        self._cfg = cfg
        self.name = str(getattr(cfg, "name", "pedestrian"))
        self._pose_at_func = pose_at_func

    def pose_at(self, elapsed_time: float) -> tuple[float, float, float, float, float]:
        return self._pose_at_func(self._cfg, elapsed_time)


def _load_pedestrian_configs(scenario: str) -> list[_PedestrianWrapper]:
    if not scenario:
        return []
    _add_repo_python_paths()
    try:
        from mrpp_gazebo.pedestrian_controller import (
            get_pedestrian_configs,
            pedestrian_pose_at,
        )
    except ModuleNotFoundError:
        return []
    return [
        _PedestrianWrapper(cfg, pedestrian_pose_at)
        for cfg in get_pedestrian_configs(scenario)
    ]


def _add_repo_python_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for package in (
        "mrpp_gazebo",
        "mrpp_experiments",
        "mrpp_rl",
        "mrpp_navigation",
    ):
        path = str(repo_root / "src" / package)
        if path not in sys.path:
            sys.path.insert(0, path)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return ordered[max(0, min(index, len(ordered) - 1))]


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 1e-9 else 0.0


def _finite_or_zero(value: float) -> float:
    return 0.0 if value == math.inf else value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit robot trajectory speed, path efficiency, and clearances."
    )
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--scenario")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--robot-robot-threshold", type=float, default=0.55)
    parser.add_argument("--robot-pedestrian-threshold", type=float, default=0.7)
    args = parser.parse_args()

    summary = audit_trajectory_csv(
        args.trajectory,
        scenario_name=args.scenario,
        robot_robot_near_miss_threshold=args.robot_robot_threshold,
        robot_pedestrian_near_miss_threshold=args.robot_pedestrian_threshold,
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
