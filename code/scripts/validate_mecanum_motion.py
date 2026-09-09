#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory

from mrpp_rl.gazebo_bridge import (
    GazeboBridge,
    ensure_processes_running,
    launch_gazebo,
    launch_topic_bridges,
    spawn_robots,
)


@dataclass(frozen=True)
class MotionCase:
    name: str
    command: tuple[float, float, float]
    duration: float
    min_forward: float = 0.0
    min_lateral: float = 0.0
    min_yaw: float = 0.0


CASES = (
    MotionCase("forward", (0.25, 0.0, 0.0), 1.8, min_forward=0.25),
    MotionCase("backward", (-0.25, 0.0, 0.0), 1.8, min_forward=0.25),
    MotionCase("strafe_left", (0.0, 0.22, 0.0), 1.8, min_lateral=0.22),
    MotionCase("strafe_right", (0.0, -0.22, 0.0), 1.8, min_lateral=0.22),
    MotionCase("diagonal", (0.18, 0.18, 0.0), 1.8, min_forward=0.18, min_lateral=0.18),
    MotionCase("rotate", (0.0, 0.0, 0.8), 1.8, min_yaw=0.8),
    MotionCase("combined", (0.16, 0.12, 0.5), 1.8, min_forward=0.14, min_lateral=0.10, min_yaw=0.45),
)


def _kill_old_processes() -> None:
    for pattern in ("ros_gz_bridge", "parameter_bridge", "ign gazebo"):
        subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)


def _configure_gazebo_resources() -> None:
    gazebo_pkg = Path(get_package_share_directory("mrpp_gazebo"))
    model_path = str(gazebo_pkg / "models")
    existing = os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "")
    paths = [p for p in existing.split(":") if p]
    if model_path not in paths:
        paths.insert(0, model_path)
    os.environ["IGN_GAZEBO_RESOURCE_PATH"] = ":".join(paths)


def _write_single_robot_config(path: Path) -> None:
    path.write_text(
        "robots:\n"
        "  - name: robot_1\n"
        "    x: 0.0\n"
        "    y: -5.0\n"
        "    yaw: 0.0\n",
        encoding="utf-8",
    )


def _angle_delta(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def _run_case(
    bridge: GazeboBridge,
    case: MotionCase,
    start_pose: dict[str, tuple[float, float, float]],
) -> dict[str, float | str | bool]:
    bridge.reset(start_pose)
    before = bridge.get_robot_states()["robot_1"]
    start_time = time.time()
    while time.time() - start_time < case.duration:
        bridge.send_commands({"robot_1": case.command})
        bridge.spin_once(timeout_sec=0.05)
        time.sleep(0.05)
    bridge.stop_all()
    for _ in range(10):
        bridge.spin_once(timeout_sec=0.05)
    after = bridge.get_robot_states()["robot_1"]

    dx = after.x - before.x
    dy = after.y - before.y
    dyaw = _angle_delta(after.yaw, before.yaw)
    forward = dx if case.command[0] >= 0.0 else -dx
    lateral = dy if case.command[1] >= 0.0 else -dy
    yaw_motion = abs(dyaw)
    tolerance = 1e-3
    passed = True
    if case.min_forward > 0.0:
        passed = passed and forward + tolerance >= case.min_forward
    if case.min_lateral > 0.0:
        passed = passed and lateral + tolerance >= case.min_lateral
    if case.min_yaw > 0.0:
        passed = passed and yaw_motion + tolerance >= case.min_yaw
    return {
        "case": case.name,
        "cmd_vx": case.command[0],
        "cmd_vy": case.command[1],
        "cmd_omega": case.command[2],
        "duration_s": case.duration,
        "start_x": before.x,
        "start_y": before.y,
        "start_yaw": before.yaw,
        "end_x": after.x,
        "end_y": after.y,
        "end_yaw": after.yaw,
        "delta_x": dx,
        "delta_y": dy,
        "delta_yaw": dyaw,
        "passed": passed,
    }


def _write_csv(path: Path, rows: list[dict[str, float | str | bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "cmd_vx",
        "cmd_vy",
        "cmd_omega",
        "duration_s",
        "start_x",
        "start_y",
        "start_yaw",
        "end_x",
        "end_y",
        "end_yaw",
        "delta_x",
        "delta_y",
        "delta_yaw",
        "passed",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate formal mecanum vx/vy/omega motion in Gazebo."
    )
    parser.add_argument(
        "--output",
        default="/tmp/mrpp_mecanum_motion_validation.csv",
    )
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    _configure_gazebo_resources()
    _kill_old_processes()
    time.sleep(2.0)

    gazebo_pkg = Path(get_package_share_directory("mrpp_gazebo"))
    world_file = gazebo_pkg / "worlds/static_clutter.sdf"
    robot_model = gazebo_pkg / "models/mecanum_robot/model.sdf"
    robot_config = Path("/tmp/mrpp_mecanum_validation_robot.yaml")
    _write_single_robot_config(robot_config)

    gazebo_proc = None
    bridge_procs: list[subprocess.Popen] = []
    bridge = None
    rclpy_started = False
    try:
        gazebo_proc = launch_gazebo(world_file, gui=args.gui)
        time.sleep(8.0)
        ensure_processes_running([gazebo_proc], "Gazebo")

        spawn_proc = spawn_robots(robot_config, robot_model, delay=0.0)
        spawn_code = spawn_proc.wait(timeout=30)
        if spawn_code != 0:
            raise RuntimeError(f"Robot spawning failed: {spawn_code}")
        time.sleep(2.0)

        bridge_procs = launch_topic_bridges(
            ["robot_1"], [], world_name="static_clutter"
        )
        time.sleep(5.0)
        ensure_processes_running(bridge_procs, "ros_gz_bridge")

        rclpy.init()
        rclpy_started = True
        bridge = GazeboBridge(["robot_1"], [], world_name="static_clutter")
        bridge.wait_until_ready(timeout_sec=30.0)
        start_pose = {"robot_1": (0.0, -5.0, 0.0)}
        bridge.set_start_positions(start_pose)

        rows = [_run_case(bridge, case, start_pose) for case in CASES]
        _write_csv(Path(args.output), rows)
        failed = [row["case"] for row in rows if not row["passed"]]
        for row in rows:
            print(
                f"{row['case']}: dx={row['delta_x']:.3f} "
                f"dy={row['delta_y']:.3f} dyaw={row['delta_yaw']:.3f} "
                f"passed={row['passed']}",
                flush=True,
            )
        print(f"CSV saved to {args.output}", flush=True)
        if failed:
            raise RuntimeError(f"Mecanum validation failed: {failed}")
        return 0
    finally:
        if bridge is not None:
            try:
                bridge.stop_all()
                bridge.destroy_node()
            except Exception:
                pass
        if rclpy_started and rclpy.ok():
            rclpy.shutdown()
        for proc in bridge_procs:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        if gazebo_proc is not None:
            gazebo_proc.terminate()
            try:
                gazebo_proc.wait(timeout=10)
            except Exception:
                gazebo_proc.kill()
        _kill_old_processes()


if __name__ == "__main__":
    raise SystemExit(main())
