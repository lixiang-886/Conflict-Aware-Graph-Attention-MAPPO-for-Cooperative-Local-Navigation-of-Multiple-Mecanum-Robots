#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import subprocess
import time
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import rclpy

from mrpp_experiments.scenario import load_scenario
from mrpp_gazebo.pedestrian_controller import (
    get_pedestrian_configs,
    spawn_pedestrians,
)
from mrpp_rl.gazebo_bridge import (
    GazeboBridge,
    build_robot_config_yaml,
    ensure_processes_running,
    launch_gazebo,
    launch_topic_bridges,
    spawn_robots,
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


def _wait_for_topics(required: set[str], timeout_sec: float = 20.0) -> set[str]:
    topics: set[str] = set()
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        output = subprocess.check_output(
            ["ros2", "topic", "list"], text=True, timeout=20
        )
        topics = {line.strip() for line in output.splitlines() if line.strip()}
        if required.issubset(topics):
            break
        time.sleep(1.0)
    missing = sorted(required - topics)
    if missing:
        raise RuntimeError(f"Missing ROS topics: {missing}")
    return topics


def _start_positions(scenario_path: Path) -> dict[str, tuple[float, float, float]]:
    scenario = load_scenario(scenario_path)
    return {
        task.name: (task.start.x, task.start.y, task.start.yaw)
        for task in scenario.robots
    }


def _cleanup(
    bridge: GazeboBridge | None,
    bridge_procs: list[subprocess.Popen],
    gazebo_proc: subprocess.Popen | None,
    rclpy_started: bool,
) -> None:
    if bridge is not None:
        try:
            bridge.stop_all()
            bridge.destroy_node()
        except Exception as exc:
            print(f"cleanup bridge warning: {exc}", flush=True)
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


def run_single_robot_motion(
    root: Path,
    move_seconds: float,
    gui: bool = False,
    hold_seconds: float = 0.0,
) -> None:
    scenario_path = root / "src/mrpp_experiments/config/scenarios/static_clutter.yaml"
    tmp_config = Path("/tmp/mrpp_live_smoke_robots.yaml")
    robot_names = build_robot_config_yaml(scenario_path, tmp_config)
    start_positions = _start_positions(scenario_path)

    gazebo_pkg = Path(get_package_share_directory("mrpp_gazebo"))
    world_file = gazebo_pkg / "worlds/static_clutter.sdf"
    robot_model = gazebo_pkg / "models/mecanum_robot/model.sdf"

    _kill_old_processes()
    time.sleep(2.0)

    gazebo_proc = None
    bridge_procs: list[subprocess.Popen] = []
    bridge = None
    rclpy_started = False
    try:
        print(f"SMOKE world={world_file}", flush=True)
        gazebo_proc = launch_gazebo(world_file, gui=gui)
        time.sleep(8.0)
        ensure_processes_running([gazebo_proc], "Gazebo")

        print("SMOKE spawning robots", flush=True)
        spawn_proc = spawn_robots(tmp_config, robot_model, delay=0.0)
        spawn_code = spawn_proc.wait(timeout=30)
        if spawn_code != 0:
            raise RuntimeError(f"Robot spawning failed: {spawn_code}")
        time.sleep(2.0)

        print("SMOKE launching bridges", flush=True)
        bridge_procs = launch_topic_bridges(
            robot_names, [], world_name="static_clutter"
        )
        time.sleep(5.0)
        ensure_processes_running(bridge_procs, "ros_gz_bridge")

        required = set()
        for name in robot_names:
            required.update(
                {f"/{name}/cmd_vel", f"/{name}/odom", f"/{name}/scan"}
            )
        topics = _wait_for_topics(required)
        forbidden = sorted(t for t in topics if t in {"/cmd_vel", "/odom", "/scan"})
        if forbidden:
            raise RuntimeError(f"Unexpected global topics: {forbidden}")
        print("SMOKE topics isolated OK", flush=True)

        rclpy.init()
        rclpy_started = True
        bridge = GazeboBridge(robot_names, [], world_name="static_clutter")
        bridge.wait_until_ready(timeout_sec=30.0)
        bridge.set_start_positions(start_positions)

        before = bridge.get_robot_states()
        before_xy = {name: (state.x, state.y) for name, state in before.items()}
        print(f"SMOKE before={before_xy}", flush=True)

        forward_seconds = max(move_seconds * 0.5, 0.5)
        lateral_seconds = max(move_seconds - forward_seconds, 0.5)
        end_time = time.time() + forward_seconds
        while time.time() < end_time:
            bridge.send_commands(
                {
                    name: (0.25, 0.0, 0.0)
                    if name == "robot_1" else (0.0, 0.0, 0.0)
                    for name in robot_names
                }
            )
            rclpy.spin_once(bridge, timeout_sec=0.05)
            time.sleep(0.05)
        for _ in range(10):
            rclpy.spin_once(bridge, timeout_sec=0.05)
        mid = bridge.get_robot_states()
        mid_xy = {name: (state.x, state.y) for name, state in mid.items()}

        end_time = time.time() + lateral_seconds
        while time.time() < end_time:
            bridge.send_commands(
                {
                    name: (0.0, 0.20, 0.0)
                    if name == "robot_1" else (0.0, 0.0, 0.0)
                    for name in robot_names
                }
            )
            rclpy.spin_once(bridge, timeout_sec=0.05)
            time.sleep(0.05)
        bridge.stop_all()
        time.sleep(0.5)
        for _ in range(20):
            rclpy.spin_once(bridge, timeout_sec=0.05)

        after = bridge.get_robot_states()
        after_xy = {name: (state.x, state.y) for name, state in after.items()}
        deltas = {
            name: math.hypot(
                after_xy[name][0] - before_xy[name][0],
                after_xy[name][1] - before_xy[name][1],
            )
            for name in robot_names
        }
        lateral_delta = math.hypot(
            after_xy["robot_1"][0] - mid_xy["robot_1"][0],
            after_xy["robot_1"][1] - mid_xy["robot_1"][1],
        )
        lasers = bridge.get_laser_data()
        scans_seen = {name: len(lasers[name].ranges) > 0 for name in robot_names}
        print(f"SMOKE mid={mid_xy}", flush=True)
        print(f"SMOKE after={after_xy}", flush=True)
        print(f"SMOKE deltas={deltas}", flush=True)
        print(f"SMOKE lateral_delta={lateral_delta:.3f}", flush=True)
        print(f"SMOKE scans_seen={scans_seen}", flush=True)

        if deltas["robot_1"] <= 0.15:
            msg = f"robot_1 did not move enough: {deltas['robot_1']:.3f}m"
            raise RuntimeError(msg)
        if lateral_delta <= 0.08:
            raise RuntimeError(
                f"robot_1 did not strafe enough: {lateral_delta:.3f}m"
            )
        stationary = {
            name: delta
            for name, delta in deltas.items()
            if name != "robot_1" and delta >= 0.05
        }
        if stationary:
            raise RuntimeError(f"Non-commanded robots moved too much: {stationary}")
        if not all(scans_seen.values()):
            raise RuntimeError(f"Laser scans missing: {scans_seen}")
        print("SMOKE single_robot_motion PASS", flush=True)
        if hold_seconds > 0.0:
            print(f"SMOKE holding Gazebo GUI for {hold_seconds:.1f}s", flush=True)
            time.sleep(hold_seconds)
    finally:
        _cleanup(bridge, bridge_procs, gazebo_proc, rclpy_started)


def run_pedestrian_initial_state(
    root: Path,
    motion_seconds: float,
    gui: bool = False,
    hold_seconds: float = 0.0,
) -> None:
    scenario_path = (
        root / "src/mrpp_experiments/config/scenarios/pedestrian_dynamic.yaml"
    )
    tmp_config = Path("/tmp/mrpp_pedestrian_smoke_robots.yaml")
    robot_names = build_robot_config_yaml(scenario_path, tmp_config)
    start_positions = _start_positions(scenario_path)

    gazebo_pkg = Path(get_package_share_directory("mrpp_gazebo"))
    world_file = gazebo_pkg / "worlds/pedestrian_dynamic.sdf"
    robot_model = gazebo_pkg / "models/mecanum_robot/model.sdf"
    ped_sdf = gazebo_pkg / "models/pedestrian_fuel/model.sdf"

    _kill_old_processes()
    time.sleep(2.0)

    gazebo_proc = None
    bridge_procs: list[subprocess.Popen] = []
    bridge = None
    rclpy_started = False
    try:
        print(f"PED_SMOKE world={world_file}", flush=True)
        gazebo_proc = launch_gazebo(world_file, gui=gui)
        time.sleep(8.0)
        ensure_processes_running([gazebo_proc], "Gazebo")

        spawn_proc = spawn_robots(tmp_config, robot_model, delay=0.0)
        spawn_code = spawn_proc.wait(timeout=30)
        if spawn_code != 0:
            raise RuntimeError(f"Robot spawning failed: {spawn_code}")
        time.sleep(2.0)

        ped_configs = get_pedestrian_configs("pedestrian_dynamic")
        ped_names = spawn_pedestrians("pedestrian_dynamic", ped_sdf)
        if len(ped_names) != len(ped_configs):
            raise RuntimeError(
                f"Expected {len(ped_configs)} pedestrians, got {len(ped_names)}"
            )
        time.sleep(2.0)

        bridge_procs = launch_topic_bridges(
            robot_names, ped_names, world_name="pedestrian_dynamic"
        )
        time.sleep(5.0)
        ensure_processes_running(bridge_procs, "ros_gz_bridge")

        rclpy.init()
        rclpy_started = True
        bridge = GazeboBridge(
            robot_names, ped_names, world_name="pedestrian_dynamic"
        )
        bridge.wait_until_ready(timeout_sec=30.0)
        bridge.set_start_positions(start_positions)
        bridge.set_pedestrian_configs(ped_configs)
        bridge.update_pedestrian_fallbacks(0.0)

        robot_xy = {
            name: (state.x, state.y)
            for name, state in bridge.get_robot_states().items()
        }
        ped_before = {
            name: (state.x, state.y)
            for name, state in bridge.get_pedestrian_states().items()
        }
        initial_collisions = bridge.get_collisions()
        print(f"PED_SMOKE robots={robot_xy}", flush=True)
        print(f"PED_SMOKE pedestrians_initial={ped_before}", flush=True)
        print(f"PED_SMOKE initial_collisions={initial_collisions}", flush=True)
        if any(initial_collisions.values()):
            raise RuntimeError("Initial false collision detected")

        start_time = time.time()
        while time.time() - start_time < motion_seconds:
            elapsed = time.time() - start_time
            bridge.update_pedestrian_fallbacks(elapsed)
            rclpy.spin_once(bridge, timeout_sec=0.05)
            time.sleep(0.05)
        bridge.update_pedestrian_fallbacks(motion_seconds)

        ped_after = {
            name: (state.x, state.y)
            for name, state in bridge.get_pedestrian_states().items()
        }
        deltas = {
            name: math.hypot(
                ped_after[name][0] - ped_before[name][0],
                ped_after[name][1] - ped_before[name][1],
            )
            for name in ped_names
        }
        print(f"PED_SMOKE pedestrians_after={ped_after}", flush=True)
        print(f"PED_SMOKE pedestrian_deltas={deltas}", flush=True)
        if max(deltas.values()) < 0.2:
            raise RuntimeError("Pedestrian states did not move")
        print("PED_SMOKE initial_state_and_motion PASS", flush=True)
        if hold_seconds > 0.0:
            print(f"PED_SMOKE holding Gazebo GUI for {hold_seconds:.1f}s", flush=True)
            time.sleep(hold_seconds)
    finally:
        _cleanup(bridge, bridge_procs, gazebo_proc, rclpy_started)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live Gazebo smoke checks.")
    parser.add_argument(
        "--mode",
        choices=("all", "motion", "pedestrians"),
        default="all",
    )
    parser.add_argument("--move-seconds", type=float, default=3.0)
    parser.add_argument("--pedestrian-seconds", type=float, default=5.0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--hold-seconds", type=float, default=0.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    _configure_gazebo_resources()
    if args.mode in ("all", "motion"):
        run_single_robot_motion(
            root, args.move_seconds, gui=args.gui, hold_seconds=args.hold_seconds
        )
    if args.mode in ("all", "pedestrians"):
        run_pedestrian_initial_state(
            root,
            args.pedestrian_seconds,
            gui=args.gui,
            hold_seconds=args.hold_seconds,
        )
    print("Gazebo live smoke checks completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
