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

import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
import random


@dataclass
class PedestrianConfig:
    name: str
    start_x: float
    start_y: float
    yaw: float
    speed: float
    direction: str
    phase_offset_sec: float = 0.0


PED_CONFIGS = {
    "pedestrian_dynamic": [
        PedestrianConfig("ped_1", -5.0, 3.0, 0.0, 0.5, "east"),
        PedestrianConfig("ped_2", 5.0, -3.0, 3.14, 0.4, "west"),
        PedestrianConfig("ped_3", -3.0, -3.0, 1.57, 0.3, "north"),
        PedestrianConfig("ped_4", 3.0, 3.0, -1.57, 0.3, "south"),
    ],
    "mixed_complex": [
        PedestrianConfig("ped_1", -4.5, 4.4, 0.0, 0.08, "east"),
        PedestrianConfig("ped_2", 4.5, -4.4, 3.14, 0.2, "west"),
        PedestrianConfig("ped_3", -3.0, -4.5, 1.57, 0.25, "north"),
    ],
}


def get_pedestrian_configs(world_name: str) -> list[PedestrianConfig]:
    return [replace(config) for config in PED_CONFIGS.get(world_name, [])]


def randomized_pedestrian_configs(
    world_name: str,
    seed: int,
    phase_jitter_sec: float = 10.0,
    speed_jitter_fraction: float = 0.10,
) -> list[PedestrianConfig]:
    if phase_jitter_sec < 0.0:
        raise ValueError("phase_jitter_sec must be non-negative")
    if not 0.0 <= speed_jitter_fraction < 1.0:
        raise ValueError("speed_jitter_fraction must be in [0, 1)")
    rng = random.Random(seed)
    randomized = []
    for config in get_pedestrian_configs(world_name):
        speed_scale = rng.uniform(
            1.0 - speed_jitter_fraction,
            1.0 + speed_jitter_fraction,
        )
        randomized.append(
            replace(
                config,
                speed=config.speed * speed_scale,
                phase_offset_sec=rng.uniform(0.0, phase_jitter_sec),
            )
        )
    return randomized


def pedestrian_pose_at(
    cfg: PedestrianConfig,
    elapsed_time: float,
    duration: float = 20.0,
) -> tuple[float, float, float, float, float]:
    loop_time = duration * 2.0
    t = (elapsed_time + cfg.phase_offset_sec) % loop_time
    forward = t <= duration
    if forward:
        dist = cfg.speed * t
        direction = cfg.direction
        x, y, yaw = _pose_from_direction(cfg.start_x, cfg.start_y, dist, direction)
    else:
        dist = cfg.speed * (loop_time - t)
        direction = _opposite_direction(cfg.direction)
        x, y, _forward_yaw = _pose_from_direction(
            cfg.start_x, cfg.start_y, dist, cfg.direction
        )
        yaw = _yaw_from_direction(direction)

    vx, vy = _velocity_from_direction(cfg.speed, direction)
    return x, y, yaw, vx, vy


def _opposite_direction(direction: str) -> str:
    opposites = {
        "east": "west",
        "west": "east",
        "north": "south",
        "south": "north",
    }
    return opposites.get(direction, "west")


def _pose_from_direction(
    start_x: float,
    start_y: float,
    dist: float,
    direction: str,
) -> tuple[float, float, float]:
    if direction == "east":
        return start_x + dist, start_y, 0.0
    if direction == "west":
        return start_x - dist, start_y, 3.14
    if direction == "north":
        return start_x, start_y + dist, 1.57
    if direction == "south":
        return start_x, start_y - dist, -1.57
    return start_x + dist, start_y, 0.0


def _yaw_from_direction(direction: str) -> float:
    if direction == "east":
        return 0.0
    if direction == "west":
        return 3.14
    if direction == "north":
        return 1.57
    if direction == "south":
        return -1.57
    return 0.0


def _velocity_from_direction(speed: float, direction: str) -> tuple[float, float]:
    if direction == "east":
        return speed, 0.0
    if direction == "west":
        return -speed, 0.0
    if direction == "north":
        return 0.0, speed
    if direction == "south":
        return 0.0, -speed
    return speed, 0.0


def _generate_waypoints(cfg: PedestrianConfig, duration: float = 20.0) -> str:
    waypoints = []
    n_points = 41
    loop_time = duration * 2.0
    for i in range(n_points):
        t = i * loop_time / (n_points - 1)
        x, y, yaw, _vx, _vy = pedestrian_pose_at(cfg, t, duration=duration)
        waypoints.append(
            f"        <waypoint>\n"
            f"          <time>{t:.1f}</time>\n"
            f"          <pose>{x:.2f} {y:.2f} 1.0 0 0 {yaw:.1f}</pose>\n"
            f"        </waypoint>"
        )
    return "\n".join(waypoints)


def spawn_pedestrians(
    world_name: str,
    sdf_template: Path,
    configs: list[PedestrianConfig] | None = None,
) -> list[str]:
    configs = configs if configs is not None else get_pedestrian_configs(world_name)
    names = []
    for cfg in configs:
        content = sdf_template.read_text(encoding="utf-8")
        content = content.replace("PLACEHOLDER_NAME", cfg.name)
        content = content.replace(
            "PLACEHOLDER_POSE",
            "0 0 0 0 0 0",
        )
        content = content.replace("PLACEHOLDER_WALK_DAE", "walk.dae")
        content = content.replace(
            "PLACEHOLDER_WAYPOINTS",
            _generate_waypoints(cfg),
        )
        tmp = sdf_template.parent / f"_actor_{cfg.name}.sdf"
        tmp.write_text(content, encoding="utf-8")
        cmd = [
            "ros2", "run", "ros_gz_sim", "create",
            "-name", cfg.name,
            "-file", str(tmp),
        ]
        subprocess.run(cmd, check=True)
        tmp.unlink(missing_ok=True)
        names.append(cfg.name)
    return names


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Spawn animated pedestrians in Gazebo."
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--sdf", required=True)
    args = parser.parse_args()

    names = spawn_pedestrians(args.scenario, Path(args.sdf))
    print(f"Spawned {len(names)} pedestrians: {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
