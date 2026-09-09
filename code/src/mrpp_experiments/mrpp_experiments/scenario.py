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

from dataclasses import dataclass
from pathlib import Path
import random

from mrpp_experiments.simple_yaml import parse_inline_mapping, parse_scalar


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class RobotTask:
    name: str
    start: Pose2D
    goal: Pose2D


@dataclass(frozen=True)
class Scenario:
    name: str
    world: str
    robots: tuple[RobotTask, ...]
    max_steps: int
    dt: float


@dataclass(frozen=True)
class EvaluationRandomizationConfig:
    profile: str = "randomized_v1"
    enabled: bool = False
    start_position_jitter_m: float = 0.05
    start_yaw_jitter_rad: float = 0.15

    def __post_init__(self) -> None:
        if self.start_position_jitter_m < 0.0:
            raise ValueError("start_position_jitter_m must be non-negative")
        if self.start_yaw_jitter_rad < 0.0:
            raise ValueError("start_yaw_jitter_rad must be non-negative")


def _pose(data: dict[str, float]) -> Pose2D:
    return Pose2D(x=float(data["x"]), y=float(data["y"]), yaw=float(data["yaw"]))


def load_scenario(path: Path) -> Scenario:
    header: dict[str, str | int | float] = {}
    robots: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line == "robots:":
            continue
        if line.startswith("- "):
            if current is not None:
                robots.append(current)
            current = {}
            line = line[2:].strip()
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if current is None:
            header[key] = parse_scalar(value)
        elif value.startswith("{"):
            current[key] = parse_inline_mapping(value)
        else:
            current[key] = parse_scalar(value)
    if current is not None:
        robots.append(current)

    tasks = tuple(
        RobotTask(
            name=str(item["name"]),
            start=_pose(item["start"]),  # type: ignore[arg-type]
            goal=_pose(item["goal"]),  # type: ignore[arg-type]
        )
        for item in robots
    )
    return Scenario(
        name=str(header["name"]),
        world=str(header["world"]),
        robots=tasks,
        max_steps=int(header["max_steps"]),
        dt=float(header["dt"]),
    )


def randomize_scenario(
    scenario: Scenario,
    config: EvaluationRandomizationConfig,
    seed: int,
) -> Scenario:
    if not config.enabled:
        return scenario
    rng = random.Random(seed)
    robots = []
    for task in scenario.robots:
        start = Pose2D(
            x=task.start.x + rng.uniform(
                -config.start_position_jitter_m,
                config.start_position_jitter_m,
            ),
            y=task.start.y + rng.uniform(
                -config.start_position_jitter_m,
                config.start_position_jitter_m,
            ),
            yaw=task.start.yaw + rng.uniform(
                -config.start_yaw_jitter_rad,
                config.start_yaw_jitter_rad,
            ),
        )
        robots.append(RobotTask(name=task.name, start=start, goal=task.goal))
    return Scenario(
        name=scenario.name,
        world=scenario.world,
        robots=tuple(robots),
        max_steps=scenario.max_steps,
        dt=scenario.dt,
    )
