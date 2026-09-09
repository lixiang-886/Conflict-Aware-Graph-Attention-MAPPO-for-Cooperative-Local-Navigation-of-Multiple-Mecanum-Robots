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

from collections import deque
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import random
from typing import Mapping

from mrpp_rl.lidar import LaserObservation, laser_scan_to_observation


@dataclass(frozen=True)
class SensorPerturbationConfig:
    profile: str = "nominal"
    enabled: bool = False
    position_noise_std: float = 0.0
    velocity_noise_std: float = 0.0
    lidar_range_noise_std: float = 0.0
    latency_steps: int = 0
    dropout_probability: float = 0.0
    lidar_sample_dropout_probability: float = 0.0
    seed: int = 0
    perturb_robot_states: bool = True
    perturb_pedestrian_states: bool = True
    perturb_lidar: bool = True

    def __post_init__(self) -> None:
        for name in (
            "position_noise_std",
            "velocity_noise_std",
            "lidar_range_noise_std",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.latency_steps < 0:
            raise ValueError("latency_steps must be non-negative")
        for name in (
            "dropout_probability",
            "lidar_sample_dropout_probability",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class SensorPerturbationStats:
    steps: int = 0
    latency_warmup_steps: int = 0
    robot_state_dropouts: int = 0
    pedestrian_state_dropouts: int = 0
    lidar_scan_dropouts: int = 0
    lidar_sample_dropouts: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class SensorSnapshot:
    robot_states: Mapping[str, object]
    pedestrian_states: Mapping[str, object]
    laser_data: Mapping[str, LaserObservation]


class SensorPerturbationPipeline:
    """Apply deterministic evaluation-time perturbations to sensor snapshots."""

    def __init__(self, config: SensorPerturbationConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)
        self._history: deque[SensorSnapshot] = deque(
            maxlen=max(config.latency_steps + 1, 1)
        )
        self._last_robot_states: dict[str, object] = {}
        self._last_pedestrian_states: dict[str, object] = {}
        self._last_laser_data: dict[str, LaserObservation] = {}
        self.stats = SensorPerturbationStats()

    def reset(self, seed: int | None = None) -> None:
        self._rng.seed(self.config.seed if seed is None else seed)
        self._history.clear()
        self._last_robot_states.clear()
        self._last_pedestrian_states.clear()
        self._last_laser_data.clear()
        self.stats = SensorPerturbationStats()

    def apply(
        self,
        robot_states: Mapping[str, object],
        pedestrian_states: Mapping[str, object],
        laser_data: Mapping[str, LaserObservation],
    ) -> SensorSnapshot:
        snapshot = SensorSnapshot(
            robot_states=dict(robot_states),
            pedestrian_states=dict(pedestrian_states),
            laser_data=dict(laser_data),
        )
        self._history.append(snapshot)
        self.stats.steps += 1
        if not self.config.enabled:
            return snapshot

        source = self._history[0]
        if len(self._history) <= self.config.latency_steps:
            self.stats.latency_warmup_steps += 1

        robots = self._perturb_agent_mapping(
            source.robot_states,
            self._last_robot_states,
            enabled=self.config.perturb_robot_states,
            dropout_counter="robot_state_dropouts",
            velocity_fields=("v", "vy"),
        )
        pedestrians = self._perturb_agent_mapping(
            source.pedestrian_states,
            self._last_pedestrian_states,
            enabled=self.config.perturb_pedestrian_states,
            dropout_counter="pedestrian_state_dropouts",
            velocity_fields=("vx", "vy"),
        )
        lidar = self._perturb_lidar_mapping(source.laser_data)
        return SensorSnapshot(
            robot_states=robots,
            pedestrian_states=pedestrians,
            laser_data=lidar,
        )

    def _perturb_agent_mapping(
        self,
        source: Mapping[str, object],
        last_delivered: dict[str, object],
        *,
        enabled: bool,
        dropout_counter: str,
        velocity_fields: tuple[str, str],
    ) -> dict[str, object]:
        output: dict[str, object] = {}
        for name, agent in source.items():
            if enabled and self._draw_dropout():
                setattr(
                    self.stats,
                    dropout_counter,
                    getattr(self.stats, dropout_counter) + 1,
                )
                delivered = last_delivered.get(name, agent)
            elif enabled:
                delivered = self._perturb_agent(agent, velocity_fields)
            else:
                delivered = agent
            output[name] = delivered
            last_delivered[name] = delivered
        return output

    def _perturb_agent(
        self,
        agent: object,
        velocity_fields: tuple[str, str],
    ) -> object:
        changes: dict[str, float] = {}
        if hasattr(agent, "x"):
            changes["x"] = float(getattr(agent, "x")) + self._gauss(
                self.config.position_noise_std
            )
        if hasattr(agent, "y"):
            changes["y"] = float(getattr(agent, "y")) + self._gauss(
                self.config.position_noise_std
            )
        for field in velocity_fields:
            if hasattr(agent, field):
                changes[field] = float(getattr(agent, field)) + self._gauss(
                    self.config.velocity_noise_std
                )
        return replace(agent, **changes) if changes else agent

    def _perturb_lidar_mapping(
        self,
        source: Mapping[str, LaserObservation],
    ) -> dict[str, LaserObservation]:
        output: dict[str, LaserObservation] = {}
        for name, observation in source.items():
            if self.config.perturb_lidar and self._draw_dropout():
                self.stats.lidar_scan_dropouts += 1
                delivered = self._last_laser_data.get(name, observation)
            elif self.config.perturb_lidar:
                delivered = self._perturb_lidar(observation)
            else:
                delivered = observation
            output[name] = delivered
            self._last_laser_data[name] = delivered
        return output

    def _perturb_lidar(self, observation: LaserObservation) -> LaserObservation:
        ranges: list[float] = []
        for raw_value in observation.ranges:
            if self._rng.random() < self.config.lidar_sample_dropout_probability:
                ranges.append(math.nan)
                self.stats.lidar_sample_dropouts += 1
                continue
            value = float(raw_value)
            if not math.isfinite(value):
                ranges.append(value)
                continue
            value += self._gauss(self.config.lidar_range_noise_std)
            ranges.append(
                min(max(value, observation.range_min), observation.range_max)
            )
        return laser_scan_to_observation(
            ranges,
            angle_min=observation.angle_min,
            angle_increment=observation.angle_increment,
            range_min=observation.range_min,
            range_max=observation.range_max,
            stamp_sec=observation.stamp_sec,
            message_stamp_sec=observation.message_stamp_sec,
        )

    def _draw_dropout(self) -> bool:
        return self._rng.random() < self.config.dropout_probability

    def _gauss(self, standard_deviation: float) -> float:
        if standard_deviation <= 0.0:
            return 0.0
        return self._rng.gauss(0.0, standard_deviation)


def load_sensor_perturbation_config(path: Path) -> SensorPerturbationConfig:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = _parse_flat_yaml(text)
    if "sensor_perturbation" in payload:
        payload = payload["sensor_perturbation"]
    if not isinstance(payload, Mapping):
        raise ValueError("sensor perturbation config must be a mapping")
    known = set(SensorPerturbationConfig.__dataclass_fields__)
    unknown = set(payload) - known
    if unknown:
        raise ValueError(
            "unknown sensor perturbation fields: " + ", ".join(sorted(unknown))
        )
    return SensorPerturbationConfig(**dict(payload))


def _parse_flat_yaml(text: str) -> dict[str, object]:
    payload: dict[str, object] = {}
    nested: dict[str, object] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.endswith(":"):
            key = stripped[:-1].strip()
            nested = {}
            payload[key] = nested
            continue
        key, raw_value = stripped.split(":", 1)
        target = nested if raw_line[:1].isspace() and nested is not None else payload
        target[key.strip()] = _parse_scalar(raw_value.strip())
    return payload


def _parse_scalar(value: str) -> object:
    lower = value.lower()
    if lower in ("true", "false"):
        return lower == "true"
    if lower in ("null", "none", "~"):
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value.strip('"\'')
