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

from dataclasses import dataclass, field
from math import atan2, cos, hypot, inf, isfinite, pi, sin


DEFAULT_LINEAR_COMMAND_SCALE = 0.6
DEFAULT_ANGULAR_COMMAND_SCALE = 2.0


@dataclass(frozen=True)
class RobotTrajectoryPoint:
    t: float
    x: float
    y: float
    v: float
    omega: float


@dataclass
class EpisodeMetrics:
    scenario: str
    algorithm: str
    trajectories: dict[str, list[RobotTrajectoryPoint]] = field(default_factory=dict)
    successes: dict[str, bool] = field(default_factory=dict)
    completion_times: dict[str, float] = field(default_factory=dict)
    waiting_times: dict[str, float] = field(default_factory=dict)
    collision_count: int = 0
    deadlock_count: int = 0
    robot_robot_collision_count: int = 0
    robot_obstacle_collision_count: int = 0
    robot_pedestrian_collision_count: int = 0
    collision_failure_count: int = 0
    timeout_failure_count: int = 0
    deadlock_failure_count: int = 0
    invalid_state_failure_count: int = 0
    process_failure_count: int = 0
    min_robot_robot_distance: float = inf
    min_robot_pedestrian_distance: float = inf
    robot_robot_near_miss_count: int = 0
    robot_pedestrian_near_miss_count: int = 0
    safety_filter_intervention_count: int = 0
    safety_filter_evaluated_command_count: int = 0
    safety_filter_intervention_magnitude_sum: float = 0.0
    max_intervention_magnitude: float = 0.0
    normalized_intervention_magnitude_sum: float = 0.0
    max_normalized_intervention_magnitude: float = 0.0
    safety_module_intervention_counts: dict[str, int] = field(default_factory=dict)
    safety_module_evaluated_counts: dict[str, int] = field(default_factory=dict)
    stale_sensor_stop_count: int = 0
    stale_sensor_evaluated_count: int = 0
    pre_filter_min_predicted_rr_distance: float = inf
    pre_filter_min_predicted_rp_distance: float = inf
    pre_filter_rr_risk_observation_count: int = 0
    pre_filter_rp_risk_observation_count: int = 0
    pre_filter_closing_speed_violation_count: int = 0
    _active_rr_near_miss_pairs: set[tuple[str, str]] = field(
        default_factory=set,
        repr=False,
    )
    _active_rp_near_miss_pairs: set[tuple[str, str]] = field(
        default_factory=set,
        repr=False,
    )

    def add_point(self, robot_name: str, point: RobotTrajectoryPoint) -> None:
        self.trajectories.setdefault(robot_name, []).append(point)

    def mark_success(
        self,
        robot_name: str,
        success: bool,
        completion_time: float | None = None,
    ) -> None:
        self.successes[robot_name] = success
        if success and completion_time is not None:
            self.completion_times.setdefault(robot_name, completion_time)

    def add_waiting_time(self, robot_name: str, dt: float) -> None:
        self.waiting_times[robot_name] = self.waiting_times.get(robot_name, 0.0) + dt

    def observe_clearances(
        self,
        robot_positions: dict[str, tuple[float, float]],
        pedestrian_positions: dict[str, tuple[float, float]] | None = None,
        robot_robot_near_miss_threshold: float = 0.55,
        robot_pedestrian_near_miss_threshold: float = 0.7,
    ) -> None:
        robot_items = sorted(robot_positions.items())
        current_rr_pairs: set[tuple[str, str]] = set()
        current_rp_pairs: set[tuple[str, str]] = set()
        for index, (name, (x, y)) in enumerate(robot_items):
            for other_name, (other_x, other_y) in robot_items[index + 1:]:
                distance = hypot(x - other_x, y - other_y)
                self.min_robot_robot_distance = min(
                    self.min_robot_robot_distance,
                    distance,
                )
                if distance < robot_robot_near_miss_threshold:
                    current_rr_pairs.add((name, other_name))

        for robot_name, (x, y) in robot_items:
            for ped_name, (ped_x, ped_y) in (pedestrian_positions or {}).items():
                distance = hypot(x - ped_x, y - ped_y)
                self.min_robot_pedestrian_distance = min(
                    self.min_robot_pedestrian_distance,
                    distance,
                )
                if distance < robot_pedestrian_near_miss_threshold:
                    current_rp_pairs.add((robot_name, ped_name))

        self.robot_robot_near_miss_count += len(
            current_rr_pairs - self._active_rr_near_miss_pairs
        )
        self.robot_pedestrian_near_miss_count += len(
            current_rp_pairs - self._active_rp_near_miss_pairs
        )
        self._active_rr_near_miss_pairs = current_rr_pairs
        self._active_rp_near_miss_pairs = current_rp_pairs

    def observe_safety_interventions(
        self,
        pre_safety_commands: dict[str, tuple[float, float, float]],
        safe_commands: dict[str, tuple[float, float, float]],
        robot_names: list[str] | None = None,
        linear_command_scale: float = DEFAULT_LINEAR_COMMAND_SCALE,
        angular_command_scale: float = DEFAULT_ANGULAR_COMMAND_SCALE,
        epsilon: float = 1e-4,
    ) -> None:
        """Record raw and dimensionless shared-safety command changes."""
        if linear_command_scale <= 0.0 or angular_command_scale <= 0.0:
            raise ValueError("command normalization scales must be positive")
        names = (
            robot_names
            if robot_names is not None
            else sorted(set(pre_safety_commands).intersection(safe_commands))
        )
        for name in names:
            if name not in pre_safety_commands or name not in safe_commands:
                continue
            raw = pre_safety_commands[name]
            safe = safe_commands[name]
            deltas = tuple(safe[i] - raw[i] for i in range(3))
            magnitude = hypot(*deltas)
            normalized_magnitude = hypot(
                deltas[0] / linear_command_scale,
                deltas[1] / linear_command_scale,
                deltas[2] / angular_command_scale,
            )
            self.safety_filter_evaluated_command_count += 1
            if normalized_magnitude <= epsilon:
                continue
            self.safety_filter_intervention_count += 1
            self.safety_filter_intervention_magnitude_sum += magnitude
            self.max_intervention_magnitude = max(
                self.max_intervention_magnitude,
                magnitude,
            )
            self.normalized_intervention_magnitude_sum += normalized_magnitude
            self.max_normalized_intervention_magnitude = max(
                self.max_normalized_intervention_magnitude,
                normalized_magnitude,
            )

    def intervention_time_ratio(self) -> float:
        if self.safety_filter_evaluated_command_count <= 0:
            return 0.0
        return (
            self.safety_filter_intervention_count
            / self.safety_filter_evaluated_command_count
        )

    def observe_safety_module(
        self,
        module: str,
        before_commands: dict[str, tuple[float, float, float]],
        after_commands: dict[str, tuple[float, float, float]],
        robot_names: list[str] | None = None,
        linear_command_scale: float = DEFAULT_LINEAR_COMMAND_SCALE,
        angular_command_scale: float = DEFAULT_ANGULAR_COMMAND_SCALE,
        epsilon: float = 1e-4,
    ) -> None:
        if linear_command_scale <= 0.0 or angular_command_scale <= 0.0:
            raise ValueError("command normalization scales must be positive")
        names = (
            robot_names
            if robot_names is not None
            else sorted(set(before_commands).intersection(after_commands))
        )
        evaluated = 0
        interventions = 0
        for name in names:
            if name not in before_commands or name not in after_commands:
                continue
            before = before_commands[name]
            after = after_commands[name]
            normalized_magnitude = hypot(
                (after[0] - before[0]) / linear_command_scale,
                (after[1] - before[1]) / linear_command_scale,
                (after[2] - before[2]) / angular_command_scale,
            )
            evaluated += 1
            if normalized_magnitude > epsilon:
                interventions += 1
        self.safety_module_evaluated_counts[module] = (
            self.safety_module_evaluated_counts.get(module, 0) + evaluated
        )
        self.safety_module_intervention_counts[module] = (
            self.safety_module_intervention_counts.get(module, 0) + interventions
        )

    def module_intervention_ratio(self, module: str) -> float:
        evaluated = self.safety_module_evaluated_counts.get(module, 0)
        if evaluated <= 0:
            return 0.0
        return self.safety_module_intervention_counts.get(module, 0) / evaluated

    def observe_stale_sensor_stops(
        self,
        laser_results: dict[str, object],
        robot_names: list[str] | None = None,
    ) -> None:
        names = robot_names if robot_names is not None else sorted(laser_results)
        for name in names:
            result = laser_results.get(name)
            if result is None:
                continue
            self.stale_sensor_evaluated_count += 1
            command = getattr(result, "command", (0.0, 0.0, 0.0))
            if (
                bool(getattr(result, "scan_stale", False))
                and hypot(float(command[0]), float(command[1])) <= 1e-6
            ):
                self.stale_sensor_stop_count += 1

    def stale_sensor_stop_ratio(self) -> float:
        if self.stale_sensor_evaluated_count <= 0:
            return 0.0
        return self.stale_sensor_stop_count / self.stale_sensor_evaluated_count

    def observe_pre_filter_risk(
        self,
        robot_states: dict[str, object],
        pedestrian_states: dict[str, object],
        commands: dict[str, tuple[float, float, float]],
        robot_names: list[str] | None = None,
        horizon_sec: float = 2.0,
        rr_threshold: float = 0.55,
        rp_threshold: float = 0.70,
    ) -> None:
        names = [
            name
            for name in (robot_names if robot_names is not None else sorted(commands))
            if name in robot_states and name in commands
        ]
        velocities = {
            name: _body_command_to_world(commands[name], robot_states[name])
            for name in names
        }
        for index, name in enumerate(names):
            robot = robot_states[name]
            for other_name in names[index + 1:]:
                other = robot_states[other_name]
                rel_x = float(getattr(other, "x")) - float(getattr(robot, "x"))
                rel_y = float(getattr(other, "y")) - float(getattr(robot, "y"))
                rel_vx = velocities[other_name][0] - velocities[name][0]
                rel_vy = velocities[other_name][1] - velocities[name][1]
                predicted, closing = _predicted_distance_and_closing(
                    rel_x,
                    rel_y,
                    rel_vx,
                    rel_vy,
                    horizon_sec,
                )
                self.pre_filter_min_predicted_rr_distance = min(
                    self.pre_filter_min_predicted_rr_distance,
                    predicted,
                )
                if predicted < rr_threshold:
                    self.pre_filter_rr_risk_observation_count += 1
                    if closing:
                        self.pre_filter_closing_speed_violation_count += 1

            for pedestrian in pedestrian_states.values():
                rel_x = float(getattr(pedestrian, "x")) - float(
                    getattr(robot, "x")
                )
                rel_y = float(getattr(pedestrian, "y")) - float(
                    getattr(robot, "y")
                )
                rel_vx = float(getattr(pedestrian, "vx", 0.0)) - velocities[name][0]
                rel_vy = float(getattr(pedestrian, "vy", 0.0)) - velocities[name][1]
                predicted, closing = _predicted_distance_and_closing(
                    rel_x,
                    rel_y,
                    rel_vx,
                    rel_vy,
                    horizon_sec,
                )
                self.pre_filter_min_predicted_rp_distance = min(
                    self.pre_filter_min_predicted_rp_distance,
                    predicted,
                )
                if predicted < rp_threshold:
                    self.pre_filter_rp_risk_observation_count += 1
                    if closing:
                        self.pre_filter_closing_speed_violation_count += 1

    def mean_intervention_magnitude(self) -> float:
        if self.safety_filter_intervention_count <= 0:
            return 0.0
        return (
            self.safety_filter_intervention_magnitude_sum
            / self.safety_filter_intervention_count
        )

    def mean_normalized_intervention_magnitude(self) -> float:
        if self.safety_filter_intervention_count <= 0:
            return 0.0
        return (
            self.normalized_intervention_magnitude_sum
            / self.safety_filter_intervention_count
        )

    def path_length(self, robot_name: str) -> float:
        points = self.trajectories.get(robot_name, [])
        if len(points) < 2:
            return 0.0
        total = 0.0
        for prev, curr in zip(points, points[1:]):
            total += hypot(curr.x - prev.x, curr.y - prev.y)
        return total

    def net_displacement(self, robot_name: str) -> float:
        points = self.trajectories.get(robot_name, [])
        if len(points) < 2:
            return 0.0
        return hypot(points[-1].x - points[0].x, points[-1].y - points[0].y)

    def path_efficiency(self, robot_name: str) -> float:
        length = self.path_length(robot_name)
        if length <= 1e-9:
            return 0.0
        return min(1.0, self.net_displacement(robot_name) / length)

    def turning_angle_sum(self, robot_name: str, min_segment_length: float = 0.02) -> float:
        headings: list[float] = []
        points = self.trajectories.get(robot_name, [])
        for prev, curr in zip(points, points[1:]):
            dx = curr.x - prev.x
            dy = curr.y - prev.y
            if hypot(dx, dy) >= min_segment_length:
                headings.append(atan2(dy, dx))
        total = 0.0
        for prev, curr in zip(headings, headings[1:]):
            total += abs(_angle_diff(prev, curr))
        return total

    def turning_angle_per_meter(self, robot_name: str) -> float:
        length = self.path_length(robot_name)
        if length <= 1e-9:
            return 0.0
        return self.turning_angle_sum(robot_name) / length

    def sharp_turn_count(
        self,
        robot_name: str,
        threshold: float = pi / 2.0,
        min_segment_length: float = 0.02,
    ) -> int:
        headings: list[float] = []
        points = self.trajectories.get(robot_name, [])
        for prev, curr in zip(points, points[1:]):
            dx = curr.x - prev.x
            dy = curr.y - prev.y
            if hypot(dx, dy) >= min_segment_length:
                headings.append(atan2(dy, dx))
        return sum(
            1
            for prev, curr in zip(headings, headings[1:])
            if abs(_angle_diff(prev, curr)) > threshold
        )

    def average_path_length(self) -> float:
        if not self.trajectories:
            return 0.0
        return sum(self.path_length(name) for name in self.trajectories) / len(
            self.trajectories
        )

    def average_path_efficiency(self) -> float:
        if not self.trajectories:
            return 0.0
        return sum(self.path_efficiency(name) for name in self.trajectories) / len(
            self.trajectories
        )

    def average_turning_angle_per_meter(self) -> float:
        if not self.trajectories:
            return 0.0
        return sum(
            self.turning_angle_per_meter(name)
            for name in self.trajectories
        ) / len(self.trajectories)

    def total_sharp_turn_count(self) -> int:
        return sum(self.sharp_turn_count(name) for name in self.trajectories)

    def success_rate(self) -> float:
        if not self.successes:
            return 0.0
        return sum(1 for value in self.successes.values() if value) / len(self.successes)

    def average_completion_time(self) -> float:
        if not self.completion_times:
            return 0.0
        return sum(self.completion_times.values()) / len(self.completion_times)

    def makespan(self) -> float:
        if not self.completion_times:
            return 0.0
        return max(self.completion_times.values())

    def flowtime(self) -> float:
        return sum(self.completion_times.values())

    def average_waiting_time(self) -> float:
        if not self.successes:
            return 0.0
        total = sum(self.waiting_times.values())
        return total / len(self.successes)

    def summary(self) -> dict[str, float | int | str]:
        return {
            "scenario": self.scenario,
            "algorithm": self.algorithm,
            "success_rate": self.success_rate(),
            "collision_count": self.collision_count,
            "deadlock_count": self.deadlock_count,
            "robot_robot_collision_count": self.robot_robot_collision_count,
            "robot_obstacle_collision_count": self.robot_obstacle_collision_count,
            "robot_pedestrian_collision_count": self.robot_pedestrian_collision_count,
            "collision_failure_count": self.collision_failure_count,
            "timeout_failure_count": self.timeout_failure_count,
            "deadlock_failure_count": self.deadlock_failure_count,
            "invalid_state_failure_count": self.invalid_state_failure_count,
            "process_failure_count": self.process_failure_count,
            "min_robot_robot_distance": _finite_or_zero(
                self.min_robot_robot_distance
            ),
            "min_robot_pedestrian_distance": _finite_or_zero(
                self.min_robot_pedestrian_distance
            ),
            "robot_robot_near_miss_count": self.robot_robot_near_miss_count,
            "robot_pedestrian_near_miss_count": self.robot_pedestrian_near_miss_count,
            "safety_filter_intervention_count": (
                self.safety_filter_intervention_count
            ),
            "safety_filter_evaluated_command_count": (
                self.safety_filter_evaluated_command_count
            ),
            "intervention_time_ratio": self.intervention_time_ratio(),
            "mean_intervention_magnitude": self.mean_intervention_magnitude(),
            "max_intervention_magnitude": self.max_intervention_magnitude,
            "mean_normalized_intervention_magnitude": (
                self.mean_normalized_intervention_magnitude()
            ),
            "max_normalized_intervention_magnitude": (
                self.max_normalized_intervention_magnitude
            ),
            "lidar_intervention_count": self.safety_module_intervention_counts.get(
                "lidar", 0
            ),
            "lidar_evaluated_command_count": self.safety_module_evaluated_counts.get(
                "lidar", 0
            ),
            "lidar_intervention_ratio": self.module_intervention_ratio("lidar"),
            "robot_robot_filter_intervention_count": (
                self.safety_module_intervention_counts.get("robot_robot", 0)
            ),
            "robot_robot_filter_evaluated_count": (
                self.safety_module_evaluated_counts.get("robot_robot", 0)
            ),
            "robot_robot_filter_intervention_ratio": (
                self.module_intervention_ratio("robot_robot")
            ),
            "robot_pedestrian_filter_intervention_count": (
                self.safety_module_intervention_counts.get("robot_pedestrian", 0)
            ),
            "robot_pedestrian_filter_evaluated_count": (
                self.safety_module_evaluated_counts.get("robot_pedestrian", 0)
            ),
            "robot_pedestrian_filter_intervention_ratio": (
                self.module_intervention_ratio("robot_pedestrian")
            ),
            "stale_sensor_stop_count": self.stale_sensor_stop_count,
            "stale_sensor_evaluated_count": self.stale_sensor_evaluated_count,
            "stale_sensor_stop_ratio": self.stale_sensor_stop_ratio(),
            "pre_filter_min_predicted_rr_distance": _finite_or_zero(
                self.pre_filter_min_predicted_rr_distance
            ),
            "pre_filter_min_predicted_rp_distance": _finite_or_zero(
                self.pre_filter_min_predicted_rp_distance
            ),
            "pre_filter_rr_risk_observation_count": (
                self.pre_filter_rr_risk_observation_count
            ),
            "pre_filter_rp_risk_observation_count": (
                self.pre_filter_rp_risk_observation_count
            ),
            "pre_filter_closing_speed_violation_count": (
                self.pre_filter_closing_speed_violation_count
            ),
            "average_path_length": self.average_path_length(),
            "average_path_efficiency": self.average_path_efficiency(),
            "average_turning_angle_per_meter": self.average_turning_angle_per_meter(),
            "total_sharp_turn_count": self.total_sharp_turn_count(),
            "average_completion_time": self.average_completion_time(),
            "makespan": self.makespan(),
            "flowtime": self.flowtime(),
            "average_waiting_time": self.average_waiting_time(),
        }


def _angle_diff(a: float, b: float) -> float:
    return atan2(sin(b - a), cos(b - a))


def _finite_or_zero(value: float) -> float:
    return float(value) if isfinite(value) else 0.0


def _body_command_to_world(
    command: tuple[float, float, float],
    state: object,
) -> tuple[float, float]:
    yaw = float(getattr(state, "yaw", 0.0))
    return (
        cos(yaw) * command[0] - sin(yaw) * command[1],
        sin(yaw) * command[0] + cos(yaw) * command[1],
    )


def _predicted_distance_and_closing(
    rel_x: float,
    rel_y: float,
    rel_vx: float,
    rel_vy: float,
    horizon_sec: float,
) -> tuple[float, bool]:
    distance = hypot(rel_x, rel_y)
    speed_sq = rel_vx * rel_vx + rel_vy * rel_vy
    time_to_closest = 0.0
    if speed_sq > 1e-12:
        time_to_closest = max(
            0.0,
            min(horizon_sec, -(rel_x * rel_vx + rel_y * rel_vy) / speed_sq),
        )
    predicted = hypot(
        rel_x + rel_vx * time_to_closest,
        rel_y + rel_vy * time_to_closest,
    )
    closing = distance > 1e-9 and (rel_x * rel_vx + rel_y * rel_vy) / distance < 0.0
    return predicted, closing
