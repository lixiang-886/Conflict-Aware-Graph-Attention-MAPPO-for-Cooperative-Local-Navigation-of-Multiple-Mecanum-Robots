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
import contextlib
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import rclpy
import torch

from mrpp_experiments.metrics import EpisodeMetrics, RobotTrajectoryPoint
from mrpp_experiments.results import result_from_metrics
from mrpp_experiments.scenario import EvaluationRandomizationConfig
from mrpp_experiments.scenario import load_scenario, randomize_scenario
from mrpp_rl.algorithm_config import SUPPORTED_ALGORITHMS
from mrpp_rl.algorithm_config import AlgorithmConfig, adjacency_matrix
from mrpp_rl.algorithm_config import get_algorithm_config
from mrpp_rl.algorithm_config import validate_checkpoint_algorithm
from mrpp_rl.control import apply_robot_proximity_safety_filter
from mrpp_rl.control import apply_pedestrian_proximity_safety_filter
from mrpp_rl.control import clamp, observation_to_features
from mrpp_rl.control import conflict_aware_graph
from mrpp_rl.control import conflict_aware_coordination
from mrpp_rl.control import control_profile_for_algorithm
from mrpp_rl.control import graph_policy_residual_gate
from mrpp_rl.control import gated_graph_policy_actions
from mrpp_rl.control import priority_yield_scale, residual_avoidance_command
from mrpp_rl.control import safety_config_for_world
from mrpp_rl.control import smooth_commands
from mrpp_rl.control import wrap_angle
from mrpp_rl.environment import GoalState, RobotState, build_observation
from mrpp_rl.environment import observation_laser_kwargs
from mrpp_rl.gazebo_bridge import (
    GazeboBridge,
    build_robot_config_from_scenario,
    ensure_processes_running,
    launch_gazebo,
    launch_topic_bridges,
    spawn_robots,
    stop_processes,
)
from mrpp_rl.maddpg import MADDPGPolicy
from mrpp_rl.lidar import apply_safety_filter_to_commands_with_results
from mrpp_rl.mappo import MOGATMAPPOPolicy
from mrpp_rl.observation_schema import OBS_DIM
from mrpp_rl.observation_schema import validate_checkpoint_observation
from mrpp_rl.sensor_perturbation import SensorPerturbationConfig
from mrpp_rl.sensor_perturbation import SensorPerturbationPipeline
from mrpp_rl.sensor_perturbation import load_sensor_perturbation_config
from mrpp_rl.sim_step import SimulationStepper
from mrpp_rl.waypoints import (
    WaypointManager,
    build_scenario_waypoint_manager,
    parse_static_obstacles,
)

HIDDEN_DIM = 64
GOAL_DIST = 0.3
MAX_SPEED = 0.6
MAX_ANGULAR_SPEED = 2.0
REALTIME_SPIN_CALLBACKS = 8
BASELINE_MIN_PRIORITY_YIELD_SCALE = 0.35
DEADLOCK_SPEED_EPS = 0.01
DEADLOCK_PROGRESS_EPS = 1e-4
DEADLOCK_TIMEOUT_SEC = 20.0
MECHANISM_TRACE_FIELDNAMES = (
    "record_type",
    "t",
    "robot_id",
    "x",
    "y",
    "vx",
    "vy",
    "omega",
    "pair_i",
    "pair_j",
    "relative_distance",
    "relative_velocity_x",
    "relative_velocity_y",
    "relative_velocity",
    "tau_cpa",
    "d_cpa",
    "edge_active",
    "gate_value",
    "priority_state",
    "residual_vx",
    "residual_vy",
    "residual_omega",
)


class PlannerController:
    algorithm_name: str
    last_policy_inference_time_s: float = 0.0
    last_pre_safety_commands: dict[str, tuple[float, float, float]] | None = None
    last_mechanism_snapshot: dict[str, object] | None = None

    def reset(self) -> None:
        pass

    def compute_commands(
        self,
        robot_names: list[str],
        states: dict[str, RobotState],
        laser_data,
        pedestrian_states,
        active_goals: dict[str, GoalState],
    ) -> dict[str, tuple[float, float, float]]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class PolicyController(PlannerController):
    def __init__(
        self,
        policy: MOGATMAPPOPolicy,
        algorithm_config: AlgorithmConfig,
        max_speed: float = MAX_SPEED,
        use_orca_prior: bool = True,
        use_proximity_safety: bool = True,
        world_name: str | None = None,
        control_dt: float = 0.1,
        communication_delay_ms: float = 0.0,
        packet_drop_rate: float = 0.0,
        communication_seed: int = 0,
    ) -> None:
        self.policy = policy
        self.algorithm_config = algorithm_config
        self.algorithm_name = algorithm_config.name
        self.max_speed = max_speed
        self.use_orca_prior = use_orca_prior
        self.use_proximity_safety = use_proximity_safety
        self.control_profile = control_profile_for_algorithm(
            algorithm_config.name,
            world_name,
        )
        self.control_dt = max(float(control_dt), 1e-6)
        self.communication_delay_ms = float(communication_delay_ms)
        self.packet_drop_rate = float(packet_drop_rate)
        self.communication_delay_steps = max(
            0,
            int(round(self.communication_delay_ms / (self.control_dt * 1000.0))),
        )
        self._communication_rng = random.Random(int(communication_seed))
        self._communication_history: list[dict[str, RobotState]] = []
        self._last_delivered_states: dict[str, RobotState] = {}
        self._communication_attempt_count = 0
        self._communication_delivery_count = 0
        self._communication_stale_count = 0
        self._communication_bytes = 0
        self._communication_steps = 0
        self._cpa_graph_build_times: list[float] = []
        self._gat_message_passing_times: list[float] = []
        self._active_conflict_edges: list[int] = []
        self._previous_conflict_edges: tuple[int, ...] | None = None
        self._edge_change_count = 0
        self._edge_transition_denominator = 0
        self._possible_edge_count = 0
        try:
            self.device = next(policy.parameters()).device
        except (AttributeError, StopIteration):
            self.device = "cpu"
        if hasattr(policy, "profile_encoder_time"):
            policy.profile_encoder_time = True
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def _communicated_states(
        self,
        states: dict[str, RobotState],
    ) -> dict[str, RobotState]:
        self._communication_history.append(dict(states))
        history_limit = self.communication_delay_steps + 1
        if len(self._communication_history) > history_limit:
            self._communication_history.pop(0)
        candidate = self._communication_history[0]
        delivered: dict[str, RobotState] = {}
        candidate_age_steps = len(self._communication_history) - 1
        delay_is_stale = candidate_age_steps > 0
        for name, state in candidate.items():
            self._communication_attempt_count += 1
            dropped = self._communication_rng.random() < self.packet_drop_rate
            if dropped and name in self._last_delivered_states:
                delivered[name] = self._last_delivered_states[name]
                self._communication_stale_count += 1
                continue
            delivered[name] = state
            self._last_delivered_states[name] = state
            self._communication_delivery_count += 1
            if delay_is_stale:
                self._communication_stale_count += 1
            payload = {
                "name": name,
                "x": state.x,
                "y": state.y,
                "yaw": state.yaw,
                "v": state.v,
                "vy": state.vy,
                "omega": state.omega,
            }
            self._communication_bytes += len(
                json.dumps(payload, separators=(",", ":")).encode("utf-8")
            )
        self._communication_steps += 1
        return delivered

    def runtime_summary(self) -> dict[str, float | int]:
        possible = self._possible_edge_count
        active_mean = (
            sum(self._active_conflict_edges) / len(self._active_conflict_edges)
            if self._active_conflict_edges else 0.0
        )
        duration = self._communication_steps * self.control_dt
        robot_count = max(len(self._last_delivered_states), 1)
        peak_gpu_memory_mb = 0.0
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            peak_gpu_memory_mb = (
                torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)
            )
        return {
            "possible_edge_count": possible,
            "active_conflict_edge_count": active_mean,
            "edge_density": active_mean / possible if possible else 0.0,
            "cpa_graph_build_time_ms": _mean_ms(self._cpa_graph_build_times),
            "gat_message_passing_time_ms": _mean_ms(
                self._gat_message_passing_times
            ),
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "communication_delay_ms": self.communication_delay_ms,
            "packet_drop_rate": self.packet_drop_rate,
            "stale_neighbor_ratio": (
                self._communication_stale_count / self._communication_attempt_count
                if self._communication_attempt_count else 0.0
            ),
            "effective_neighbor_update_rate_hz": (
                self._communication_delivery_count / duration / robot_count
                if duration > 0.0 else 0.0
            ),
            "mean_message_size_bytes": (
                self._communication_bytes / self._communication_delivery_count
                if self._communication_delivery_count else 0.0
            ),
            "estimated_bandwidth_per_robot_bps": (
                self._communication_bytes / duration / robot_count
                if duration > 0.0 else 0.0
            ),
            "cpa_edge_change_rate": (
                self._edge_change_count / self._edge_transition_denominator
                if self._edge_transition_denominator else 0.0
            ),
        }

    def compute_commands(
        self,
        robot_names: list[str],
        states: dict[str, RobotState],
        laser_data,
        pedestrian_states,
        active_goals: dict[str, GoalState],
    ) -> dict[str, tuple[float, float, float]]:
        ped_positions = [(p.x, p.y) for p in pedestrian_states.values()]
        communicated_states = self._communicated_states(states)
        obs_list = []
        observations = {}
        for name in robot_names:
            state = states[name]
            others = [
                communicated_states.get(other_name, other_state)
                for other_name, other_state in states.items()
                if other_name != name
            ]
            obs = build_observation(
                state, active_goals[name], others,
                **observation_laser_kwargs(laser_data.get(name)),
                pedestrians=ped_positions,
            )
            observations[name] = obs
            obs_list.append(
                torch.tensor(observation_to_features(obs), dtype=torch.float32)
            )
        obs_tensor = torch.stack(obs_list)
        action_linear_by_name = None
        cpa_adjacency_values = adjacency_matrix(len(robot_names), False)
        graph_diagnostics: list[dict[str, float | int | str]] = []
        if (
            self.algorithm_config.use_graph_attention
            or self.algorithm_config.use_conflict_coordinator
            or self.algorithm_config.use_interaction_gate
        ):
            graph_start = time.perf_counter()
            cpa_adjacency_values, graph_diagnostics = conflict_aware_graph(
                robot_names,
                communicated_states,
                active_goals,
                self.max_speed,
                self.control_profile,
            )
            self._cpa_graph_build_times.append(time.perf_counter() - graph_start)
        if (
            self.algorithm_config.use_graph_attention
            and not self.algorithm_config.use_fully_connected_graph
        ):
            adjacency_values = cpa_adjacency_values
        else:
            adjacency_values = adjacency_matrix(
                len(robot_names),
                self.algorithm_config.use_graph_attention,
            )
        if (
            self.algorithm_config.use_conflict_coordinator
            or self.algorithm_config.use_interaction_gate
        ):
            interaction_adjacency_values = cpa_adjacency_values
        else:
            interaction_adjacency_values = adjacency_values
        adjacency = torch.tensor(adjacency_values, dtype=torch.float32)

        possible_edge_count = len(robot_names) * (len(robot_names) - 1) // 2
        active_edges = tuple(
            int(cpa_adjacency_values[row][column] > 0.0)
            for row in range(len(robot_names))
            for column in range(row + 1, len(robot_names))
        )
        self._possible_edge_count = possible_edge_count
        self._active_conflict_edges.append(sum(active_edges))
        if self._previous_conflict_edges is not None:
            self._edge_change_count += sum(
                int(before != after)
                for before, after in zip(self._previous_conflict_edges, active_edges)
            )
            self._edge_transition_denominator += possible_edge_count
        self._previous_conflict_edges = active_edges

        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        inference_start = time.perf_counter()
        with torch.no_grad():
            action, _, _, _ = self.policy.get_action_and_value(
                obs_tensor.to(self.device).unsqueeze(0),
                adjacency.to(self.device).unsqueeze(0),
                deterministic=True,
            )
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        self.last_policy_inference_time_s = time.perf_counter() - inference_start
        encoder_time = float(getattr(self.policy, "last_encoder_time_s", 0.0))
        self._gat_message_passing_times.append(
            encoder_time if self.algorithm_config.use_graph_attention else 0.0
        )
        action = action.squeeze(0).detach().cpu()

        ped_list = list(pedestrian_states.values())
        if self.algorithm_config.use_conflict_coordinator:
            action_linear_by_name = {
                name: action[index, 0].item()
                for index, name in enumerate(robot_names)
            }
            directives = conflict_aware_coordination(
                robot_names,
                communicated_states,
                active_goals,
                self.max_speed,
                self.control_profile,
                action_linear_by_name=action_linear_by_name,
            )
        else:
            directives = {}
        commands = {}
        pre_safety_commands = {}
        gate_values: dict[str, float] = {}
        residual_actions: dict[str, tuple[float, float, float]] = {}
        priority_states: dict[str, str] = {}
        for i, name in enumerate(robot_names):
            directive = directives.get(name)
            residual_gate = 1.0
            if self.algorithm_config.use_interaction_gate:
                has_robot_conflict = any(
                    value > 0.0
                    for column, value in enumerate(interaction_adjacency_values[i])
                    if column != i
                )
                residual_gate = graph_policy_residual_gate(
                    has_robot_conflict,
                    observations[name],
                    self.control_profile,
                )
            action_linear, action_lateral, action_angular = gated_graph_policy_actions(
                action[i, 0].item(),
                action[i, 1].item(),
                action[i, 2].item(),
                residual_gate,
            )
            gate_values[name] = residual_gate
            residual_actions[name] = (
                action_linear,
                action_lateral,
                action_angular,
            )
            priority_states[name] = (
                "yield"
                if directive is not None and directive.preferred_speed_scale < 0.999
                else "proceed"
            )
            command_options = {
                "use_orca_prior": self.use_orca_prior,
                "control_profile": self.control_profile,
                "preferred_speed_scale": (
                    directive.preferred_speed_scale
                    if directive is not None else 1.0
                ),
                "coordination_world_vx": (
                    directive.world_lateral_vx if directive is not None else 0.0
                ),
                "coordination_world_vy": (
                    directive.world_lateral_vy if directive is not None else 0.0
                ),
                "priority_yield_scale_override": (
                    directive.priority_yield_scale
                    if directive is not None else None
                ),
            }
            commands[name] = residual_avoidance_command(
                action_linear,
                action_lateral,
                action_angular,
                states[name],
                active_goals[name],
                [
                    communicated_states.get(other, other_state)
                    for other, other_state in states.items()
                    if other != name
                ],
                ped_list,
                self.max_speed,
                use_proximity_safety=self.use_proximity_safety,
                **command_options,
            )
            if self.use_proximity_safety:
                pre_safety_commands[name] = residual_avoidance_command(
                    action_linear,
                    action_lateral,
                    action_angular,
                    states[name],
                    active_goals[name],
                    [
                        communicated_states.get(other, other_state)
                        for other, other_state in states.items()
                        if other != name
                    ],
                    ped_list,
                    self.max_speed,
                    use_proximity_safety=False,
                    **command_options,
                )
            else:
                pre_safety_commands[name] = commands[name]
        self.last_pre_safety_commands = pre_safety_commands
        self.last_mechanism_snapshot = {
            "graph_diagnostics": graph_diagnostics,
            "gate_values": gate_values,
            "priority_states": priority_states,
            "residual_actions": residual_actions,
        }
        return commands


class ORCAController(PlannerController):
    algorithm_name = "orca"

    def __init__(
        self,
        use_proximity_safety: bool = True,
        world_name: str | None = None,
    ) -> None:
        self.use_proximity_safety = use_proximity_safety
        self.control_profile = control_profile_for_algorithm(
            "sensors_mo_gat_mappo",
            world_name,
        )

    def compute_commands(
        self,
        robot_names: list[str],
        states: dict[str, RobotState],
        laser_data,
        pedestrian_states,
        active_goals: dict[str, GoalState],
    ) -> dict[str, tuple[float, float, float]]:
        del laser_data
        ped_list = list(pedestrian_states.values())
        commands = {}
        pre_safety_commands = {}
        for name in robot_names:
            command_args = (
                0.0,
                0.0,
                0.0,
                states[name],
                active_goals[name],
                [state for other, state in states.items() if other != name],
                ped_list,
                MAX_SPEED,
            )
            commands[name] = residual_avoidance_command(
                *command_args,
                control_profile=self.control_profile,
                use_proximity_safety=self.use_proximity_safety,
            )
            if self.use_proximity_safety:
                pre_safety_commands[name] = residual_avoidance_command(
                    *command_args,
                    control_profile=self.control_profile,
                    use_proximity_safety=False,
                )
            else:
                pre_safety_commands[name] = commands[name]
        self.last_pre_safety_commands = pre_safety_commands
        return commands


class RVO2Controller(PlannerController):
    """Canonical RVO2 reciprocal-agent controller with A* waypoint guidance."""

    algorithm_name = "rvo2"

    def __init__(self, control_dt: float = 0.1) -> None:
        try:
            import rvo2
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The canonical rvo2 baseline requires the pinned pyrvo2 "
                "dependency described in rvo2_dependency.json."
            ) from exc
        self.rvo2 = rvo2
        self.control_dt = max(float(control_dt), 1e-3)

    @staticmethod
    def _world_velocity(state: RobotState) -> tuple[float, float]:
        return (
            math.cos(state.yaw) * state.v - math.sin(state.yaw) * state.vy,
            math.sin(state.yaw) * state.v + math.cos(state.yaw) * state.vy,
        )

    def compute_commands(
        self,
        robot_names: list[str],
        states: dict[str, RobotState],
        laser_data,
        pedestrian_states,
        active_goals: dict[str, GoalState],
    ) -> dict[str, tuple[float, float, float]]:
        del laser_data
        max_neighbors = max(10, len(robot_names) + len(pedestrian_states))
        simulator = self.rvo2.PyRVOSimulator(
            self.control_dt,
            5.0,
            max_neighbors,
            5.0,
            5.0,
            0.30,
            MAX_SPEED,
        )
        agent_index: dict[str, int] = {}
        for name in robot_names:
            state = states[name]
            index = simulator.addAgent((state.x, state.y))
            agent_index[name] = index
            simulator.setAgentVelocity(index, self._world_velocity(state))
            goal = active_goals[name]
            dx = goal.x - state.x
            dy = goal.y - state.y
            distance = math.hypot(dx, dy)
            speed = min(MAX_SPEED, distance)
            preferred = (
                (speed * dx / distance, speed * dy / distance)
                if distance > 1e-9 else (0.0, 0.0)
            )
            simulator.setAgentPrefVelocity(index, preferred)
        for pedestrian in pedestrian_states.values():
            index = simulator.addAgent((pedestrian.x, pedestrian.y))
            velocity = self._world_velocity(pedestrian)
            simulator.setAgentVelocity(index, velocity)
            simulator.setAgentPrefVelocity(index, velocity)
        simulator.doStep()

        commands: dict[str, tuple[float, float, float]] = {}
        for name in robot_names:
            state = states[name]
            world_vx, world_vy = simulator.getAgentVelocity(agent_index[name])
            body_vx = (
                math.cos(state.yaw) * world_vx
                + math.sin(state.yaw) * world_vy
            )
            body_vy = (
                -math.sin(state.yaw) * world_vx
                + math.cos(state.yaw) * world_vy
            )
            speed = math.hypot(world_vx, world_vy)
            desired_heading = (
                math.atan2(world_vy, world_vx)
                if speed > 1e-6
                else state.yaw
            )
            omega = clamp(
                1.8 * wrap_angle(desired_heading - state.yaw),
                -MAX_ANGULAR_SPEED,
                MAX_ANGULAR_SPEED,
            )
            commands[name] = (body_vx, body_vy, omega)
        self.last_pre_safety_commands = dict(commands)
        return commands


class PriorityAStarController(PlannerController):
    algorithm_name = "priority_astar"

    def compute_commands(
        self,
        robot_names: list[str],
        states: dict[str, RobotState],
        laser_data,
        pedestrian_states,
        active_goals: dict[str, GoalState],
    ) -> dict[str, tuple[float, float, float]]:
        del laser_data, pedestrian_states
        commands = {}
        for name in robot_names:
            robot = states[name]
            goal = active_goals[name]
            neighbors = [state for other, state in states.items() if other != name]
            commands[name] = _direct_holonomic_command(
                robot,
                goal,
                neighbors,
                MAX_SPEED,
                min_yield_scale=BASELINE_MIN_PRIORITY_YIELD_SCALE,
            )
        self.last_pre_safety_commands = dict(commands)
        return commands


def build_controller(
    algorithm: str,
    checkpoint: dict[str, object] | None = None,
    hidden_dim: int = HIDDEN_DIM,
    device: str = "cpu",
    use_orca_prior: bool = True,
    use_safety_filter: bool = True,
    world_name: str | None = None,
    modular_safety_stages: bool = False,
    control_dt: float = 0.1,
    communication_delay_ms: float = 0.0,
    packet_drop_rate: float = 0.0,
    communication_seed: int = 0,
) -> PlannerController:
    get_algorithm_config(algorithm)
    if algorithm == "priority_astar":
        return PriorityAStarController()
    if algorithm == "orca":
        if not use_orca_prior:
            raise ValueError("--disable-orca-prior is not valid with --algorithm orca")
        return ORCAController(
            use_proximity_safety=(
                use_safety_filter and not modular_safety_stages
            ),
            world_name=world_name,
        )
    if algorithm == "rvo2":
        return RVO2Controller(control_dt=control_dt)
    if checkpoint is None:
        raise ValueError(f"--checkpoint is required for algorithm '{algorithm}'")
    checkpoint_config = validate_checkpoint_algorithm(checkpoint, algorithm)
    validate_checkpoint_observation(checkpoint)
    if checkpoint_config.update_mode == "maddpg":
        policy = MADDPGPolicy(node_dim=OBS_DIM, hidden_dim=hidden_dim)
        policy.load_state_dict(checkpoint["policy_state_dict"])
    else:
        policy = MOGATMAPPOPolicy(
            node_dim=OBS_DIM,
            hidden_dim=hidden_dim,
            centralized_critic=checkpoint_config.use_centralized_critic,
        )
        from mrpp_rl.policy import load_mogat_policy_state_dict

        load_mogat_policy_state_dict(policy, checkpoint["policy_state_dict"])
    policy.to(device)
    policy.eval()
    return PolicyController(
        policy,
        checkpoint_config,
        use_orca_prior=use_orca_prior,
        use_proximity_safety=(
            use_safety_filter and not modular_safety_stages
        ),
        world_name=world_name,
        control_dt=control_dt,
        communication_delay_ms=communication_delay_ms,
        packet_drop_rate=packet_drop_rate,
        communication_seed=communication_seed,
    )


def _direct_holonomic_command(
    robot: RobotState,
    goal: GoalState,
    neighbors: list[RobotState],
    max_speed: float,
    min_yield_scale: float = 0.0,
) -> tuple[float, float, float]:
    dx = goal.x - robot.x
    dy = goal.y - robot.y
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
        return 0.0, 0.0, 0.0
    speed = min(max_speed, distance)
    yield_scale = priority_yield_scale(robot, neighbors)
    if yield_scale < 1.0 and min_yield_scale > 0.0:
        yield_scale = max(yield_scale, clamp(min_yield_scale, 0.0, 1.0))
    speed *= yield_scale
    world_vx = speed * dx / distance
    world_vy = speed * dy / distance
    body_vx = math.cos(robot.yaw) * world_vx + math.sin(robot.yaw) * world_vy
    body_vy = -math.sin(robot.yaw) * world_vx + math.cos(robot.yaw) * world_vy
    heading_error = wrap_angle(math.atan2(dy, dx) - robot.yaw)
    omega = clamp(1.8 * heading_error, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)
    return body_vx, body_vy, omega


def _mean_ms(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values) * 1000.0


def _p95_ms(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index] * 1000.0


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and (
        not hasattr(torch, "cuda") or not torch.cuda.is_available()
    ):
        raise RuntimeError(
            f"Requested --device {device}, but PyTorch CUDA is not available."
        )
    return device


def _spin_bridge_callbacks(bridge: GazeboBridge, max_callbacks: int = 1) -> None:
    spin_some = getattr(bridge, "spin_some", None)
    if spin_some is not None:
        spin_some(max_callbacks=max_callbacks, timeout_sec=0.0)
    else:
        bridge.spin_once(timeout_sec=0.0)


def _override_max_steps(scenario, max_steps: int | None):
    if max_steps is None:
        return scenario
    return replace(scenario, max_steps=max_steps)


def evaluate_episode(
    bridge: GazeboBridge,
    scenario,
    robot_names: list[str],
    goals: dict[str, GoalState],
    seed: int,
    log_interval: int = 0,
    waypoint_manager: WaypointManager | None = None,
    controller: PlannerController | None = None,
    policy: MOGATMAPPOPolicy | None = None,
    sim_stepper: SimulationStepper | None = None,
    use_orca_prior: bool = True,
    use_safety_filter: bool = True,
    policy_sensor_pipeline: SensorPerturbationPipeline | None = None,
    safety_sensor_pipeline: SensorPerturbationPipeline | None = None,
    modular_safety_stages: bool = False,
    mechanism_trace_rows: list[dict[str, object]] | None = None,
) -> EpisodeMetrics:
    if controller is None:
        if policy is None:
            raise ValueError("Either controller or policy must be provided.")
        controller = PolicyController(
            policy,
            get_algorithm_config("mo_gat_mappo"),
            use_orca_prior=use_orca_prior,
            use_proximity_safety=(
                use_safety_filter and not modular_safety_stages
            ),
            world_name=getattr(scenario, "world", ""),
        )
    if policy_sensor_pipeline is not None:
        policy_sensor_pipeline.reset()
    if safety_sensor_pipeline is not None:
        safety_sensor_pipeline.reset()
    metrics = EpisodeMetrics(
        scenario=scenario.name,
        algorithm=controller.algorithm_name,
    )
    for name in robot_names:
        metrics.mark_success(name, False)
    laser_safety_config = safety_config_for_world(getattr(scenario, "world", ""))
    reached = {name: False for name in robot_names}
    failed = {name: False for name in robot_names}
    prev_collision_flags = {name: False for name in robot_names}
    deadlock_active = {name: False for name in robot_names}
    deadlock_stall_times = {name: 0.0 for name in robot_names}
    previous_commands = {name: (0.0, 0.0, 0.0) for name in robot_names}
    policy_times: list[float] = []
    control_step_times: list[float] = []
    diagnostic_robot = os.environ.get("MRPP_CONTROL_DIAGNOSTICS", "").strip()
    episode_wall_start = time.perf_counter()
    steps_run = 0

    for step in range(scenario.max_steps):
        steps_run = step + 1
        control_step_start = time.perf_counter()
        bridge.update_pedestrian_fallbacks(step * scenario.dt)
        _spin_bridge_callbacks(bridge, max_callbacks=REALTIME_SPIN_CALLBACKS)
        states = bridge.get_robot_states()
        laser_data = bridge.get_laser_data()
        ped_states = bridge.get_pedestrian_states()
        policy_snapshot = (
            policy_sensor_pipeline.apply(states, ped_states, laser_data)
            if policy_sensor_pipeline is not None else None
        )
        safety_snapshot = (
            safety_sensor_pipeline.apply(states, ped_states, laser_data)
            if safety_sensor_pipeline is not None else None
        )
        policy_states = (
            dict(policy_snapshot.robot_states)
            if policy_snapshot is not None else states
        )
        policy_laser_data = (
            dict(policy_snapshot.laser_data)
            if policy_snapshot is not None else laser_data
        )
        policy_ped_states = (
            dict(policy_snapshot.pedestrian_states)
            if policy_snapshot is not None else ped_states
        )
        safety_states = (
            dict(safety_snapshot.robot_states)
            if safety_snapshot is not None else states
        )
        safety_laser_data = (
            dict(safety_snapshot.laser_data)
            if safety_snapshot is not None else laser_data
        )
        safety_ped_states = (
            dict(safety_snapshot.pedestrian_states)
            if safety_snapshot is not None else ped_states
        )
        tracking_states = (
            waypoint_manager.tracking_states(states, goals)
            if waypoint_manager is not None
            and hasattr(waypoint_manager, "tracking_states")
            else {}
        )
        active_goals = (
            {name: state.local_goal for name, state in tracking_states.items()}
            if tracking_states
            else (
                waypoint_manager.current_goals(states, goals)
                if waypoint_manager is not None else goals
            )
        )
        before_path_progress = {
            name: state.progress_distance
            for name, state in tracking_states.items()
        }
        before_active_dists = {
            name: math.hypot(
                states[name].x - active_goals[name].x,
                states[name].y - active_goals[name].y,
            )
            for name in robot_names
            if name in states and name in active_goals
        }

        raw_commands = controller.compute_commands(
            robot_names,
            policy_states,
            policy_laser_data,
            policy_ped_states,
            active_goals,
        )
        if mechanism_trace_rows is not None:
            mechanism = getattr(controller, "last_mechanism_snapshot", None) or {}
            gate_values = mechanism.get("gate_values", {})
            priority_states = mechanism.get("priority_states", {})
            residual_actions = mechanism.get("residual_actions", {})
            trace_time = step * scenario.dt
            for name in robot_names:
                state = states.get(name)
                if state is None:
                    continue
                residual = residual_actions.get(name, (0.0, 0.0, 0.0))
                mechanism_trace_rows.append(
                    {
                        "record_type": "robot",
                        "t": trace_time,
                        "robot_id": name,
                        "x": state.x,
                        "y": state.y,
                        "vx": state.v,
                        "vy": state.vy,
                        "omega": state.omega,
                        "gate_value": gate_values.get(name, 0.0),
                        "priority_state": priority_states.get(name, ""),
                        "residual_vx": residual[0],
                        "residual_vy": residual[1],
                        "residual_omega": residual[2],
                    }
                )
            for diagnostic in mechanism.get("graph_diagnostics", []):
                mechanism_trace_rows.append(
                    {
                        "record_type": "pair",
                        "t": trace_time,
                        **diagnostic,
                    }
                )
        pre_safety_commands = (
            getattr(controller, "last_pre_safety_commands", None)
            or raw_commands
        )
        current_time_sec = (
            bridge.current_time_sec()
            if hasattr(bridge, "current_time_sec") else None
        )
        smoothed_commands = smooth_commands(
            raw_commands,
            previous_commands,
            scenario.dt,
        )
        smoothed_pre_safety_commands = smooth_commands(
            pre_safety_commands,
            previous_commands,
            scenario.dt,
        )
        active_robot_names = [
            name for name in robot_names
            if not reached[name] and not failed[name]
        ]
        world_name = getattr(scenario, "world", "")
        control_profile = getattr(
            controller,
            "control_profile",
            control_profile_for_algorithm(
                getattr(controller, "algorithm_name", ""),
                world_name,
            ),
        )
        metrics.observe_pre_filter_risk(
            states,
            ped_states,
            smoothed_pre_safety_commands,
            robot_names=active_robot_names,
            horizon_sec=max(control_profile.robot_proximity_time_horizon, 2.0),
        )
        laser_results = {}
        if use_safety_filter:
            if modular_safety_stages:
                pedestrian_commands = apply_pedestrian_proximity_safety_filter(
                    smoothed_commands,
                    safety_states,
                    list(safety_ped_states.values()),
                    MAX_SPEED,
                    control_profile,
                )
            else:
                pedestrian_commands = smoothed_commands
            metrics.observe_safety_module(
                "robot_pedestrian",
                smoothed_pre_safety_commands,
                pedestrian_commands,
                robot_names=active_robot_names,
                linear_command_scale=MAX_SPEED,
                angular_command_scale=MAX_ANGULAR_SPEED,
            )
            laser_results = apply_safety_filter_to_commands_with_results(
                pedestrian_commands,
                safety_laser_data,
                config=laser_safety_config,
                current_time_sec=current_time_sec,
            )
            laser_commands = {
                name: result.command for name, result in laser_results.items()
            }
            metrics.observe_safety_module(
                "lidar",
                pedestrian_commands,
                laser_commands,
                robot_names=active_robot_names,
                linear_command_scale=MAX_SPEED,
                angular_command_scale=MAX_ANGULAR_SPEED,
            )
            metrics.observe_stale_sensor_stops(
                laser_results,
                robot_names=active_robot_names,
            )
            commands = apply_robot_proximity_safety_filter(
                laser_commands,
                safety_states,
                stop_radius=control_profile.robot_proximity_stop_radius,
                slow_radius=control_profile.robot_proximity_slow_radius,
                predicted_radius=control_profile.robot_proximity_predicted_radius,
                time_horizon=control_profile.robot_proximity_time_horizon,
                pedestrians=list(safety_ped_states.values()),
                pedestrian_skip_radius=(
                    control_profile.pedestrian_proximity_slow_radius
                    if world_name == "mixed_complex" else 0.0
                ),
            )
            metrics.observe_safety_module(
                "robot_robot",
                laser_commands,
                commands,
                robot_names=active_robot_names,
                linear_command_scale=MAX_SPEED,
                angular_command_scale=MAX_ANGULAR_SPEED,
            )
            metrics.observe_safety_interventions(
                smoothed_pre_safety_commands,
                commands,
                robot_names=active_robot_names,
                linear_command_scale=MAX_SPEED,
                angular_command_scale=MAX_ANGULAR_SPEED,
            )
        else:
            pedestrian_commands = smoothed_commands
            commands = smoothed_commands
        if diagnostic_robot in states and step % 10 == 0:
            diagnostic_state = states[diagnostic_robot]
            diagnostic_goal = active_goals[diagnostic_robot]
            result = laser_results.get(diagnostic_robot)
            diagnostic_laser_command = (
                result.command
                if result is not None
                else pedestrian_commands[diagnostic_robot]
            )
            nearest_pedestrian = min(
                (
                    math.hypot(
                        diagnostic_state.x - pedestrian.x,
                        diagnostic_state.y - pedestrian.y,
                    )
                    for pedestrian in ped_states.values()
                ),
                default=math.inf,
            )
            print(
                "  control_diag "
                f"step={step} robot={diagnostic_robot} "
                f"pose=({diagnostic_state.x:.3f},{diagnostic_state.y:.3f},"
                f"{diagnostic_state.yaw:.3f}) "
                f"goal=({diagnostic_goal.x:.3f},{diagnostic_goal.y:.3f}) "
                f"raw={tuple(round(v, 3) for v in raw_commands[diagnostic_robot])} "
                f"smooth={tuple(round(v, 3) for v in smoothed_commands[diagnostic_robot])} "
                f"laser={tuple(round(v, 3) for v in diagnostic_laser_command)} "
                f"final={tuple(round(v, 3) for v in commands[diagnostic_robot])} "
                f"laser_reason={result.reason if result else 'disabled'} "
                f"motion_clearance={result.motion_clearance if result else math.inf:.3f} "
                f"nearest_pedestrian={nearest_pedestrian:.3f}",
                flush=True,
            )
        policy_times.append(getattr(controller, "last_policy_inference_time_s", 0.0))
        for name in robot_names:
            if reached[name] or failed[name]:
                commands[name] = (0.0, 0.0, 0.0)
        previous_commands = {
            name: commands.get(name, (0.0, 0.0, 0.0))
            for name in robot_names
        }
        bridge.send_commands(commands)
        _spin_bridge_callbacks(bridge, max_callbacks=REALTIME_SPIN_CALLBACKS)
        if sim_stepper is None:
            time.sleep(scenario.dt)
            bridge.update_pedestrian_fallbacks((step + 1) * scenario.dt)
            _spin_bridge_callbacks(bridge, max_callbacks=REALTIME_SPIN_CALLBACKS)
        else:
            sim_stepper.advance(step * scenario.dt)

        states = bridge.get_robot_states()
        laser_data = bridge.get_laser_data()
        ped_states = bridge.get_pedestrian_states()
        metrics.observe_clearances(
            {
                name: (states[name].x, states[name].y)
                for name in robot_names
                if name in states
                and not reached.get(name, False)
                and not failed.get(name, False)
            },
            {
                name: (ped.x, ped.y)
                for name, ped in ped_states.items()
            },
        )
        get_collision_categories = getattr(
            bridge,
            "get_collision_categories",
            None,
        )
        collision_categories = (
            get_collision_categories()
            if get_collision_categories is not None else {}
        )
        collision_flags = (
            {
                robot_name: category is not None
                for robot_name, category in collision_categories.items()
            }
            if get_collision_categories is not None
            else bridge.get_collisions()
        )
        t = (step + 1) * scenario.dt
        for name in robot_names:
            state = states[name]
            metrics.add_point(
                name,
                RobotTrajectoryPoint(
                    t=t, x=state.x, y=state.y, v=state.v, omega=state.omega
                ),
            )
            goal = goals[name]
            dist = math.hypot(state.x - goal.x, state.y - goal.y)
            already_done = reached[name] or failed[name]
            speed = math.hypot(state.v, state.vy)
            if (
                not already_done
                and speed < 0.01
                and dist >= GOAL_DIST
            ):
                metrics.add_waiting_time(name, scenario.dt)
            new_collision = (
                collision_flags.get(name, False)
                and not prev_collision_flags.get(name, False)
                and not already_done
            )
            if new_collision:
                metrics.collision_count += 1
                metrics.collision_failure_count += 1
                category = collision_categories.get(name) or "robot_obstacle"
                laser = laser_data.get(name)
                min_laser = laser.min_range if laser is not None else 999.0
                nearest_robot = min(
                    (
                        math.hypot(state.x - other.x, state.y - other.y)
                        for other_name, other in states.items()
                        if other_name != name
                    ),
                    default=999.0,
                )
                nearest_pedestrian = min(
                    (
                        math.hypot(state.x - ped.x, state.y - ped.y)
                        for ped in ped_states.values()
                    ),
                    default=999.0,
                )
                print(
                    "  collision "
                    f"robot={name} category={category} "
                    f"t={t:.1f}s x={state.x:.2f} y={state.y:.2f} "
                    f"min_laser={min_laser:.3f} "
                    f"nearest_robot={nearest_robot:.3f} "
                    f"nearest_pedestrian={nearest_pedestrian:.3f}",
                    flush=True,
                )
                if category == "robot_robot":
                    metrics.robot_robot_collision_count += 1
                elif category == "robot_pedestrian":
                    metrics.robot_pedestrian_collision_count += 1
                else:
                    metrics.robot_obstacle_collision_count += 1
                failed[name] = True
            elif dist < GOAL_DIST and not already_done:
                reached[name] = True
                metrics.mark_success(name, True, completion_time=t)
            elif not already_done:
                active_goal = active_goals.get(name, goal)
                active_dist = math.hypot(
                    state.x - active_goal.x,
                    state.y - active_goal.y,
                )
                after_tracking = None
                if (
                    waypoint_manager is not None
                    and hasattr(waypoint_manager, "tracking_state")
                ):
                    after_tracking = waypoint_manager.tracking_state(
                        name,
                        state,
                        goal,
                    )
                progress = (
                    after_tracking.progress_distance - before_path_progress[name]
                    if after_tracking is not None and name in before_path_progress
                    else before_active_dists.get(name, active_dist) - active_dist
                )
                stalled_without_progress = (
                    speed < DEADLOCK_SPEED_EPS
                    and progress <= DEADLOCK_PROGRESS_EPS
                    and dist >= GOAL_DIST
                )
                if stalled_without_progress:
                    deadlock_stall_times[name] += scenario.dt
                else:
                    deadlock_stall_times[name] = 0.0
                    deadlock_active[name] = False
                if (
                    deadlock_stall_times[name] >= DEADLOCK_TIMEOUT_SEC
                    and not deadlock_active[name]
                ):
                    deadlock_active[name] = True
                    metrics.deadlock_count += 1
                    print(
                        "  deadlock "
                        f"robot={name} t={t:.1f}s x={state.x:.2f} "
                        f"y={state.y:.2f} stall_time="
                        f"{deadlock_stall_times[name]:.1f}s",
                        flush=True,
                    )
        prev_collision_flags = collision_flags

        if log_interval > 0 and (step + 1) % log_interval == 0:
            print(
                f"  eval step={step + 1}/{scenario.max_steps} "
                f"reached={sum(1 for v in reached.values() if v)}/"
                f"{len(robot_names)} "
                f"failed={sum(1 for v in failed.values() if v)}/"
                f"{len(robot_names)} "
                f"collisions={metrics.collision_count}",
                flush=True,
            )

        if all(reached[name] or failed[name] for name in robot_names):
            control_step_times.append(time.perf_counter() - control_step_start)
            break
        control_step_times.append(time.perf_counter() - control_step_start)

    bridge.stop_all()
    controller.close()
    unfinished = [
        name for name in robot_names
        if not reached.get(name, False) and not failed.get(name, False)
    ]
    if unfinished:
        deadlocked_unfinished = [
            name for name in unfinished
            if deadlock_active.get(name, False)
            or deadlock_stall_times.get(name, 0.0) >= DEADLOCK_TIMEOUT_SEC
        ]
        metrics.deadlock_failure_count = len(deadlocked_unfinished)
        metrics.timeout_failure_count = len(unfinished) - len(deadlocked_unfinished)
    wall_time = time.perf_counter() - episode_wall_start
    simulated_time = steps_run * scenario.dt
    metrics.runtime_summary = {
        "steps_run": steps_run,
        "mean_policy_inference_time_ms": _mean_ms(policy_times),
        "p95_policy_inference_time_ms": _p95_ms(policy_times),
        "mean_control_step_time_ms": _mean_ms(control_step_times),
        "episode_wall_time_s": wall_time,
        "real_time_factor": simulated_time / wall_time if wall_time > 0 else 0.0,
        "policy_sensor_perturbation": (
            policy_sensor_pipeline.stats.to_dict()
            if policy_sensor_pipeline is not None else {}
        ),
        "safety_sensor_perturbation": (
            safety_sensor_pipeline.stats.to_dict()
            if safety_sensor_pipeline is not None else {}
        ),
    }
    runtime_summary = getattr(controller, "runtime_summary", None)
    if callable(runtime_summary):
        metrics.runtime_summary.update(runtime_summary())
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a path-planning algorithm on a scenario."
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--algorithm",
        choices=SUPPORTED_ALGORITHMS,
        default="mo_gat_mappo",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--training-seed",
        type=int,
        default=None,
        help="Training seed of the evaluated checkpoint; distinct from --seed.",
    )
    parser.add_argument(
        "--episode-id",
        type=int,
        default=0,
        help="Randomized evaluation episode identifier written to result metadata.",
    )
    parser.add_argument(
        "--randomize-evaluation",
        action="store_true",
        help="Randomize start pose, initial yaw, pedestrian phase, and speed.",
    )
    parser.add_argument("--randomization-profile", default="randomized_v1")
    parser.add_argument("--start-position-jitter-m", type=float, default=0.05)
    parser.add_argument("--start-yaw-jitter-rad", type=float, default=0.15)
    parser.add_argument("--pedestrian-phase-jitter-sec", type=float, default=10.0)
    parser.add_argument(
        "--pedestrian-speed-jitter-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--world", default=None)
    parser.add_argument("--robot-model", default=None)
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--sim-step-mode",
        choices=("realtime", "fast"),
        default="realtime",
        help=(
            "Simulation stepping mode. realtime sleeps for scenario dt; "
            "fast drives Gazebo to the next simulation timestamp through "
            "world control and waits for fresh odom/scan."
        ),
    )
    parser.add_argument(
        "--gazebo-update-rate",
        type=float,
        default=None,
        help=(
            "Gazebo server update rate passed to ign gazebo -z. "
            "Defaults to 10000 in fast mode and Gazebo's default in realtime."
        ),
    )
    parser.add_argument(
        "--disable-orca-prior",
        action="store_true",
        help=(
            "Disable the reciprocal-avoidance velocity prior for learned "
            "policy controllers. This is not valid with --algorithm orca."
        ),
    )
    parser.add_argument(
        "--disable-safety-filter",
        action="store_true",
        help=(
            "Disable post-policy laser and robot-proximity safety filters, "
            "and disable command-level pedestrian proximity limiting in "
            "learned controllers."
        ),
    )
    parser.add_argument(
        "--sensor-perturbation-config",
        default=None,
        help="YAML or JSON perturbation profile applied to policy observations.",
    )
    parser.add_argument(
        "--safety-sensor-perturbation-config",
        default=None,
        help=(
            "Optional independent perturbation profile for final safety inputs. "
            "When omitted, the safety layer receives nominal simulator inputs."
        ),
    )
    parser.add_argument(
        "--modular-safety-audit",
        action="store_true",
        help=(
            "Run pedestrian, LiDAR, robot proximity, and stale-scan safety "
            "stages separately and record their intervention ratios."
        ),
    )
    parser.add_argument(
        "--communication-profile",
        default="nominal",
        help="Name of the deterministic communication-emulation profile.",
    )
    parser.add_argument(
        "--communication-delay-ms",
        type=float,
        default=0.0,
        help="Emulated delay on policy neighbour-state updates only.",
    )
    parser.add_argument(
        "--packet-drop-rate",
        type=float,
        default=0.0,
        help="Emulated policy neighbour-state packet drop probability.",
    )
    parser.add_argument(
        "--communication-seed",
        type=int,
        default=None,
        help="Seed for deterministic delay/drop emulation; defaults to --seed.",
    )
    parser.add_argument(
        "--checkpoint-sha256",
        default="",
        help="Frozen checkpoint digest written to the raw episode row.",
    )
    parser.add_argument(
        "--source-tree-sha256",
        default="",
        help="Frozen source-tree digest written to the raw episode row.",
    )
    parser.add_argument(
        "--config-sha256",
        default="",
        help="Frozen training-config digest written to the raw episode row.",
    )
    parser.add_argument(
        "--mechanism-trace-output",
        default=None,
        help="Optional time-aligned robot, CPA-edge, and gate trace CSV.",
    )
    parser.add_argument(
        "--variant-label",
        default="",
        help=(
            "Optional experiment variant label written to result/config files. "
            "Use this for ablation rows without changing the checkpoint algorithm."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the full Gazebo GUI instead of the headless server.",
    )
    parser.add_argument(
        "--trajectory-output",
        default=None,
        help=(
            "Optional trajectory CSV path. If omitted, a sibling "
            "*_trajectories.csv file is written next to the result CSV."
        ),
    )
    parser.add_argument(
        "--manual-start-file",
        default=None,
        help=(
            "Optional file gate for interactive GUI captures. When set, "
            "evaluation waits until this file exists before the episode starts."
        ),
    )
    parser.add_argument(
        "--final-hold-file",
        default=None,
        help=(
            "Optional file gate for GUI captures. After the episode finishes, "
            "Gazebo is paused and kept open until this file is created."
        ),
    )
    parser.add_argument(
        "--final-hold-timeout-sec",
        type=float,
        default=120.0,
        help="Maximum time to wait for --final-hold-file before closing Gazebo.",
    )
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be greater than zero")
    if args.log_interval < 0:
        parser.error("--log-interval must be zero or greater")
    if args.final_hold_timeout_sec <= 0:
        parser.error("--final-hold-timeout-sec must be greater than zero")
    if args.episode_id < 0:
        parser.error("--episode-id must be non-negative")
    if args.start_position_jitter_m < 0.0:
        parser.error("--start-position-jitter-m must be non-negative")
    if args.start_yaw_jitter_rad < 0.0:
        parser.error("--start-yaw-jitter-rad must be non-negative")
    if args.pedestrian_phase_jitter_sec < 0.0:
        parser.error("--pedestrian-phase-jitter-sec must be non-negative")
    if not 0.0 <= args.pedestrian_speed_jitter_fraction < 1.0:
        parser.error("--pedestrian-speed-jitter-fraction must be in [0, 1)")
    if args.communication_delay_ms < 0.0:
        parser.error("--communication-delay-ms must be non-negative")
    if not 0.0 <= args.packet_drop_rate <= 1.0:
        parser.error("--packet-drop-rate must be in [0, 1]")
    if args.gui and args.sim_step_mode == "fast":
        parser.error("--sim-step-mode fast is only supported in headless mode")
    if args.final_hold_file and not args.gui:
        parser.error("--final-hold-file requires --gui")
    if args.gazebo_update_rate is None and args.sim_step_mode == "fast":
        args.gazebo_update_rate = 10000.0
    if args.algorithm == "orca" and args.disable_orca_prior:
        parser.error("--disable-orca-prior is not valid with --algorithm orca")

    scenario_path = Path(args.scenario)
    randomization_seed = args.seed * 1_000_003 + args.episode_id
    randomization_config = EvaluationRandomizationConfig(
        profile=args.randomization_profile,
        enabled=args.randomize_evaluation,
        start_position_jitter_m=args.start_position_jitter_m,
        start_yaw_jitter_rad=args.start_yaw_jitter_rad,
    )
    scenario = randomize_scenario(
        load_scenario(scenario_path),
        randomization_config,
        randomization_seed,
    )
    scenario = _override_max_steps(scenario, args.max_steps)
    _set_random_seed(args.seed)
    args.device = _resolve_device(args.device)
    policy_sensor_config = (
        load_sensor_perturbation_config(Path(args.sensor_perturbation_config))
        if args.sensor_perturbation_config else SensorPerturbationConfig()
    )
    safety_sensor_config = (
        load_sensor_perturbation_config(
            Path(args.safety_sensor_perturbation_config)
        )
        if args.safety_sensor_perturbation_config else SensorPerturbationConfig()
    )
    policy_sensor_config = replace(
        policy_sensor_config,
        seed=policy_sensor_config.seed + args.seed * 1009 + args.episode_id,
    )
    safety_sensor_config = replace(
        safety_sensor_config,
        seed=safety_sensor_config.seed + args.seed * 1013 + args.episode_id,
    )
    policy_sensor_pipeline = SensorPerturbationPipeline(policy_sensor_config)
    safety_sensor_pipeline = SensorPerturbationPipeline(safety_sensor_config)
    from ament_index_python.packages import get_package_share_directory

    algorithm_config = get_algorithm_config(args.algorithm)
    checkpoint = None
    if algorithm_config.requires_checkpoint:
        if args.checkpoint is None:
            parser.error(f"--checkpoint is required for {args.algorithm}")
        checkpoint_path = Path(args.checkpoint)
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    elif args.checkpoint is not None:
        print(
            f"Warning: ignoring --checkpoint for baseline algorithm {args.algorithm}",
            flush=True,
        )
    training_seed = args.training_seed
    if training_seed is None and isinstance(checkpoint, dict):
        stored_training_seed = checkpoint.get("training_seed", checkpoint.get("seed"))
        if stored_training_seed is not None:
            training_seed = int(stored_training_seed)
    if training_seed is None:
        training_seed = args.seed

    output_dir = Path(args.output).parent if args.output else Path("results/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_config = output_dir / "tmp_robot_config.yaml"

    if args.world is None:
        gazebo_dir = Path(get_package_share_directory("mrpp_gazebo"))
        world_file = gazebo_dir / "worlds" / f"{scenario.world}.sdf"
    else:
        world_file = Path(args.world)

    if args.robot_model is None:
        gazebo_pkg = Path(get_package_share_directory("mrpp_gazebo"))
        robot_model = gazebo_pkg / "models" / "mecanum_robot" / "model.sdf"
    else:
        robot_model = Path(args.robot_model)

    robot_names = build_robot_config_from_scenario(scenario, tmp_config)

    import os
    import time as _time

    gazebo_pkg = Path(get_package_share_directory("mrpp_gazebo"))
    os.environ["IGN_GAZEBO_RESOURCE_PATH"] = str(gazebo_pkg / "models")

    gazebo_proc = None
    bridge_procs = []
    bridge = None
    controller = None
    rclpy_initialized = False
    pedestrian_configs = []

    def _stop_processes(processes, timeout_sec: float = 2.0) -> None:
        stop_processes(processes, timeout_sec=timeout_sec)

    try:
        print("Starting Gazebo...")
        gazebo_proc = launch_gazebo(
            world_file,
            gui=args.gui,
            update_rate=args.gazebo_update_rate,
            run_on_start=args.sim_step_mode != "fast",
        )
        _time.sleep(8)
        ensure_processes_running([gazebo_proc], "Gazebo")

        print("Spawning robots...")
        spawn_proc = spawn_robots(tmp_config, robot_model, delay=0.0)
        spawn_code = spawn_proc.wait(timeout=30)
        if spawn_code != 0:
            raise RuntimeError(f"Robot spawning failed with exit code {spawn_code}")
        _time.sleep(2)

        ped_names = []
        if scenario.world in ("pedestrian_dynamic", "mixed_complex"):
            print("Spawning pedestrians...")
            from mrpp_gazebo.pedestrian_controller import get_pedestrian_configs
            from mrpp_gazebo.pedestrian_controller import (
                randomized_pedestrian_configs,
            )
            from mrpp_gazebo.pedestrian_controller import spawn_pedestrians
            from ament_index_python.packages import get_package_share_directory
            gazebo_pkg = Path(get_package_share_directory("mrpp_gazebo"))
            ped_sdf = gazebo_pkg / "models" / "pedestrian_fuel" / "model.sdf"
            pedestrian_configs = (
                randomized_pedestrian_configs(
                    scenario.world,
                    seed=randomization_seed + 97,
                    phase_jitter_sec=args.pedestrian_phase_jitter_sec,
                    speed_jitter_fraction=(
                        args.pedestrian_speed_jitter_fraction
                    ),
                )
                if args.randomize_evaluation
                else get_pedestrian_configs(scenario.world)
            )
            ped_names = spawn_pedestrians(
                scenario.world,
                ped_sdf,
                configs=pedestrian_configs,
            )
            print(f"  Spawned {len(ped_names)} pedestrians")
            _time.sleep(5)

        print("Launching topic bridges...")
        bridge_procs = launch_topic_bridges(
            robot_names, ped_names, world_name=scenario.world
        )
        _time.sleep(5)
        ensure_processes_running(bridge_procs, "ros_gz_bridge")

        rclpy.init()
        rclpy_initialized = True
        robot_names = [r.name for r in scenario.robots]
        bridge = GazeboBridge(robot_names, ped_names, world_name=scenario.world)
        sim_stepper = SimulationStepper(
            bridge,
            scenario.dt,
            mode=args.sim_step_mode,
        )
        sim_stepper.initialize()
        if args.sim_step_mode == "fast":
            sim_stepper.synchronize_after_reset(resume=False)
        try:
            bridge.wait_until_ready(timeout_sec=45.0)
        except RuntimeError as exc:
            print(
                "Gazebo bridge was not ready; restarting topic bridges once "
                f"before failing. Details: {exc}",
                flush=True,
            )
            _stop_processes(bridge_procs)
            bridge_procs = launch_topic_bridges(
                robot_names, ped_names, world_name=scenario.world
            )
            _time.sleep(8)
            ensure_processes_running(bridge_procs, "ros_gz_bridge")
            bridge.wait_until_ready(timeout_sec=45.0)
        if ped_names:
            bridge.set_pedestrian_configs(pedestrian_configs)

        start_positions = {
            rt.name: (rt.start.x, rt.start.y, rt.start.yaw)
            for rt in scenario.robots
        }
        bridge.set_start_positions(start_positions)

        goals = {}
        for robot_task in scenario.robots:
            goals[robot_task.name] = GoalState(
                x=robot_task.goal.x,
                y=robot_task.goal.y,
                yaw=robot_task.goal.yaw,
            )
        bridge.set_goals(goals)
        bridge.set_static_obstacles(parse_static_obstacles(world_file))
        waypoint_manager = build_scenario_waypoint_manager(scenario, world_file)
        if waypoint_manager is not None:
            waypoint_counts = {
                name: len(waypoint_manager.path_for(name)) for name in robot_names
            }
            print(f"Planned waypoints: {waypoint_counts}")
        else:
            print("Planned waypoints: disabled for this scenario")

        if args.randomize_evaluation:
            sim_stepper.before_reset()
            bridge.reset(start_positions)
            if waypoint_manager is not None:
                waypoint_manager.reset()
            sim_stepper.synchronize_after_reset(resume=False)

        final_hold_file = None
        if args.final_hold_file:
            final_hold_file = Path(args.final_hold_file)
            final_hold_file.parent.mkdir(parents=True, exist_ok=True)
            final_hold_file.unlink(missing_ok=True)

        print(f"Simulation step mode: {args.sim_step_mode}")
        if args.manual_start_file:
            start_file = Path(args.manual_start_file)
            start_file.parent.mkdir(parents=True, exist_ok=True)
            if start_file.exists():
                start_file.unlink()
            if args.sim_step_mode == "realtime":
                with contextlib.suppress(Exception):
                    bridge.pause_world(timeout_sec=3.0)
            print(
                f"Manual GUI adjustment ready. Create {start_file} to start evaluation.",
                flush=True,
            )
            while not start_file.exists():
                ensure_processes_running([gazebo_proc], "Gazebo")
                _time.sleep(0.2)
            print("Manual start signal received.", flush=True)
        sim_stepper.resume()
        if args.manual_start_file and args.sim_step_mode == "realtime":
            with contextlib.suppress(Exception):
                bridge.resume_world(timeout_sec=3.0)

        if args.sim_step_mode == "realtime":
            time.sleep(2.0)

        print(
            f"Evaluating {scenario.name} with {args.algorithm} "
            f"for {scenario.max_steps} steps (training_seed={training_seed}, "
            f"evaluation_seed={args.seed}, episode_id={args.episode_id}, "
            f"device={args.device}, orca_prior={not args.disable_orca_prior}, "
            f"safety_filter={not args.disable_safety_filter}, "
            f"sensor_profile={policy_sensor_config.profile}, "
            f"safety_sensor_profile={safety_sensor_config.profile})..."
        )
        controller = build_controller(
            args.algorithm,
            checkpoint=checkpoint,
            hidden_dim=args.hidden_dim,
            device=args.device,
            use_orca_prior=not args.disable_orca_prior,
            use_safety_filter=not args.disable_safety_filter,
            world_name=getattr(scenario, "world", ""),
            modular_safety_stages=args.modular_safety_audit,
            control_dt=scenario.dt,
            communication_delay_ms=args.communication_delay_ms,
            packet_drop_rate=args.packet_drop_rate,
            communication_seed=(
                args.seed
                if args.communication_seed is None
                else args.communication_seed
            ),
        )
        mechanism_trace_rows: list[dict[str, object]] | None = (
            [] if args.mechanism_trace_output else None
        )
        metrics = evaluate_episode(
            bridge,
            scenario,
            robot_names,
            goals,
            args.seed,
            log_interval=args.log_interval,
            waypoint_manager=waypoint_manager,
            controller=controller,
            sim_stepper=sim_stepper,
            use_orca_prior=not args.disable_orca_prior,
            use_safety_filter=not args.disable_safety_filter,
            policy_sensor_pipeline=policy_sensor_pipeline,
            safety_sensor_pipeline=safety_sensor_pipeline,
            modular_safety_stages=args.modular_safety_audit,
            mechanism_trace_rows=mechanism_trace_rows,
        )

        reference_lengths: list[float] = []
        executed_to_reference: list[float] = []
        if waypoint_manager is not None:
            for name in robot_names:
                path = waypoint_manager.path_for(name)
                reference_length = sum(
                    math.hypot(end.x - start.x, end.y - start.y)
                    for start, end in zip(path, path[1:])
                )
                if reference_length <= 1e-9:
                    continue
                reference_lengths.append(reference_length)
                executed_to_reference.append(
                    metrics.path_length(name) / reference_length
                )
        mean_reference_length = (
            sum(reference_lengths) / len(reference_lengths)
            if reference_lengths else 0.0
        )
        mean_executed_to_reference = (
            sum(executed_to_reference) / len(executed_to_reference)
            if executed_to_reference else 0.0
        )
        runtime_summary = getattr(metrics, "runtime_summary", {})

        result = result_from_metrics(
            metrics,
            seed=args.seed,
            robot_count=len(robot_names),
            a_star_reference_path_length=mean_reference_length,
            executed_to_astar_ratio=mean_executed_to_reference,
            trajectory_straightness=metrics.average_path_efficiency(),
            mean_policy_inference_time_ms=getattr(
                metrics, "runtime_summary", {}
            ).get("mean_policy_inference_time_ms", 0.0),
            p95_policy_inference_time_ms=getattr(
                metrics, "runtime_summary", {}
            ).get("p95_policy_inference_time_ms", 0.0),
            mean_control_step_time_ms=getattr(
                metrics, "runtime_summary", {}
            ).get("mean_control_step_time_ms", 0.0),
            episode_wall_time_s=getattr(
                metrics, "runtime_summary", {}
            ).get("episode_wall_time_s", 0.0),
            real_time_factor=getattr(
                metrics, "runtime_summary", {}
            ).get("real_time_factor", 0.0),
            possible_edge_count=int(
                runtime_summary.get("possible_edge_count", 0)
            ),
            active_conflict_edge_count=float(
                runtime_summary.get("active_conflict_edge_count", 0.0)
            ),
            edge_density=float(runtime_summary.get("edge_density", 0.0)),
            cpa_graph_build_time_ms=float(
                runtime_summary.get("cpa_graph_build_time_ms", 0.0)
            ),
            gat_message_passing_time_ms=float(
                runtime_summary.get("gat_message_passing_time_ms", 0.0)
            ),
            peak_gpu_memory_mb=float(
                runtime_summary.get("peak_gpu_memory_mb", 0.0)
            ),
            message_delay_profile=args.communication_profile,
            communication_delay_ms=float(
                runtime_summary.get("communication_delay_ms", 0.0)
            ),
            packet_drop_rate=float(
                runtime_summary.get("packet_drop_rate", 0.0)
            ),
            stale_neighbor_ratio=float(
                runtime_summary.get("stale_neighbor_ratio", 0.0)
            ),
            effective_neighbor_update_rate_hz=float(
                runtime_summary.get("effective_neighbor_update_rate_hz", 0.0)
            ),
            mean_message_size_bytes=float(
                runtime_summary.get("mean_message_size_bytes", 0.0)
            ),
            estimated_bandwidth_per_robot_bps=float(
                runtime_summary.get("estimated_bandwidth_per_robot_bps", 0.0)
            ),
            cpa_edge_change_rate=float(
                runtime_summary.get("cpa_edge_change_rate", 0.0)
            ),
            variant=args.variant_label,
            orca_prior_enabled=not args.disable_orca_prior,
            safety_filter_enabled=not args.disable_safety_filter,
            training_seed=training_seed,
            evaluation_seed=args.seed,
            episode_id=args.episode_id,
            sensor_profile=policy_sensor_config.profile,
            safety_sensor_profile=safety_sensor_config.profile,
            randomization_profile=(
                randomization_config.profile
                if randomization_config.enabled else "fixed"
            ),
            checkpoint_sha256=args.checkpoint_sha256,
            source_tree_sha256=args.source_tree_sha256,
            config_sha256=args.config_sha256,
            data_source="gazebo",
        )

        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path(f"results/raw/{args.algorithm}_seed{args.seed}.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        from mrpp_experiments.results import write_results_csv, write_trajectory_csv

        write_results_csv(output_path, [result])
        config_output = output_path.with_name(f"{output_path.stem}_config.json")
        config_output.write_text(
            json.dumps(
                {
                    "scenario": scenario.name,
                    "scenario_path": str(scenario_path),
                    "algorithm": args.algorithm,
                    "variant": args.variant_label,
                    "seed": args.seed,
                    "training_seed": training_seed,
                    "evaluation_seed": args.seed,
                    "episode_id": args.episode_id,
                    "randomization_seed": randomization_seed,
                    "randomization": asdict(randomization_config),
                    "randomized_start_positions": start_positions,
                    "pedestrian_configs": [
                        asdict(config) for config in pedestrian_configs
                    ],
                    "checkpoint": args.checkpoint,
                    "checkpoint_sha256": args.checkpoint_sha256,
                    "source_tree_sha256": args.source_tree_sha256,
                    "config_sha256": args.config_sha256,
                    "orca_prior_enabled": not args.disable_orca_prior,
                    "safety_filter_enabled": not args.disable_safety_filter,
                    "modular_safety_audit": args.modular_safety_audit,
                    "policy_sensor_perturbation": policy_sensor_config.to_dict(),
                    "safety_sensor_perturbation": safety_sensor_config.to_dict(),
                    "policy_sensor_perturbation_stats": (
                        policy_sensor_pipeline.stats.to_dict()
                    ),
                    "safety_sensor_perturbation_stats": (
                        safety_sensor_pipeline.stats.to_dict()
                    ),
                    "sim_step_mode": args.sim_step_mode,
                    "gazebo_update_rate": args.gazebo_update_rate,
                    "max_steps": scenario.max_steps,
                    "dt": scenario.dt,
                    "device": args.device,
                    "communication": {
                        "profile": args.communication_profile,
                        "delay_ms": args.communication_delay_ms,
                        "packet_drop_rate": args.packet_drop_rate,
                        "seed": (
                            args.seed
                            if args.communication_seed is None
                            else args.communication_seed
                        ),
                        "scope": "policy_neighbour_state_channel",
                        "safety_sensor_channel": "nominal",
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if args.trajectory_output:
            trajectory_output = Path(args.trajectory_output)
        else:
            trajectory_output = output_path.with_name(
                f"{output_path.stem}_trajectories.csv"
            )
        write_trajectory_csv(trajectory_output, metrics, args.seed)
        if args.mechanism_trace_output and mechanism_trace_rows is not None:
            mechanism_trace_output = Path(args.mechanism_trace_output)
            mechanism_trace_output.parent.mkdir(parents=True, exist_ok=True)
            with mechanism_trace_output.open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=MECHANISM_TRACE_FIELDNAMES,
                    lineterminator="\n",
                    extrasaction="ignore",
                )
                writer.writeheader()
                for row in mechanism_trace_rows:
                    writer.writerow(
                        {
                            name: row.get(name, "")
                            for name in MECHANISM_TRACE_FIELDNAMES
                        }
                    )
        print(f"Results saved to {output_path}")
        print(f"Trajectories saved to {trajectory_output}")
        if args.mechanism_trace_output:
            print(f"Mechanism trace saved to {args.mechanism_trace_output}")
        print(f"Summary: {metrics.summary()}")
        if final_hold_file is not None:
            with contextlib.suppress(Exception):
                bridge.pause_world(timeout_sec=3.0)
            print(
                "Final GUI hold ready at step "
                f"{int(metrics.runtime_summary.get('steps_run', 0))}. "
                f"Create {final_hold_file} to close evaluation.",
                flush=True,
            )
            deadline = _time.monotonic() + args.final_hold_timeout_sec
            while not final_hold_file.exists() and _time.monotonic() < deadline:
                ensure_processes_running([gazebo_proc], "Gazebo")
                _time.sleep(0.2)
            if final_hold_file.exists():
                final_hold_file.unlink(missing_ok=True)
                print("Final hold release signal received.", flush=True)
            else:
                print("Final hold timed out; closing Gazebo.", flush=True)
        return 0
    finally:
        if controller is not None:
            with contextlib.suppress(Exception):
                controller.close()
        if bridge is not None:
            with contextlib.suppress(Exception):
                bridge.stop_all()
            with contextlib.suppress(Exception):
                bridge.destroy_node()
        if rclpy_initialized:
            with contextlib.suppress(Exception):
                rclpy.shutdown()
        _stop_processes(bridge_procs)
        if gazebo_proc is not None and gazebo_proc.poll() is None:
            gazebo_proc.terminate()
            with contextlib.suppress(Exception):
                gazebo_proc.wait(timeout=3)
            if gazebo_proc.poll() is None:
                with contextlib.suppress(Exception):
                    gazebo_proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
