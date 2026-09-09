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
import copy
import csv
import json
import math
import os
import platform
import random
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import rclpy
import torch

from mrpp_experiments.scenario import load_scenario
from mrpp_rl.algorithm_config import CHECKPOINT_ALGORITHMS
from mrpp_rl.algorithm_config import MODULAR_SAFETY_ALGORITHMS
from mrpp_rl.algorithm_config import AlgorithmConfig, adjacency_matrix
from mrpp_rl.algorithm_config import get_algorithm_config
from mrpp_rl.algorithm_config import validate_checkpoint_algorithm
from mrpp_rl.adaptive_weights import SceneContext, adaptive_reward_weights
from mrpp_rl.control import command_change_cost
from mrpp_rl.control import apply_pedestrian_proximity_safety_filter
from mrpp_rl.control import apply_robot_proximity_safety_filter
from mrpp_rl.control import conflict_aware_adjacency_matrix
from mrpp_rl.control import conflict_aware_coordination
from mrpp_rl.control import control_profile_for_algorithm
from mrpp_rl.control import graph_policy_residual_gate
from mrpp_rl.control import gated_graph_policy_actions
from mrpp_rl.control import observation_to_features, residual_avoidance_command
from mrpp_rl.control import safety_config_for_world
from mrpp_rl.control import smooth_commands
from mrpp_rl.environment import GoalState, build_observation
from mrpp_rl.environment import observation_laser_kwargs
from mrpp_rl.gazebo_bridge import (
    GazeboBridge,
    build_robot_config_yaml,
    ensure_processes_running,
    launch_gazebo,
    launch_topic_bridges,
    spawn_robots,
    stop_processes,
)
from mrpp_rl.lidar import apply_safety_filter_to_commands_with_results
from mrpp_rl.maddpg import MADDPGConfig, MADDPGPolicy, MADDPGTrainer
from mrpp_rl.mappo import MAPPOConfig, MAPPOTrainer, MOGATMAPPOPolicy
from mrpp_rl.observation_schema import OBS_DIM
from mrpp_rl.observation_schema import checkpoint_observation_fields
from mrpp_rl.observation_schema import observation_metadata
from mrpp_rl.observation_schema import validate_checkpoint_observation
from mrpp_rl.reward import RewardWeights, compute_reward
from mrpp_rl.sim_step import SimulationStepper
from mrpp_rl.training_config import mappo_config_for_algorithm
from mrpp_rl.waypoints import (
    WaypointManager,
    build_scenario_waypoint_manager,
    parse_static_obstacles,
)


HIDDEN_DIM = 64
COLLISION_DIST = 0.25
GOAL_DIST = 0.3
MAX_SPEED = 0.6
REALTIME_SPIN_CALLBACKS = 8
TRAIN_LOG_FIELDNAMES = (
    "episode",
    "seed",
    "scenario",
    "algorithm",
    "variant",
    "orca_prior_enabled",
    "safety_filter_enabled",
    "episode_reward",
    "mean_agent_reward",
    "rollout_success_rate",
    "rollout_collision_count",
    "rollout_deadlock_count",
    "eval_success_rate",
    "eval_collision_count",
    "eval_average_path_length",
    "eval_average_completion_time",
    "success_rate",
    "collision_count",
    "deadlock_count",
    "average_path_length",
    "average_completion_time",
    "episode_steps",
    "policy_loss",
    "value_loss",
    "entropy",
    "update_skipped",
    "update_skip_reason",
    "learning_rate",
    "wall_time",
    "checkpoint_path",
    "selection_source",
)
VALIDATION_LOG_FIELDNAMES = (
    "training_seed",
    "validation_index",
    "training_episode",
    "success",
    "success_count",
    "collision",
    "timeout",
    "deadlock",
    "average_completion_time",
    "makespan",
    "average_waiting_time",
    "average_path_length",
    "trajectory_straightness",
    "min_robot_robot_distance",
    "min_robot_pedestrian_distance",
    "selection_score",
    "is_selected_checkpoint",
    "checkpoint_path",
)


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


def _best_checkpoint_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_name(
            f"{output_path.stem}_best{output_path.suffix}"
        )
    return output_path.with_name(f"{output_path.name}_best")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_config_snapshot(
    output_dir: Path,
    args,
    scenario,
    scenario_path: Path,
    world_file: Path,
    robot_model: Path,
    algorithm_config: AlgorithmConfig,
    trainer_config: MAPPOConfig | MADDPGConfig,
) -> None:
    payload = {
            "scenario": scenario.name,
            "scenario_path": str(scenario_path),
            "world": scenario.world,
            "world_file": str(world_file),
            "robot_model": str(robot_model),
            "algorithm": algorithm_config.to_dict(),
            "variant": args.variant_label,
            "seed": args.seed,
            "episodes": args.episodes,
            "max_steps": scenario.max_steps,
            "dt": scenario.dt,
            "hidden_dim": args.hidden_dim,
            "device": args.device,
            "sim_step_mode": args.sim_step_mode,
            "gazebo_update_rate": args.gazebo_update_rate,
            "resume_checkpoint": args.resume_checkpoint,
            "validation_interval": args.validation_interval,
            "validation_log_interval": args.validation_log_interval,
            "validation_log": args.validation_log,
            "validation_rollback": not args.no_validation_rollback,
            "initial_validation_diagnostic_only": (
                args.initial_validation_diagnostic_only
            ),
            "orca_prior_enabled": not args.disable_orca_prior,
            "safety_filter_enabled": not args.disable_safety_filter,
            "trainer_config": trainer_config.__dict__,
            "observation": observation_metadata(),
        }
    _write_json(output_dir / "config_snapshot.json", payload)
    # JSON is a valid YAML 1.2 subset; keep the required submission-run name.
    _write_json(output_dir / "config_snapshot.yaml", payload)


def _write_environment_snapshot(output_dir: Path) -> None:
    def _run(command: list[str]) -> str:
        try:
            return subprocess.check_output(command, text=True, timeout=10).strip()
        except Exception as exc:
            return f"unavailable: {exc}"

    _write_json(
        output_dir / "environment.json",
        {
            "git_commit_sha": _run(["git", "rev-parse", "HEAD"]),
            "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "hostname": socket.gethostname(),
            "os_version": platform.platform(),
            "ros_distro": os.environ.get("ROS_DISTRO", ""),
            "gazebo_version": _run(["ign", "gazebo", "--version"]),
            "python_version": sys.version,
            "torch_version": getattr(torch, "__version__", ""),
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        },
    )


def _trainer_optimizer_state_dict(trainer) -> dict[str, object]:
    if hasattr(trainer, "optimizer_state_dict"):
        return trainer.optimizer_state_dict()
    return trainer.optimizer.state_dict()


def _load_trainer_optimizer_state_dict(trainer, state: dict[str, object]) -> None:
    if hasattr(trainer, "load_optimizer_state_dict"):
        trainer.load_optimizer_state_dict(state)
    else:
        trainer.optimizer.load_state_dict(state)


def _trainer_learning_rate(trainer) -> float:
    if hasattr(trainer, "learning_rate"):
        return trainer.learning_rate()
    return float(trainer.optimizer.param_groups[0]["lr"])


def _append_training_log(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAIN_LOG_FIELDNAMES, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in TRAIN_LOG_FIELDNAMES})


def _append_validation_log(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=VALIDATION_LOG_FIELDNAMES,
            lineterminator="\n",
        )
        if not exists:
            writer.writeheader()
        writer.writerow(
            {name: row.get(name, "") for name in VALIDATION_LOG_FIELDNAMES}
        )


def _load_training_checkpoint(
    checkpoint_path: Path,
    algorithm: str,
    policy,
    trainer,
    scenario_name: str | None = None,
) -> dict[str, object]:
    checkpoint = torch.load(str(checkpoint_path), map_location=trainer.device)
    validate_checkpoint_algorithm(checkpoint, algorithm)
    validate_checkpoint_observation(checkpoint)
    checkpoint_scenario = checkpoint.get("scenario")
    if scenario_name is not None:
        if checkpoint_scenario is None:
            raise RuntimeError(
                "Checkpoint is missing scenario metadata; refusing to resume "
                "because cross-scenario resume is disabled."
            )
        if checkpoint_scenario != scenario_name:
            raise RuntimeError(
                "Checkpoint scenario mismatch: "
                f"checkpoint={checkpoint_scenario}, requested={scenario_name}. "
                "Cross-scenario resume is disabled."
            )
    if isinstance(policy, MADDPGPolicy) or not hasattr(policy, "encoder"):
        policy.load_state_dict(checkpoint["policy_state_dict"])
    else:
        from mrpp_rl.policy import load_mogat_policy_state_dict

        load_mogat_policy_state_dict(policy, checkpoint["policy_state_dict"])
    if hasattr(trainer, "sync_targets"):
        trainer.sync_targets()
    if "optimizer_state_dict" in checkpoint:
        _load_trainer_optimizer_state_dict(
            trainer,
            checkpoint["optimizer_state_dict"],
        )
    return checkpoint


def _make_observations(bridge: GazeboBridge, goals: dict[str, GoalState]):
    states = bridge.get_robot_states()
    neighbor_dists = bridge.get_neighbor_distances()
    laser_data = bridge.get_laser_data()
    ped_states = bridge.get_pedestrian_states()
    ped_positions = [(p.x, p.y) for p in ped_states.values()]

    obs_dict = {}
    for name, state in states.items():
        others = [s for n, s in states.items() if n != name]
        obs = build_observation(
            state, goals[name], others,
            **observation_laser_kwargs(laser_data.get(name)),
            pedestrians=ped_positions,
        )
        obs_dict[name] = obs
    return obs_dict, states, neighbor_dists


def _make_tensors(
    obs_dict,
    robot_names,
    algorithm_config: AlgorithmConfig | None = None,
    states=None,
    goals=None,
    control_profile=None,
):
    if algorithm_config is None:
        algorithm_config = get_algorithm_config("mo_gat_mappo")
    obs_list = [
        torch.tensor(observation_to_features(obs_dict[name]), dtype=torch.float32)
        for name in robot_names
    ]
    obs_tensor = torch.stack(obs_list)
    obs_tensor = torch.nan_to_num(obs_tensor, nan=0.0)
    if (
        algorithm_config.use_graph_attention
        and not algorithm_config.use_fully_connected_graph
        and states is not None
        and goals is not None
    ):
        profile = control_profile or control_profile_for_algorithm(
            algorithm_config.name,
        )
        adjacency_values = conflict_aware_adjacency_matrix(
            robot_names,
            states,
            goals,
            MAX_SPEED,
            profile,
        )
    else:
        adjacency_values = adjacency_matrix(
            len(robot_names),
            algorithm_config.use_graph_attention,
        )
    adjacency = torch.tensor(adjacency_values, dtype=torch.float32)
    return obs_tensor, adjacency


def _soft_clearance_penalty(
    distance: float,
    safe_distance: float,
    hard_distance: float,
) -> float:
    if not math.isfinite(distance) or distance >= safe_distance:
        return 0.0
    if distance <= hard_distance:
        return 1.0
    span = max(safe_distance - hard_distance, 1e-6)
    risk = (safe_distance - distance) / span
    return max(0.0, min(1.0, risk * risk))


def _progress_speed_reward(progress: float, dt: float) -> float:
    if dt <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, progress / max(dt * MAX_SPEED, 1e-6)))


def _safety_intervention_cost(raw_command, filtered_command, result) -> float:
    raw_vx, raw_vy, raw_omega = raw_command
    safe_vx, safe_vy, safe_omega = filtered_command
    linear_delta = math.hypot(safe_vx - raw_vx, safe_vy - raw_vy) / max(MAX_SPEED, 1e-6)
    angular_delta = abs(safe_omega - raw_omega) / 2.0
    translation_penalty = 0.0
    emergency_penalty = 0.0
    if result is not None:
        translation_penalty = max(0.0, 1.0 - float(result.translation_scale))
        emergency_penalty = 1.0 if result.emergency_stop else 0.0
    return max(0.0, linear_delta + 0.25 * angular_delta + translation_penalty + emergency_penalty)


def _run_deterministic_validation(
    bridge: GazeboBridge,
    policy: MOGATMAPPOPolicy | MADDPGPolicy,
    scenario,
    robot_names: list[str],
    goals: dict[str, GoalState],
    seed: int,
    log_interval: int,
    waypoint_manager: WaypointManager | None,
    algorithm_config: AlgorithmConfig,
    sim_stepper: SimulationStepper | None = None,
    use_orca_prior: bool = True,
    use_safety_filter: bool = True,
):
    from mrpp_rl.evaluate import PolicyController, evaluate_episode

    if waypoint_manager is not None:
        waypoint_manager.reset()
    was_training = policy.training
    policy.eval()
    try:
        modular_safety_stages = algorithm_config.name in MODULAR_SAFETY_ALGORITHMS
        controller = PolicyController(
            policy,
            algorithm_config,
            use_orca_prior=use_orca_prior,
            use_proximity_safety=(
                use_safety_filter and not modular_safety_stages
            ),
            world_name=getattr(scenario, "world", ""),
        )
        return evaluate_episode(
            bridge=bridge,
            scenario=scenario,
            robot_names=robot_names,
            goals=goals,
            seed=seed,
            log_interval=log_interval,
            waypoint_manager=waypoint_manager,
            controller=controller,
            sim_stepper=sim_stepper,
            use_orca_prior=use_orca_prior,
            use_safety_filter=use_safety_filter,
            modular_safety_stages=modular_safety_stages,
        )
    finally:
        policy.train(was_training)
        if waypoint_manager is not None:
            waypoint_manager.reset()


def _validation_selection_score(metrics, robot_count: int) -> tuple:
    success_count = sum(1 for value in metrics.successes.values() if value)
    min_robot_robot_distance = _finite_score_value(
        getattr(metrics, "min_robot_robot_distance", 0.0)
    )
    min_robot_pedestrian_distance = _finite_score_value(
        getattr(metrics, "min_robot_pedestrian_distance", 0.0)
    )
    return (
        success_count,
        -metrics.collision_count,
        -metrics.timeout_failure_count,
        -getattr(metrics, "deadlock_failure_count", 0),
        -getattr(metrics, "robot_pedestrian_near_miss_count", 0),
        -getattr(metrics, "robot_robot_near_miss_count", 0),
        # Once hard safety events tie, prefer the controller that clears the
        # task sooner.  Raw clearance remains a tie-breaker, so it cannot
        # reward unnecessary waiting merely to create a larger separation.
        -metrics.average_completion_time(),
        -metrics.makespan(),
        -metrics.average_waiting_time(),
        -metrics.average_path_length(),
        metrics.average_path_efficiency(),
        min_robot_pedestrian_distance,
        min_robot_robot_distance,
        robot_count,
    )


def _rollout_selection_score(
    success_count: int,
    total_collisions: int,
    failure_count: int,
    deadlock_steps: int,
    details: Mapping[str, object],
) -> tuple:
    def detail_float(key: str) -> float:
        try:
            return float(details.get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0

    return (
        success_count,
        -total_collisions,
        -failure_count,
        -deadlock_steps,
        -detail_float("average_waiting_time"),
        -detail_float("average_completion_time"),
        -detail_float("average_path_length"),
    )


def _finite_score_value(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def _selection_summary(
    metrics,
    robot_count: int,
    source: str,
) -> dict[str, object]:
    success_count = sum(1 for value in metrics.successes.values() if value)
    return {
        "success_count": success_count,
        "success_rate": metrics.success_rate(),
        "failure_count": robot_count - success_count,
        "collision_count": metrics.collision_count,
        "timeout_failure_count": metrics.timeout_failure_count,
        "deadlock_failure_count": getattr(metrics, "deadlock_failure_count", 0),
        "deadlock_steps": metrics.deadlock_count,
        "robot_count": robot_count,
        "average_path_length": metrics.average_path_length(),
        "average_path_efficiency": metrics.average_path_efficiency(),
        "average_completion_time": metrics.average_completion_time(),
        "makespan": metrics.makespan(),
        "average_waiting_time": metrics.average_waiting_time(),
        "min_robot_robot_distance": _finite_score_value(
            getattr(metrics, "min_robot_robot_distance", 0.0)
        ),
        "min_robot_pedestrian_distance": _finite_score_value(
            getattr(metrics, "min_robot_pedestrian_distance", 0.0)
        ),
        "robot_robot_near_miss_count": getattr(
            metrics, "robot_robot_near_miss_count", 0
        ),
        "robot_pedestrian_near_miss_count": getattr(
            metrics, "robot_pedestrian_near_miss_count", 0
        ),
        "source": source,
    }


def _save_training_checkpoint(
    path: Path,
    episode: int,
    policy_state,
    optimizer_state,
    algorithm_config: AlgorithmConfig,
    scenario_name: str,
    trainer: MAPPOTrainer,
    selection: dict[str, object] | None = None,
    control_config: dict[str, object] | None = None,
    training_seed: int | None = None,
) -> None:
    payload = {
        "episode": episode,
        "policy_state_dict": policy_state,
        "optimizer_state_dict": optimizer_state,
        "algorithm": algorithm_config.name,
        "scenario": scenario_name,
        "algorithm_config": algorithm_config.to_dict(),
        **checkpoint_observation_fields(),
        "config": trainer.describe(),
    }
    if training_seed is not None:
        payload["training_seed"] = training_seed
    if control_config is not None:
        payload["control_config"] = control_config
    if selection is not None:
        payload["selection"] = selection
    torch.save(payload, str(path))


def train_episode(
    bridge,
    trainer,
    scenario,
    robot_names,
    goals,
    log_interval: int = 0,
    waypoint_manager: WaypointManager | None = None,
    algorithm_config: AlgorithmConfig | None = None,
    min_success_for_update: int | None = None,
    max_collisions_for_update: int | None = None,
    sim_stepper: SimulationStepper | None = None,
    return_details: bool = False,
    use_orca_prior: bool = True,
    use_safety_filter: bool = True,
):
    if algorithm_config is None:
        algorithm_config = get_algorithm_config("mo_gat_mappo")
    control_profile = control_profile_for_algorithm(
        algorithm_config.name,
        getattr(scenario, "world", ""),
    )
    modular_safety_stages = algorithm_config.name in MODULAR_SAFETY_ALGORITHMS
    laser_safety_config = safety_config_for_world(getattr(scenario, "world", ""))
    cfg = trainer.config
    device = getattr(trainer, "device", "cpu")
    max_steps = scenario.max_steps
    collisions = {n: 0 for n in robot_names}
    collision_details: dict[str, dict[str, float | str]] = {}
    deadlock_counters = {n: 0 for n in robot_names}
    reached = {n: False for n in robot_names}
    failed = {n: False for n in robot_names}
    prev_collision_flags = {n: False for n in robot_names}
    deadlock_active = {n: False for n in robot_names}
    episode_rewards = {n: 0.0 for n in robot_names}
    completion_times: dict[str, float] = {}
    path_lengths = {n: 0.0 for n in robot_names}
    initial_states = bridge.get_robot_states()
    last_positions = {
        name: (initial_states[name].x, initial_states[name].y)
        for name in robot_names
        if name in initial_states
    }
    previous_commands = {name: (0.0, 0.0, 0.0) for name in robot_names}
    last_loss_info = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    update_skipped = False
    update_skip_reason = ""
    defer_quality_gated_update = (
        min_success_for_update is not None
        or max_collisions_for_update is not None
    )
    steps_run = 0

    for step in range(max_steps):
        steps_run = step + 1
        bridge.update_pedestrian_fallbacks(step * scenario.dt)
        _spin_bridge_callbacks(bridge, max_callbacks=REALTIME_SPIN_CALLBACKS)
        states = bridge.get_robot_states()
        tracking_states = (
            waypoint_manager.tracking_states(states, goals)
            if waypoint_manager is not None
            and hasattr(waypoint_manager, "tracking_states")
            else {}
        )
        active_goals = (
            {name: state.local_goal for name, state in tracking_states.items()}
            if tracking_states else (
                waypoint_manager.current_goals(states, goals)
                if waypoint_manager is not None else goals
            )
        )
        obs_dict, states, neighbor_dists = _make_observations(bridge, active_goals)
        before_active_dists = {
            name: math.hypot(
                states[name].x - active_goals[name].x,
                states[name].y - active_goals[name].y,
            )
            for name in robot_names
        }
        before_path_progress = {
            name: tracking_states[name].progress_distance
            for name in robot_names
            if name in tracking_states
        }
        obs_tensor, adjacency = _make_tensors(
            obs_dict,
            robot_names,
            algorithm_config=algorithm_config,
            states=states,
            goals=active_goals,
            control_profile=control_profile,
        )

        with torch.no_grad():
            policy_obs = obs_tensor.to(device)
            policy_adj = adjacency.to(device)
            if hasattr(trainer.policy, "sample_action_and_value"):
                (
                    action,
                    raw_action,
                    log_prob,
                    _,
                    value,
                ) = trainer.policy.sample_action_and_value(
                    policy_obs.unsqueeze(0),
                    policy_adj.unsqueeze(0),
                )
            else:
                action, log_prob, _, value = (
                    trainer.policy.get_action_and_value(
                        policy_obs.unsqueeze(0),
                        policy_adj.unsqueeze(0),
                    )
                )
                raw_action = None
        action = action.squeeze(0)
        if raw_action is not None:
            raw_action = raw_action.squeeze(0)
        log_prob = log_prob.squeeze(0)
        value = value.squeeze(0)
        ped_states = list(bridge.get_pedestrian_states().values())
        if (
            algorithm_config.use_conflict_coordinator
            or algorithm_config.use_interaction_gate
        ):
            coordination_adjacency = conflict_aware_adjacency_matrix(
                robot_names,
                states,
                active_goals,
                MAX_SPEED,
                control_profile,
            )
        else:
            coordination_adjacency = adjacency_matrix(
                len(robot_names),
                use_graph_attention=False,
            )
        if algorithm_config.use_conflict_coordinator:
            directives = conflict_aware_coordination(
                robot_names,
                states,
                active_goals,
                MAX_SPEED,
                control_profile,
                action_linear_by_name={
                    name: action[index, 0].item()
                    for index, name in enumerate(robot_names)
                },
            )
        else:
            directives = {}

        commands = {}
        for i, name in enumerate(robot_names):
            if reached[name] or failed[name]:
                commands[name] = (0.0, 0.0, 0.0)
                continue
            residual_gate = 1.0
            if algorithm_config.use_interaction_gate:
                has_robot_conflict = any(
                    value > 0.0
                    for column, value in enumerate(coordination_adjacency[i])
                    if column != i
                )
                residual_gate = graph_policy_residual_gate(
                    has_robot_conflict,
                    obs_dict[name],
                    control_profile,
                )
            action_linear, action_lateral, action_angular = gated_graph_policy_actions(
                action[i, 0].item(),
                action[i, 1].item(),
                action[i, 2].item(),
                residual_gate,
            )
            commands[name] = residual_avoidance_command(
                action_linear,
                action_lateral,
                action_angular,
                states[name],
                active_goals[name],
                [state for other, state in states.items() if other != name],
                ped_states,
                MAX_SPEED,
                use_orca_prior=use_orca_prior,
                use_proximity_safety=(
                    use_safety_filter and not modular_safety_stages
                ),
                control_profile=control_profile,
                preferred_speed_scale=(
                    directives[name].preferred_speed_scale
                    if name in directives else 1.0
                ),
                coordination_world_vx=(
                    directives[name].world_lateral_vx
                    if name in directives else 0.0
                ),
                coordination_world_vy=(
                    directives[name].world_lateral_vy
                    if name in directives else 0.0
                ),
                priority_yield_scale_override=(
                    directives[name].priority_yield_scale
                    if name in directives else None
                ),
            )
        laser_data = bridge.get_laser_data()
        current_time_sec = (
            bridge.current_time_sec()
            if hasattr(bridge, "current_time_sec") else None
        )
        smoothed_commands = smooth_commands(commands, previous_commands, scenario.dt)
        if use_safety_filter and modular_safety_stages:
            smoothed_commands = apply_pedestrian_proximity_safety_filter(
                smoothed_commands,
                states,
                ped_states,
                MAX_SPEED,
                control_profile,
            )
        if use_safety_filter:
            safety_results = apply_safety_filter_to_commands_with_results(
                smoothed_commands,
                laser_data,
                config=laser_safety_config,
                current_time_sec=current_time_sec,
            )
            commands = {
                name: result.command
                for name, result in safety_results.items()
            }
        else:
            safety_results = {}
            commands = smoothed_commands
        if use_safety_filter:
            world_name = getattr(scenario, "world", "")
            commands = apply_robot_proximity_safety_filter(
                commands,
                states,
                stop_radius=control_profile.robot_proximity_stop_radius,
                slow_radius=control_profile.robot_proximity_slow_radius,
                predicted_radius=control_profile.robot_proximity_predicted_radius,
                time_horizon=control_profile.robot_proximity_time_horizon,
                pedestrians=ped_states,
                pedestrian_skip_radius=(
                    control_profile.pedestrian_proximity_slow_radius
                    if world_name == "mixed_complex" else 0.0
                ),
            )
        for name in robot_names:
            if reached[name] or failed[name]:
                commands[name] = (0.0, 0.0, 0.0)
        safety_intervention_costs = {
            name: _safety_intervention_cost(
                smoothed_commands.get(name, (0.0, 0.0, 0.0)),
                commands.get(name, (0.0, 0.0, 0.0)),
                safety_results.get(name),
            )
            for name in robot_names
        }
        command_change_costs = {
            name: command_change_cost(
                commands.get(name, (0.0, 0.0, 0.0)),
                previous_commands.get(name, (0.0, 0.0, 0.0)),
                MAX_SPEED,
            )
            for name in robot_names
        }
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

        post_obs_dict, states, neighbor_dists = _make_observations(bridge, goals)
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
        unsafe_flags = bridge.get_unsafe_distances()

        rewards = []
        dones = []
        for name in robot_names:
            state = states[name]
            goal = goals[name]
            dist = math.hypot(state.x - goal.x, state.y - goal.y)
            already_done = reached[name] or failed[name]

            if already_done:
                rewards.append(0.0)
                dones.append(1.0)
                continue

            active_goal = active_goals[name]
            active_dist = math.hypot(
                state.x - active_goal.x,
                state.y - active_goal.y,
            )
            after_tracking = None
            if waypoint_manager is not None and hasattr(waypoint_manager, "tracking_state"):
                after_tracking = waypoint_manager.tracking_state(name, state, goal)
            progress = (
                after_tracking.progress_distance - before_path_progress[name]
                if after_tracking is not None and name in before_path_progress
                else before_active_dists[name] - active_dist
            )
            is_collision = collision_flags.get(name, False)
            is_unsafe = (
                unsafe_flags.get(name, False)
                if algorithm_config.use_safety_distance_reward
                else False
            )
            goal_reached = dist < GOAL_DIST
            speed = math.hypot(state.v, state.vy)
            is_waiting = speed < 0.01 and not goal_reached

            if is_collision and not prev_collision_flags.get(name, False):
                collisions[name] += 1
            new_collision = (
                is_collision and not prev_collision_flags.get(name, False)
            )
            newly_reached = goal_reached and not new_collision
            if new_collision:
                failed[name] = True
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
                        for ped in bridge.get_pedestrian_states().values()
                    ),
                    default=999.0,
                )
                collision_details[name] = {
                    "category": category,
                    "x": state.x,
                    "y": state.y,
                    "min_laser": min_laser,
                    "nearest_robot": nearest_robot,
                    "nearest_pedestrian": nearest_pedestrian,
                }
                print(
                    "  collision "
                    f"robot={name} category={category} "
                    f"t={(step + 1) * scenario.dt:.1f}s "
                    f"x={state.x:.2f} y={state.y:.2f} "
                    f"min_laser={min_laser:.3f} "
                    f"nearest_robot={nearest_robot:.3f} "
                    f"nearest_pedestrian={nearest_pedestrian:.3f}",
                    flush=True,
                )
            elif goal_reached:
                reached[name] = True
                completion_times.setdefault(name, (step + 1) * scenario.dt)
            stalled_without_progress = (
                speed < 0.01 and not goal_reached and progress <= 1e-4
            )
            if stalled_without_progress and not deadlock_active[name]:
                deadlock_counters[name] += 1
                deadlock_active[name] = True
            elif not stalled_without_progress:
                deadlock_active[name] = False

            density = sum(
                1 for n, d in neighbor_dists.items()
                if n != name and d < 1.5
            ) / max(len(robot_names) - 1, 1)
            robot_clearance_penalty = _soft_clearance_penalty(
                post_obs_dict[name].nearest_robot_distance,
                safe_distance=1.20,
                hard_distance=0.55,
            )
            pedestrian_clearance_penalty = _soft_clearance_penalty(
                post_obs_dict[name].nearest_ped_distance,
                safe_distance=1.50,
                hard_distance=0.70,
            )
            obstacle_clearance_penalty = _soft_clearance_penalty(
                post_obs_dict[name].min_laser_range,
                safe_distance=laser_safety_config.slow_distance,
                hard_distance=laser_safety_config.stop_distance,
            )
            context = SceneContext(
                local_robot_density=density,
                nearest_obstacle_distance=post_obs_dict[name].min_laser_range,
                nearest_robot_distance=post_obs_dict[name].nearest_robot_distance,
                nearest_pedestrian_distance=post_obs_dict[name].nearest_ped_distance,
            )
            if algorithm_config.use_adaptive_reward:
                weights = adaptive_reward_weights(context)
            else:
                weights = RewardWeights()
            command = commands.get(name, (0.0, 0.0, 0.0))
            command_effort = (
                math.hypot(command[0], command[1])
                + 0.25 * abs(command[2])
                + 0.30 * abs(command[1])
            )

            reward = compute_reward(
                progress=progress,
                reached_goal=newly_reached,
                collision=new_collision,
                unsafe_distance=is_unsafe,
                control_effort=command_effort,
                waiting=is_waiting,
                weights=weights,
                smoothness=command_change_costs.get(name, 0.0),
                clearance_penalty=obstacle_clearance_penalty,
                robot_clearance_penalty=robot_clearance_penalty,
                pedestrian_clearance_penalty=pedestrian_clearance_penalty,
                safety_intervention=safety_intervention_costs.get(name, 0.0),
                speed_reward=_progress_speed_reward(progress, scenario.dt),
            )
            rewards.append(reward)
            episode_rewards[name] += reward
            dones.append(float(reached[name] or failed[name]))

        prev_collision_flags = collision_flags
        for name in robot_names:
            if name not in states:
                continue
            prev_x, prev_y = last_positions.get(name, (states[name].x, states[name].y))
            path_lengths[name] += math.hypot(states[name].x - prev_x, states[name].y - prev_y)
            last_positions[name] = (states[name].x, states[name].y)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        done_tensor = torch.tensor(dones, dtype=torch.float32)
        bridge.get_robot_states()

        if log_interval > 0 and (step + 1) % log_interval == 0:
            avg_speed = sum(
                math.hypot(states[name].v, states[name].vy)
                for name in robot_names
            )
            avg_speed /= max(len(robot_names), 1)
            print(
                f"  progress step={step + 1}/{max_steps} "
                f"reached={sum(1 for v in reached.values() if v)}/"
                f"{len(robot_names)} "
                f"failed={sum(1 for v in failed.values() if v)}/"
                f"{len(robot_names)} "
                f"collisions={sum(collisions.values())} "
                f"avg_speed={avg_speed:.3f}",
                flush=True,
            )

        buffer_kwargs = {
            "adjacency": adjacency.detach().cpu(),
        }
        if raw_action is not None:
            buffer_kwargs["raw_action"] = raw_action.detach().cpu()
        trainer.buffer.add(
            obs_tensor,
            action.detach().cpu(),
            reward_tensor,
            done_tensor,
            log_prob.detach().cpu(),
            value.detach().cpu(),
            **buffer_kwargs,
        )

        if (
            len(trainer.buffer.observations) >= cfg.rollout_steps
            and not defer_quality_gated_update
        ):
            with torch.no_grad():
                last_obs_dict, last_states, _ = _make_observations(bridge, goals)
                last_obs, last_adj = _make_tensors(
                    last_obs_dict,
                    robot_names,
                    algorithm_config=algorithm_config,
                    states=last_states,
                    goals=goals,
                    control_profile=control_profile,
                )
                policy_last_obs = last_obs.to(device)
                policy_last_adj = last_adj.to(device)
                _, _, _, last_value = trainer.policy.get_action_and_value(
                    policy_last_obs.unsqueeze(0),
                    policy_last_adj.unsqueeze(0),
                    deterministic=True,
                )
            loss_info = trainer.update(last_obs, last_adj)
            last_loss_info = dict(loss_info)
            print(
                f"  step={step} "
                f"policy_loss={loss_info['policy_loss']:.4f} "
                f"value_loss={loss_info['value_loss']:.4f} "
                f"entropy={loss_info['entropy']:.4f}"
            )

        if all(reached[name] or failed[name] for name in robot_names):
            break

    success_count_for_update = sum(1 for value in reached.values() if value)
    collision_count_for_update = sum(collisions.values())
    update_allowed = True
    if (
        min_success_for_update is not None
        and success_count_for_update < min_success_for_update
    ):
        update_allowed = False
        update_skip_reason = (
            f"rollout_success {success_count_for_update}/"
            f"{len(robot_names)} below required {min_success_for_update}/"
            f"{len(robot_names)}"
        )
    if (
        max_collisions_for_update is not None
        and collision_count_for_update > max_collisions_for_update
    ):
        update_allowed = False
        collision_reason = (
            f"rollout_collisions {collision_count_for_update} above "
            f"allowed {max_collisions_for_update}"
        )
        update_skip_reason = (
            f"{update_skip_reason}; {collision_reason}"
            if update_skip_reason else collision_reason
        )

    if len(trainer.buffer.observations) > 0 and update_allowed:
        with torch.no_grad():
            last_obs_dict, last_states, _ = _make_observations(bridge, goals)
            last_obs, last_adj = _make_tensors(
                last_obs_dict,
                robot_names,
                algorithm_config=algorithm_config,
                states=last_states,
                goals=goals,
                control_profile=control_profile,
            )
        loss_info = trainer.update(last_obs, last_adj)
        last_loss_info = dict(loss_info)
    elif len(trainer.buffer.observations) > 0:
        trainer.buffer.clear()
        update_skipped = True
        print(f"  PPO update skipped: {update_skip_reason}", flush=True)

    final_states = bridge.get_robot_states()
    final_laser = bridge.get_laser_data()
    final_tracking_states = (
        waypoint_manager.tracking_states(final_states, goals)
        if waypoint_manager is not None
        and hasattr(waypoint_manager, "tracking_states")
        else {}
    )
    robot_outcomes = {}
    for name in robot_names:
        state = final_states.get(name)
        goal = goals[name]
        laser = final_laser.get(name)
        tracking = final_tracking_states.get(name)
        if state is None:
            robot_outcomes[name] = {
                "reached": reached.get(name, False),
                "failed": failed.get(name, False),
                "missing_state": True,
            }
            continue
        robot_outcomes[name] = {
            "reached": reached.get(name, False),
            "failed": failed.get(name, False),
            "collision_category": collision_details.get(name, {}).get("category", ""),
            "collision_nearest_robot": (
                collision_details.get(name, {}).get("nearest_robot", 999.0)
            ),
            "collision_nearest_pedestrian": (
                collision_details.get(name, {}).get("nearest_pedestrian", 999.0)
            ),
            "x": state.x,
            "y": state.y,
            "goal_distance": math.hypot(state.x - goal.x, state.y - goal.y),
            "speed": math.hypot(state.v, state.vy),
            "min_laser": (
                getattr(laser, "min_range", 999.0)
                if laser is not None else 999.0
            ),
            "path_progress": (
                tracking.progress if tracking is not None else 0.0
            ),
            "remaining_path_distance": (
                tracking.remaining_distance if tracking is not None else 0.0
            ),
            "deadlock_count": deadlock_counters.get(name, 0),
            "collision_count": collisions.get(name, 0),
        }

    bridge.stop_all()
    details = {
        "episode_reward": sum(episode_rewards.values()),
        "mean_agent_reward": (
            sum(episode_rewards.values()) / max(len(robot_names), 1)
        ),
        "episode_steps": steps_run,
        "average_path_length": (
            sum(path_lengths.values()) / max(len(robot_names), 1)
        ),
        "average_completion_time": (
            sum(completion_times.values()) / len(completion_times)
            if completion_times else 0.0
        ),
        "policy_loss": last_loss_info.get("policy_loss", 0.0),
        "value_loss": last_loss_info.get("value_loss", 0.0),
        "entropy": last_loss_info.get("entropy", 0.0),
        "update_skipped": int(update_skipped),
        "update_skip_reason": update_skip_reason,
        "robot_outcomes": robot_outcomes,
    }
    if return_details:
        return collisions, deadlock_counters, reached, failed, details
    return collisions, deadlock_counters, reached, failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train MO-GAT-MAPPO for multi-robot path planning."
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument(
        "--algorithm",
        choices=CHECKPOINT_ALGORITHMS,
        default="mo_gat_mappo",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", default="models/checkpoints/mo_gat_mappo.pt")
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
            "Disable the reciprocal-avoidance velocity prior during training "
            "and deterministic validation."
        ),
    )
    parser.add_argument(
        "--disable-safety-filter",
        action="store_true",
        help=(
            "Disable post-policy laser and robot-proximity safety filters, "
            "and disable command-level pedestrian proximity limiting."
        ),
    )
    parser.add_argument(
        "--variant-label",
        default="",
        help=(
            "Optional experiment variant label written to checkpoints, config "
            "snapshots, and training logs."
        ),
    )
    parser.add_argument("--keep-gazebo", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--training-log", default=None)
    parser.add_argument(
        "--validation-log",
        default=None,
        help="Optional CSV path for one row per checkpoint-selection validation.",
    )
    parser.add_argument("--tensorboard-dir", default=None)
    parser.add_argument(
        "--min-success-for-update",
        type=int,
        default=None,
        help=(
            "Skip PPO updates for rollout episodes with fewer than this many "
            "successful robots. Disabled by default."
        ),
    )
    parser.add_argument(
        "--max-collisions-for-update",
        type=int,
        default=None,
        help=(
            "Skip PPO updates for rollout episodes with more than this many "
            "collisions. Disabled by default."
        ),
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=0,
        help=(
            "Run a deterministic no-exploration validation episode every N "
            "training episodes. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--validation-log-interval",
        type=int,
        default=0,
        help="Progress print interval for deterministic validation episodes.",
    )
    parser.add_argument(
        "--no-validation-rollback",
        action="store_true",
        help=(
            "Keep training from the current policy even when deterministic "
            "validation regresses below the best checkpoint."
        ),
    )
    parser.add_argument(
        "--initial-validation-diagnostic-only",
        action="store_true",
        help=(
            "Run the initial deterministic validation for diagnostics but do "
            "not allow the untrained episode-0 policy to become the best "
            "checkpoint."
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help=(
            "Resume a checkpoint from the same scenario. Cross-scenario "
            "resume is rejected."
        ),
    )
    parser.add_argument("--no-config-snapshot", action="store_true")
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be greater than zero")
    if args.log_interval < 0:
        parser.error("--log-interval must be zero or greater")
    if args.validation_interval < 0:
        parser.error("--validation-interval must be zero or greater")
    if args.validation_log_interval < 0:
        parser.error("--validation-log-interval must be zero or greater")
    if args.min_success_for_update is not None and args.min_success_for_update < 0:
        parser.error("--min-success-for-update must be zero or greater")
    if args.max_collisions_for_update is not None and args.max_collisions_for_update < 0:
        parser.error("--max-collisions-for-update must be zero or greater")
    if args.gui and args.sim_step_mode == "fast":
        parser.error("--sim-step-mode fast is only supported in headless mode")
    if args.gazebo_update_rate is None and args.sim_step_mode == "fast":
        args.gazebo_update_rate = 10000.0

    scenario_path = Path(args.scenario)
    scenario = _override_max_steps(load_scenario(scenario_path), args.max_steps)
    algorithm_config = get_algorithm_config(args.algorithm)
    _set_random_seed(args.seed)
    args.device = _resolve_device(args.device)
    control_config = {
        "orca_prior_enabled": not args.disable_orca_prior,
        "safety_filter_enabled": not args.disable_safety_filter,
        "variant": args.variant_label,
    }
    from ament_index_python.packages import get_package_share_directory

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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_config = output_path.parent / "tmp_robot_config.yaml"

    robot_names = build_robot_config_yaml(scenario_path, tmp_config)
    print(
        f"Scenario: {scenario.name} | Robots: {len(robot_names)} | "
        f"Max steps: {scenario.max_steps} | Seed: {args.seed} | "
        f"Algorithm: {algorithm_config.name} | Device: {args.device} | "
        f"ORCA prior: {control_config['orca_prior_enabled']} | "
        f"Safety filter: {control_config['safety_filter_enabled']}"
    )
    print(f"World: {world_file}")

    import os
    import subprocess
    import time

    gazebo_pkg = Path(get_package_share_directory("mrpp_gazebo"))
    os.environ["IGN_GAZEBO_RESOURCE_PATH"] = str(gazebo_pkg / "models")

    global_cleanup = os.environ.get("MRPP_GLOBAL_PROCESS_CLEANUP", "1") == "1"
    if global_cleanup:
        print("Cleaning up old processes...")
        subprocess.run(["pkill", "-9", "-f", "ros_gz_bridge"], capture_output=True)
        subprocess.run(["pkill", "-9", "-f", "parameter_bridge"], capture_output=True)
        subprocess.run(["pkill", "-9", "-f", "ign gazebo"], capture_output=True)
        time.sleep(2)
    else:
        print("Global process cleanup disabled for isolated concurrent job.")

    print("Starting Gazebo...")
    gazebo_proc = launch_gazebo(
        world_file,
        gui=args.gui,
        update_rate=args.gazebo_update_rate,
        run_on_start=args.sim_step_mode != "fast",
    )
    time.sleep(8)
    ensure_processes_running([gazebo_proc], "Gazebo")

    print("Spawning robots...")
    spawn_proc = spawn_robots(tmp_config, robot_model, delay=0.0)
    spawn_code = spawn_proc.wait(timeout=30)
    if spawn_code != 0:
        raise RuntimeError(f"Robot spawning failed with exit code {spawn_code}")
    time.sleep(2)

    ped_names = []
    if scenario.world in ("pedestrian_dynamic", "mixed_complex"):
        print("Spawning pedestrians...")
        from mrpp_gazebo.pedestrian_controller import spawn_pedestrians
        from ament_index_python.packages import get_package_share_directory
        gazebo_pkg = Path(get_package_share_directory("mrpp_gazebo"))
        ped_sdf = gazebo_pkg / "models" / "pedestrian_fuel" / "model.sdf"
        ped_names = spawn_pedestrians(scenario.world, ped_sdf)
        print(f"  Spawned {len(ped_names)} pedestrians")

    print("Launching topic bridges...")
    bridge_procs = launch_topic_bridges(
        robot_names, ped_names, world_name=scenario.world
    )
    time.sleep(5)
    ensure_processes_running(bridge_procs, "ros_gz_bridge")

    rclpy.init()
    bridge = GazeboBridge(robot_names, ped_names, world_name=scenario.world)
    sim_stepper = SimulationStepper(
        bridge,
        scenario.dt,
        mode=args.sim_step_mode,
    )
    sim_stepper.initialize()
    if args.sim_step_mode == "fast":
        sim_stepper.synchronize_after_reset(resume=False)
    bridge.wait_until_ready(timeout_sec=20.0)
    if ped_names:
        from mrpp_gazebo.pedestrian_controller import get_pedestrian_configs
        bridge.set_pedestrian_configs(get_pedestrian_configs(scenario.world))

    start_positions = {
        rt.name: (rt.start.x, rt.start.y, rt.start.yaw)
        for rt in scenario.robots
    }
    bridge.set_start_positions(start_positions)

    goals = {}
    for robot_task in scenario.robots:
        goals[robot_task.name] = GoalState(
            x=robot_task.goal.x, y=robot_task.goal.y, yaw=robot_task.goal.yaw
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

    print(f"Simulation step mode: {args.sim_step_mode}")
    if args.validation_interval == 0:
        sim_stepper.resume()

    if algorithm_config.update_mode == "maddpg":
        policy = MADDPGPolicy(node_dim=OBS_DIM, hidden_dim=args.hidden_dim)
        config = MADDPGConfig(rollout_steps=scenario.max_steps + 1)
        trainer = MADDPGTrainer(policy=policy, config=config, device=args.device)
    else:
        policy = MOGATMAPPOPolicy(
            node_dim=OBS_DIM,
            hidden_dim=args.hidden_dim,
            centralized_critic=algorithm_config.use_centralized_critic,
        )
        config = mappo_config_for_algorithm(
            algorithm_config,
            rollout_steps=scenario.max_steps + 1,
        )
        trainer = MAPPOTrainer(policy=policy, config=config, device=args.device)
    if args.resume_checkpoint:
        loaded = _load_training_checkpoint(
            Path(args.resume_checkpoint),
            algorithm_config.name,
            policy,
            trainer,
            scenario_name=scenario.name,
        )
        print(
            "Resumed checkpoint "
            f"{args.resume_checkpoint} "
            f"(episode={loaded.get('episode', 'unknown')})",
            flush=True,
        )
    training_log = (
        Path(args.training_log)
        if args.training_log else output_path.parent / "training_log.csv"
    )
    validation_log = (
        Path(args.validation_log)
        if args.validation_log else output_path.parent / "validation.csv"
    )
    if not args.no_config_snapshot:
        _write_config_snapshot(
            output_path.parent,
            args,
            scenario,
            scenario_path,
            world_file,
            robot_model,
            algorithm_config,
            config,
        )
        _write_environment_snapshot(output_path.parent)

    summary_writer = None
    if args.tensorboard_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            summary_writer = SummaryWriter(args.tensorboard_dir)
        except Exception as exc:
            print(f"TensorBoard disabled: {exc}", flush=True)

    running = True

    def _cleanup():
        try:
            bridge.stop_all()
        except Exception:
            pass
        try:
            bridge.destroy_node()
        except Exception:
            pass
        stop_processes(bridge_procs)
        if not args.keep_gazebo:
            gazebo_proc.terminate()

    def _signal_handler(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    import atexit
    atexit.register(_cleanup)

    print(f"Starting training for {args.episodes} episodes...")
    best_score: tuple | None = None
    best_output_path = _best_checkpoint_path(output_path)
    best_policy_state = None
    best_optimizer_state = None
    best_episode: int | None = None
    best_selection: dict[str, object] | None = None
    if args.validation_interval > 0:
        print("Running initial deterministic validation...")
        sim_stepper.before_reset()
        bridge.reset(start_positions)
        if waypoint_manager is not None:
            waypoint_manager.reset()
        sim_stepper.synchronize_after_reset()
        initial_metrics = _run_deterministic_validation(
            bridge=bridge,
            policy=policy,
            scenario=scenario,
            robot_names=robot_names,
            goals=goals,
            seed=args.seed,
            log_interval=args.validation_log_interval,
            waypoint_manager=waypoint_manager,
            algorithm_config=algorithm_config,
            sim_stepper=sim_stepper,
            use_orca_prior=control_config["orca_prior_enabled"],
            use_safety_filter=control_config["safety_filter_enabled"],
        )
        initial_success_count = sum(
            1 for value in initial_metrics.successes.values() if value
        )
        print(
            f"  Initial Eval: {initial_success_count}/{len(robot_names)} | "
            f"Collisions: {initial_metrics.collision_count} | "
            f"Timeouts: {initial_metrics.timeout_failure_count}"
        )
        if args.initial_validation_diagnostic_only:
            print("  Initial validation is diagnostic only; no checkpoint saved.")
        else:
            best_score = _validation_selection_score(
                initial_metrics,
                len(robot_names),
            )
            best_policy_state = copy.deepcopy(policy.state_dict())
            best_optimizer_state = copy.deepcopy(
                _trainer_optimizer_state_dict(trainer)
            )
            _save_training_checkpoint(
                best_output_path,
                0,
                best_policy_state,
                best_optimizer_state,
                algorithm_config,
                scenario.name,
                trainer,
                selection=_selection_summary(
                    initial_metrics,
                    len(robot_names),
                    "initial_deterministic_validation",
                ),
                control_config=control_config,
                training_seed=args.seed,
            )
            best_episode = 0
            best_selection = _selection_summary(
                initial_metrics,
                len(robot_names),
                "initial_deterministic_validation",
            )
            print(f"  Initial best checkpoint saved to {best_output_path}")
        sim_stepper.before_reset()
        bridge.reset(start_positions)
        if waypoint_manager is not None:
            waypoint_manager.reset()
        sim_stepper.synchronize_after_reset()

    for episode in range(args.episodes):
        if not running:
            break
        if episode > 0:
            sim_stepper.before_reset()
            bridge.reset(start_positions)
            if waypoint_manager is not None:
                waypoint_manager.reset()
            sim_stepper.synchronize_after_reset()
        episode_start_policy_state = copy.deepcopy(policy.state_dict())
        episode_start_optimizer_state = copy.deepcopy(
            _trainer_optimizer_state_dict(trainer)
        )
        print(f"\nEpisode {episode + 1}/{args.episodes}")
        episode_wall_start = time.time()
        collisions, deadlocks, reached, failed, details = train_episode(
            bridge,
            trainer,
            scenario,
            robot_names,
            goals,
            log_interval=args.log_interval,
            waypoint_manager=waypoint_manager,
            algorithm_config=algorithm_config,
            min_success_for_update=args.min_success_for_update,
            max_collisions_for_update=args.max_collisions_for_update,
            sim_stepper=sim_stepper,
            return_details=True,
            use_orca_prior=control_config["orca_prior_enabled"],
            use_safety_filter=control_config["safety_filter_enabled"],
        )
        episode_wall_time = time.time() - episode_wall_start
        success_count = sum(1 for v in reached.values() if v)
        failure_count = sum(1 for v in failed.values() if v)
        deadlock_steps = sum(deadlocks.values())
        total_collisions = sum(collisions.values())
        print(
            f"  Success: {success_count}/{len(robot_names)} | "
            f"Failed: {failure_count}/{len(robot_names)} | "
            f"Collisions: {total_collisions} | "
            f"DeadlockSteps: {deadlock_steps}"
        )
        for name, outcome in details.get("robot_outcomes", {}).items():
            print(
                "  outcome "
                f"{name}: reached={outcome.get('reached')} "
                f"failed={outcome.get('failed')} "
                f"dist={float(outcome.get('goal_distance', 999.0)):.3f} "
                f"pos=({float(outcome.get('x', 0.0)):.2f},"
                f"{float(outcome.get('y', 0.0)):.2f}) "
                f"speed={float(outcome.get('speed', 0.0)):.3f} "
                f"min_laser={float(outcome.get('min_laser', 999.0)):.3f} "
                f"path={float(outcome.get('path_progress', 0.0)):.2f} "
                f"remaining={float(outcome.get('remaining_path_distance', 0.0)):.2f} "
                f"deadlocks={outcome.get('deadlock_count', 0)}",
                flush=True,
            )
        checkpoint_for_log = str(output_path)
        selection_source = ""
        selection_score = None
        checkpoint_policy_state = None
        checkpoint_optimizer_state = None
        rollout_success_rate = success_count / max(len(robot_names), 1)
        eval_success_rate = ""
        eval_collision_count = ""
        eval_average_path_length = ""
        eval_average_completion_time = ""

        validation_metrics = None
        checkpoint_selected = False
        if (
            args.validation_interval > 0
            and (
                (episode + 1) % args.validation_interval == 0
                or episode + 1 == args.episodes
            )
        ):
            print("  Running deterministic validation...")
            sim_stepper.before_reset()
            bridge.reset(start_positions)
            if waypoint_manager is not None:
                waypoint_manager.reset()
            sim_stepper.synchronize_after_reset()
            validation_metrics = _run_deterministic_validation(
                bridge=bridge,
                policy=policy,
                scenario=scenario,
                robot_names=robot_names,
                goals=goals,
                seed=args.seed,
                log_interval=args.validation_log_interval,
                waypoint_manager=waypoint_manager,
                algorithm_config=algorithm_config,
                sim_stepper=sim_stepper,
                use_orca_prior=control_config["orca_prior_enabled"],
                use_safety_filter=control_config["safety_filter_enabled"],
            )
            eval_success_count = sum(
                1 for value in validation_metrics.successes.values() if value
            )
            eval_success_rate = validation_metrics.success_rate()
            eval_collision_count = validation_metrics.collision_count
            eval_average_path_length = validation_metrics.average_path_length()
            eval_average_completion_time = (
                validation_metrics.average_completion_time()
            )
            print(
                f"  Eval: {eval_success_count}/{len(robot_names)} | "
                f"Collisions: {validation_metrics.collision_count} | "
                f"Timeouts: {validation_metrics.timeout_failure_count}"
            )
            selection_source = "deterministic_validation"
            selection_score = _validation_selection_score(
                validation_metrics,
                len(robot_names),
            )
            checkpoint_policy_state = copy.deepcopy(policy.state_dict())
            checkpoint_optimizer_state = copy.deepcopy(
                _trainer_optimizer_state_dict(trainer)
            )
        elif args.validation_interval == 0:
            selection_source = "rollout_episode_start_policy"
            selection_score = _rollout_selection_score(
                success_count,
                total_collisions,
                failure_count,
                deadlock_steps,
                details,
            )
            checkpoint_policy_state = episode_start_policy_state
            checkpoint_optimizer_state = episode_start_optimizer_state

        if (
            selection_score is not None
            and (best_score is None or selection_score > best_score)
            and checkpoint_policy_state is not None
            and checkpoint_optimizer_state is not None
        ):
            checkpoint_selected = True
            best_score = selection_score
            checkpoint_for_log = str(best_output_path)
            best_policy_state = checkpoint_policy_state
            best_optimizer_state = checkpoint_optimizer_state
            if validation_metrics is not None:
                selection = _selection_summary(
                    validation_metrics,
                    len(robot_names),
                    selection_source,
                )
            else:
                selection = {
                    "success_count": success_count,
                    "success_rate": rollout_success_rate,
                    "failure_count": failure_count,
                    "collision_count": total_collisions,
                    "deadlock_steps": deadlock_steps,
                    "robot_count": len(robot_names),
                    "source": selection_source,
                }
            _save_training_checkpoint(
                best_output_path,
                episode + 1,
                checkpoint_policy_state,
                checkpoint_optimizer_state,
                algorithm_config,
                scenario.name,
                trainer,
                selection=selection,
                control_config=control_config,
                training_seed=args.seed,
            )
            best_episode = episode + 1
            best_selection = selection
            print(f"  Best checkpoint saved to {best_output_path}")
        elif (
            validation_metrics is not None
            and best_policy_state is not None
            and best_optimizer_state is not None
            and not args.no_validation_rollback
        ):
            policy.load_state_dict(best_policy_state)
            _load_trainer_optimizer_state_dict(trainer, best_optimizer_state)
            print("  Validation regressed; restored best checkpoint state")

        if validation_metrics is not None:
            validation_summary = _selection_summary(
                validation_metrics,
                len(robot_names),
                "deterministic_validation",
            )
            validation_index = (episode + 1) // args.validation_interval
            _append_validation_log(
                validation_log,
                {
                    "training_seed": args.seed,
                    "validation_index": validation_index,
                    "training_episode": episode + 1,
                    "success": int(
                        validation_summary["success_count"] == len(robot_names)
                    ),
                    "success_count": validation_summary["success_count"],
                    "collision": validation_summary["collision_count"],
                    "timeout": validation_summary["timeout_failure_count"],
                    "deadlock": validation_summary["deadlock_failure_count"],
                    "average_completion_time": validation_summary[
                        "average_completion_time"
                    ],
                    "makespan": validation_summary["makespan"],
                    "average_waiting_time": validation_summary[
                        "average_waiting_time"
                    ],
                    "average_path_length": validation_summary[
                        "average_path_length"
                    ],
                    "trajectory_straightness": validation_summary[
                        "average_path_efficiency"
                    ],
                    "min_robot_robot_distance": validation_summary[
                        "min_robot_robot_distance"
                    ],
                    "min_robot_pedestrian_distance": validation_summary[
                        "min_robot_pedestrian_distance"
                    ],
                    "selection_score": json.dumps(list(selection_score)),
                    "is_selected_checkpoint": int(checkpoint_selected),
                    "checkpoint_path": str(best_output_path),
                },
            )
            _write_json(
                output_path.parent / "checkpoint_selection.json",
                {
                    "training_seed": args.seed,
                    "best_training_episode": best_episode,
                    "best_selection_score": (
                        list(best_score) if best_score is not None else None
                    ),
                    "best_selection": best_selection,
                    "checkpoint_path": str(best_output_path),
                    "validation_records": validation_index,
                },
            )

        if (episode + 1) % 10 == 0:
            _save_training_checkpoint(
                output_path,
                episode + 1,
                policy.state_dict(),
                _trainer_optimizer_state_dict(trainer),
                algorithm_config,
                scenario.name,
                trainer,
                control_config=control_config,
                training_seed=args.seed,
            )
            print(f"  Checkpoint saved to {output_path}")
        lr = _trainer_learning_rate(trainer)
        log_row = {
            "episode": episode + 1,
            "seed": args.seed,
            "scenario": scenario.name,
            "algorithm": algorithm_config.name,
            "variant": args.variant_label,
            "orca_prior_enabled": int(control_config["orca_prior_enabled"]),
            "safety_filter_enabled": int(control_config["safety_filter_enabled"]),
            "episode_reward": details["episode_reward"],
            "mean_agent_reward": details["mean_agent_reward"],
            "rollout_success_rate": rollout_success_rate,
            "rollout_collision_count": total_collisions,
            "rollout_deadlock_count": deadlock_steps,
            "eval_success_rate": eval_success_rate,
            "eval_collision_count": eval_collision_count,
            "eval_average_path_length": eval_average_path_length,
            "eval_average_completion_time": eval_average_completion_time,
            "success_rate": (
                eval_success_rate
                if eval_success_rate != "" else rollout_success_rate
            ),
            "collision_count": (
                eval_collision_count
                if eval_collision_count != "" else total_collisions
            ),
            "deadlock_count": deadlock_steps,
            "average_path_length": (
                eval_average_path_length
                if eval_average_path_length != ""
                else details["average_path_length"]
            ),
            "average_completion_time": (
                eval_average_completion_time
                if eval_average_completion_time != ""
                else details["average_completion_time"]
            ),
            "episode_steps": details["episode_steps"],
            "policy_loss": details["policy_loss"],
            "value_loss": details["value_loss"],
            "entropy": details["entropy"],
            "update_skipped": details["update_skipped"],
            "update_skip_reason": details["update_skip_reason"],
            "learning_rate": lr,
            "wall_time": episode_wall_time,
            "checkpoint_path": checkpoint_for_log,
            "selection_source": selection_source,
        }
        _append_training_log(training_log, log_row)
        if summary_writer is not None:
            global_step = episode + 1
            for key in (
                "episode_reward",
                "rollout_success_rate",
                "rollout_collision_count",
                "eval_success_rate",
                "eval_collision_count",
                "success_rate",
                "collision_count",
                "policy_loss",
                "value_loss",
                "entropy",
            ):
                value = log_row[key]
                if value != "":
                    summary_writer.add_scalar(key, float(value), global_step)

    _save_training_checkpoint(
        output_path,
        args.episodes,
        policy.state_dict(),
        _trainer_optimizer_state_dict(trainer),
        algorithm_config,
        scenario.name,
        trainer,
        control_config=control_config,
        training_seed=args.seed,
    )
    print(f"\nTraining complete. Final model saved to {output_path}")
    if best_score is not None:
        print(f"Best model saved to {best_output_path}")
    if summary_writer is not None:
        summary_writer.close()

    _cleanup()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
