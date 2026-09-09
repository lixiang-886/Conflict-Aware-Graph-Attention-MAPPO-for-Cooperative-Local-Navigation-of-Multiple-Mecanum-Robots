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

from dataclasses import dataclass, replace
import math
import re

from mrpp_rl.environment import ObservationVector
from mrpp_rl.environment import GoalState, RobotState
from mrpp_rl.lidar import DEFAULT_SAFETY_CONFIG, LaserSafetyConfig
from mrpp_rl.lidar import LaserSectors, apply_laser_safety_filter
from mrpp_rl.lidar import SECTOR_NAMES
from mrpp_rl.lidar import apply_safety_filter_to_commands as _apply_laser_safety
from mrpp_rl.lidar import laser_observation_from_data
from mrpp_rl.lidar import laser_scan_to_sectors
from mrpp_navigation.orca import AgentState, VelocityCommand2D
from mrpp_navigation.orca import reciprocal_avoidance_velocity

DISTANCE_SCALE = 8.0
ROBOT_DISTANCE_SCALE = 5.0
LASER_SCALE = 10.0
PEDESTRIAN_SCALE = 10.0
MAX_ABS_ANGULAR_SPEED = 2.0
MAX_LINEAR_ACCEL = 1.2
MAX_ANGULAR_ACCEL = 3.0
HEADING_GAIN = 1.8
LINEAR_RESIDUAL_FRACTION = 0.2
ANGULAR_RESIDUAL_LIMIT = 0.5
AVOIDANCE_RADIUS = 1.05
AVOIDANCE_GAIN = 1.25
PRIORITY_YIELD_RADIUS = 1.05
PRIORITY_STOP_RADIUS = 0.50
PEDESTRIAN_AVOIDANCE_RADIUS = 1.40
PEDESTRIAN_AVOIDANCE_GAIN = 1.45
PEDESTRIAN_AVOIDANCE_TIME_HORIZON = 2.6
PEDESTRIAN_YIELD_RADIUS = 1.15
PEDESTRIAN_STOP_RADIUS = 0.55
PEDESTRIAN_PROXIMITY_HARD_STOP_RADIUS = 0.53
PEDESTRIAN_PROXIMITY_STOP_RADIUS = 0.74
PEDESTRIAN_PROXIMITY_SLOW_RADIUS = 1.35
PEDESTRIAN_PROXIMITY_TIME_HORIZON = 2.0
ROBOT_PROXIMITY_STOP_RADIUS = 0.58
ROBOT_PROXIMITY_SLOW_RADIUS = 0.95
ROBOT_PROXIMITY_PREDICTED_RADIUS = 0.64
ROBOT_PROXIMITY_TIME_HORIZON = 2.0
SAFETY_STOP_DISTANCE = 0.45
SAFETY_SLOW_DISTANCE = 1.00
SAFETY_EMERGENCY_STOP_DISTANCE = 0.25
MIXED_COMPLEX_SAFETY_CONFIG = LaserSafetyConfig(
    sector_statistic=DEFAULT_SAFETY_CONFIG.sector_statistic,
    percentile=DEFAULT_SAFETY_CONFIG.percentile,
    stale_timeout_sec=DEFAULT_SAFETY_CONFIG.stale_timeout_sec,
    stale_or_missing_policy=DEFAULT_SAFETY_CONFIG.stale_or_missing_policy,
    emergency_stop_distance=0.30,
    stop_distance=0.54,
    slow_distance=1.05,
    emergency_max_angular_speed=DEFAULT_SAFETY_CONFIG.emergency_max_angular_speed,
    max_angular_speed=DEFAULT_SAFETY_CONFIG.max_angular_speed,
    stop_rotation_at_emergency=DEFAULT_SAFETY_CONFIG.stop_rotation_at_emergency,
)
WORLD_SAFETY_CONFIGS = {
    "mixed_complex": MIXED_COMPLEX_SAFETY_CONFIG,
}


@dataclass(frozen=True)
class ControlProfile:
    linear_residual_fraction: float = LINEAR_RESIDUAL_FRACTION
    lateral_residual_fraction: float = LINEAR_RESIDUAL_FRACTION
    angular_residual_limit: float = ANGULAR_RESIDUAL_LIMIT
    robot_avoidance_radius: float = AVOIDANCE_RADIUS
    robot_avoidance_gain: float = AVOIDANCE_GAIN
    priority_yield_radius: float = PRIORITY_YIELD_RADIUS
    priority_stop_radius: float = PRIORITY_STOP_RADIUS
    min_priority_yield_scale: float = 0.0
    pedestrian_avoidance_radius: float = PEDESTRIAN_AVOIDANCE_RADIUS
    pedestrian_avoidance_gain: float = PEDESTRIAN_AVOIDANCE_GAIN
    pedestrian_avoidance_time_horizon: float = PEDESTRIAN_AVOIDANCE_TIME_HORIZON
    pedestrian_yield_radius: float = PEDESTRIAN_YIELD_RADIUS
    pedestrian_stop_radius: float = PEDESTRIAN_STOP_RADIUS
    pedestrian_proximity_hard_stop_radius: float = PEDESTRIAN_PROXIMITY_HARD_STOP_RADIUS
    pedestrian_proximity_stop_radius: float = PEDESTRIAN_PROXIMITY_STOP_RADIUS
    pedestrian_proximity_slow_radius: float = PEDESTRIAN_PROXIMITY_SLOW_RADIUS
    pedestrian_proximity_time_horizon: float = PEDESTRIAN_PROXIMITY_TIME_HORIZON
    robot_proximity_stop_radius: float = ROBOT_PROXIMITY_STOP_RADIUS
    robot_proximity_slow_radius: float = ROBOT_PROXIMITY_SLOW_RADIUS
    robot_proximity_predicted_radius: float = ROBOT_PROXIMITY_PREDICTED_RADIUS
    robot_proximity_time_horizon: float = ROBOT_PROXIMITY_TIME_HORIZON
    coordination_horizon: float = 0.0
    coordination_radius: float = 0.0
    coordination_lateral_speed: float = 0.0
    coordination_min_speed_scale: float = 1.0


DEFAULT_CONTROL_PROFILE = ControlProfile()
MOGAT_EFFICIENCY_PROFILE = ControlProfile(
    linear_residual_fraction=0.35,
    lateral_residual_fraction=0.35,
    angular_residual_limit=0.65,
    robot_avoidance_radius=1.00,
    robot_avoidance_gain=1.15,
    priority_yield_radius=0.92,
    priority_stop_radius=0.46,
    min_priority_yield_scale=0.25,
    pedestrian_avoidance_radius=1.25,
    pedestrian_avoidance_gain=1.35,
    pedestrian_avoidance_time_horizon=2.20,
    pedestrian_yield_radius=1.00,
    pedestrian_stop_radius=0.52,
    pedestrian_proximity_hard_stop_radius=0.50,
    pedestrian_proximity_stop_radius=0.68,
    pedestrian_proximity_slow_radius=1.15,
    pedestrian_proximity_time_horizon=1.50,
    robot_proximity_stop_radius=0.58,
    robot_proximity_slow_radius=0.94,
    robot_proximity_predicted_radius=0.64,
    robot_proximity_time_horizon=1.90,
    coordination_horizon=4.0,
    coordination_radius=0.92,
    coordination_lateral_speed=0.10,
    coordination_min_speed_scale=0.86,
)
MOGAT_STATIC_CLUTTER_PROFILE = replace(
    MOGAT_EFFICIENCY_PROFILE,
    lateral_residual_fraction=0.25,
    angular_residual_limit=0.0,
)
MOGAT_MIXED_COMPLEX_PROFILE = ControlProfile(
    linear_residual_fraction=0.18,
    lateral_residual_fraction=0.18,
    angular_residual_limit=0.45,
    robot_avoidance_radius=0.98,
    robot_avoidance_gain=1.08,
    priority_yield_radius=0.95,
    priority_stop_radius=0.48,
    min_priority_yield_scale=0.22,
    pedestrian_avoidance_radius=1.30,
    pedestrian_avoidance_gain=1.25,
    pedestrian_avoidance_time_horizon=1.80,
    pedestrian_yield_radius=1.05,
    pedestrian_stop_radius=0.56,
    pedestrian_proximity_hard_stop_radius=0.53,
    pedestrian_proximity_stop_radius=0.70,
    pedestrian_proximity_slow_radius=1.20,
    pedestrian_proximity_time_horizon=1.60,
    robot_proximity_stop_radius=0.56,
    robot_proximity_slow_radius=0.88,
    robot_proximity_predicted_radius=0.62,
    robot_proximity_time_horizon=1.75,
    coordination_horizon=3.5,
    coordination_radius=0.86,
    coordination_lateral_speed=0.14,
    coordination_min_speed_scale=0.72,
)
MOGAT_PROFILE_ALGORITHMS = {
    "ippo",
    "mappo",
    "maddpg",
    "matched_mappo",
    "gat_mappo",
    "sensors_mo_gat_mappo",
    "mo_gat_mappo",
    "fc_gat_mappo",
    "ablation_no_graph_attention",
    "ablation_fixed_reward",
    "ablation_no_safety_distance",
    "ablation_no_gate",
    "ablation_no_coordinator",
}


def control_profile_for_algorithm(
    algorithm_name: str | None,
    world_name: str | None = None,
) -> ControlProfile:
    if str(algorithm_name or "") in MOGAT_PROFILE_ALGORITHMS:
        if str(world_name or "") == "mixed_complex":
            return MOGAT_MIXED_COMPLEX_PROFILE
        if str(world_name or "") == "static_clutter":
            return MOGAT_STATIC_CLUTTER_PROFILE
        return MOGAT_EFFICIENCY_PROFILE
    return DEFAULT_CONTROL_PROFILE


@dataclass(frozen=True)
class CoordinationDirective:
    """A graph-coordinated preference applied before local collision avoidance."""

    preferred_speed_scale: float = 1.0
    world_lateral_vx: float = 0.0
    world_lateral_vy: float = 0.0
    priority_yield_scale: float | None = None


def conflict_aware_adjacency_matrix(
    robot_names: list[str],
    states: dict[str, RobotState],
    goals: dict[str, GoalState],
    max_speed: float,
    profile: ControlProfile,
    action_linear_by_name: dict[str, float] | None = None,
) -> list[list[float]]:
    """
    Build a sparse interaction graph from predicted robot conflicts.

    A complete graph makes every robot aggregate irrelevant state at every step.
    The cooperative policy instead exchanges messages only with robots whose
    nominal paths can enter the same local conflict region.
    """
    adjacency, _ = conflict_aware_graph(
        robot_names,
        states,
        goals,
        max_speed,
        profile,
        action_linear_by_name=action_linear_by_name,
    )
    return adjacency


def conflict_aware_graph(
    robot_names: list[str],
    states: dict[str, RobotState],
    goals: dict[str, GoalState],
    max_speed: float,
    profile: ControlProfile,
    action_linear_by_name: dict[str, float] | None = None,
) -> tuple[list[list[float]], list[dict[str, float | int | str]]]:
    """Return the CPA adjacency and per-pair diagnostics from one calculation."""
    count = len(robot_names)
    adjacency = [
        [1.0 if row == column else 0.0 for column in range(count)]
        for row in range(count)
    ]
    diagnostics: list[dict[str, float | int | str]] = []
    if (
        profile.coordination_horizon <= 0.0
        or profile.coordination_radius <= 0.0
        or max_speed <= 0.0
    ):
        return adjacency, diagnostics

    predicted = _nominal_goal_velocities(
        robot_names,
        states,
        goals,
        max_speed,
        profile,
        action_linear_by_name,
    )
    graph_horizon = profile.coordination_horizon * 1.25
    graph_radius = profile.coordination_radius * 1.35
    for index, name in enumerate(robot_names):
        robot = states.get(name)
        if robot is None:
            continue
        for other_index in range(index + 1, count):
            other_name = robot_names[other_index]
            other = states.get(other_name)
            if other is None:
                continue
            velocity = predicted.get(name, (0.0, 0.0))
            other_velocity = predicted.get(other_name, (0.0, 0.0))
            rel_x = other.x - robot.x
            rel_y = other.y - robot.y
            rel_vx = other_velocity[0] - velocity[0]
            rel_vy = other_velocity[1] - velocity[1]
            relative_distance = math.hypot(rel_x, rel_y)
            tau_cpa = _time_to_closest_approach(
                rel_x,
                rel_y,
                rel_vx,
                rel_vy,
            )
            if math.isfinite(tau_cpa):
                closest_x = rel_x + rel_vx * tau_cpa
                closest_y = rel_y + rel_vy * tau_cpa
                d_cpa = math.hypot(closest_x, closest_y)
            else:
                d_cpa = relative_distance
            conflict = _predicted_robot_conflict(
                robot,
                other,
                velocity,
                other_velocity,
                graph_horizon,
                graph_radius,
            )
            edge_active = int(conflict is not None)
            if edge_active:
                adjacency[index][other_index] = 1.0
                adjacency[other_index][index] = 1.0
            diagnostics.append(
                {
                    "pair_i": name,
                    "pair_j": other_name,
                    "relative_distance": relative_distance,
                    "relative_velocity_x": rel_vx,
                    "relative_velocity_y": rel_vy,
                    "relative_velocity": math.hypot(rel_vx, rel_vy),
                    "tau_cpa": tau_cpa,
                    "d_cpa": d_cpa,
                    "edge_active": edge_active,
                }
            )
    return adjacency, diagnostics


def conflict_aware_coordination(
    robot_names: list[str],
    states: dict[str, RobotState],
    goals: dict[str, GoalState],
    max_speed: float,
    profile: ControlProfile,
    action_linear_by_name: dict[str, float] | None = None,
) -> dict[str, CoordinationDirective]:
    """
    Assign local right-of-way and a bounded lateral passing preference.

    The rule is intentionally limited to predicted conflicts.  It replaces the
    legacy robot-name priority only for the graph-aware MO-GAT controller; the
    shared ORCA, laser, and proximity safety layers remain the final authority.
    """
    if (
        profile.coordination_horizon <= 0.0
        or profile.coordination_radius <= 0.0
        or profile.coordination_lateral_speed <= 0.0
        or max_speed <= 0.0
    ):
        return {name: CoordinationDirective() for name in robot_names}

    predicted = _nominal_goal_velocities(
        robot_names,
        states,
        goals,
        max_speed,
        profile,
        action_linear_by_name,
    )
    speed_scales = {name: 1.0 for name in robot_names}
    lateral_vectors = {name: (0.0, 0.0) for name in robot_names}
    lateral_risks = {name: -1.0 for name in robot_names}

    for index, name in enumerate(robot_names):
        robot = states.get(name)
        goal = goals.get(name)
        if robot is None or goal is None:
            continue
        for other_name in robot_names[index + 1:]:
            other = states.get(other_name)
            other_goal = goals.get(other_name)
            if other is None or other_goal is None:
                continue
            conflict = _predicted_robot_conflict(
                robot,
                other,
                predicted.get(name, (0.0, 0.0)),
                predicted.get(other_name, (0.0, 0.0)),
                profile.coordination_horizon,
                profile.coordination_radius,
            )
            if conflict is None:
                continue
            time_to_conflict, closest_distance = conflict
            leader_name, follower_name = _coordination_leader(
                name,
                other_name,
                robot,
                other,
                goal,
                other_goal,
                predicted.get(name, (0.0, 0.0)),
                predicted.get(other_name, (0.0, 0.0)),
            )
            leader = states[leader_name]
            follower = states[follower_name]
            leader_velocity = predicted.get(leader_name, (0.0, 0.0))
            risk = _coordination_risk(
                time_to_conflict,
                closest_distance,
                profile.coordination_horizon,
                profile.coordination_radius,
            )
            speed_scales[follower_name] = min(
                speed_scales[follower_name],
                1.0 - (1.0 - profile.coordination_min_speed_scale) * risk,
            )
            if risk > lateral_risks[follower_name]:
                lateral_vectors[follower_name] = _passing_lateral_velocity(
                    leader,
                    follower,
                    leader_velocity,
                    profile.coordination_lateral_speed * (0.55 + 0.45 * risk),
                    leader_name,
                    follower_name,
                )
                lateral_risks[follower_name] = risk

    return {
        name: CoordinationDirective(
            preferred_speed_scale=clamp(speed_scales[name], 0.0, 1.0),
            world_lateral_vx=lateral_vectors[name][0],
            world_lateral_vy=lateral_vectors[name][1],
            # The graph coordinator has already selected a right-of-way.  Do
            # not apply the unrelated, static robot-name yielding rule again.
            priority_yield_scale=1.0,
        )
        for name in robot_names
    }


def graph_policy_residual_gate(
    has_robot_conflict: bool,
    observation: ObservationVector,
    profile: ControlProfile,
) -> float:
    """
    Activate learned residual motion only around graph interactions.

    The waypoint and ORCA stack already handles nominal free-space travel.  A
    risk gate keeps the learned policy from adding persistent lateral or yaw
    noise on clear path segments while retaining authority near robot and
    pedestrian conflicts.
    """
    if has_robot_conflict:
        return 1.0
    activation_distance = max(
        profile.pedestrian_proximity_slow_radius * 1.35,
        profile.pedestrian_proximity_stop_radius + 0.20,
    )
    distance = float(observation.nearest_ped_distance)
    if not math.isfinite(distance) or distance >= activation_distance:
        return 0.0
    if distance <= profile.pedestrian_proximity_stop_radius:
        return 1.0
    span = max(
        activation_distance - profile.pedestrian_proximity_stop_radius,
        1e-6,
    )
    return clamp((activation_distance - distance) / span, 0.0, 1.0)


def gated_graph_policy_actions(
    action_linear: float,
    action_lateral: float,
    action_angular: float,
    interaction_gate: float,
) -> tuple[float, float, float]:
    """Keep clear-path residuals monotonic while preserving conflict authority."""
    gate = clamp(interaction_gate, 0.0, 1.0)
    linear = max(action_linear, 0.0) + min(action_linear, 0.0) * gate
    return linear, action_lateral * gate, action_angular * gate


def _nominal_goal_velocities(
    robot_names: list[str],
    states: dict[str, RobotState],
    goals: dict[str, GoalState],
    max_speed: float,
    profile: ControlProfile,
    action_linear_by_name: dict[str, float] | None,
) -> dict[str, tuple[float, float]]:
    velocities = {}
    for name in robot_names:
        robot = states.get(name)
        goal = goals.get(name)
        if robot is None or goal is None:
            continue
        dx = goal.x - robot.x
        dy = goal.y - robot.y
        distance = math.hypot(dx, dy)
        if distance <= 1e-9:
            velocities[name] = (0.0, 0.0)
            continue
        action_linear = 0.0
        if action_linear_by_name is not None:
            action_linear = action_linear_by_name.get(name, 0.0)
        speed_scale = 1.0 + profile.linear_residual_fraction * clamp(
            action_linear, -1.0, 1.0
        )
        speed = min(max_speed, distance) * clamp(speed_scale, 0.0, 1.5)
        velocities[name] = (speed * dx / distance, speed * dy / distance)
    return velocities


def _predicted_robot_conflict(
    robot: RobotState,
    other: RobotState,
    velocity: tuple[float, float],
    other_velocity: tuple[float, float],
    horizon: float,
    radius: float,
) -> tuple[float, float] | None:
    if horizon <= 0.0 or radius <= 0.0:
        return None
    rel_x = other.x - robot.x
    rel_y = other.y - robot.y
    rel_vx = other_velocity[0] - velocity[0]
    rel_vy = other_velocity[1] - velocity[1]
    rel_speed_sq = rel_vx * rel_vx + rel_vy * rel_vy
    if rel_speed_sq <= 1e-9:
        return None
    time_to_closest = -(rel_x * rel_vx + rel_y * rel_vy) / rel_speed_sq
    if time_to_closest <= 0.0 or time_to_closest > horizon:
        return None
    closest_x = rel_x + rel_vx * time_to_closest
    closest_y = rel_y + rel_vy * time_to_closest
    closest_distance = math.hypot(closest_x, closest_y)
    if closest_distance > radius:
        return None
    return time_to_closest, closest_distance


def _coordination_leader(
    name: str,
    other_name: str,
    robot: RobotState,
    other: RobotState,
    goal: GoalState,
    other_goal: GoalState,
    velocity: tuple[float, float],
    other_velocity: tuple[float, float],
) -> tuple[str, str]:
    """
    Prefer the robot that can clear its current segment first.

    Name ordering is only a deterministic tie-breaker for perfectly symmetric
    intersections, rather than the primary policy for every nearby pair.
    """
    own_speed = max(math.hypot(*velocity), 1e-6)
    other_speed = max(math.hypot(*other_velocity), 1e-6)
    own_eta = math.hypot(goal.x - robot.x, goal.y - robot.y) / own_speed
    other_eta = (
        math.hypot(other_goal.x - other.x, other_goal.y - other.y) / other_speed
    )
    if abs(own_eta - other_eta) > 0.20:
        return (name, other_name) if own_eta < other_eta else (other_name, name)
    if _priority_key(name) <= _priority_key(other_name):
        return name, other_name
    return other_name, name


def _coordination_risk(
    time_to_conflict: float,
    closest_distance: float,
    horizon: float,
    radius: float,
) -> float:
    temporal = 1.0 - clamp(time_to_conflict / max(horizon, 1e-6), 0.0, 1.0)
    clearance = 1.0 - clamp(closest_distance / max(radius, 1e-6), 0.0, 1.0)
    return clamp(0.30 + 0.70 * max(temporal, clearance), 0.0, 1.0)


def _passing_lateral_velocity(
    leader: RobotState,
    follower: RobotState,
    leader_velocity: tuple[float, float],
    speed: float,
    leader_name: str,
    follower_name: str,
) -> tuple[float, float]:
    heading_norm = math.hypot(*leader_velocity)
    if heading_norm <= 1e-9:
        heading_x = math.cos(leader.yaw)
        heading_y = math.sin(leader.yaw)
    else:
        heading_x = leader_velocity[0] / heading_norm
        heading_y = leader_velocity[1] / heading_norm
    normal_x = -heading_y
    normal_y = heading_x
    lateral_position = (
        (follower.x - leader.x) * normal_x
        + (follower.y - leader.y) * normal_y
    )
    if abs(lateral_position) <= 1e-6:
        side = 1.0 if _priority_key(follower_name) > _priority_key(leader_name) else -1.0
    else:
        side = 1.0 if lateral_position > 0.0 else -1.0
    return speed * side * normal_x, speed * side * normal_y


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def slew_rate_limit_command(
    command: tuple[float, float, float],
    previous: tuple[float, float, float],
    dt: float,
    max_linear_accel: float = MAX_LINEAR_ACCEL,
    max_angular_accel: float = MAX_ANGULAR_ACCEL,
) -> tuple[float, float, float]:
    if dt <= 0.0:
        return command
    vx, vy, omega = command
    prev_vx, prev_vy, prev_omega = previous
    delta_vx = vx - prev_vx
    delta_vy = vy - prev_vy
    delta_speed = math.hypot(delta_vx, delta_vy)
    max_delta_speed = max(0.0, max_linear_accel) * dt
    if delta_speed > max_delta_speed > 0.0:
        scale = max_delta_speed / delta_speed
        vx = prev_vx + delta_vx * scale
        vy = prev_vy + delta_vy * scale
    max_delta_omega = max(0.0, max_angular_accel) * dt
    omega = prev_omega + clamp(
        omega - prev_omega,
        -max_delta_omega,
        max_delta_omega,
    )
    return vx, vy, omega


def smooth_commands(
    commands: dict[str, tuple[float, float, float]],
    previous_commands: dict[str, tuple[float, float, float]],
    dt: float,
) -> dict[str, tuple[float, float, float]]:
    return {
        name: slew_rate_limit_command(
            command,
            previous_commands.get(name, (0.0, 0.0, 0.0)),
            dt,
        )
        for name, command in commands.items()
    }


def command_change_cost(
    command: tuple[float, float, float],
    previous: tuple[float, float, float],
    max_speed: float,
    max_angular_speed: float = MAX_ABS_ANGULAR_SPEED,
) -> float:
    linear_scale = max(max_speed, 1e-6)
    angular_scale = max(max_angular_speed, 1e-6)
    linear_delta = math.hypot(command[0] - previous[0], command[1] - previous[1])
    angular_delta = abs(command[2] - previous[2])
    return linear_delta / linear_scale + angular_delta / angular_scale


def safety_config_for_world(world_name: str | None) -> LaserSafetyConfig:
    return WORLD_SAFETY_CONFIGS.get(str(world_name or ""), DEFAULT_SAFETY_CONFIG)


def observation_to_features(obs: ObservationVector) -> list[float]:
    heading_error = wrap_angle(obs.heading_error)
    laser_range_max = max(float(obs.laser_range_max), 1e-6)
    laser_features = [
        clamp(float(getattr(obs, f"laser_{name}")) / laser_range_max, 0.0, 1.0)
        for name in SECTOR_NAMES
    ]
    return [
        clamp(obs.distance_to_goal / DISTANCE_SCALE, 0.0, 2.0),
        clamp(heading_error / math.pi, -1.0, 1.0),
        clamp(obs.nearest_robot_distance / ROBOT_DISTANCE_SCALE, 0.0, 2.0),
        clamp(obs.speed, -1.0, 1.0),
        clamp(obs.angular_speed / MAX_ABS_ANGULAR_SPEED, -1.0, 1.0),
        *laser_features,
        clamp(obs.nearest_ped_distance / PEDESTRIAN_SCALE, 0.0, 2.0),
        clamp(obs.nearest_ped_dx / PEDESTRIAN_SCALE, -1.0, 1.0),
        clamp(obs.nearest_ped_dy / PEDESTRIAN_SCALE, -1.0, 1.0),
    ]


def apply_static_obstacle_safety_filter(
    command: tuple[float, float, float],
    laser_sectors: LaserSectors,
    stop_distance: float = SAFETY_STOP_DISTANCE,
    slow_distance: float = SAFETY_SLOW_DISTANCE,
    emergency_stop_distance: float = SAFETY_EMERGENCY_STOP_DISTANCE,
) -> tuple[float, float, float]:
    config = LaserSafetyConfig(
        sector_statistic=DEFAULT_SAFETY_CONFIG.sector_statistic,
        percentile=DEFAULT_SAFETY_CONFIG.percentile,
        stale_timeout_sec=DEFAULT_SAFETY_CONFIG.stale_timeout_sec,
        stale_or_missing_policy=DEFAULT_SAFETY_CONFIG.stale_or_missing_policy,
        emergency_stop_distance=emergency_stop_distance,
        stop_distance=stop_distance,
        slow_distance=slow_distance,
        emergency_max_angular_speed=DEFAULT_SAFETY_CONFIG.emergency_max_angular_speed,
        max_angular_speed=DEFAULT_SAFETY_CONFIG.max_angular_speed,
        stop_rotation_at_emergency=DEFAULT_SAFETY_CONFIG.stop_rotation_at_emergency,
    )
    return apply_laser_safety_filter(command, laser_sectors, config).command


def laser_sectors_from_scan(
    ranges: list[float],
    angle_min: float,
    angle_increment: float,
    max_range: float = 999.0,
) -> LaserSectors:
    return laser_scan_to_sectors(
        ranges,
        angle_min,
        angle_increment,
        range_max=max_range,
    )


def laser_sectors_from_laser_data(laser_data) -> LaserSectors:
    observation = laser_observation_from_data(laser_data)
    return observation.sectors if observation is not None else LaserSectors(
        front=999.0,
        front_left=999.0,
        left=999.0,
        rear_left=999.0,
        rear=999.0,
        rear_right=999.0,
        right=999.0,
        front_right=999.0,
    )


def apply_safety_filter_to_commands(
    commands: dict[str, tuple[float, float, float]],
    laser_data: dict[str, object],
    config: LaserSafetyConfig = DEFAULT_SAFETY_CONFIG,
    current_time_sec: float | None = None,
) -> dict[str, tuple[float, float, float]]:
    return _apply_laser_safety(
        commands,
        laser_data,
        config=config,
        current_time_sec=current_time_sec,
    )


def apply_robot_proximity_safety_filter(
    commands: dict[str, tuple[float, float, float]],
    states: dict[str, RobotState],
    stop_radius: float = ROBOT_PROXIMITY_STOP_RADIUS,
    slow_radius: float = ROBOT_PROXIMITY_SLOW_RADIUS,
    predicted_radius: float = ROBOT_PROXIMITY_PREDICTED_RADIUS,
    time_horizon: float = ROBOT_PROXIMITY_TIME_HORIZON,
    pedestrians: list[object] | None = None,
    pedestrian_skip_radius: float = 0.0,
) -> dict[str, tuple[float, float, float]]:
    """Limit only the velocity component that closes robot-robot clearance."""
    adjusted_world = {}
    omega_by_name = {}
    for name, command in commands.items():
        state = states.get(name)
        if state is None:
            adjusted_world[name] = (command[0], command[1])
            omega_by_name[name] = command[2]
            continue
        adjusted_world[name] = (
            _body_to_world_vx(command[0], command[1], state.yaw),
            _body_to_world_vy(command[0], command[1], state.yaw),
        )
        omega_by_name[name] = command[2]

    names = sorted(name for name in commands if name in states)
    for index, name in enumerate(names):
        robot = states[name]
        for other_name in names[index + 1:]:
            other = states[other_name]
            if pedestrians and pedestrian_skip_radius > 0.0:
                if (
                    _nearest_pedestrian_distance(robot, pedestrians)
                    <= pedestrian_skip_radius
                    or _nearest_pedestrian_distance(other, pedestrians)
                    <= pedestrian_skip_radius
                ):
                    continue
            rel_x = other.x - robot.x
            rel_y = other.y - robot.y
            distance = math.hypot(rel_x, rel_y)
            if distance <= 1e-9:
                adjusted_world[name] = (0.0, 0.0)
                adjusted_world[other_name] = (0.0, 0.0)
                continue

            vx, vy = adjusted_world[name]
            other_vx, other_vy = adjusted_world[other_name]
            rel_vx = other_vx - vx
            rel_vy = other_vy - vy
            time_to_close = _time_to_closest_approach(
                rel_x,
                rel_y,
                rel_vx,
                rel_vy,
            )
            closest_distance = distance
            predicted_risk = False
            if 0.0 < time_to_close <= time_horizon:
                closest_x = rel_x + rel_vx * time_to_close
                closest_y = rel_y + rel_vy * time_to_close
                closest_distance = math.hypot(closest_x, closest_y)
                predicted_risk = closest_distance < predicted_radius

            closing_now = (
                rel_x * rel_vx + rel_y * rel_vy
            ) / distance < -1e-4
            if distance >= slow_radius and not (closing_now and predicted_risk):
                continue

            risk_distance = min(distance, closest_distance)
            scale = _robot_proximity_scale(
                risk_distance,
                stop_radius=stop_radius,
                slow_radius=slow_radius,
            )
            unit_x = rel_x / distance
            unit_y = rel_y / distance
            if distance <= stop_radius:
                adjusted_world[name] = _scale_toward_neighbor_component(
                    adjusted_world[name],
                    unit_x,
                    unit_y,
                    scale,
                )
                adjusted_world[other_name] = _scale_toward_neighbor_component(
                    adjusted_world[other_name],
                    -unit_x,
                    -unit_y,
                    scale,
                )
            elif _priority_key(name) <= _priority_key(other_name):
                adjusted_world[other_name] = _scale_toward_neighbor_component(
                    adjusted_world[other_name],
                    -unit_x,
                    -unit_y,
                    scale,
                )
            else:
                adjusted_world[name] = _scale_toward_neighbor_component(
                    adjusted_world[name],
                    unit_x,
                    unit_y,
                    scale,
                )

    filtered = {}
    for name, command in commands.items():
        state = states.get(name)
        world_vx, world_vy = adjusted_world.get(name, (command[0], command[1]))
        if state is None:
            filtered[name] = (world_vx, world_vy, omega_by_name.get(name, command[2]))
            continue
        filtered[name] = (
            _world_to_body_vx(world_vx, world_vy, state.yaw),
            _world_to_body_vy(world_vx, world_vy, state.yaw),
            omega_by_name.get(name, command[2]),
        )
    return filtered


def apply_pedestrian_proximity_safety_filter(
    commands: dict[str, tuple[float, float, float]],
    states: dict[str, RobotState],
    pedestrians: list[object],
    max_speed: float,
    profile: ControlProfile,
) -> dict[str, tuple[float, float, float]]:
    """Apply the command-level robot--pedestrian limiter as an auditable stage."""
    if not pedestrians:
        return dict(commands)
    filtered: dict[str, tuple[float, float, float]] = {}
    for name, command in commands.items():
        robot = states.get(name)
        if robot is None:
            filtered[name] = command
            continue
        world_vx = _body_to_world_vx(command[0], command[1], robot.yaw)
        world_vy = _body_to_world_vy(command[0], command[1], robot.yaw)
        escape_vx, escape_vy = pedestrian_escape_velocity(
            robot,
            pedestrians,
            world_vx,
            world_vy,
            max_speed=max_speed,
            stop_radius=profile.pedestrian_proximity_stop_radius,
            slow_radius=profile.pedestrian_proximity_slow_radius,
        )
        world_vx += escape_vx
        world_vy += escape_vy
        speed = math.hypot(world_vx, world_vy)
        if speed > max_speed > 0.0:
            speed_scale = max_speed / speed
            world_vx *= speed_scale
            world_vy *= speed_scale
        scale = pedestrian_proximity_scale(
            robot,
            pedestrians,
            world_vx,
            world_vy,
            hard_stop_radius=profile.pedestrian_proximity_hard_stop_radius,
            stop_radius=profile.pedestrian_proximity_stop_radius,
            slow_radius=profile.pedestrian_proximity_slow_radius,
            time_horizon=profile.pedestrian_proximity_time_horizon,
        )
        filtered[name] = (
            _world_to_body_vx(world_vx * scale, world_vy * scale, robot.yaw),
            _world_to_body_vy(world_vx * scale, world_vy * scale, robot.yaw),
            command[2],
        )
    return filtered


def residual_goal_command(
    action_linear: float,
    action_angular: float,
    obs: ObservationVector,
    max_speed: float,
) -> tuple[float, float]:
    heading_error = wrap_angle(obs.heading_error)
    alignment = max(0.0, math.cos(heading_error))
    distance_scale = clamp(obs.distance_to_goal, 0.0, 1.0)
    base_v = max_speed * alignment * distance_scale
    residual_v = clamp(action_linear, -1.0, 1.0)
    residual_v *= max_speed * LINEAR_RESIDUAL_FRACTION
    v = clamp(base_v + residual_v, 0.0, max_speed)

    base_omega = HEADING_GAIN * heading_error
    residual_omega = clamp(action_angular, -1.0, 1.0) * ANGULAR_RESIDUAL_LIMIT
    omega = clamp(
        base_omega + residual_omega,
        -MAX_ABS_ANGULAR_SPEED,
        MAX_ABS_ANGULAR_SPEED,
    )
    return v, omega


def residual_avoidance_command(
    action_linear: float,
    action_lateral: float,
    action_angular: float,
    robot: RobotState,
    goal: GoalState,
    neighbors: list[RobotState],
    pedestrians: list[object],
    max_speed: float,
    min_priority_yield_scale: float = 0.0,
    use_orca_prior: bool = True,
    use_proximity_safety: bool = True,
    control_profile: ControlProfile | None = None,
    preferred_speed_scale: float = 1.0,
    coordination_world_vx: float = 0.0,
    coordination_world_vy: float = 0.0,
    priority_yield_scale_override: float | None = None,
) -> tuple[float, float, float]:
    profile = control_profile or DEFAULT_CONTROL_PROFILE
    dx = goal.x - robot.x
    dy = goal.y - robot.y
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
        return 0.0, 0.0, 0.0

    speed_scale = clamp(distance, 0.0, 1.0)
    residual_scale = 1.0 + profile.linear_residual_fraction * clamp(
        action_linear, -1.0, 1.0
    )
    preferred_speed = clamp(
        max_speed
        * speed_scale
        * residual_scale
        * clamp(preferred_speed_scale, 0.0, 1.0),
        0.0,
        max_speed,
    )
    preferred = VelocityCommand2D(
        vx=preferred_speed * dx / distance + coordination_world_vx,
        vy=preferred_speed * dy / distance + coordination_world_vy,
    )
    preferred_norm = math.hypot(preferred.vx, preferred.vy)
    if preferred_norm > max_speed:
        preferred = VelocityCommand2D(
            vx=preferred.vx / preferred_norm * max_speed,
            vy=preferred.vy / preferred_norm * max_speed,
        )
    agent = AgentState(
        x=robot.x,
        y=robot.y,
        vx=_body_to_world_vx(robot.v, robot.vy, robot.yaw),
        vy=_body_to_world_vy(robot.v, robot.vy, robot.yaw),
    )
    safe_velocity = preferred
    if use_orca_prior:
        avoid_neighbors = [
            AgentState(
                x=neighbor.x,
                y=neighbor.y,
                vx=_body_to_world_vx(neighbor.v, neighbor.vy, neighbor.yaw),
                vy=_body_to_world_vy(neighbor.v, neighbor.vy, neighbor.yaw),
            )
            for neighbor in neighbors
        ]
        avoid_pedestrians = [
            AgentState(
                x=float(getattr(ped, "x")),
                y=float(getattr(ped, "y")),
                vx=float(getattr(ped, "vx", 0.0)),
                vy=float(getattr(ped, "vy", 0.0)),
            )
            for ped in pedestrians
        ]
        safe_velocity = reciprocal_avoidance_velocity(
            agent,
            avoid_neighbors,
            safe_velocity,
            safety_radius=profile.robot_avoidance_radius,
            gain=profile.robot_avoidance_gain,
            max_speed=max_speed,
        )
        safe_velocity = reciprocal_avoidance_velocity(
            agent,
            avoid_pedestrians,
            safe_velocity,
            safety_radius=profile.pedestrian_avoidance_radius,
            gain=profile.pedestrian_avoidance_gain,
            max_speed=max_speed,
            time_horizon=profile.pedestrian_avoidance_time_horizon,
        )
    speed = math.hypot(safe_velocity.vx, safe_velocity.vy)
    safe_velocity_stopped = speed < 1e-6
    if safe_velocity_stopped and not use_proximity_safety:
        return 0.0, 0.0, 0.0

    if priority_yield_scale_override is None:
        interaction_scale = priority_yield_scale(
            robot,
            neighbors,
            yield_radius=profile.priority_yield_radius,
            stop_radius=profile.priority_stop_radius,
        )
        effective_min_priority_yield_scale = max(
            min_priority_yield_scale,
            profile.min_priority_yield_scale,
        )
        if interaction_scale < 1.0 and effective_min_priority_yield_scale > 0.0:
            interaction_scale = max(
                interaction_scale,
                clamp(effective_min_priority_yield_scale, 0.0, 1.0),
            )
    else:
        interaction_scale = clamp(priority_yield_scale_override, 0.0, 1.0)
    world_vx = safe_velocity.vx
    world_vy = safe_velocity.vy
    body_vx = _world_to_body_vx(world_vx, world_vy, robot.yaw)
    body_vy = _world_to_body_vy(world_vx, world_vy, robot.yaw)
    if not safe_velocity_stopped:
        body_vy += (
            clamp(action_lateral, -1.0, 1.0)
            * max_speed
            * profile.lateral_residual_fraction
            * pedestrian_yield_scale(
                robot,
                pedestrians,
                yield_radius=profile.pedestrian_yield_radius,
                stop_radius=profile.pedestrian_stop_radius,
            )
        )
    cmd_speed = math.hypot(body_vx, body_vy)
    if cmd_speed > max_speed:
        scale = max_speed / cmd_speed
        body_vx *= scale
        body_vy *= scale
    body_vx *= interaction_scale
    body_vy *= interaction_scale
    world_vx = _body_to_world_vx(body_vx, body_vy, robot.yaw)
    world_vy = _body_to_world_vy(body_vx, body_vy, robot.yaw)
    if use_proximity_safety:
        escape_vx, escape_vy = pedestrian_escape_velocity(
            robot,
            pedestrians,
            world_vx,
            world_vy,
            max_speed=max_speed,
            stop_radius=profile.pedestrian_proximity_stop_radius,
            slow_radius=profile.pedestrian_proximity_slow_radius,
        )
        if escape_vx or escape_vy:
            world_vx += escape_vx
            world_vy += escape_vy
            escape_speed = math.hypot(world_vx, world_vy)
            if escape_speed > max_speed:
                escape_scale = max_speed / escape_speed
                world_vx *= escape_scale
                world_vy *= escape_scale
            body_vx = _world_to_body_vx(world_vx, world_vy, robot.yaw)
            body_vy = _world_to_body_vy(world_vx, world_vy, robot.yaw)
        pedestrian_scale = pedestrian_proximity_scale(
            robot,
            pedestrians,
            world_vx,
            world_vy,
            hard_stop_radius=profile.pedestrian_proximity_hard_stop_radius,
            stop_radius=profile.pedestrian_proximity_stop_radius,
            slow_radius=profile.pedestrian_proximity_slow_radius,
            time_horizon=profile.pedestrian_proximity_time_horizon,
        )
        body_vx *= pedestrian_scale
        body_vy *= pedestrian_scale

    desired_heading = math.atan2(dy, dx)
    heading_error = wrap_angle(desired_heading - robot.yaw)
    omega = HEADING_GAIN * heading_error
    omega += clamp(action_angular, -1.0, 1.0) * profile.angular_residual_limit
    return body_vx, body_vy, clamp(
        omega,
        -MAX_ABS_ANGULAR_SPEED,
        MAX_ABS_ANGULAR_SPEED,
    )


def priority_yield_scale(
    robot: RobotState,
    neighbors: list[RobotState],
    yield_radius: float = PRIORITY_YIELD_RADIUS,
    stop_radius: float = PRIORITY_STOP_RADIUS,
) -> float:
    scale = 1.0
    for neighbor in neighbors:
        if _priority_key(neighbor.name) >= _priority_key(robot.name):
            continue
        dx = neighbor.x - robot.x
        dy = neighbor.y - robot.y
        distance = math.hypot(dx, dy)
        if distance >= yield_radius:
            continue
        if distance <= stop_radius:
            scale = 0.0
        else:
            room = max(yield_radius - stop_radius, 1e-6)
            scale = min(scale, (distance - stop_radius) / room)
    return clamp(scale, 0.0, 1.0)


def pedestrian_yield_scale(
    robot: RobotState,
    pedestrians: list[object],
    yield_radius: float = PEDESTRIAN_YIELD_RADIUS,
    stop_radius: float = PEDESTRIAN_STOP_RADIUS,
) -> float:
    scale = 1.0
    for ped in pedestrians:
        ped_x = float(getattr(ped, "x", 999.0))
        ped_y = float(getattr(ped, "y", 999.0))
        if not math.isfinite(ped_x) or not math.isfinite(ped_y):
            continue
        dx = ped_x - robot.x
        dy = ped_y - robot.y
        distance = math.hypot(dx, dy)
        if distance >= yield_radius:
            continue
        if distance <= stop_radius:
            scale = 0.0
        else:
            room = max(yield_radius - stop_radius, 1e-6)
            scale = min(scale, (distance - stop_radius) / room)
    return clamp(scale, 0.0, 1.0)


def pedestrian_escape_velocity(
    robot: RobotState,
    pedestrians: list[object],
    world_vx: float,
    world_vy: float,
    max_speed: float,
    stop_radius: float = PEDESTRIAN_PROXIMITY_STOP_RADIUS,
    slow_radius: float = PEDESTRIAN_PROXIMITY_SLOW_RADIUS,
) -> tuple[float, float]:
    escape_x = 0.0
    escape_y = 0.0
    for ped in pedestrians:
        ped_x = float(getattr(ped, "x", 999.0))
        ped_y = float(getattr(ped, "y", 999.0))
        if not math.isfinite(ped_x) or not math.isfinite(ped_y):
            continue
        away_x = robot.x - ped_x
        away_y = robot.y - ped_y
        distance = math.hypot(away_x, away_y)
        if distance <= 1e-9 or distance >= slow_radius:
            continue
        ped_vx = float(getattr(ped, "vx", 0.0))
        ped_vy = float(getattr(ped, "vy", 0.0))
        distance_rate = (
            away_x * (world_vx - ped_vx)
            + away_y * (world_vy - ped_vy)
        ) / distance
        if distance_rate > 0.03 and distance > stop_radius:
            continue
        unit_x = away_x / distance
        unit_y = away_y / distance
        room = max(slow_radius - stop_radius, 1e-6)
        risk = clamp((slow_radius - distance) / room, 0.0, 1.0)
        speed = max_speed * (0.15 + 0.35 * risk)
        escape_x += speed * unit_x
        escape_y += speed * unit_y
    escape_speed = math.hypot(escape_x, escape_y)
    max_escape_speed = max_speed * 0.45
    if escape_speed > max_escape_speed > 0.0:
        scale = max_escape_speed / escape_speed
        escape_x *= scale
        escape_y *= scale
    return escape_x, escape_y


def pedestrian_proximity_scale(
    robot: RobotState,
    pedestrians: list[object],
    world_vx: float,
    world_vy: float,
    hard_stop_radius: float = PEDESTRIAN_PROXIMITY_HARD_STOP_RADIUS,
    stop_radius: float = PEDESTRIAN_PROXIMITY_STOP_RADIUS,
    slow_radius: float = PEDESTRIAN_PROXIMITY_SLOW_RADIUS,
    time_horizon: float = PEDESTRIAN_PROXIMITY_TIME_HORIZON,
) -> float:
    command_speed = math.hypot(world_vx, world_vy)
    if command_speed <= 1e-9:
        return 1.0

    scale = 1.0
    for ped in pedestrians:
        ped_x = float(getattr(ped, "x", 999.0))
        ped_y = float(getattr(ped, "y", 999.0))
        if not math.isfinite(ped_x) or not math.isfinite(ped_y):
            continue
        rel_x = ped_x - robot.x
        rel_y = ped_y - robot.y
        distance = math.hypot(rel_x, rel_y)
        if distance <= 1e-9:
            return 0.0
        motion_alignment = (
            rel_x * world_vx + rel_y * world_vy
        ) / (distance * command_speed)
        ped_vx = float(getattr(ped, "vx", 0.0))
        ped_vy = float(getattr(ped, "vy", 0.0))
        distance_rate = (
            rel_x * (ped_vx - world_vx)
            + rel_y * (ped_vy - world_vy)
        ) / distance
        separating = distance_rate > 0.03
        command_away = motion_alignment < -0.05
        escaping_or_crossing_away = motion_alignment <= 0.05 and separating
        moving_closer = distance_rate < -1e-4
        if distance <= hard_stop_radius and not (separating or command_away):
            return 0.0
        if (
            distance <= stop_radius
            and moving_closer
            and not (escaping_or_crossing_away or command_away)
        ):
            return 0.0
        if distance >= slow_radius:
            continue
        time_to_close = _time_to_closest_approach(
            rel_x,
            rel_y,
            ped_vx - world_vx,
            ped_vy - world_vy,
        )
        predicted_risk = 0.0 < time_to_close <= time_horizon
        if escaping_or_crossing_away or command_away:
            continue
        if not (moving_closer or motion_alignment > 0.15 or predicted_risk):
            continue

        room = max(slow_radius - stop_radius, 1e-6)
        scale = min(scale, clamp((distance - stop_radius) / room, 0.0, 1.0))
    return clamp(scale, 0.0, 1.0)


def _robot_proximity_scale(
    distance: float,
    stop_radius: float = ROBOT_PROXIMITY_STOP_RADIUS,
    slow_radius: float = ROBOT_PROXIMITY_SLOW_RADIUS,
) -> float:
    if not math.isfinite(distance):
        return 1.0
    if distance <= stop_radius:
        return 0.0
    if distance >= slow_radius:
        return 1.0
    room = max(slow_radius - stop_radius, 1e-6)
    ratio = clamp((distance - stop_radius) / room, 0.0, 1.0)
    return ratio * ratio


def _nearest_pedestrian_distance(
    robot: RobotState,
    pedestrians: list[object],
) -> float:
    nearest = math.inf
    for pedestrian in pedestrians:
        ped_x = float(getattr(pedestrian, "x", math.inf))
        ped_y = float(getattr(pedestrian, "y", math.inf))
        if not math.isfinite(ped_x) or not math.isfinite(ped_y):
            continue
        nearest = min(nearest, math.hypot(ped_x - robot.x, ped_y - robot.y))
    return nearest


def _scale_toward_neighbor_component(
    velocity: tuple[float, float],
    unit_x: float,
    unit_y: float,
    scale: float,
) -> tuple[float, float]:
    vx, vy = velocity
    toward = vx * unit_x + vy * unit_y
    if toward <= 1e-9:
        return velocity
    reduction = toward * (1.0 - clamp(scale, 0.0, 1.0))
    return vx - reduction * unit_x, vy - reduction * unit_y


def _time_to_closest_approach(
    rel_x: float,
    rel_y: float,
    rel_vx: float,
    rel_vy: float,
) -> float:
    rel_speed_sq = rel_vx * rel_vx + rel_vy * rel_vy
    if rel_speed_sq <= 1e-9:
        return math.inf
    return -(rel_x * rel_vx + rel_y * rel_vy) / rel_speed_sq


def _body_to_world_vx(vx: float, vy: float, yaw: float) -> float:
    return math.cos(yaw) * vx - math.sin(yaw) * vy


def _body_to_world_vy(vx: float, vy: float, yaw: float) -> float:
    return math.sin(yaw) * vx + math.cos(yaw) * vy


def _world_to_body_vx(vx: float, vy: float, yaw: float) -> float:
    return math.cos(yaw) * vx + math.sin(yaw) * vy


def _world_to_body_vy(vx: float, vy: float, yaw: float) -> float:
    return -math.sin(yaw) * vx + math.cos(yaw) * vy


def _priority_key(name: str) -> tuple[str, int]:
    match = re.match(r"^(.*?)(\d+)$", name)
    if not match:
        return name, 0
    return match.group(1), int(match.group(2))
