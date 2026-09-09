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

import csv
from dataclasses import MISSING, dataclass, fields
from pathlib import Path

from mrpp_experiments.metrics import EpisodeMetrics

SCHEMA_VERSION = "mrpp-results-v2"


@dataclass(frozen=True)
class EpisodeResult:
    scenario: str
    algorithm: str
    seed: int
    robot_count: int
    success_rate: float
    collision_count: int
    deadlock_count: int
    average_path_length: float
    average_completion_time: float
    makespan: float
    flowtime: float
    average_waiting_time: float
    average_path_efficiency: float = 0.0
    a_star_reference_path_length: float = 0.0
    executed_to_astar_ratio: float = 0.0
    trajectory_straightness: float = 0.0
    average_turning_angle_per_meter: float = 0.0
    total_sharp_turn_count: int = 0
    mean_policy_inference_time_ms: float = 0.0
    p95_policy_inference_time_ms: float = 0.0
    mean_control_step_time_ms: float = 0.0
    episode_wall_time_s: float = 0.0
    real_time_factor: float = 0.0
    possible_edge_count: int = 0
    active_conflict_edge_count: float = 0.0
    edge_density: float = 0.0
    cpa_graph_build_time_ms: float = 0.0
    gat_message_passing_time_ms: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    message_delay_profile: str = "nominal"
    communication_delay_ms: float = 0.0
    packet_drop_rate: float = 0.0
    stale_neighbor_ratio: float = 0.0
    effective_neighbor_update_rate_hz: float = 0.0
    mean_message_size_bytes: float = 0.0
    estimated_bandwidth_per_robot_bps: float = 0.0
    cpa_edge_change_rate: float = 0.0
    robot_robot_collision_count: int = 0
    robot_obstacle_collision_count: int = 0
    robot_pedestrian_collision_count: int = 0
    collision_failure_count: int = 0
    timeout_failure_count: int = 0
    deadlock_failure_count: int = 0
    invalid_state_failure_count: int = 0
    process_failure_count: int = 0
    min_robot_robot_distance: float = 0.0
    min_robot_pedestrian_distance: float = 0.0
    robot_robot_near_miss_count: int = 0
    robot_pedestrian_near_miss_count: int = 0
    safety_filter_intervention_count: int = 0
    safety_filter_evaluated_command_count: int = 0
    intervention_time_ratio: float = 0.0
    mean_intervention_magnitude: float = 0.0
    max_intervention_magnitude: float = 0.0
    mean_normalized_intervention_magnitude: float = 0.0
    max_normalized_intervention_magnitude: float = 0.0
    lidar_intervention_count: int = 0
    lidar_evaluated_command_count: int = 0
    lidar_intervention_ratio: float = 0.0
    robot_robot_filter_intervention_count: int = 0
    robot_robot_filter_evaluated_count: int = 0
    robot_robot_filter_intervention_ratio: float = 0.0
    robot_pedestrian_filter_intervention_count: int = 0
    robot_pedestrian_filter_evaluated_count: int = 0
    robot_pedestrian_filter_intervention_ratio: float = 0.0
    stale_sensor_stop_count: int = 0
    stale_sensor_evaluated_count: int = 0
    stale_sensor_stop_ratio: float = 0.0
    pre_filter_min_predicted_rr_distance: float = 0.0
    pre_filter_min_predicted_rp_distance: float = 0.0
    pre_filter_rr_risk_observation_count: int = 0
    pre_filter_rp_risk_observation_count: int = 0
    pre_filter_closing_speed_violation_count: int = 0
    training_seed: int = -1
    evaluation_seed: int = -1
    episode_id: int = 0
    sensor_profile: str = "nominal"
    safety_sensor_profile: str = "nominal"
    randomization_profile: str = "fixed"
    variant: str = ""
    orca_prior_enabled: int = 1
    safety_filter_enabled: int = 1
    checkpoint_sha256: str = ""
    source_tree_sha256: str = ""
    config_sha256: str = ""
    data_source: str = ""
    notes: str = ""
    schema_version: str = SCHEMA_VERSION


RESULT_FIELDNAMES = tuple(field.name for field in fields(EpisodeResult))
RESULT_FIELD_DEFAULTS = {
    field.name: field.default
    for field in fields(EpisodeResult)
    if field.default is not MISSING
}
TRAJECTORY_FIELDNAMES = (
    "scenario",
    "algorithm",
    "seed",
    "robot_name",
    "point_index",
    "t",
    "x",
    "y",
    "v",
    "omega",
    "success",
    "completion_time",
    "schema_version",
)
INT_FIELDS = {
    "seed",
    "robot_count",
    "possible_edge_count",
    "collision_count",
    "deadlock_count",
    "robot_robot_collision_count",
    "robot_obstacle_collision_count",
    "robot_pedestrian_collision_count",
    "collision_failure_count",
    "timeout_failure_count",
    "deadlock_failure_count",
    "invalid_state_failure_count",
    "process_failure_count",
    "total_sharp_turn_count",
    "robot_robot_near_miss_count",
    "robot_pedestrian_near_miss_count",
    "safety_filter_intervention_count",
    "safety_filter_evaluated_command_count",
    "lidar_intervention_count",
    "lidar_evaluated_command_count",
    "robot_robot_filter_intervention_count",
    "robot_robot_filter_evaluated_count",
    "robot_pedestrian_filter_intervention_count",
    "robot_pedestrian_filter_evaluated_count",
    "stale_sensor_stop_count",
    "stale_sensor_evaluated_count",
    "pre_filter_rr_risk_observation_count",
    "pre_filter_rp_risk_observation_count",
    "pre_filter_closing_speed_violation_count",
    "training_seed",
    "evaluation_seed",
    "episode_id",
    "orca_prior_enabled",
    "safety_filter_enabled",
}
FLOAT_FIELDS = {
    "success_rate",
    "average_path_length",
    "average_path_efficiency",
    "a_star_reference_path_length",
    "executed_to_astar_ratio",
    "trajectory_straightness",
    "average_turning_angle_per_meter",
    "average_completion_time",
    "makespan",
    "flowtime",
    "average_waiting_time",
    "mean_policy_inference_time_ms",
    "p95_policy_inference_time_ms",
    "mean_control_step_time_ms",
    "episode_wall_time_s",
    "real_time_factor",
    "active_conflict_edge_count",
    "edge_density",
    "cpa_graph_build_time_ms",
    "gat_message_passing_time_ms",
    "peak_gpu_memory_mb",
    "communication_delay_ms",
    "packet_drop_rate",
    "stale_neighbor_ratio",
    "effective_neighbor_update_rate_hz",
    "mean_message_size_bytes",
    "estimated_bandwidth_per_robot_bps",
    "cpa_edge_change_rate",
    "min_robot_robot_distance",
    "min_robot_pedestrian_distance",
    "intervention_time_ratio",
    "mean_intervention_magnitude",
    "max_intervention_magnitude",
    "mean_normalized_intervention_magnitude",
    "max_normalized_intervention_magnitude",
    "lidar_intervention_ratio",
    "robot_robot_filter_intervention_ratio",
    "robot_pedestrian_filter_intervention_ratio",
    "stale_sensor_stop_ratio",
    "pre_filter_min_predicted_rr_distance",
    "pre_filter_min_predicted_rp_distance",
}


def result_from_metrics(
    metrics: EpisodeMetrics,
    seed: int,
    robot_count: int,
    average_completion_time: float | None = None,
    makespan: float | None = None,
    flowtime: float | None = None,
    average_waiting_time: float | None = None,
    a_star_reference_path_length: float = 0.0,
    executed_to_astar_ratio: float = 0.0,
    trajectory_straightness: float | None = None,
    mean_policy_inference_time_ms: float = 0.0,
    p95_policy_inference_time_ms: float = 0.0,
    mean_control_step_time_ms: float = 0.0,
    episode_wall_time_s: float = 0.0,
    real_time_factor: float = 0.0,
    possible_edge_count: int = 0,
    active_conflict_edge_count: float = 0.0,
    edge_density: float = 0.0,
    cpa_graph_build_time_ms: float = 0.0,
    gat_message_passing_time_ms: float = 0.0,
    peak_gpu_memory_mb: float = 0.0,
    message_delay_profile: str = "nominal",
    communication_delay_ms: float = 0.0,
    packet_drop_rate: float = 0.0,
    stale_neighbor_ratio: float = 0.0,
    effective_neighbor_update_rate_hz: float = 0.0,
    mean_message_size_bytes: float = 0.0,
    estimated_bandwidth_per_robot_bps: float = 0.0,
    cpa_edge_change_rate: float = 0.0,
    variant: str = "",
    orca_prior_enabled: bool = True,
    safety_filter_enabled: bool = True,
    training_seed: int | None = None,
    evaluation_seed: int | None = None,
    episode_id: int = 0,
    sensor_profile: str = "nominal",
    safety_sensor_profile: str = "nominal",
    randomization_profile: str = "fixed",
    checkpoint_sha256: str = "",
    source_tree_sha256: str = "",
    config_sha256: str = "",
    data_source: str = "",
    notes: str = "",
) -> EpisodeResult:
    summary = metrics.summary()
    return EpisodeResult(
        scenario=str(summary["scenario"]),
        algorithm=str(summary["algorithm"]),
        seed=seed,
        robot_count=robot_count,
        success_rate=float(summary["success_rate"]),
        collision_count=int(summary["collision_count"]),
        deadlock_count=int(summary["deadlock_count"]),
        average_path_length=float(summary["average_path_length"]),
        average_path_efficiency=float(summary.get("average_path_efficiency", 0.0)),
        a_star_reference_path_length=a_star_reference_path_length,
        executed_to_astar_ratio=executed_to_astar_ratio,
        trajectory_straightness=(
            float(summary.get("average_path_efficiency", 0.0))
            if trajectory_straightness is None else trajectory_straightness
        ),
        average_turning_angle_per_meter=float(
            summary.get("average_turning_angle_per_meter", 0.0)
        ),
        total_sharp_turn_count=int(summary.get("total_sharp_turn_count", 0)),
        average_completion_time=_metric_value(
            average_completion_time, summary, "average_completion_time"
        ),
        makespan=_metric_value(makespan, summary, "makespan"),
        flowtime=_metric_value(flowtime, summary, "flowtime"),
        average_waiting_time=_metric_value(
            average_waiting_time, summary, "average_waiting_time"
        ),
        mean_policy_inference_time_ms=mean_policy_inference_time_ms,
        p95_policy_inference_time_ms=p95_policy_inference_time_ms,
        mean_control_step_time_ms=mean_control_step_time_ms,
        episode_wall_time_s=episode_wall_time_s,
        real_time_factor=real_time_factor,
        possible_edge_count=possible_edge_count,
        active_conflict_edge_count=active_conflict_edge_count,
        edge_density=edge_density,
        cpa_graph_build_time_ms=cpa_graph_build_time_ms,
        gat_message_passing_time_ms=gat_message_passing_time_ms,
        peak_gpu_memory_mb=peak_gpu_memory_mb,
        message_delay_profile=message_delay_profile,
        communication_delay_ms=communication_delay_ms,
        packet_drop_rate=packet_drop_rate,
        stale_neighbor_ratio=stale_neighbor_ratio,
        effective_neighbor_update_rate_hz=effective_neighbor_update_rate_hz,
        mean_message_size_bytes=mean_message_size_bytes,
        estimated_bandwidth_per_robot_bps=(
            estimated_bandwidth_per_robot_bps
        ),
        cpa_edge_change_rate=cpa_edge_change_rate,
        robot_robot_collision_count=int(summary.get("robot_robot_collision_count", 0)),
        robot_obstacle_collision_count=int(summary.get("robot_obstacle_collision_count", 0)),
        robot_pedestrian_collision_count=int(
            summary.get("robot_pedestrian_collision_count", 0)
        ),
        collision_failure_count=int(summary.get("collision_failure_count", 0)),
        timeout_failure_count=int(summary.get("timeout_failure_count", 0)),
        deadlock_failure_count=int(summary.get("deadlock_failure_count", 0)),
        invalid_state_failure_count=int(summary.get("invalid_state_failure_count", 0)),
        process_failure_count=int(summary.get("process_failure_count", 0)),
        min_robot_robot_distance=float(summary.get("min_robot_robot_distance", 0.0)),
        min_robot_pedestrian_distance=float(
            summary.get("min_robot_pedestrian_distance", 0.0)
        ),
        robot_robot_near_miss_count=int(
            summary.get("robot_robot_near_miss_count", 0)
        ),
        robot_pedestrian_near_miss_count=int(
            summary.get("robot_pedestrian_near_miss_count", 0)
        ),
        safety_filter_intervention_count=int(
            summary.get("safety_filter_intervention_count", 0)
        ),
        safety_filter_evaluated_command_count=int(
            summary.get("safety_filter_evaluated_command_count", 0)
        ),
        intervention_time_ratio=float(
            summary.get("intervention_time_ratio", 0.0)
        ),
        mean_intervention_magnitude=float(
            summary.get("mean_intervention_magnitude", 0.0)
        ),
        max_intervention_magnitude=float(
            summary.get("max_intervention_magnitude", 0.0)
        ),
        mean_normalized_intervention_magnitude=float(
            summary.get("mean_normalized_intervention_magnitude", 0.0)
        ),
        max_normalized_intervention_magnitude=float(
            summary.get("max_normalized_intervention_magnitude", 0.0)
        ),
        lidar_intervention_count=int(summary.get("lidar_intervention_count", 0)),
        lidar_evaluated_command_count=int(
            summary.get("lidar_evaluated_command_count", 0)
        ),
        lidar_intervention_ratio=float(summary.get("lidar_intervention_ratio", 0.0)),
        robot_robot_filter_intervention_count=int(
            summary.get("robot_robot_filter_intervention_count", 0)
        ),
        robot_robot_filter_evaluated_count=int(
            summary.get("robot_robot_filter_evaluated_count", 0)
        ),
        robot_robot_filter_intervention_ratio=float(
            summary.get("robot_robot_filter_intervention_ratio", 0.0)
        ),
        robot_pedestrian_filter_intervention_count=int(
            summary.get("robot_pedestrian_filter_intervention_count", 0)
        ),
        robot_pedestrian_filter_evaluated_count=int(
            summary.get("robot_pedestrian_filter_evaluated_count", 0)
        ),
        robot_pedestrian_filter_intervention_ratio=float(
            summary.get("robot_pedestrian_filter_intervention_ratio", 0.0)
        ),
        stale_sensor_stop_count=int(summary.get("stale_sensor_stop_count", 0)),
        stale_sensor_evaluated_count=int(
            summary.get("stale_sensor_evaluated_count", 0)
        ),
        stale_sensor_stop_ratio=float(summary.get("stale_sensor_stop_ratio", 0.0)),
        pre_filter_min_predicted_rr_distance=float(
            summary.get("pre_filter_min_predicted_rr_distance", 0.0)
        ),
        pre_filter_min_predicted_rp_distance=float(
            summary.get("pre_filter_min_predicted_rp_distance", 0.0)
        ),
        pre_filter_rr_risk_observation_count=int(
            summary.get("pre_filter_rr_risk_observation_count", 0)
        ),
        pre_filter_rp_risk_observation_count=int(
            summary.get("pre_filter_rp_risk_observation_count", 0)
        ),
        pre_filter_closing_speed_violation_count=int(
            summary.get("pre_filter_closing_speed_violation_count", 0)
        ),
        training_seed=seed if training_seed is None else training_seed,
        evaluation_seed=seed if evaluation_seed is None else evaluation_seed,
        episode_id=episode_id,
        sensor_profile=sensor_profile,
        safety_sensor_profile=safety_sensor_profile,
        randomization_profile=randomization_profile,
        variant=variant,
        orca_prior_enabled=int(orca_prior_enabled),
        safety_filter_enabled=int(safety_filter_enabled),
        checkpoint_sha256=checkpoint_sha256,
        source_tree_sha256=source_tree_sha256,
        config_sha256=config_sha256,
        data_source=data_source,
        notes=notes,
    )


def _metric_value(
    explicit_value: float | None,
    summary: dict[str, float | int | str],
    key: str,
) -> float:
    if explicit_value is not None:
        return explicit_value
    return float(summary.get(key, 0.0))


def write_results_csv(path: Path, rows: list[EpisodeResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in RESULT_FIELDNAMES})


def trajectory_rows_from_metrics(
    metrics: EpisodeMetrics,
    seed: int,
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for robot_name, points in sorted(metrics.trajectories.items()):
        success = bool(metrics.successes.get(robot_name, False))
        completion_time = metrics.completion_times.get(robot_name, "")
        for index, point in enumerate(points):
            rows.append(
                {
                    "scenario": metrics.scenario,
                    "algorithm": metrics.algorithm,
                    "seed": seed,
                    "robot_name": robot_name,
                    "point_index": index,
                    "t": point.t,
                    "x": point.x,
                    "y": point.y,
                    "v": point.v,
                    "omega": point.omega,
                    "success": int(success),
                    "completion_time": completion_time,
                    "schema_version": SCHEMA_VERSION,
                }
            )
    return rows


def write_trajectory_csv(path: Path, metrics: EpisodeMetrics, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=TRAJECTORY_FIELDNAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in trajectory_rows_from_metrics(metrics, seed):
            writer.writerow(row)


def read_results_csv(path: Path) -> list[EpisodeResult]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [_parse_row(row) for row in reader]


def _parse_row(row: dict[str, str]) -> EpisodeResult:
    values: dict[str, str | int | float] = {}
    for name in RESULT_FIELDNAMES:
        raw_value = row.get(name, "")
        if raw_value == "":
            raw_value = RESULT_FIELD_DEFAULTS.get(
                name,
                "0" if name in INT_FIELDS or name in FLOAT_FIELDS else "",
            )
        if name in INT_FIELDS:
            values[name] = int(raw_value)
        elif name in FLOAT_FIELDS:
            values[name] = float(raw_value)
        else:
            values[name] = raw_value
    return EpisodeResult(**values)  # type: ignore[arg-type]
