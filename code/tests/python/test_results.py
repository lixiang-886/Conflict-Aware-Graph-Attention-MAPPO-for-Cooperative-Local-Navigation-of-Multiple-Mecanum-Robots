from pathlib import Path

from mrpp_experiments.metrics import EpisodeMetrics, RobotTrajectoryPoint
from mrpp_experiments.results import (
    EpisodeResult,
    RESULT_FIELDNAMES,
    SCHEMA_VERSION,
    read_results_csv,
    result_from_metrics,
    trajectory_rows_from_metrics,
    write_results_csv,
    write_trajectory_csv,
)


ROOT = Path(__file__).resolve().parents[2]


def test_data_dictionary_tracks_canonical_result_schema() -> None:
    text = (ROOT / "docs/paper/data_dictionary.md").read_text(encoding="utf-8")
    episode_schema = text.split("```text\n", 1)[1].split("\n```", 1)[0]

    assert episode_schema == ",".join(RESULT_FIELDNAMES)
    assert f"Current canonical value: `{SCHEMA_VERSION}`" in text
    assert "mrpp-observation-v3-body-relative-pedestrian" in text


def test_result_csv_round_trip(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    result = EpisodeResult(
        scenario="static_clutter",
        algorithm="mo_gat_mappo",
        seed=7,
        robot_count=4,
        success_rate=0.75,
        collision_count=1,
        deadlock_count=0,
        average_path_length=6.5,
        average_completion_time=18.0,
        makespan=21.0,
        flowtime=72.0,
        average_waiting_time=1.5,
        mean_policy_inference_time_ms=2.5,
        p95_policy_inference_time_ms=4.0,
        mean_control_step_time_ms=8.0,
        episode_wall_time_s=30.0,
        real_time_factor=0.8,
        collision_failure_count=1,
        variant="ablation_no_orca_prior",
        orca_prior_enabled=0,
        safety_filter_enabled=1,
        safety_filter_intervention_count=12,
        safety_filter_evaluated_command_count=200,
        intervention_time_ratio=0.06,
        mean_intervention_magnitude=0.14,
        max_intervention_magnitude=0.52,
        mean_normalized_intervention_magnitude=0.23,
        max_normalized_intervention_magnitude=0.61,
        data_source="gazebo",
        notes="unit-test",
    )

    write_results_csv(csv_path, [result])
    rows = read_results_csv(csv_path)

    assert rows == [result]


def test_result_from_metrics_preserves_summary_fields() -> None:
    metrics = EpisodeMetrics(scenario="static_clutter", algorithm="priority_astar")
    metrics.add_point("robot_1", RobotTrajectoryPoint(t=0.0, x=0.0, y=0.0, v=0.0, omega=0.0))
    metrics.add_point("robot_1", RobotTrajectoryPoint(t=1.0, x=3.0, y=4.0, v=1.0, omega=0.0))
    metrics.mark_success("robot_1", True, completion_time=1.0)
    metrics.add_waiting_time("robot_1", 0.2)
    metrics.observe_clearances(
        {
            "robot_1": (0.0, 0.0),
            "robot_2": (0.8, 0.0),
        },
        {
            "ped_1": (0.0, 0.6),
        },
        robot_pedestrian_near_miss_threshold=0.7,
    )
    metrics.observe_safety_interventions(
        {"robot_1": (0.4, 0.0, 0.0)},
        {"robot_1": (0.2, 0.0, 0.0)},
    )

    result = result_from_metrics(
        metrics,
        seed=11,
        robot_count=1,
        data_source="unit",
    )

    assert result.scenario == "static_clutter"
    assert result.algorithm == "priority_astar"
    assert result.seed == 11
    assert result.success_rate == 1.0
    assert result.average_path_length == 5.0
    assert result.average_path_efficiency == 1.0
    assert result.average_turning_angle_per_meter == 0.0
    assert result.total_sharp_turn_count == 0
    assert result.average_completion_time == 1.0
    assert result.makespan == 1.0
    assert result.flowtime == 1.0
    assert result.average_waiting_time == 0.2
    assert result.mean_policy_inference_time_ms == 0.0
    assert result.min_robot_robot_distance == 0.8
    assert result.min_robot_pedestrian_distance == 0.6
    assert result.robot_pedestrian_near_miss_count == 1
    assert result.safety_filter_intervention_count == 1
    assert result.safety_filter_evaluated_command_count == 1
    assert result.intervention_time_ratio == 1.0
    assert result.mean_intervention_magnitude == 0.2
    assert result.max_intervention_magnitude == 0.2
    assert abs(result.mean_normalized_intervention_magnitude - 1.0 / 3.0) < 1e-12
    assert abs(result.max_normalized_intervention_magnitude - 1.0 / 3.0) < 1e-12
    assert result.orca_prior_enabled == 1
    assert result.safety_filter_enabled == 1


def test_result_csv_reads_legacy_rows_with_default_control_flags(tmp_path: Path) -> None:
    csv_path = tmp_path / "legacy.csv"
    csv_path.write_text(
        "scenario,algorithm,seed,robot_count,success_rate,collision_count,"
        "deadlock_count,average_path_length,average_completion_time,makespan,"
        "flowtime,average_waiting_time\n"
        "static_clutter,mo_gat_mappo,0,4,1.0,0,0,12.0,20.0,22.0,80.0,1.0\n",
        encoding="utf-8",
    )

    rows = read_results_csv(csv_path)

    assert rows[0].variant == ""
    assert rows[0].orca_prior_enabled == 1
    assert rows[0].safety_filter_enabled == 1
    assert rows[0].safety_filter_intervention_count == 0
    assert rows[0].intervention_time_ratio == 0.0


def test_trajectory_csv_preserves_robot_points(tmp_path: Path) -> None:
    metrics = EpisodeMetrics(scenario="static_clutter", algorithm="mo_gat_mappo")
    metrics.add_point("robot_1", RobotTrajectoryPoint(t=0.2, x=1.0, y=2.0, v=0.3, omega=0.1))
    metrics.add_point("robot_1", RobotTrajectoryPoint(t=0.4, x=1.2, y=2.1, v=0.2, omega=0.0))
    metrics.mark_success("robot_1", True, completion_time=0.4)
    csv_path = tmp_path / "trajectories.csv"

    write_trajectory_csv(csv_path, metrics, seed=3)
    rows = trajectory_rows_from_metrics(metrics, seed=3)
    text = csv_path.read_text(encoding="utf-8")

    assert rows[0]["scenario"] == "static_clutter"
    assert rows[0]["robot_name"] == "robot_1"
    assert rows[0]["success"] == 1
    assert rows[0]["completion_time"] == 0.4
    assert "robot_1" in text
