from math import sqrt

from mrpp_experiments.metrics import EpisodeMetrics, RobotTrajectoryPoint


def test_episode_metrics_summary() -> None:
    metrics = EpisodeMetrics(scenario="static_clutter", algorithm="unit_test")
    metrics.add_point("robot_1", RobotTrajectoryPoint(t=0.0, x=0.0, y=0.0, v=0.0, omega=0.0))
    metrics.add_point("robot_1", RobotTrajectoryPoint(t=1.0, x=3.0, y=4.0, v=1.0, omega=0.5))
    metrics.mark_success("robot_1", True)
    metrics.collision_count = 1

    summary = metrics.summary()

    assert summary["scenario"] == "static_clutter"
    assert summary["algorithm"] == "unit_test"
    assert summary["success_rate"] == 1.0
    assert summary["collision_count"] == 1
    assert summary["average_path_length"] == 5.0


def test_episode_metrics_reports_trajectory_quality() -> None:
    metrics = EpisodeMetrics(scenario="static_clutter", algorithm="unit_test")
    metrics.add_point("robot_1", RobotTrajectoryPoint(t=0.0, x=0.0, y=0.0, v=0.0, omega=0.0))
    metrics.add_point("robot_1", RobotTrajectoryPoint(t=1.0, x=1.0, y=0.0, v=1.0, omega=0.0))
    metrics.add_point("robot_1", RobotTrajectoryPoint(t=2.0, x=1.0, y=1.0, v=1.0, omega=0.0))

    summary = metrics.summary()

    assert metrics.path_length("robot_1") == 2.0
    assert round(metrics.path_efficiency("robot_1"), 3) == 0.707
    assert round(metrics.turning_angle_per_meter("robot_1"), 3) == 0.785
    assert metrics.sharp_turn_count("robot_1") == 0
    assert summary["average_path_efficiency"] == metrics.average_path_efficiency()
    assert summary["total_sharp_turn_count"] == 0


def test_episode_metrics_reports_clearance_audit() -> None:
    metrics = EpisodeMetrics(scenario="pedestrian_dynamic", algorithm="unit_test")

    metrics.observe_clearances(
        {
            "robot_1": (0.0, 0.0),
            "robot_2": (0.5, 0.0),
        },
        {
            "ped_1": (0.0, 0.6),
        },
        robot_robot_near_miss_threshold=0.7,
        robot_pedestrian_near_miss_threshold=0.7,
    )
    summary = metrics.summary()

    assert summary["min_robot_robot_distance"] == 0.5
    assert summary["min_robot_pedestrian_distance"] == 0.6
    assert summary["robot_robot_near_miss_count"] == 1
    assert summary["robot_pedestrian_near_miss_count"] == 1


def test_near_misses_are_threshold_entry_events_not_per_step_observations() -> None:
    metrics = EpisodeMetrics(scenario="mixed_complex", algorithm="unit_test")
    near = {"robot_1": (0.0, 0.0), "robot_2": (0.5, 0.0)}
    clear = {"robot_1": (0.0, 0.0), "robot_2": (1.0, 0.0)}

    metrics.observe_clearances(near, robot_robot_near_miss_threshold=0.7)
    metrics.observe_clearances(near, robot_robot_near_miss_threshold=0.7)
    metrics.observe_clearances(clear, robot_robot_near_miss_threshold=0.7)
    metrics.observe_clearances(near, robot_robot_near_miss_threshold=0.7)

    assert metrics.robot_robot_near_miss_count == 2


def test_episode_metrics_reports_safety_filter_interventions() -> None:
    metrics = EpisodeMetrics(scenario="mixed_complex", algorithm="unit_test")

    metrics.observe_safety_interventions(
        {
            "robot_1": (0.6, 0.0, 0.0),
            "robot_2": (0.4, 0.0, 0.2),
        },
        {
            "robot_1": (0.3, 0.0, 0.0),
            "robot_2": (0.4, 0.0, 0.2),
        },
    )
    summary = metrics.summary()

    assert summary["safety_filter_intervention_count"] == 1
    assert summary["safety_filter_evaluated_command_count"] == 2
    assert summary["intervention_time_ratio"] == 0.5
    assert summary["mean_intervention_magnitude"] == 0.3
    assert summary["max_intervention_magnitude"] == 0.3
    assert summary["mean_normalized_intervention_magnitude"] == 0.5
    assert summary["max_normalized_intervention_magnitude"] == 0.5


def test_safety_intervention_normalizes_linear_and_angular_axes() -> None:
    metrics = EpisodeMetrics(scenario="mixed_complex", algorithm="unit_test")

    metrics.observe_safety_interventions(
        {"robot_1": (0.6, 0.0, 2.0)},
        {"robot_1": (0.0, 0.0, 0.0)},
    )

    assert abs(metrics.mean_normalized_intervention_magnitude() - sqrt(2.0)) < 1e-12
    assert abs(metrics.max_normalized_intervention_magnitude - sqrt(2.0)) < 1e-12


def test_safety_intervention_empty_active_set_records_nothing() -> None:
    metrics = EpisodeMetrics(scenario="mixed_complex", algorithm="unit_test")

    metrics.observe_safety_interventions(
        {"robot_1": (0.6, 0.0, 0.0)},
        {"robot_1": (0.0, 0.0, 0.0)},
        robot_names=[],
    )

    assert metrics.safety_filter_evaluated_command_count == 0


def test_safety_module_interventions_and_prefilter_risk_are_separate() -> None:
    from mrpp_rl.environment import RobotState

    metrics = EpisodeMetrics(scenario="mixed_complex", algorithm="unit_test")
    before = {"robot_1": (0.5, 0.0, 0.0), "robot_2": (-0.5, 0.0, 0.0)}
    after = {"robot_1": (0.0, 0.0, 0.0), "robot_2": (-0.5, 0.0, 0.0)}
    states = {
        "robot_1": RobotState("robot_1", 0.0, 0.0, 0.0),
        "robot_2": RobotState("robot_2", 1.0, 0.0, 0.0),
    }

    metrics.observe_safety_module("robot_robot", before, after)
    metrics.observe_pre_filter_risk(states, {}, before, horizon_sec=2.0)
    summary = metrics.summary()

    assert summary["robot_robot_filter_intervention_count"] == 1
    assert summary["robot_robot_filter_evaluated_count"] == 2
    assert summary["robot_robot_filter_intervention_ratio"] == 0.5
    assert summary["pre_filter_min_predicted_rr_distance"] == 0.0
    assert summary["pre_filter_rr_risk_observation_count"] == 1
