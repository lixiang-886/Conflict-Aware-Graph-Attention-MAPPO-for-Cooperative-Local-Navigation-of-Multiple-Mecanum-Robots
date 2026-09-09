import math
from types import SimpleNamespace

from mrpp_rl.control import apply_robot_proximity_safety_filter
from mrpp_rl.control import LaserSectors, apply_static_obstacle_safety_filter
from mrpp_rl.control import command_change_cost
from mrpp_rl.control import conflict_aware_graph
from mrpp_rl.control import conflict_aware_adjacency_matrix
from mrpp_rl.control import conflict_aware_coordination
from mrpp_rl.control import control_profile_for_algorithm
from mrpp_rl.control import laser_sectors_from_scan
from mrpp_rl.control import graph_policy_residual_gate
from mrpp_rl.control import gated_graph_policy_actions
from mrpp_rl.control import pedestrian_escape_velocity
from mrpp_rl.control import pedestrian_proximity_scale
from mrpp_rl.control import pedestrian_yield_scale
from mrpp_rl.control import observation_to_features, priority_yield_scale
from mrpp_rl.control import residual_avoidance_command
from mrpp_rl.control import residual_goal_command
from mrpp_rl.control import safety_config_for_world
from mrpp_rl.control import slew_rate_limit_command
from mrpp_rl.environment import GoalState, ObservationVector, RobotState


def test_observation_features_are_wrapped_and_scaled() -> None:
    obs = ObservationVector(
        distance_to_goal=4.0,
        heading_error=3.5 * math.pi,
        nearest_robot_distance=999.0,
        speed=2.0,
        angular_speed=-4.0,
        min_laser_range=999.0,
        laser_front=4.5,
        laser_front_left=2.25,
        laser_left=0.0,
        laser_rear_left=9.0,
        laser_rear=4.5,
        laser_rear_right=4.5,
        laser_right=4.5,
        laser_front_right=4.5,
        laser_range_max=4.5,
        nearest_ped_distance=999.0,
        nearest_ped_dx=20.0,
        nearest_ped_dy=-20.0,
    )

    features = observation_to_features(obs)

    assert features[0] == 0.5
    assert -1.0 <= features[1] <= 1.0
    assert features[2] == 2.0
    assert features[3] == 1.0
    assert features[4] == -1.0
    assert features[5] == 1.0
    assert features[6] == 0.5
    assert features[7] == 0.0
    assert features[8] == 1.0
    assert features[13] == 2.0
    assert features[14] == 1.0
    assert features[15] == -1.0
    assert len(features) == 16


def test_residual_goal_command_moves_toward_aligned_goal() -> None:
    obs = ObservationVector(
        distance_to_goal=2.0,
        heading_error=0.0,
        nearest_robot_distance=999.0,
        speed=0.0,
        angular_speed=0.0,
    )

    v, omega = residual_goal_command(0.0, 0.0, obs, max_speed=0.6)

    assert v > 0.5
    assert abs(omega) < 1e-6


def test_slew_rate_limit_command_limits_linear_and_angular_changes() -> None:
    command = slew_rate_limit_command(
        (0.6, 0.0, 2.0),
        (0.0, 0.0, 0.0),
        dt=0.1,
        max_linear_accel=1.0,
        max_angular_accel=2.0,
    )

    assert math.isclose(command[0], 0.1)
    assert math.isclose(command[1], 0.0)
    assert math.isclose(command[2], 0.2)


def test_command_change_cost_normalizes_linear_and_angular_delta() -> None:
    cost = command_change_cost(
        (0.3, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        max_speed=0.6,
        max_angular_speed=2.0,
    )

    assert math.isclose(cost, 1.0)


def test_residual_avoidance_command_uses_neighbor_repulsion() -> None:
    robot = RobotState(name="robot_1", x=0.0, y=0.0, yaw=0.0)
    goal = GoalState(x=2.0, y=0.0)
    neighbor = RobotState(name="robot_2", x=0.2, y=0.0, yaw=0.0)

    vx, vy, omega = residual_avoidance_command(
        0.0,
        0.0,
        0.0,
        robot,
        goal,
        [neighbor],
        [],
        max_speed=0.6,
    )

    assert vx < 0.0
    assert math.hypot(vx, vy) <= 0.6
    assert abs(omega) <= 2.0


def test_residual_avoidance_command_can_disable_orca_prior() -> None:
    robot = RobotState(name="robot_1", x=0.0, y=0.0, yaw=0.0)
    goal = GoalState(x=2.0, y=0.0)
    neighbor = RobotState(name="robot_2", x=0.2, y=0.0, yaw=0.0)

    vx, vy, omega = residual_avoidance_command(
        0.0,
        0.0,
        0.0,
        robot,
        goal,
        [neighbor],
        [],
        max_speed=0.6,
        use_orca_prior=False,
    )

    assert vx > 0.5
    assert abs(vy) < 1e-6
    assert abs(omega) <= 2.0


def test_residual_avoidance_command_accepts_pedestrians() -> None:
    robot = RobotState(name="robot_1", x=0.0, y=0.0, yaw=0.0)
    goal = GoalState(x=2.0, y=0.0)
    pedestrian = SimpleNamespace(x=0.3, y=0.0, vx=0.0, vy=0.0)

    vx, vy, omega = residual_avoidance_command(
        0.0,
        0.0,
        0.0,
        robot,
        goal,
        [],
        [pedestrian],
        max_speed=0.6,
    )

    assert vx < 0.0
    assert math.hypot(vx, vy) <= 0.6
    assert abs(omega) <= 2.0


def test_residual_avoidance_command_can_command_lateral_motion() -> None:
    robot = RobotState(name="robot_1", x=0.0, y=0.0, yaw=0.0)
    goal = GoalState(x=0.0, y=2.0)

    vx, vy, omega = residual_avoidance_command(
        0.0,
        0.0,
        0.0,
        robot,
        goal,
        [],
        [],
        max_speed=0.6,
    )

    assert abs(vx) < 0.05
    assert vy > 0.5
    assert omega > 0.0


def test_priority_yield_scale_slows_for_higher_priority_robot_ahead() -> None:
    robot = RobotState(name="robot_2", x=0.0, y=0.0, yaw=0.0)
    higher_priority = RobotState(name="robot_1", x=0.8, y=0.0, yaw=0.0)

    scale = priority_yield_scale(robot, [higher_priority])

    assert 0.0 < scale < 1.0


def test_priority_yield_scale_ignores_lower_priority_robot() -> None:
    robot = RobotState(name="robot_2", x=0.0, y=0.0, yaw=0.0)
    lower_priority = RobotState(name="robot_10", x=0.7, y=0.0, yaw=0.0)

    assert priority_yield_scale(robot, [lower_priority]) == 1.0


def test_conflict_aware_graph_connects_only_predicted_interactions() -> None:
    states = {
        "robot_1": RobotState(name="robot_1", x=-1.0, y=0.0, yaw=0.0),
        "robot_2": RobotState(name="robot_2", x=1.0, y=0.0, yaw=math.pi),
        "robot_3": RobotState(name="robot_3", x=0.0, y=3.0, yaw=math.pi / 2.0),
    }
    goals = {
        "robot_1": GoalState(x=2.0, y=0.0),
        "robot_2": GoalState(x=-2.0, y=0.0),
        "robot_3": GoalState(x=0.0, y=5.0),
    }

    adjacency, diagnostics = conflict_aware_graph(
        ["robot_1", "robot_2", "robot_3"],
        states,
        goals,
        max_speed=0.6,
        profile=control_profile_for_algorithm("mo_gat_mappo"),
    )

    assert adjacency == [
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert len(diagnostics) == 3
    diagnostic_by_pair = {
        (item["pair_i"], item["pair_j"]): item for item in diagnostics
    }
    assert diagnostic_by_pair[("robot_1", "robot_2")]["edge_active"] == 1
    assert diagnostic_by_pair[("robot_1", "robot_3")]["edge_active"] == 0
    assert diagnostic_by_pair[("robot_2", "robot_3")]["edge_active"] == 0

    assert adjacency == conflict_aware_adjacency_matrix(
        ["robot_1", "robot_2", "robot_3"],
        states,
        goals,
        max_speed=0.6,
        profile=control_profile_for_algorithm("mo_gat_mappo"),
    )


def test_conflict_aware_coordination_assigns_lateral_yield_to_follower() -> None:
    states = {
        "robot_1": RobotState(name="robot_1", x=-1.0, y=0.0, yaw=0.0),
        "robot_2": RobotState(name="robot_2", x=1.0, y=0.0, yaw=math.pi),
    }
    goals = {
        "robot_1": GoalState(x=2.0, y=0.0),
        "robot_2": GoalState(x=-2.0, y=0.0),
    }

    directives = conflict_aware_coordination(
        ["robot_1", "robot_2"],
        states,
        goals,
        max_speed=0.6,
        profile=control_profile_for_algorithm("mo_gat_mappo"),
    )

    assert directives["robot_1"].preferred_speed_scale == 1.0
    assert directives["robot_2"].preferred_speed_scale < 1.0
    assert abs(directives["robot_2"].world_lateral_vy) > 0.05
    assert directives["robot_2"].priority_yield_scale == 1.0


def test_graph_policy_residual_gate_stays_closed_on_clear_path() -> None:
    observation = ObservationVector(
        distance_to_goal=4.0,
        heading_error=0.0,
        nearest_robot_distance=3.0,
        speed=0.5,
        angular_speed=0.0,
        nearest_ped_distance=5.0,
    )

    gate = graph_policy_residual_gate(
        False,
        observation,
        control_profile_for_algorithm("mo_gat_mappo"),
    )

    assert gate == 0.0


def test_graph_policy_residual_gate_opens_for_robot_or_pedestrian_risk() -> None:
    profile = control_profile_for_algorithm("mo_gat_mappo")
    pedestrian_risk = ObservationVector(
        distance_to_goal=4.0,
        heading_error=0.0,
        nearest_robot_distance=3.0,
        speed=0.5,
        angular_speed=0.0,
        nearest_ped_distance=profile.pedestrian_proximity_stop_radius,
    )

    assert graph_policy_residual_gate(True, pedestrian_risk, profile) == 1.0
    assert graph_policy_residual_gate(False, pedestrian_risk, profile) == 1.0


def test_gated_graph_policy_actions_only_allow_clear_path_acceleration() -> None:
    assert gated_graph_policy_actions(0.4, 0.5, -0.6, 0.0) == (
        0.4,
        0.0,
        -0.0,
    )
    assert gated_graph_policy_actions(-0.4, 0.5, -0.6, 0.0) == (
        0.0,
        0.0,
        -0.0,
    )


def test_gated_graph_policy_actions_restore_full_conflict_control() -> None:
    assert gated_graph_policy_actions(-0.4, 0.5, -0.6, 1.0) == (
        -0.4,
        0.5,
        -0.6,
    )


def test_residual_avoidance_soft_priority_yield_is_opt_in() -> None:
    robot = RobotState(name="robot_2", x=0.0, y=0.0, yaw=0.0)
    goal = GoalState(x=2.0, y=0.0)
    higher_priority = RobotState(name="robot_1", x=-0.2, y=0.0, yaw=0.0)

    stopped = residual_avoidance_command(
        0.0,
        0.0,
        0.0,
        robot,
        goal,
        [higher_priority],
        [],
        max_speed=0.6,
    )
    softened = residual_avoidance_command(
        0.0,
        0.0,
        0.0,
        robot,
        goal,
        [higher_priority],
        [],
        max_speed=0.6,
        min_priority_yield_scale=0.35,
    )

    assert stopped[0] == 0.0
    assert softened[0] == 0.21


def test_electronics_main_methods_share_control_profile() -> None:
    methods = ("ippo", "mappo", "maddpg", "mo_gat_mappo")

    for scenario in ("static_clutter", "pedestrian_dynamic", "mixed_complex"):
        profiles = [
            control_profile_for_algorithm(method, scenario) for method in methods
        ]
        assert profiles[1:] == [profiles[0]] * (len(profiles) - 1)


def test_mogat_mixed_profile_keeps_pedestrian_margin() -> None:
    pedestrian = control_profile_for_algorithm("mo_gat_mappo", "pedestrian_dynamic")
    mixed = control_profile_for_algorithm("mo_gat_mappo", "mixed_complex")

    assert mixed.linear_residual_fraction < pedestrian.linear_residual_fraction
    assert mixed.pedestrian_avoidance_radius > pedestrian.pedestrian_avoidance_radius
    assert mixed.pedestrian_proximity_slow_radius > pedestrian.pedestrian_proximity_slow_radius


def test_mogat_static_profile_limits_yaw_oscillation() -> None:
    static = control_profile_for_algorithm("mo_gat_mappo", "static_clutter")
    pedestrian = control_profile_for_algorithm("mo_gat_mappo", "pedestrian_dynamic")

    assert static.linear_residual_fraction == pedestrian.linear_residual_fraction
    assert static.lateral_residual_fraction < pedestrian.lateral_residual_fraction
    assert static.angular_residual_limit < pedestrian.angular_residual_limit
    assert static.coordination_lateral_speed == pedestrian.coordination_lateral_speed


def test_mogat_control_profile_softens_priority_waiting() -> None:
    robot = RobotState(name="robot_2", x=0.0, y=0.0, yaw=0.0)
    goal = GoalState(x=2.0, y=0.0)
    higher_priority = RobotState(name="robot_1", x=-0.2, y=0.0, yaw=0.0)

    command = residual_avoidance_command(
        0.0,
        0.0,
        0.0,
        robot,
        goal,
        [higher_priority],
        [],
        max_speed=0.6,
        control_profile=control_profile_for_algorithm("mo_gat_mappo"),
    )

    assert command[0] > 0.0


def test_electronics_mappo_and_mogat_share_lateral_residual_authority() -> None:
    robot = RobotState(name="robot_1", x=0.0, y=0.0, yaw=0.0)
    goal = GoalState(x=2.0, y=0.0)

    baseline = residual_avoidance_command(
        0.0,
        1.0,
        0.0,
        robot,
        goal,
        [],
        [],
        max_speed=0.6,
        control_profile=control_profile_for_algorithm("mappo"),
    )
    mogat = residual_avoidance_command(
        0.0,
        1.0,
        0.0,
        robot,
        goal,
        [],
        [],
        max_speed=0.6,
        control_profile=control_profile_for_algorithm("mo_gat_mappo"),
    )

    assert mogat == baseline
    assert math.hypot(mogat[0], mogat[1]) <= 0.6


def test_pedestrian_yield_scale_stops_for_close_pedestrian() -> None:
    robot = RobotState(name="robot_1", x=0.0, y=0.0, yaw=0.0)
    pedestrian = SimpleNamespace(x=0.5, y=0.0)

    assert pedestrian_yield_scale(robot, [pedestrian]) == 0.0


def test_residual_avoidance_pedestrian_yield_scales_lateral_residual() -> None:
    robot = RobotState(name="robot_1", x=0.0, y=0.0, yaw=0.0)
    goal = GoalState(x=2.0, y=0.0)
    pedestrian = SimpleNamespace(x=0.5, y=0.0, vx=0.0, vy=0.0)

    command = residual_avoidance_command(
        0.0,
        1.0,
        0.0,
        robot,
        goal,
        [],
        [pedestrian],
        0.6,
    )

    assert command[0] < 0.0
    assert command[1] == 0.0


def test_residual_avoidance_command_can_disable_proximity_safety() -> None:
    robot = RobotState(name="robot_1", x=0.0, y=0.0, yaw=0.0)
    goal = GoalState(x=2.0, y=0.0)
    pedestrian = SimpleNamespace(x=0.45, y=0.0, vx=0.0, vy=0.0)

    command = residual_avoidance_command(
        0.0,
        0.0,
        0.0,
        robot,
        goal,
        [],
        [pedestrian],
        max_speed=0.6,
        use_orca_prior=False,
        use_proximity_safety=False,
    )

    assert command[0] > 0.5
    assert abs(command[1]) < 1e-6


def test_pedestrian_proximity_scale_stops_for_hard_close_pedestrian() -> None:
    robot = RobotState(name="robot_1", x=0.0, y=0.0, yaw=0.0)
    pedestrian = SimpleNamespace(x=0.45, y=0.0, vx=0.0, vy=0.0)

    assert pedestrian_proximity_scale(robot, [pedestrian], 0.2, 0.0) == 0.0


def test_pedestrian_proximity_scale_allows_escape_from_close_pedestrian() -> None:
    robot = RobotState(name="robot_1", x=0.0, y=0.0, yaw=0.0)
    pedestrian = SimpleNamespace(x=0.45, y=0.0, vx=0.0, vy=0.0)

    assert pedestrian_proximity_scale(robot, [pedestrian], -0.2, 0.0) == 1.0


def test_pedestrian_proximity_scale_preserves_command_away_from_closing_pedestrian() -> None:
    robot = RobotState(name="robot_4", x=-1.86, y=4.53, yaw=0.0)
    pedestrian = SimpleNamespace(x=-1.48, y=4.40, vx=0.2, vy=0.0)

    assert pedestrian_proximity_scale(robot, [pedestrian], -0.12, 0.20) == 1.0


def test_pedestrian_escape_velocity_moves_away_from_crossing_pedestrian() -> None:
    robot = RobotState(name="robot_4", x=-1.86, y=4.77, yaw=0.0)
    pedestrian = SimpleNamespace(x=-1.48, y=4.40, vx=0.2, vy=0.0)

    escape_vx, escape_vy = pedestrian_escape_velocity(
        robot,
        [pedestrian],
        world_vx=0.0,
        world_vy=0.0,
        max_speed=0.6,
        stop_radius=0.76,
        slow_radius=1.32,
    )

    assert escape_vx < -0.05
    assert escape_vy > 0.05
    assert math.hypot(escape_vx, escape_vy) <= 0.27


def test_residual_avoidance_keeps_pedestrian_escape_when_orca_stops() -> None:
    robot = RobotState(name="robot_4", x=-1.86, y=4.53, yaw=0.0)
    goal = GoalState(x=-1.65, y=5.35)
    pedestrian = SimpleNamespace(x=-1.48, y=4.40, vx=0.2, vy=0.0)

    command = residual_avoidance_command(
        0.0,
        0.0,
        0.0,
        robot,
        goal,
        [],
        [pedestrian],
        max_speed=0.6,
        control_profile=control_profile_for_algorithm(
            "mo_gat_mappo",
            "mixed_complex",
        ),
    )

    assert math.hypot(command[0], command[1]) > 0.05


def test_pedestrian_proximity_scale_blocks_close_crossing_motion() -> None:
    robot = RobotState(name="robot_1", x=0.0, y=0.0, yaw=0.0)
    pedestrian = SimpleNamespace(x=0.45, y=0.0, vx=0.0, vy=0.0)

    assert pedestrian_proximity_scale(robot, [pedestrian], 0.0, 0.2) == 0.0


def test_pedestrian_proximity_scale_slows_for_closing_crossing() -> None:
    robot = RobotState(name="robot_1", x=0.0, y=0.0, yaw=0.0)
    pedestrian = SimpleNamespace(x=0.9, y=0.2, vx=-0.2, vy=0.0)

    scale = pedestrian_proximity_scale(robot, [pedestrian], 0.3, 0.0)

    assert 0.0 < scale < 1.0


def test_robot_proximity_filter_slows_closing_pair() -> None:
    states = {
        "robot_1": RobotState(name="robot_1", x=0.0, y=0.0, yaw=0.0),
        "robot_2": RobotState(name="robot_2", x=0.8, y=0.0, yaw=math.pi),
    }
    commands = {
        "robot_1": (0.6, 0.0, 0.0),
        "robot_2": (0.6, 0.0, 0.0),
    }

    filtered = apply_robot_proximity_safety_filter(commands, states)

    assert filtered["robot_1"][0] == commands["robot_1"][0]
    assert 0.0 <= filtered["robot_2"][0] < commands["robot_2"][0]


def test_robot_proximity_filter_allows_separating_motion() -> None:
    states = {
        "robot_1": RobotState(name="robot_1", x=0.0, y=0.0, yaw=0.0),
        "robot_2": RobotState(name="robot_2", x=0.5, y=0.0, yaw=0.0),
    }
    commands = {
        "robot_1": (-0.4, 0.0, 0.0),
        "robot_2": (0.4, 0.0, 0.0),
    }

    filtered = apply_robot_proximity_safety_filter(commands, states)

    assert filtered == commands


def test_robot_proximity_filter_skips_pairs_near_pedestrians() -> None:
    states = {
        "robot_1": RobotState(name="robot_1", x=0.0, y=0.0, yaw=0.0),
        "robot_2": RobotState(name="robot_2", x=0.8, y=0.0, yaw=math.pi),
    }
    commands = {
        "robot_1": (0.6, 0.0, 0.0),
        "robot_2": (0.6, 0.0, 0.0),
    }
    pedestrians = [SimpleNamespace(x=0.1, y=0.0)]

    filtered = apply_robot_proximity_safety_filter(
        commands,
        states,
        pedestrians=pedestrians,
        pedestrian_skip_radius=1.0,
    )

    assert filtered == commands


def test_laser_sectors_assign_obstacles_to_body_directions() -> None:
    sectors = laser_sectors_from_scan(
        ranges=[1.0, 2.0, 0.4, float("nan"), 3.0],
        angle_min=-math.pi / 2.0,
        angle_increment=math.pi / 4.0,
    )

    assert sectors.right == 1.0
    assert sectors.front == 0.4
    assert sectors.left == 3.0


def test_static_safety_filter_limits_forward_velocity() -> None:
    command = apply_static_obstacle_safety_filter(
        (0.6, 0.0, 0.1),
        LaserSectors(front=0.4),
        stop_distance=0.2,
        slow_distance=0.6,
    )

    assert 0.0 < command[0] < 0.6
    assert command[1] == 0.0
    assert command[2] == 0.1


def test_mixed_complex_uses_conservative_laser_safety() -> None:
    default_config = safety_config_for_world("static_clutter")
    mixed_config = safety_config_for_world("mixed_complex")

    assert mixed_config.stop_distance > default_config.stop_distance
    assert mixed_config.slow_distance > default_config.slow_distance

    command = apply_static_obstacle_safety_filter(
        (0.6, 0.0, 0.0),
        LaserSectors(front=0.50),
        stop_distance=mixed_config.stop_distance,
        slow_distance=mixed_config.slow_distance,
        emergency_stop_distance=mixed_config.emergency_stop_distance,
    )

    assert command == (0.0, 0.0, 0.0)


def test_static_safety_filter_uses_front_diagonal_for_forward_velocity() -> None:
    command = apply_static_obstacle_safety_filter(
        (0.6, 0.0, 0.0),
        LaserSectors(front=999.0, front_left=0.3),
        stop_distance=0.2,
        slow_distance=0.6,
    )

    assert 0.0 < command[0] < 0.6


def test_static_safety_filter_limits_lateral_velocity() -> None:
    command = apply_static_obstacle_safety_filter(
        (0.0, 0.5, 0.0),
        LaserSectors(left=0.18),
        stop_distance=0.2,
        slow_distance=0.6,
    )

    assert command[0] == 0.0
    assert command[1] == 0.0


def test_static_safety_filter_limits_diagonal_mecanum_motion() -> None:
    command = apply_static_obstacle_safety_filter(
        (0.5, 0.5, 0.0),
        LaserSectors(front=999.0, left=999.0, front_left=0.3),
        stop_distance=0.2,
        slow_distance=0.6,
    )

    assert 0.0 < command[0] < 0.5
    assert 0.0 < command[1] < 0.5


def test_static_safety_filter_scales_rotation_near_any_obstacle() -> None:
    command = apply_static_obstacle_safety_filter(
        (0.0, 0.0, 2.0),
        LaserSectors(right=0.4),
        stop_distance=0.2,
        slow_distance=0.6,
    )

    assert 0.0 < command[2] < 2.0
