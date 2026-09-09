from pathlib import Path
import math

import pytest

from mrpp_experiments.scenario import Pose2D, RobotTask, Scenario, load_scenario
from mrpp_rl.environment import GoalState, RobotState
from mrpp_rl.waypoints import WaypointManager, build_scenario_waypoint_manager
from mrpp_rl.waypoints import build_waypoint_manager
from mrpp_rl.waypoints import densify_goal_path
from mrpp_rl.waypoints import WorldGrid, build_world_grid, parse_static_obstacles
from mrpp_rl.waypoints import _drop_collinear_grid_points, simplify_grid_path


def _world(path: Path) -> Path:
    text = """<?xml version="1.0"?>
<sdf version="1.9">
  <world name="unit">
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>8 8</size></plane></geometry>
        </collision>
      </link>
    </model>
    <model name="block">
      <static>true</static>
      <pose>0 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="c">
          <geometry><box><size>1.0 1.0 1.0</size></box></geometry>
        </collision>
      </link>
    </model>
  </world>
</sdf>
"""
    path.write_text(text, encoding="utf-8")
    return path


def _world_with_internal_wall(path: Path) -> Path:
    text = """<?xml version="1.0"?>
<sdf version="1.9">
  <world name="unit">
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>20 20</size></plane></geometry>
        </collision>
      </link>
    </model>
    <model name="wall_w"><static>true</static><pose>-8 0 1 0 0 0</pose><link name="link"><collision name="c"><geometry><box><size>0.2 14 2</size></box></geometry></collision></link></model>
    <model name="wall_e"><static>true</static><pose>8 0 1 0 0 0</pose><link name="link"><collision name="c"><geometry><box><size>0.2 14 2</size></box></geometry></collision></link></model>
    <model name="wall_s"><static>true</static><pose>0 -7 1 0 0 0</pose><link name="link"><collision name="c"><geometry><box><size>16 0.2 2</size></box></geometry></collision></link></model>
    <model name="wall_n"><static>true</static><pose>0 7 1 0 0 0</pose><link name="link"><collision name="c"><geometry><box><size>16 0.2 2</size></box></geometry></collision></link></model>
    <model name="internal"><static>true</static><pose>0 2 1 0 0 0</pose><link name="link"><collision name="c"><geometry><box><size>0.15 3 2</size></box></geometry></collision></link></model>
  </world>
</sdf>
"""
    path.write_text(text, encoding="utf-8")
    return path


def _scenario() -> Scenario:
    return Scenario(
        name="unit",
        world="unit",
        max_steps=100,
        dt=0.1,
        robots=(
            RobotTask(
                name="robot_1",
                start=Pose2D(x=-3.0, y=0.0, yaw=0.0),
                goal=Pose2D(x=3.0, y=0.0, yaw=0.0),
            ),
        ),
    )


def test_parse_static_obstacles_reads_box_models(tmp_path: Path) -> None:
    obstacles = parse_static_obstacles(_world(tmp_path / "world.sdf"))

    assert len(obstacles) == 1
    assert obstacles[0].x == 0.0
    assert obstacles[0].y == 0.0


def test_build_waypoint_manager_routes_around_static_obstacle(tmp_path: Path) -> None:
    scenario = _scenario()
    manager = build_waypoint_manager(
        scenario,
        _world(tmp_path / "world.sdf"),
        resolution=0.25,
        inflation=0.3,
    )
    path = manager.path_for("robot_1")

    assert path
    assert path[-1] == GoalState(x=3.0, y=0.0, yaw=0.0)
    assert any(abs(point.y) > 0.5 for point in path[:-1])


def test_waypoint_manager_uses_lookahead_target_on_path() -> None:
    manager = WaypointManager(
        {
            "robot_1": [
                GoalState(x=0.0, y=0.0),
                GoalState(x=1.0, y=0.0),
                GoalState(x=2.0, y=0.0),
            ]
        },
        lookahead_distance=0.5,
        minimum_lookahead=0.5,
        maximum_lookahead=0.5,
        lookahead_speed_gain=0.0,
        final_goal_switch_distance=0.25,
    )
    final = GoalState(x=2.0, y=0.0)

    first = manager.current_goal(
        "robot_1",
        RobotState(name="robot_1", x=0.0, y=0.0, yaw=0.0),
        final,
    )
    second = manager.current_goal(
        "robot_1",
        RobotState(name="robot_1", x=1.0, y=0.0, yaw=0.0),
        final,
    )

    assert first.x == pytest.approx(0.5)
    assert second.x == pytest.approx(1.5)


def test_waypoint_tracking_reports_progress_and_cross_track_error() -> None:
    manager = WaypointManager(
        {
            "robot_1": [
                GoalState(x=0.0, y=0.0),
                GoalState(x=2.0, y=0.0),
            ]
        },
        lookahead_distance=0.5,
        minimum_lookahead=0.5,
        maximum_lookahead=0.5,
        lookahead_speed_gain=0.0,
    )

    state = manager.tracking_state(
        "robot_1",
        RobotState(name="robot_1", x=0.75, y=0.25, yaw=0.0),
        GoalState(x=2.0, y=0.0),
    )

    assert state.local_goal.x == pytest.approx(1.25)
    assert state.progress == pytest.approx(0.375)
    assert state.remaining_distance == pytest.approx(1.25)
    assert state.cross_track_error == pytest.approx(0.25)


def test_waypoint_tracking_progress_does_not_move_backward() -> None:
    manager = WaypointManager(
        {
            "robot_1": [
                GoalState(x=0.0, y=0.0),
                GoalState(x=2.0, y=0.0),
            ]
        },
        lookahead_distance=0.5,
        minimum_lookahead=0.5,
        maximum_lookahead=0.5,
        lookahead_speed_gain=0.0,
    )
    final = GoalState(x=2.0, y=0.0)

    manager.tracking_state(
        "robot_1",
        RobotState(name="robot_1", x=1.2, y=0.0, yaw=0.0),
        final,
    )
    state = manager.tracking_state(
        "robot_1",
        RobotState(name="robot_1", x=0.4, y=0.0, yaw=0.0),
        final,
    )

    assert state.progress_distance == pytest.approx(1.2)
    assert state.local_goal.x == pytest.approx(1.7)


def test_waypoint_tracking_rejects_large_projection_jump() -> None:
    manager = WaypointManager(
        {
            "robot_1": [
                GoalState(x=0.0, y=0.0),
                GoalState(x=10.0, y=0.0),
                GoalState(x=10.0, y=10.0),
                GoalState(x=0.0, y=10.0),
            ]
        },
        lookahead_distance=0.5,
        minimum_lookahead=0.5,
        maximum_lookahead=0.5,
        lookahead_speed_gain=0.0,
        final_goal_switch_distance=0.25,
        max_projection_advance=2.0,
    )

    state = manager.tracking_state(
        "robot_1",
        RobotState(name="robot_1", x=0.1, y=9.0, yaw=0.0),
        GoalState(x=0.0, y=10.0),
    )

    assert state.progress_distance <= 2.0
    assert state.local_goal.y == pytest.approx(0.0)


def test_static_clutter_uses_more_conservative_waypoints() -> None:
    scenario = Scenario(
        name="static_clutter",
        world="static_clutter",
        max_steps=100,
        dt=0.1,
        robots=(
            RobotTask(
                name="robot_4",
                start=Pose2D(x=5.0, y=4.0, yaw=-2.4669),
                goal=Pose2D(x=-5.0, y=-4.0, yaw=-2.4669),
            ),
        ),
    )
    manager = build_scenario_waypoint_manager(
        scenario,
        Path("src/mrpp_gazebo/worlds/static_clutter.sdf"),
    )

    assert manager is not None
    path = manager.path_for("robot_4")
    assert len(path) >= 8
    assert any(point.x < -3.0 and point.y > 1.0 for point in path)


def test_world_grid_moves_occupied_start_to_nearest_free_cell(tmp_path: Path) -> None:
    grid = build_world_grid(
        _world(tmp_path / "world.sdf"),
        _scenario(),
        resolution=0.25,
        inflation=0.3,
    )
    occupied = grid.to_grid(0.0, 0.0)
    free = grid.nearest_free(occupied)

    assert occupied in grid.obstacles
    assert free not in grid.obstacles


def test_path_simplification_does_not_cut_between_obstacle_corners() -> None:
    grid = WorldGrid(
        min_x=0.0,
        min_y=0.0,
        resolution=1.0,
        width=3,
        height=3,
        obstacles=frozenset({(0, 1)}),
    )
    path = [(0, 0), (1, 0), (1, 1)]

    assert simplify_grid_path(path, grid) == path


def test_collinear_waypoint_compaction_keeps_corners() -> None:
    path = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3)]

    assert _drop_collinear_grid_points(path) == [(0, 0), (0, 2), (2, 2), (2, 3)]


def test_world_grid_bounds_use_outer_walls_when_internal_walls_exist(
    tmp_path: Path,
) -> None:
    grid = build_world_grid(
        _world_with_internal_wall(tmp_path / "world.sdf"),
        _scenario(),
        resolution=0.25,
        inflation=0.3,
    )

    assert grid.min_x < -7.5
    assert grid.min_y < -6.5
    assert grid.to_grid(-6.0, 6.0)[0] > 0
    assert grid.to_grid(-6.0, 6.0)[1] > 0


def test_scenario_waypoints_are_enabled_for_dynamic_pedestrian_scene(
    tmp_path: Path,
) -> None:
    scenario = Scenario(
        name="pedestrian_dynamic",
        world="pedestrian_dynamic",
        max_steps=100,
        dt=0.1,
        robots=(
            RobotTask(
                name="robot_1",
                start=Pose2D(x=-5.0, y=4.0, yaw=0.0),
                goal=Pose2D(x=5.0, y=-4.0, yaw=0.0),
            ),
            RobotTask(
                name="robot_2",
                start=Pose2D(x=-5.0, y=-4.0, yaw=0.0),
                goal=Pose2D(x=5.0, y=4.0, yaw=0.0),
            ),
            RobotTask(
                name="robot_3",
                start=Pose2D(x=5.0, y=-4.0, yaw=0.0),
                goal=Pose2D(x=-5.0, y=4.0, yaw=0.0),
            ),
            RobotTask(
                name="robot_4",
                start=Pose2D(x=5.0, y=4.0, yaw=0.0),
                goal=Pose2D(x=-5.0, y=-4.0, yaw=0.0),
            ),
        ),
    )

    manager = build_scenario_waypoint_manager(
        scenario,
        _world(tmp_path / "world.sdf"),
    )

    assert manager is not None
    for task in scenario.robots:
        path = manager.path_for(task.name)
        assert len(path) == 2
        assert path[0] == GoalState(
            x=task.start.x,
            y=task.start.y,
            yaw=task.start.yaw,
        )
        assert path[-1] == GoalState(
            x=task.goal.x,
            y=task.goal.y,
            yaw=task.goal.yaw,
        )


def test_scenario_waypoints_are_enabled_for_static_clutter(
    tmp_path: Path,
) -> None:
    scenario = Scenario(
        name="static_clutter",
        world="static_clutter",
        max_steps=100,
        dt=0.1,
        robots=_scenario().robots,
    )

    assert build_scenario_waypoint_manager(
        scenario,
        _world(tmp_path / "world.sdf"),
    ) is not None


def test_scenario_waypoints_are_enabled_for_mixed_dynamic_scene(
    tmp_path: Path,
) -> None:
    scenario = Scenario(
        name="mixed_complex",
        world="mixed_complex",
        max_steps=100,
        dt=0.1,
        robots=_scenario().robots,
    )

    assert build_scenario_waypoint_manager(
        scenario,
        _world(tmp_path / "world.sdf"),
    ) is not None


def test_densify_goal_path_limits_segment_length() -> None:
    path = [
        GoalState(x=0.0, y=0.0, yaw=0.0),
        GoalState(x=2.4, y=0.0, yaw=0.0),
    ]

    dense = densify_goal_path(path, max_segment_length=1.0)

    assert len(dense) == 4
    assert dense[0] == path[0]
    assert dense[-1] == path[-1]
    assert all(
        math.hypot(next_point.x - point.x, next_point.y - point.y) <= 1.0 + 1e-9
        for point, next_point in zip(dense, dense[1:])
    )


def test_mixed_complex_uses_simplified_densified_lookahead_waypoints() -> None:
    scenario = load_scenario(
        Path("src/mrpp_experiments/config/scenarios/mixed_complex.yaml")
    )

    manager = build_scenario_waypoint_manager(
        scenario,
        Path("src/mrpp_gazebo/worlds/mixed_complex.sdf"),
    )

    assert manager is not None
    for task in scenario.robots:
        path = manager.path_for(task.name)
        path_length = sum(
            math.hypot(next_point.x - point.x, next_point.y - point.y)
            for point, next_point in zip(path, path[1:])
        )
        max_segment = max(
            math.hypot(next_point.x - point.x, next_point.y - point.y)
            for point, next_point in zip(path, path[1:])
        )
        assert 16 <= len(path) <= 40
        assert path[-1] == GoalState(
            x=task.goal.x,
            y=task.goal.y,
            yaw=task.goal.yaw,
        )
        assert max_segment <= 1.0 + 1e-9
        assert path_length <= 26.0
