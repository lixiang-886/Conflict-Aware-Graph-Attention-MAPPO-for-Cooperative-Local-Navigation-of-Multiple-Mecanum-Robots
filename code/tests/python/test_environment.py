from math import pi

import pytest

from mrpp_rl.environment import GoalState, RobotState, build_observation
from mrpp_rl.lidar import LaserSectors


def test_build_observation_includes_goal_and_neighbor_distance() -> None:
    robot = RobotState(name="robot_1", x=0.0, y=0.0, yaw=0.0, v=0.1, omega=0.2)
    goal = GoalState(x=3.0, y=4.0)
    neighbor = RobotState(name="robot_2", x=1.0, y=0.0, yaw=0.0)

    observation = build_observation(robot, goal, [neighbor])

    assert observation.distance_to_goal == 5.0
    assert observation.nearest_robot_distance == 1.0
    assert observation.speed == 0.1


def test_build_observation_carries_laser_sectors() -> None:
    robot = RobotState(name="robot_1", x=0.0, y=0.0, yaw=0.0)
    goal = GoalState(x=1.0, y=0.0)
    sectors = LaserSectors(front=0.5, left=1.5, rear=4.5, right=2.0)

    observation = build_observation(
        robot,
        goal,
        [],
        min_laser=0.5,
        laser_sectors=sectors,
        laser_range_max=4.5,
    )

    assert observation.min_laser_range == 0.5
    assert observation.laser_front == 0.5
    assert observation.laser_left == 1.5
    assert observation.laser_right == 2.0
    assert observation.laser_range_max == 4.5


def test_build_observation_wraps_heading_error() -> None:
    robot = RobotState(name="robot_1", x=0.0, y=0.0, yaw=3.13)
    goal = GoalState(x=-1.0, y=-0.01)

    observation = build_observation(robot, goal, [])

    assert -3.14159 <= observation.heading_error <= 3.14159


def test_pedestrian_relative_position_is_expressed_in_robot_body_frame() -> None:
    robot = RobotState(name="robot_1", x=2.0, y=3.0, yaw=pi / 2.0)
    goal = GoalState(x=2.0, y=4.0)

    observation = build_observation(
        robot,
        goal,
        [],
        pedestrians=[(2.0, 5.0)],
    )

    assert observation.nearest_ped_distance == 2.0
    assert observation.nearest_ped_dx == pytest.approx(2.0)
    assert observation.nearest_ped_dy == pytest.approx(0.0, abs=1e-12)
