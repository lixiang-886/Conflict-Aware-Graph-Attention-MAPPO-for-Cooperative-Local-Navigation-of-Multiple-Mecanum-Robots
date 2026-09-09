import pytest

from mrpp_rl.lidar import LaserSafetyConfig, LaserSectors
from mrpp_rl.lidar import apply_laser_safety_filter, laser_scan_to_observation


def _config() -> LaserSafetyConfig:
    return LaserSafetyConfig(
        stop_distance=0.2,
        slow_distance=0.6,
        emergency_stop_distance=0.1,
        emergency_max_angular_speed=0.4,
        max_angular_speed=2.0,
    )


def test_front_obstacle_limits_forward_velocity() -> None:
    result = apply_laser_safety_filter(
        (0.6, 0.0, 0.0),
        LaserSectors(front=0.4),
        _config(),
    )

    assert 0.0 < result.command[0] < 0.6
    assert result.translation_scale == pytest.approx(0.5)
    assert result.reason == "motion_clearance"


def test_front_obstacle_allows_backing_away() -> None:
    result = apply_laser_safety_filter(
        (-0.4, 0.0, 0.0),
        LaserSectors(front=0.12, rear=4.5, rear_left=4.5, rear_right=4.5),
        _config(),
    )

    assert result.command[0] == -0.4
    assert result.translation_scale == 1.0


def test_left_obstacle_limits_left_motion_but_allows_right_motion() -> None:
    sectors = LaserSectors(left=0.15, front_left=4.5, rear_left=4.5)

    left = apply_laser_safety_filter((0.0, 0.5, 0.0), sectors, _config())
    right = apply_laser_safety_filter((0.0, -0.5, 0.0), sectors, _config())

    assert left.command[1] == 0.0
    assert right.command[1] == -0.5


def test_close_side_obstacle_projects_motion_along_clear_direction() -> None:
    result = apply_laser_safety_filter(
        (0.3, -0.3, 0.0),
        LaserSectors(right=0.15, front=4.5, front_right=4.5),
        _config(),
    )

    assert result.command[0] == pytest.approx(0.3)
    assert abs(result.command[1]) < 1e-9
    assert "obstacle_projection" in result.reason


def test_rear_right_obstacle_does_not_limit_front_left_motion() -> None:
    result = apply_laser_safety_filter(
        (0.4, 0.4, 0.0),
        LaserSectors(rear_right=0.12),
        _config(),
    )

    assert result.command[:2] == (0.4, 0.4)


def test_diagonal_motion_is_scaled_once() -> None:
    result = apply_laser_safety_filter(
        (0.4, 0.4, 0.0),
        LaserSectors(front_left=0.4),
        _config(),
    )

    assert result.command[0] == pytest.approx(0.2)
    assert result.command[1] == pytest.approx(0.2)


def test_front_obstacle_preserves_tangential_mecanum_motion() -> None:
    result = apply_laser_safety_filter(
        (0.4, 0.4, 0.0),
        LaserSectors(front=0.4, front_left=4.5, left=4.5),
        _config(),
    )

    assert result.command[0] == pytest.approx(0.2)
    assert result.command[1] == pytest.approx(0.4)
    assert "obstacle_projection" in result.reason


def test_angular_velocity_is_limited_but_not_cleared_near_obstacle() -> None:
    result = apply_laser_safety_filter(
        (0.0, 0.0, 1.5),
        LaserSectors(right=0.15),
        _config(),
    )

    assert 0.0 < result.command[2] <= 0.4
    assert result.angular_limited


def test_missing_scan_stops_translation_and_flags_stale() -> None:
    result = apply_laser_safety_filter(
        (0.5, 0.1, 1.0),
        None,
        _config(),
    )

    assert result.command[0] == 0.0
    assert result.command[1] == 0.0
    assert result.scan_stale
    assert result.reason == "stale_scan"


def test_stale_scan_stops_translation() -> None:
    observation = laser_scan_to_observation(
        [4.5],
        angle_min=0.0,
        angle_increment=1.0,
        range_min=0.1,
        range_max=4.5,
        stamp_sec=1.0,
    )
    config = LaserSafetyConfig(stale_timeout_sec=0.5)

    result = apply_laser_safety_filter(
        (0.5, 0.0, 0.0),
        observation,
        config,
        current_time_sec=2.0,
    )

    assert result.command[0] == 0.0
    assert result.scan_stale
