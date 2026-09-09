import math

import pytest

from mrpp_rl.lidar import LaserSectors, laser_scan_to_observation
from mrpp_rl.lidar import laser_scan_to_sectors


def test_laser_observation_sanitizes_ranges_and_preserves_stamp() -> None:
    observation = laser_scan_to_observation(
        [float("nan"), float("inf"), float("-inf"), 0.05, 9.0, 1.2],
        angle_min=0.0,
        angle_increment=math.pi / 8.0,
        range_min=0.1,
        range_max=4.5,
        stamp_sec=12.5,
        message_stamp_sec=3.25,
    )

    assert math.isnan(observation.ranges[0])
    assert observation.ranges[1] == 4.5
    assert math.isnan(observation.ranges[2])
    assert math.isnan(observation.ranges[3])
    assert observation.ranges[4] == 4.5
    assert observation.minimum_distance == 1.2
    assert observation.valid_sample_count == 3
    assert observation.stamp_sec == 12.5
    assert observation.message_stamp_sec == 3.25


def test_laser_observation_empty_scan_has_no_valid_samples() -> None:
    observation = laser_scan_to_observation(
        [],
        angle_min=0.0,
        angle_increment=0.0,
        range_min=0.1,
        range_max=4.5,
    )

    assert observation.valid_sample_count == 0
    assert observation.minimum_distance == 4.5
    assert observation.sectors == LaserSectors()


def test_laser_sectors_follow_angle_min_and_increment() -> None:
    sectors = laser_scan_to_sectors(
        [1.0, 2.0, 0.4, float("nan"), 3.0],
        angle_min=-math.pi / 2.0,
        angle_increment=math.pi / 4.0,
        range_min=0.1,
        range_max=4.5,
        sector_statistic="minimum",
    )

    assert sectors.right == 1.0
    assert sectors.front == 0.4
    assert sectors.left == 3.0


def test_laser_sectors_handle_positive_negative_pi_boundary() -> None:
    sectors = laser_scan_to_sectors(
        [0.8, 2.0, 0.7],
        angle_min=math.pi - 0.05,
        angle_increment=0.05,
        range_min=0.1,
        range_max=4.5,
        sector_statistic="minimum",
    )

    assert sectors.rear == pytest.approx(0.7)


def test_laser_sector_percentile_ignores_single_low_noise_point() -> None:
    sectors = laser_scan_to_sectors(
        [0.11, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
        angle_min=-0.01,
        angle_increment=0.002,
        range_min=0.1,
        range_max=4.5,
        sector_statistic="percentile",
        percentile=10.0,
    )

    assert sectors.front == pytest.approx(1.0)
