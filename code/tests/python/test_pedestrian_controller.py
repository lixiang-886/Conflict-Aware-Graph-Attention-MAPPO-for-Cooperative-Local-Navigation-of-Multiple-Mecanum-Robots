from pathlib import Path

from mrpp_gazebo.pedestrian_controller import (
    PedestrianConfig,
    get_pedestrian_configs,
    pedestrian_pose_at,
    randomized_pedestrian_configs,
)
from mrpp_rl.waypoints import parse_static_obstacles


def test_pedestrian_pose_at_moves_forward_and_back() -> None:
    cfg = PedestrianConfig(
        name="ped_1",
        start_x=-5.0,
        start_y=3.0,
        yaw=0.0,
        speed=0.5,
        direction="east",
    )

    start_x, start_y, _yaw, start_vx, _start_vy = pedestrian_pose_at(cfg, 0.0)
    mid_x, mid_y, _yaw, mid_vx, _mid_vy = pedestrian_pose_at(cfg, 10.0)
    back_x, back_y, _yaw, back_vx, _back_vy = pedestrian_pose_at(cfg, 30.0)

    assert (start_x, start_y) == (-5.0, 3.0)
    assert (mid_x, mid_y) == (0.0, 3.0)
    assert mid_vx > 0.0
    assert (back_x, back_y) == (0.0, 3.0)
    assert back_vx < 0.0


def test_pedestrian_dynamic_uses_four_formal_pedestrians() -> None:
    configs = get_pedestrian_configs("pedestrian_dynamic")

    assert [cfg.name for cfg in configs] == ["ped_1", "ped_2", "ped_3", "ped_4"]


def test_randomized_pedestrians_have_repeatable_phase_and_speed() -> None:
    first = randomized_pedestrian_configs("pedestrian_dynamic", seed=23)
    second = randomized_pedestrian_configs("pedestrian_dynamic", seed=23)

    assert first == second
    assert any(config.phase_offset_sec > 0.0 for config in first)
    assert any(
        randomized.speed != nominal.speed
        for randomized, nominal in zip(
            first,
            get_pedestrian_configs("pedestrian_dynamic"),
        )
    )


def test_mixed_complex_crossing_pedestrians_avoid_fixed_top_corridor_conflict() -> None:
    configs = {
        cfg.name: cfg
        for cfg in get_pedestrian_configs("mixed_complex")
    }

    assert list(configs) == ["ped_1", "ped_2", "ped_3"]
    assert configs["ped_1"].speed == 0.08
    assert configs["ped_2"].speed == 0.2

    ped_1_x, ped_1_y, *_ = pedestrian_pose_at(configs["ped_1"], 30.0)
    ped_2_x, ped_2_y, *_ = pedestrian_pose_at(configs["ped_2"], 30.0)
    assert configs["ped_1"].start_y == 4.4
    assert configs["ped_2"].start_y == -4.4
    assert ((ped_1_x + 1.65) ** 2 + (ped_1_y - 5.6) ** 2) ** 0.5 > 0.75
    assert ((ped_2_x - 1.65) ** 2 + (ped_2_y + 5.6) ** 2) ** 0.5 > 0.75
    for t in (20.0, 24.0, 28.0):
        ped_1_x, ped_1_y, *_ = pedestrian_pose_at(configs["ped_1"], t)
        assert ((ped_1_x + 1.86) ** 2 + (ped_1_y - 4.53) ** 2) ** 0.5 > 0.75


def test_mixed_complex_pedestrians_do_not_start_inside_static_obstacles() -> None:
    obstacles = parse_static_obstacles(
        Path("src/mrpp_gazebo/worlds/mixed_complex.sdf")
    )
    configs = get_pedestrian_configs("mixed_complex")

    for cfg in configs:
        assert not any(
            _contains(obstacle, cfg.start_x, cfg.start_y)
            for obstacle in obstacles
        )


def _contains(obstacle, x: float, y: float) -> bool:
    radius = getattr(obstacle, "radius", None)
    if radius is not None:
        return ((x - obstacle.x) ** 2 + (y - obstacle.y) ** 2) ** 0.5 <= radius
    sx = getattr(obstacle, "sx", None)
    sy = getattr(obstacle, "sy", None)
    if sx is None or sy is None:
        return False
    return abs(x - obstacle.x) <= sx / 2.0 and abs(y - obstacle.y) <= sy / 2.0
