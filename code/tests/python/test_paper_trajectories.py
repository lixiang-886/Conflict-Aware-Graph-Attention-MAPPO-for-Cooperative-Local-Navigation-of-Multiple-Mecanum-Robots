from pathlib import Path

from mrpp_paper.plot_trajectories import (
    _default_world_path,
    _pedestrian_paths_for_world,
    read_trajectory_csv,
)
from mrpp_experiments.scenario import load_scenario
from mrpp_rl.waypoints import parse_static_obstacles


def test_read_trajectory_csv_groups_and_sorts_points(tmp_path: Path) -> None:
    csv_path = tmp_path / "trajectories.csv"
    csv_path.write_text(
        "scenario,algorithm,seed,robot_name,point_index,t,x,y,v,omega,success,completion_time,schema_version\n"
        "static_clutter,mo_gat_mappo,0,robot_1,1,0.2,1.0,2.0,0.3,0.1,1,0.3,mrpp-results-v1\n"
        "static_clutter,mo_gat_mappo,0,robot_1,0,0.1,0.5,1.5,0.2,0.0,1,0.3,mrpp-results-v1\n"
        "static_clutter,mo_gat_mappo,0,robot_2,0,0.1,-0.5,1.0,0.1,0.0,0,,mrpp-results-v1\n",
        encoding="utf-8",
    )

    trajectories = read_trajectory_csv(csv_path)

    assert list(trajectories) == ["robot_1", "robot_2"]
    assert trajectories["robot_1"][0]["t"] == 0.1
    assert trajectories["robot_1"][1]["x"] == 1.0
    assert trajectories["robot_2"][0]["success"] == 0


def test_pedestrian_paths_for_world_follow_spawn_configs() -> None:
    dynamic_paths = _pedestrian_paths_for_world("pedestrian_dynamic")
    mixed_paths = _pedestrian_paths_for_world("mixed_complex")

    assert list(dynamic_paths) == ["ped_1", "ped_2", "ped_3", "ped_4"]
    assert list(mixed_paths) == ["ped_1", "ped_2", "ped_3"]
    assert dynamic_paths["ped_1"][0] == (-5.0, 3.0)
    assert mixed_paths["ped_3"][0] == (-3.0, -4.5)


def test_default_world_path_resolves_static_obstacles() -> None:
    scenario = load_scenario(
        Path("src/mrpp_experiments/config/scenarios/static_clutter.yaml")
    )

    world_path = _default_world_path(scenario)

    assert world_path is not None
    assert world_path.name == "static_clutter.sdf"
    assert parse_static_obstacles(world_path)
