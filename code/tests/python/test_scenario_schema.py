from pathlib import Path

from mrpp_experiments.scenario import EvaluationRandomizationConfig
from mrpp_experiments.scenario import load_scenario, randomize_scenario

FORMAL_SCENARIOS = {
    "static_clutter",
    "pedestrian_dynamic",
    "mixed_complex",
    "unseen_dense_crossing_n4",
    "unseen_dense_crossing_n6",
    "unseen_dense_crossing_n8",
    "unseen_dense_crossing_n10",
}


def test_load_static_clutter_scenario() -> None:
    scenario = load_scenario(Path("src/mrpp_experiments/config/scenarios/static_clutter.yaml"))

    assert scenario.name == "static_clutter"
    assert scenario.world == "static_clutter"
    assert scenario.max_steps == 1800
    assert scenario.dt == 0.1
    assert len(scenario.robots) == 4
    assert scenario.robots[0].name == "robot_1"
    assert scenario.robots[0].start.x == -5.0
    assert scenario.robots[0].goal.y == 4.0


def test_only_formal_scenarios_are_kept() -> None:
    scenario_dir = Path("src/mrpp_experiments/config/scenarios")
    scenario_names = {path.stem for path in scenario_dir.glob("*.yaml")}

    assert scenario_names == FORMAL_SCENARIOS


def test_unseen_dense_crossing_team_sizes_and_routes() -> None:
    scenario_dir = Path("src/mrpp_experiments/config/scenarios")
    for count in (4, 6, 8, 10):
        scenario = load_scenario(
            scenario_dir / f"unseen_dense_crossing_n{count}.yaml"
        )
        assert scenario.world == "unseen_dense_crossing"
        assert len(scenario.robots) == count
        for robot in scenario.robots:
            assert abs(robot.start.x + robot.goal.x) < 1e-6
            assert abs(robot.start.y + robot.goal.y) < 1e-6


def test_evaluation_randomization_is_seeded_and_keeps_goals_fixed() -> None:
    scenario = load_scenario(
        Path("src/mrpp_experiments/config/scenarios/pedestrian_dynamic.yaml")
    )
    config = EvaluationRandomizationConfig(enabled=True)

    first = randomize_scenario(scenario, config, seed=19)
    second = randomize_scenario(scenario, config, seed=19)

    assert first == second
    assert first.robots[0].start != scenario.robots[0].start
    assert first.robots[0].goal == scenario.robots[0].goal
    assert abs(first.robots[0].start.x - scenario.robots[0].start.x) <= 0.05
