from pathlib import Path

from mrpp_experiments.baseline_results import build_template_results, write_template_results


def test_build_template_results_covers_algorithms() -> None:
    rows = build_template_results(
        Path("src/mrpp_experiments/config/scenarios/static_clutter.yaml"),
        algorithms=["priority_astar", "orca"],
        seed=3,
    )

    assert [row.algorithm for row in rows] == ["priority_astar", "orca"]
    assert rows[0].scenario == "static_clutter"
    assert rows[0].robot_count == 4
    assert rows[0].data_source == "template"
    assert rows[0].average_path_length > 0.0


def test_write_template_results(tmp_path: Path) -> None:
    output = tmp_path / "baseline_templates.csv"

    write_template_results(
        scenario_paths=[Path("src/mrpp_experiments/config/scenarios/static_clutter.yaml")],
        output=output,
        algorithms=["priority_astar"],
        seed=0,
    )

    text = output.read_text(encoding="utf-8")
    assert "scenario,algorithm,seed" in text
    assert "static_clutter,priority_astar,0" in text
