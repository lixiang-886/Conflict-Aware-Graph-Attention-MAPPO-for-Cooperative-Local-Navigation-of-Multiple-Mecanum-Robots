from pathlib import Path

from mrpp_experiments.dry_run_results import build_dry_run_results, write_dry_run_results
from mrpp_experiments.experiment_matrix import build_experiment_jobs, write_experiment_plan
from mrpp_experiments.results import read_results_csv


def test_build_dry_run_results_preserves_plan_identity(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.csv"
    jobs = build_experiment_jobs(
        scenario_dir=Path("src/mrpp_experiments/config/scenarios"),
        algorithms=["mo_gat_mappo"],
        seeds=[2],
        output_root=Path("results/raw"),
    )
    write_experiment_plan(plan_path, jobs[:1])

    rows = build_dry_run_results(plan_path)

    assert len(rows) == 1
    assert rows[0].scenario == jobs[0].scenario
    assert rows[0].algorithm == "mo_gat_mappo"
    assert rows[0].seed == 2
    assert rows[0].data_source == "dry_run"
    assert rows[0].robot_count > 0


def test_write_dry_run_results_creates_canonical_csv(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.csv"
    output = tmp_path / "all_metrics_dry_run.csv"
    jobs = build_experiment_jobs(
        scenario_dir=Path("src/mrpp_experiments/config/scenarios"),
        algorithms=["priority_astar"],
        seeds=[0, 1],
        output_root=Path("results/raw"),
    )
    write_experiment_plan(plan_path, jobs[:2])

    write_dry_run_results(plan_path=plan_path, output=output)
    rows = read_results_csv(output)

    assert len(rows) == 2
    assert {row.schema_version for row in rows} == {"mrpp-results-v2"}
    assert {row.data_source for row in rows} == {"dry_run"}
