from pathlib import Path

import pytest

from mrpp_experiments.experiment_matrix import (
    build_experiment_jobs,
    read_experiment_plan,
    write_experiment_plan,
)
from mrpp_experiments.run_batch import command_for_job, create_batch_manifest
from mrpp_experiments.run_batch import checkpoint_for_job
from mrpp_experiments.run_batch import filter_jobs


def test_build_experiment_jobs_expands_scenarios_algorithms_and_seeds(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    (scenario_dir / "static_clutter.yaml").write_text(
        "name: static_clutter\n", encoding="utf-8"
    )
    (scenario_dir / "pedestrian_dynamic.yaml").write_text(
        "name: pedestrian_dynamic\n", encoding="utf-8"
    )

    jobs = build_experiment_jobs(
        scenario_dir=scenario_dir,
        algorithms=["orca", "mo_gat_mappo"],
        seeds=[0, 1],
        output_root=Path("results/raw"),
    )

    assert len(jobs) == 8
    assert jobs[0].scenario == "pedestrian_dynamic"
    assert jobs[0].algorithm == "orca"
    assert jobs[0].seed == 0
    assert jobs[0].output_csv == Path("results/raw/orca/pedestrian_dynamic_seed0.csv")
    assert jobs[-1].scenario == "static_clutter"
    assert jobs[-1].algorithm == "mo_gat_mappo"
    assert jobs[-1].seed == 1


def test_build_experiment_jobs_can_select_formal_scenarios(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    for name in [
        "static_clutter",
        "pedestrian_dynamic",
        "mixed_complex",
    ]:
        (scenario_dir / f"{name}.yaml").write_text(
            f"name: {name}\n", encoding="utf-8"
        )

    jobs = build_experiment_jobs(
        scenario_dir=scenario_dir,
        algorithms=["mo_gat_mappo"],
        seeds=[0],
        output_root=Path("results/raw"),
        scenario_names=["static_clutter", "pedestrian_dynamic", "mixed_complex"],
    )

    assert [job.scenario for job in jobs] == [
        "static_clutter",
        "pedestrian_dynamic",
        "mixed_complex",
    ]


def test_build_experiment_jobs_rejects_missing_selected_scenario(
    tmp_path: Path,
) -> None:
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="missing_scene"):
        build_experiment_jobs(
            scenario_dir=scenario_dir,
            algorithms=["mo_gat_mappo"],
            seeds=[0],
            output_root=Path("results/raw"),
            scenario_names=["missing_scene"],
        )


def test_experiment_plan_csv_round_trip(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    (scenario_dir / "static_clutter.yaml").write_text(
        "name: static_clutter\n", encoding="utf-8"
    )
    plan_path = tmp_path / "plan.csv"

    jobs = build_experiment_jobs(
        scenario_dir=scenario_dir,
        algorithms=["priority_astar"],
        seeds=[3],
        output_root=Path("results/raw"),
    )

    write_experiment_plan(plan_path, jobs)
    loaded = read_experiment_plan(plan_path)

    assert loaded == jobs


def test_create_batch_manifest_uses_plan_rows(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    scenario_path = scenario_dir / "static_clutter.yaml"
    scenario_path.write_text("name: static_clutter\n", encoding="utf-8")
    plan_path = tmp_path / "plan.csv"
    output_dir = tmp_path / "batch"

    jobs = build_experiment_jobs(
        scenario_dir=scenario_dir,
        algorithms=["mo_gat_mappo"],
        seeds=[5],
        output_root=Path("results/raw"),
    )
    write_experiment_plan(plan_path, jobs)

    manifest = create_batch_manifest(plan_path=plan_path, output_dir=output_dir)

    assert manifest.exists()
    assert "scenario=static_clutter" in manifest.read_text(encoding="utf-8")
    assert "seed=5" in manifest.read_text(encoding="utf-8")


def test_batch_runner_builds_real_evaluate_command(tmp_path: Path) -> None:
    scenario_path = tmp_path / "static_clutter.yaml"
    scenario_path.write_text("name: static_clutter\n", encoding="utf-8")
    jobs = build_experiment_jobs(
        scenario_dir=tmp_path,
        algorithms=["orca"],
        seeds=[7],
        output_root=Path("results/raw"),
    )

    command = command_for_job(jobs[0], checkpoint_dir=Path("models/checkpoints"))

    assert command[:4] == ["ros2", "run", "mrpp_rl", "evaluate_mrpp"]
    assert "--algorithm" in command
    assert "orca" in command
    assert "--checkpoint" not in command


def test_batch_runner_can_cap_smoke_steps(tmp_path: Path) -> None:
    scenario_path = tmp_path / "static_clutter.yaml"
    scenario_path.write_text("name: static_clutter\n", encoding="utf-8")
    jobs = build_experiment_jobs(
        scenario_dir=tmp_path,
        algorithms=["orca"],
        seeds=[7],
        output_root=Path("results/raw"),
    )

    command = command_for_job(
        jobs[0],
        checkpoint_dir=Path("models/checkpoints"),
        max_steps=40,
    )

    assert command[-2:] == ["--max-steps", "40"]


def test_batch_runner_requires_checkpoint_for_learning_algorithm(tmp_path: Path) -> None:
    scenario_path = tmp_path / "static_clutter.yaml"
    scenario_path.write_text("name: static_clutter\n", encoding="utf-8")
    jobs = build_experiment_jobs(
        scenario_dir=tmp_path,
        algorithms=["mo_gat_mappo"],
        seeds=[7],
        output_root=Path("results/raw"),
    )

    command = command_for_job(jobs[0], checkpoint_dir=Path("models/checkpoints"))

    assert "--checkpoint" in command
    assert (
        "models/checkpoints/independent_learning_seed7/"
        "static_clutter/mo_gat_mappo_best.pt"
    ) in command


def test_batch_runner_requires_checkpoint_for_ippo(tmp_path: Path) -> None:
    scenario_path = tmp_path / "static_clutter.yaml"
    scenario_path.write_text("name: static_clutter\n", encoding="utf-8")
    jobs = build_experiment_jobs(
        scenario_dir=tmp_path,
        algorithms=["ippo"],
        seeds=[7],
        output_root=Path("results/raw"),
    )

    command = command_for_job(jobs[0], checkpoint_dir=Path("models/checkpoints"))

    assert "--checkpoint" in command
    assert (
        "models/checkpoints/independent_learning_seed7/"
        "static_clutter/ippo_best.pt"
    ) in command


def test_batch_runner_requires_checkpoint_for_maddpg(tmp_path: Path) -> None:
    scenario_path = tmp_path / "static_clutter.yaml"
    scenario_path.write_text("name: static_clutter\n", encoding="utf-8")
    jobs = build_experiment_jobs(
        scenario_dir=tmp_path,
        algorithms=["maddpg"],
        seeds=[7],
        output_root=Path("results/raw"),
    )

    command = command_for_job(jobs[0], checkpoint_dir=Path("models/checkpoints"))

    assert "--checkpoint" in command
    assert (
        "models/checkpoints/independent_learning_seed7/"
        "static_clutter/maddpg_best.pt"
    ) in command


def test_batch_runner_prefers_scenario_best_checkpoint(tmp_path: Path) -> None:
    scenario_path = tmp_path / "static_clutter.yaml"
    scenario_path.write_text("name: static_clutter\n", encoding="utf-8")
    jobs = build_experiment_jobs(
        scenario_dir=tmp_path,
        algorithms=["mappo"],
        seeds=[0],
        output_root=Path("results/raw"),
    )
    checkpoint_dir = tmp_path / "checkpoints"
    scenario_checkpoint = checkpoint_dir / "static_clutter" / "mappo_best.pt"
    scenario_checkpoint.parent.mkdir(parents=True)
    scenario_checkpoint.write_text("placeholder", encoding="utf-8")

    command = command_for_job(jobs[0], checkpoint_dir=checkpoint_dir)

    assert checkpoint_for_job(jobs[0], checkpoint_dir) == scenario_checkpoint
    assert str(scenario_checkpoint) in command


def test_batch_runner_filters_plan_jobs(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    for name in ("static_clutter", "pedestrian_dynamic"):
        (scenario_dir / f"{name}.yaml").write_text(
            f"name: {name}\n", encoding="utf-8"
        )
    jobs = build_experiment_jobs(
        scenario_dir=scenario_dir,
        algorithms=["orca", "mo_gat_mappo"],
        seeds=[0, 1],
        output_root=Path("results/raw"),
    )

    selected = filter_jobs(
        jobs,
        scenarios=["static_clutter"],
        algorithm="orca",
        seeds=[1],
    )

    assert len(selected) == 1
    assert selected[0].scenario == "static_clutter"
    assert selected[0].algorithm == "orca"
    assert selected[0].seed == 1
