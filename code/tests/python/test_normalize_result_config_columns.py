from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_normalizer() -> ModuleType:
    path = Path("scripts/normalize_result_config_columns.py")
    spec = importlib.util.spec_from_file_location("normalize_result_config_columns", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_result_config_columns_preserves_existing_values(tmp_path: Path) -> None:
    normalizer = load_normalizer()
    csv_path = tmp_path / "all_metrics.csv"
    csv_path.write_text(
        "scenario,algorithm,variant,success_rate,orca_prior_enabled,safety_filter_enabled,data_source\n"
        "mixed_complex,mo_gat_mappo,ablation_no_orca_prior,0.0,0,1,gazebo\n",
        encoding="utf-8",
    )

    changed = normalizer.normalize_csv(csv_path)

    assert changed is False
    assert "ablation_no_orca_prior,0.0,0,1,gazebo" in csv_path.read_text(encoding="utf-8")


def test_normalize_result_config_columns_adds_legacy_defaults(tmp_path: Path) -> None:
    normalizer = load_normalizer()
    csv_path = tmp_path / "all_metrics.csv"
    csv_path.write_text(
        "scenario,algorithm,success_rate,data_source,schema_version\n"
        "static_clutter,ippo,1.0,gazebo,mrpp-results-v1\n",
        encoding="utf-8",
    )

    changed = normalizer.normalize_csv(csv_path)

    assert changed is True
    assert csv_path.read_text(encoding="utf-8") == (
        "scenario,algorithm,success_rate,variant,orca_prior_enabled,safety_filter_enabled,data_source,schema_version\n"
        "static_clutter,ippo,1.0,,1,1,gazebo,mrpp-results-v1\n"
    )


def test_normalize_result_config_columns_accepts_empty_main_variant(tmp_path: Path) -> None:
    normalizer = load_normalizer()
    csv_path = tmp_path / "all_metrics.csv"
    csv_path.write_text(
        "scenario,algorithm,success_rate,variant,orca_prior_enabled,safety_filter_enabled,data_source\n"
        "static_clutter,ippo,1.0,,1,1,gazebo\n",
        encoding="utf-8",
    )

    changed = normalizer.normalize_csv(csv_path)

    assert changed is False


def test_normalize_result_config_columns_skips_trajectory_csv_inputs(tmp_path: Path) -> None:
    normalizer = load_normalizer()
    trajectory = tmp_path / "eval_seed0_trajectories.csv"
    trajectory.write_text("robot,time,x,y\nrobot_1,0.0,0.0,0.0\n", encoding="utf-8")

    assert normalizer.iter_inputs([trajectory]) == []
