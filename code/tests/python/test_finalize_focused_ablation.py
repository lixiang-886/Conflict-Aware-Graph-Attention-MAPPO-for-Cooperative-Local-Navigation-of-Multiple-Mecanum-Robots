from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_finalizer() -> ModuleType:
    path = Path("scripts/finalize_focused_ablation.py")
    spec = importlib.util.spec_from_file_location("finalize_focused_ablation", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_config_requires_matched_realtime_protocol() -> None:
    finalizer = load_finalizer()
    config = {
        "scenario": "mixed_complex",
        "variant": "ablation_no_gate",
        "seed": 1,
        "episodes": 30,
        "dt": 0.1,
        "device": "cuda",
        "sim_step_mode": "realtime",
        "resume_checkpoint": None,
        "validation_interval": 5,
        "validation_rollback": False,
        "initial_validation_diagnostic_only": True,
        "orca_prior_enabled": True,
        "safety_filter_enabled": True,
        "algorithm": {"name": "ablation_no_gate"},
    }

    assert not finalizer.validate_training_config(
        config,
        "ablation_no_gate",
        "ablation_no_gate",
        1,
        30,
    )

    config["validation_rollback"] = True
    errors = finalizer.validate_training_config(
        config,
        "ablation_no_gate",
        "ablation_no_gate",
        1,
        30,
    )
    assert any("validation_rollback" in error for error in errors)


def test_expected_paths_keep_variant_runs_isolated() -> None:
    finalizer = load_finalizer()
    paths = finalizer.expected_paths(
        Path("eval"),
        Path("checkpoints"),
        Path("training"),
        "ablation_no_graph_attention",
        2,
    )

    assert "formal_ablation_no_gat_seed2" in str(paths["checkpoint"])
    assert paths["checkpoint"].name == "ablation_no_graph_attention_best.pt"
    assert "formal_ablation_no_gat_seed2" in str(paths["trajectory"])


def test_hash_table_records_relative_artifact_and_digest() -> None:
    finalizer = load_finalizer()
    lines = finalizer.build_hash_table(
        "Best Checkpoints",
        {"checkpoints/full_seed0/best.pt": "abc123"},
        relative_to=Path("checkpoints"),
    )
    rendered = "\n".join(lines)

    assert "### Best Checkpoints" in rendered
    assert "`full_seed0/best.pt`" in rendered
    assert "`abc123`" in rendered
