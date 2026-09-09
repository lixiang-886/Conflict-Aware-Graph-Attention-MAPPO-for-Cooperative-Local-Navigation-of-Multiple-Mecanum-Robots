from pathlib import Path

from mrpp_experiments.experiment_matrix import resolve_algorithm_names
from mrpp_experiments.method_registry import load_method_registry, select_method_names


def test_load_method_registry_includes_proposed_and_ablations() -> None:
    methods = load_method_registry(Path("src/mrpp_experiments/config/methods/paper_methods.csv"))

    names = [method.name for method in methods]
    roles = {method.name: method.role for method in methods}

    assert "ippo" in names
    assert "maddpg" in names
    assert "mo_gat_mappo" in names
    assert "matched_mappo" in names
    assert "gat_mappo" in names
    assert "sensors_mo_gat_mappo" in names
    assert "orca" in names
    assert "ablation_no_graph_attention" in names
    assert "ablation_fixed_reward" in names
    assert "ablation_no_gate" in names
    assert "ablation_no_coordinator" in names
    assert roles["ippo"] == "legacy_baseline"
    assert roles["maddpg"] == "legacy_baseline"
    assert roles["mo_gat_mappo"] == "legacy_proposed"
    assert roles["matched_mappo"] == "baseline"
    assert roles["gat_mappo"] == "baseline"
    assert roles["sensors_mo_gat_mappo"] == "proposed"
    assert roles["priority_astar"] == "diagnostic"
    assert roles["orca"] == "baseline"
    assert roles["ablation_no_graph_attention"] == "ablation"


def test_method_registry_describes_shared_path_layer_and_learning_baselines() -> None:
    methods = load_method_registry(Path("src/mrpp_experiments/config/methods/paper_methods.csv"))
    descriptions = {method.name: method.description for method in methods}

    assert all("path/lookahead" in method.description.lower() for method in methods)
    assert all("safety filter" in method.description.lower() for method in methods)
    assert "independent PPO" in descriptions["ippo"]
    assert "parameter-sharing PPO" in descriptions["mappo"]
    assert "MADDPG-style" in descriptions["maddpg"]
    assert "graph-attention MAPPO" in descriptions["mo_gat_mappo"]
    assert "centralized-critic MAPPO" in descriptions["matched_mappo"]
    assert "CPA graph attention" in descriptions["gat_mappo"]


def test_select_method_names_can_exclude_ablations() -> None:
    methods = load_method_registry(Path("src/mrpp_experiments/config/methods/paper_methods.csv"))

    selected = select_method_names(methods, include_ablations=False)

    assert "priority_astar" not in selected
    assert "ippo" not in selected
    assert "maddpg" not in selected
    assert "orca" in selected
    assert "mappo" not in selected
    assert "mo_gat_mappo" not in selected
    assert "matched_mappo" in selected
    assert "gat_mappo" in selected
    assert "sensors_mo_gat_mappo" in selected
    assert "ablation_fixed_reward" not in selected


def test_resolve_algorithm_names_uses_registry_when_cli_algorithms_are_absent() -> None:
    algorithms = resolve_algorithm_names(
        cli_algorithms=None,
        method_registry=Path("src/mrpp_experiments/config/methods/paper_methods.csv"),
        include_ablations=True,
    )

    assert algorithms == [
        "priority_astar",
        "orca",
        "matched_mappo",
        "gat_mappo",
        "sensors_mo_gat_mappo",
        "ippo",
        "mappo",
        "maddpg",
        "mo_gat_mappo",
        "ablation_no_graph_attention",
        "ablation_fixed_reward",
        "ablation_no_safety_distance",
        "ablation_no_gate",
        "ablation_no_coordinator",
    ]
