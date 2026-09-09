import pytest

from mrpp_rl.algorithm_config import adjacency_matrix
from mrpp_rl.algorithm_config import get_algorithm_config
from mrpp_rl.algorithm_config import validate_checkpoint_algorithm
from mrpp_rl.control import control_profile_for_algorithm
from mrpp_rl.training_config import mappo_config_for_algorithm


def test_algorithm_configs_make_ablations_distinct() -> None:
    full = get_algorithm_config("mo_gat_mappo")
    no_graph = get_algorithm_config("ablation_no_graph_attention")
    fixed_reward = get_algorithm_config("ablation_fixed_reward")
    no_safety = get_algorithm_config("ablation_no_safety_distance")
    no_gate = get_algorithm_config("ablation_no_gate")
    no_coordinator = get_algorithm_config("ablation_no_coordinator")

    assert full.use_graph_attention
    assert full.use_adaptive_reward
    assert full.use_safety_distance_reward
    assert not no_graph.use_graph_attention
    assert not fixed_reward.use_adaptive_reward
    assert not no_safety.use_safety_distance_reward
    assert full.use_conflict_coordinator
    assert full.use_interaction_gate
    assert no_graph.use_conflict_coordinator
    assert no_graph.use_interaction_gate
    assert not no_gate.use_interaction_gate
    assert no_gate.use_conflict_coordinator
    assert no_coordinator.use_interaction_gate
    assert not no_coordinator.use_conflict_coordinator


def test_baselines_do_not_require_checkpoints() -> None:
    assert not get_algorithm_config("priority_astar").requires_checkpoint
    assert not get_algorithm_config("orca").requires_checkpoint
    assert not get_algorithm_config("rvo2").requires_checkpoint
    assert get_algorithm_config("ippo").requires_checkpoint
    assert get_algorithm_config("mappo").requires_checkpoint
    assert get_algorithm_config("maddpg").requires_checkpoint


def test_ippo_uses_independent_ppo_update_mode() -> None:
    assert get_algorithm_config("ippo").update_mode == "independent_ppo"
    assert get_algorithm_config("mappo").update_mode == "joint_ppo"
    assert get_algorithm_config("maddpg").update_mode == "maddpg"
    assert get_algorithm_config("mo_gat_mappo").update_mode == "joint_ppo"


def test_canonical_rvo2_is_distinct_from_the_custom_orca_style_baseline() -> None:
    rvo2 = get_algorithm_config("rvo2")
    assert rvo2.update_mode == "none"
    assert rvo2.name != get_algorithm_config("orca").name


def test_sensors_methods_have_matched_centralized_training_contract() -> None:
    matched = get_algorithm_config("matched_mappo")
    gat = get_algorithm_config("gat_mappo")
    full = get_algorithm_config("sensors_mo_gat_mappo")

    assert all(
        config.use_centralized_critic
        for config in (matched, gat, full)
    )
    assert all(config.use_adaptive_reward for config in (matched, gat, full))
    assert not matched.use_graph_attention
    assert gat.use_graph_attention
    assert not gat.use_conflict_coordinator
    assert not gat.use_interaction_gate
    assert full.use_graph_attention
    assert full.use_conflict_coordinator
    assert full.use_interaction_gate


def test_electronics_mappo_matches_proposed_ppo_contract() -> None:
    mappo = get_algorithm_config("mappo")
    proposed = get_algorithm_config("mo_gat_mappo")
    assert mappo.use_adaptive_reward
    assert mappo.use_centralized_critic
    mappo_ppo = mappo_config_for_algorithm(mappo, 2001)
    proposed_ppo = mappo_config_for_algorithm(proposed, 2001)
    shared_fields = (
        "gamma",
        "gae_lambda",
        "clip_ratio",
        "actor_lr",
        "critic_lr",
        "entropy_coef",
        "value_coef",
        "ppo_epochs",
        "mini_batch_size",
        "max_grad_norm",
        "action_std",
        "centralized_critic",
        "team_advantage_mix",
    )
    assert all(
        getattr(mappo_ppo, field) == getattr(proposed_ppo, field)
        for field in shared_fields
    )


def test_fc_gat_differs_from_cpa_gat_only_in_graph_topology() -> None:
    cpa = get_algorithm_config("mo_gat_mappo").to_dict()
    fully_connected = get_algorithm_config("fc_gat_mappo").to_dict()
    assert not cpa["use_fully_connected_graph"]
    assert fully_connected["use_fully_connected_graph"]
    cpa.pop("name")
    fully_connected.pop("name")
    cpa.pop("use_fully_connected_graph")
    fully_connected.pop("use_fully_connected_graph")
    assert cpa == fully_connected


def test_electronics_ablations_keep_centralized_critic() -> None:
    methods = (
        "ablation_no_graph_attention",
        "ablation_fixed_reward",
        "ablation_no_gate",
        "ablation_no_coordinator",
    )
    assert all(get_algorithm_config(method).use_centralized_critic for method in methods)


@pytest.mark.parametrize(
    "scenario",
    ["static_clutter", "pedestrian_dynamic", "mixed_complex"],
)
def test_electronics_main_methods_share_control_profile(scenario: str) -> None:
    profiles = {
        control_profile_for_algorithm(method, scenario)
        for method in ("ippo", "mappo", "maddpg", "mo_gat_mappo")
    }
    assert len(profiles) == 1


def test_graph_disabled_uses_identity_adjacency() -> None:
    assert adjacency_matrix(3, use_graph_attention=False) == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert adjacency_matrix(2, use_graph_attention=True) == [
        [1.0, 1.0],
        [1.0, 1.0],
    ]


def test_checkpoint_algorithm_mismatch_is_rejected() -> None:
    checkpoint = {
        "algorithm_config": get_algorithm_config("mappo").to_dict(),
        "policy_state_dict": {},
    }

    with pytest.raises(ValueError, match="Checkpoint algorithm mismatch"):
        validate_checkpoint_algorithm(checkpoint, "mo_gat_mappo")


def test_checkpoint_without_update_mode_remains_compatible() -> None:
    checkpoint_config = get_algorithm_config("mappo").to_dict()
    checkpoint_config.pop("update_mode")
    checkpoint = {
        "algorithm_config": checkpoint_config,
        "policy_state_dict": {},
    }

    assert validate_checkpoint_algorithm(checkpoint, "mappo").name == "mappo"


def test_checkpoint_without_new_component_flags_remains_compatible() -> None:
    checkpoint_config = get_algorithm_config("mo_gat_mappo").to_dict()
    checkpoint_config.pop("use_conflict_coordinator")
    checkpoint_config.pop("use_interaction_gate")
    checkpoint = {
        "algorithm_config": checkpoint_config,
        "policy_state_dict": {},
    }

    assert validate_checkpoint_algorithm(checkpoint, "mo_gat_mappo").name == (
        "mo_gat_mappo"
    )
