# Copyright 2026 lixiang-886
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class AlgorithmConfig:
    name: str
    requires_checkpoint: bool
    use_graph_attention: bool
    use_adaptive_reward: bool
    use_safety_distance_reward: bool
    use_fully_connected_graph: bool = False
    use_conflict_coordinator: bool = False
    use_interaction_gate: bool = False
    use_centralized_critic: bool = False
    update_mode: str = "joint_ppo"

    def to_dict(self) -> dict[str, bool | str]:
        return asdict(self)


_CONFIGS = {
    "priority_astar": AlgorithmConfig(
        name="priority_astar",
        requires_checkpoint=False,
        use_graph_attention=False,
        use_adaptive_reward=False,
        use_safety_distance_reward=True,
        update_mode="none",
    ),
    "orca": AlgorithmConfig(
        name="orca",
        requires_checkpoint=False,
        use_graph_attention=False,
        use_adaptive_reward=False,
        use_safety_distance_reward=True,
        update_mode="none",
    ),
    "rvo2": AlgorithmConfig(
        name="rvo2",
        requires_checkpoint=False,
        use_graph_attention=False,
        use_adaptive_reward=False,
        use_safety_distance_reward=True,
        update_mode="none",
    ),
    "ippo": AlgorithmConfig(
        name="ippo",
        requires_checkpoint=True,
        use_graph_attention=False,
        use_adaptive_reward=False,
        use_safety_distance_reward=True,
        update_mode="independent_ppo",
    ),
    "mappo": AlgorithmConfig(
        name="mappo",
        requires_checkpoint=True,
        use_graph_attention=False,
        use_adaptive_reward=True,
        use_safety_distance_reward=True,
        use_centralized_critic=True,
        update_mode="joint_ppo",
    ),
    "maddpg": AlgorithmConfig(
        name="maddpg",
        requires_checkpoint=True,
        use_graph_attention=False,
        use_adaptive_reward=False,
        use_safety_distance_reward=True,
        update_mode="maddpg",
    ),
    "matched_mappo": AlgorithmConfig(
        name="matched_mappo",
        requires_checkpoint=True,
        use_graph_attention=False,
        use_adaptive_reward=True,
        use_safety_distance_reward=True,
        use_centralized_critic=True,
        update_mode="joint_ppo",
    ),
    "gat_mappo": AlgorithmConfig(
        name="gat_mappo",
        requires_checkpoint=True,
        use_graph_attention=True,
        use_adaptive_reward=True,
        use_safety_distance_reward=True,
        use_centralized_critic=True,
        update_mode="joint_ppo",
    ),
    "sensors_mo_gat_mappo": AlgorithmConfig(
        name="sensors_mo_gat_mappo",
        requires_checkpoint=True,
        use_graph_attention=True,
        use_adaptive_reward=True,
        use_safety_distance_reward=True,
        use_conflict_coordinator=True,
        use_interaction_gate=True,
        use_centralized_critic=True,
        update_mode="joint_ppo",
    ),
    "mo_gat_mappo": AlgorithmConfig(
        name="mo_gat_mappo",
        requires_checkpoint=True,
        use_graph_attention=True,
        use_adaptive_reward=True,
        use_safety_distance_reward=True,
        use_conflict_coordinator=True,
        use_interaction_gate=True,
        use_centralized_critic=True,
        update_mode="joint_ppo",
    ),
    "fc_gat_mappo": AlgorithmConfig(
        name="fc_gat_mappo",
        requires_checkpoint=True,
        use_graph_attention=True,
        use_adaptive_reward=True,
        use_safety_distance_reward=True,
        use_fully_connected_graph=True,
        use_conflict_coordinator=True,
        use_interaction_gate=True,
        use_centralized_critic=True,
        update_mode="joint_ppo",
    ),
    "ablation_no_graph_attention": AlgorithmConfig(
        name="ablation_no_graph_attention",
        requires_checkpoint=True,
        use_graph_attention=False,
        use_adaptive_reward=True,
        use_safety_distance_reward=True,
        use_conflict_coordinator=True,
        use_interaction_gate=True,
        use_centralized_critic=True,
        update_mode="joint_ppo",
    ),
    "ablation_fixed_reward": AlgorithmConfig(
        name="ablation_fixed_reward",
        requires_checkpoint=True,
        use_graph_attention=True,
        use_adaptive_reward=False,
        use_safety_distance_reward=True,
        use_conflict_coordinator=True,
        use_interaction_gate=True,
        use_centralized_critic=True,
        update_mode="joint_ppo",
    ),
    "ablation_no_safety_distance": AlgorithmConfig(
        name="ablation_no_safety_distance",
        requires_checkpoint=True,
        use_graph_attention=True,
        use_adaptive_reward=True,
        use_safety_distance_reward=False,
        use_conflict_coordinator=True,
        use_interaction_gate=True,
        use_centralized_critic=True,
        update_mode="joint_ppo",
    ),
    "ablation_no_gate": AlgorithmConfig(
        name="ablation_no_gate",
        requires_checkpoint=True,
        use_graph_attention=True,
        use_adaptive_reward=True,
        use_safety_distance_reward=True,
        use_conflict_coordinator=True,
        use_interaction_gate=False,
        use_centralized_critic=True,
        update_mode="joint_ppo",
    ),
    "ablation_no_coordinator": AlgorithmConfig(
        name="ablation_no_coordinator",
        requires_checkpoint=True,
        use_graph_attention=True,
        use_adaptive_reward=True,
        use_safety_distance_reward=True,
        use_conflict_coordinator=False,
        use_interaction_gate=True,
        use_centralized_critic=True,
        update_mode="joint_ppo",
    ),
}

SUPPORTED_ALGORITHMS = tuple(_CONFIGS)
CHECKPOINT_ALGORITHMS = tuple(
    name for name, config in _CONFIGS.items() if config.requires_checkpoint
)
BASELINE_ALGORITHMS = tuple(
    name for name, config in _CONFIGS.items() if not config.requires_checkpoint
)
MATCHED_ALGORITHMS = (
    "matched_mappo",
    "gat_mappo",
    "sensors_mo_gat_mappo",
)
MAIN_ALGORITHMS = (
    "ippo",
    "mappo",
    "maddpg",
    "mo_gat_mappo",
)
GRAPH_ALGORITHMS = (
    "mo_gat_mappo",
    "fc_gat_mappo",
)
MODULAR_SAFETY_ALGORITHMS = (
    *MATCHED_ALGORITHMS,
    *MAIN_ALGORITHMS,
    "fc_gat_mappo",
    "ablation_no_graph_attention",
    "ablation_fixed_reward",
    "ablation_no_safety_distance",
    "ablation_no_gate",
    "ablation_no_coordinator",
)


def get_algorithm_config(name: str) -> AlgorithmConfig:
    try:
        return _CONFIGS[name]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_ALGORITHMS)
        raise ValueError(f"Unsupported algorithm '{name}'. Supported: {supported}") from exc


def adjacency_matrix(robot_count: int, use_graph_attention: bool) -> list[list[float]]:
    if robot_count <= 0:
        return []
    if use_graph_attention:
        return [
            [1.0 for _ in range(robot_count)]
            for _ in range(robot_count)
        ]
    return [
        [1.0 if row == col else 0.0 for col in range(robot_count)]
        for row in range(robot_count)
    ]


def algorithm_config_from_checkpoint(
    checkpoint: Mapping[str, object],
) -> AlgorithmConfig:
    raw = checkpoint.get("algorithm_config")
    if not isinstance(raw, Mapping):
        raise ValueError(
            "Checkpoint is missing algorithm_config. Re-train with the current "
            "code or use a checkpoint produced after algorithm configs were added."
        )
    name = raw.get("name")
    if not isinstance(name, str):
        raise ValueError("Checkpoint algorithm_config.name is missing or invalid.")
    expected = get_algorithm_config(name)
    for field in (
        "requires_checkpoint",
        "use_graph_attention",
        "use_fully_connected_graph",
        "use_adaptive_reward",
        "use_safety_distance_reward",
        "use_conflict_coordinator",
        "use_interaction_gate",
        "use_centralized_critic",
        "update_mode",
    ):
        raw_value = raw.get(field)
        expected_value = getattr(expected, field)
        if raw_value is None:
            raw_value = expected_value
        if isinstance(expected_value, bool):
            matches = bool(raw_value) == expected_value
        else:
            matches = raw_value == expected_value
        if not matches:
            raise ValueError(
                f"Checkpoint algorithm_config.{field} does not match "
                f"registered settings for {name}."
            )
    return expected


def validate_checkpoint_algorithm(
    checkpoint: Mapping[str, object],
    requested_algorithm: str,
) -> AlgorithmConfig:
    requested = get_algorithm_config(requested_algorithm)
    stored = algorithm_config_from_checkpoint(checkpoint)
    if stored.name != requested.name:
        raise ValueError(
            "Checkpoint algorithm mismatch: "
            f"checkpoint={stored.name}, requested={requested.name}."
        )
    return stored
