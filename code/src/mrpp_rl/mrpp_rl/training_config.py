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

from mrpp_rl.algorithm_config import AlgorithmConfig, MATCHED_ALGORITHMS
from mrpp_rl.mappo import MAPPOConfig


def mappo_config_for_algorithm(
    algorithm_config: AlgorithmConfig,
    rollout_steps: int,
) -> MAPPOConfig:
    """Return the registered PPO schedule without importing ROS runtime code."""
    config_kwargs = {
        "rollout_steps": rollout_steps,
        "use_graph_attention": algorithm_config.use_graph_attention,
        "independent_agents": algorithm_config.update_mode == "independent_ppo",
        "centralized_critic": algorithm_config.use_centralized_critic,
    }
    if algorithm_config.name == "ippo":
        config_kwargs.update(
            actor_lr=3e-5,
            critic_lr=1e-4,
            clip_ratio=0.10,
            gae_lambda=0.95,
            gamma=0.99,
            entropy_coef=0.001,
            ppo_epochs=1,
            mini_batch_size=64,
            action_std=0.02,
            team_advantage_mix=0.0,
        )
    elif (
        algorithm_config.name in MATCHED_ALGORITHMS
        or algorithm_config.name in {"mappo", "mo_gat_mappo", "fc_gat_mappo"}
        or algorithm_config.name.startswith("ablation_")
    ):
        config_kwargs.update(
            actor_lr=3e-5,
            critic_lr=1e-4,
            clip_ratio=0.10,
            gae_lambda=0.95,
            gamma=0.99,
            entropy_coef=0.001,
            ppo_epochs=1,
            mini_batch_size=64,
            action_std=0.015,
            team_advantage_mix=0.25,
        )
    return MAPPOConfig(**config_kwargs)
