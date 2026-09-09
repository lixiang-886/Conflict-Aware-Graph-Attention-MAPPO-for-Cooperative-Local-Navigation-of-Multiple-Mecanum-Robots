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

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from mrpp_rl.mappo import RolloutBuffer


@dataclass(frozen=True)
class MADDPGConfig:
    gamma: float = 0.99
    actor_lr: float = 3e-5
    critic_lr: float = 1e-4
    tau: float = 0.01
    rollout_steps: int = 256
    mini_batch_size: int = 128
    action_noise_std: float = 0.05


class MADDPGPolicy(nn.Module):
    """Shared actor with a centralized, permutation-scalable critic."""

    def __init__(self, node_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.action_noise_std = 0.05
        self.actor = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Tanh(),
        )
        critic_dim = node_dim + 3 + node_dim + 3
        self.critic = nn.Sequential(
            nn.Linear(critic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def actor_actions(self, nodes, deterministic: bool = False):
        nodes = torch.nan_to_num(nodes, nan=0.0)
        action = self.actor(nodes)
        if not deterministic and self.action_noise_std > 0.0:
            action = action + torch.randn_like(action) * self.action_noise_std
        return torch.clamp(torch.nan_to_num(action, nan=0.0), -1.0, 1.0)

    def critic_values(self, nodes, actions):
        nodes = torch.nan_to_num(nodes, nan=0.0)
        actions = torch.nan_to_num(actions, nan=0.0)
        joint_obs = nodes.mean(dim=-2, keepdim=True).expand_as(nodes)
        joint_action = actions.mean(dim=-2, keepdim=True).expand_as(actions)
        critic_input = torch.cat([nodes, actions, joint_obs, joint_action], dim=-1)
        values = self.critic(critic_input).squeeze(-1)
        return torch.nan_to_num(values, nan=0.0)

    def get_action_and_value(
        self,
        nodes,
        adjacency,
        action=None,
        deterministic: bool = False,
    ):
        del adjacency
        if action is None:
            action = self.actor_actions(nodes, deterministic=deterministic)
        else:
            action = torch.clamp(torch.nan_to_num(action, nan=0.0), -1.0, 1.0)
        value = self.critic_values(nodes, action)
        log_prob = torch.zeros_like(value)
        entropy = torch.zeros_like(value)
        return action, log_prob, entropy, value


class MADDPGTrainer:
    """MADDPG-style trainer using the repository rollout interface."""

    def __init__(
        self,
        policy: MADDPGPolicy,
        config: MADDPGConfig,
        device: str = "cpu",
    ) -> None:
        self.policy = policy
        self.config = config
        self.device = torch.device(device)
        self.policy.action_noise_std = config.action_noise_std
        self.policy.to(self.device)
        self.target_policy = copy.deepcopy(policy).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(),
            lr=config.actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.policy.critic.parameters(),
            lr=config.critic_lr,
        )
        self.optimizer = self.actor_optimizer
        self.buffer = RolloutBuffer()

    def describe(self) -> dict[str, float | int | str]:
        return {
            "gamma": self.config.gamma,
            "actor_lr": self.config.actor_lr,
            "critic_lr": self.config.critic_lr,
            "tau": self.config.tau,
            "rollout_steps": self.config.rollout_steps,
            "mini_batch_size": self.config.mini_batch_size,
            "action_noise_std": self.config.action_noise_std,
            "trainer": "maddpg",
        }

    def optimizer_state_dict(self) -> dict[str, object]:
        return {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_optimizer_state_dict(self, state: dict[str, object]) -> None:
        if "actor_optimizer" in state:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        if "critic_optimizer" in state:
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])

    def sync_targets(self) -> None:
        self.target_policy.load_state_dict(self.policy.state_dict())

    def learning_rate(self) -> float:
        return float(self.actor_optimizer.param_groups[0]["lr"])

    def update(self, last_obs, last_adjacency):
        del last_adjacency
        if not self.buffer.observations:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        cfg = self.config
        obs = torch.nan_to_num(torch.stack(self.buffer.observations), nan=0.0).to(
            self.device
        )
        actions = torch.nan_to_num(torch.stack(self.buffer.actions), nan=0.0).to(
            self.device
        )
        rewards = torch.nan_to_num(torch.stack(self.buffer.rewards), nan=0.0).to(
            self.device
        )
        dones = torch.nan_to_num(torch.stack(self.buffer.dones), nan=0.0).to(
            self.device
        )
        last_obs = torch.nan_to_num(last_obs, nan=0.0).to(self.device)
        next_obs = torch.cat([obs[1:], last_obs.unsqueeze(0)], dim=0)

        batch_size = obs.shape[0]
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        updates = 0

        for start in range(0, batch_size, cfg.mini_batch_size):
            end = start + cfg.mini_batch_size
            mb_obs = obs[start:end]
            mb_actions = actions[start:end]
            mb_rewards = rewards[start:end]
            mb_dones = dones[start:end]
            mb_next_obs = next_obs[start:end]

            with torch.no_grad():
                next_actions = self.target_policy.actor_actions(
                    mb_next_obs,
                    deterministic=True,
                )
                target_q = self.target_policy.critic_values(
                    mb_next_obs,
                    next_actions,
                )
                q_target = mb_rewards + cfg.gamma * (1.0 - mb_dones) * target_q

            q_values = self.policy.critic_values(mb_obs, mb_actions)
            critic_loss = F.mse_loss(q_values, q_target)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.critic.parameters(), 0.5)
            self.critic_optimizer.step()

            for param in self.policy.critic.parameters():
                param.requires_grad_(False)
            actor_actions = self.policy.actor_actions(mb_obs, deterministic=True)
            actor_loss = -self.policy.critic_values(mb_obs, actor_actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            for param in self.policy.critic.parameters():
                param.requires_grad_(True)

            self._soft_update_targets()
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            updates += 1

        self.buffer.clear()
        return {
            "policy_loss": total_actor_loss / max(updates, 1),
            "value_loss": total_critic_loss / max(updates, 1),
            "entropy": 0.0,
        }

    def _soft_update_targets(self) -> None:
        tau = self.config.tau
        with torch.no_grad():
            for target_param, param in zip(
                self.target_policy.parameters(),
                self.policy.parameters(),
            ):
                target_param.mul_(1.0 - tau)
                target_param.add_(tau * param)
