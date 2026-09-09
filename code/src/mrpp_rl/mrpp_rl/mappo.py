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

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from mrpp_rl.policy import MOGATMAPPOPolicy as _BasePolicy


ACTION_DISTRIBUTION = "tanh_squashed_gaussian_v1"
_LOG_2 = math.log(2.0)


def _inverse_tanh(action: torch.Tensor) -> torch.Tensor:
    """Return a numerically safe inverse tanh for bounded rollout actions."""
    limit = 1.0 - torch.finfo(action.dtype).eps
    bounded_action = action.clamp(min=-limit, max=limit)
    return 0.5 * (
        torch.log1p(bounded_action) - torch.log1p(-bounded_action)
    )


def _squashed_gaussian_log_prob(
    base_dist: torch.distributions.Normal,
    raw_action: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the tanh-transformed Normal density from its latent sample."""
    log_abs_det_jacobian = 2.0 * (
        _LOG_2 - raw_action - F.softplus(-2.0 * raw_action)
    )
    return (
        base_dist.log_prob(raw_action) - log_abs_det_jacobian
    ).sum(dim=-1)


@dataclass(frozen=True)
class MAPPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.10
    actor_lr: float = 3e-5
    critic_lr: float = 1e-4
    entropy_coef: float = 0.001
    value_coef: float = 0.5
    rollout_steps: int = 256
    ppo_epochs: int = 1
    mini_batch_size: int = 64
    max_grad_norm: float = 0.5
    action_std: float = 0.02
    use_graph_attention: bool = True
    independent_agents: bool = False
    centralized_critic: bool = False
    team_advantage_mix: float = 0.5


class MOGATMAPPOPolicy(_BasePolicy):
    """Extended policy with critic head for MAPPO training."""

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        centralized_critic: bool = False,
    ) -> None:
        super().__init__(node_dim, hidden_dim)
        self.action_std = 0.02
        self.centralized_critic_enabled = bool(centralized_critic)
        self.profile_encoder_time = False
        self.last_encoder_time_s = 0.0
        if hasattr(self.actor, "__getitem__") and hasattr(self.actor[2], "weight"):
            nn.init.zeros_(self.actor[2].weight)
            nn.init.zeros_(self.actor[2].bias)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.central_state_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.Tanh(),
        )
        self.central_critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def _action_and_value(
        self,
        nodes,
        adjacency,
        action=None,
        deterministic: bool = False,
        raw_action=None,
    ):
        nodes = torch.nan_to_num(nodes, nan=0.0)
        adjacency = torch.nan_to_num(adjacency, nan=1.0)
        if self.profile_encoder_time and nodes.is_cuda:
            torch.cuda.synchronize(nodes.device)
        encoder_start = time.perf_counter()
        encoded = self.encoder(nodes, adjacency)
        if self.profile_encoder_time and nodes.is_cuda:
            torch.cuda.synchronize(nodes.device)
        if self.profile_encoder_time:
            self.last_encoder_time_s = time.perf_counter() - encoder_start
        encoded = torch.nan_to_num(encoded, nan=0.0)
        mean = self.actor(encoded)
        mean = torch.nan_to_num(
            mean,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        )
        if self.action_std <= 0.0:
            raise ValueError("action_std must be positive")
        std = torch.ones_like(mean) * self.action_std
        base_dist = torch.distributions.Normal(mean, std)

        if raw_action is not None:
            raw_action = torch.nan_to_num(
                raw_action,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            )
            action = torch.tanh(raw_action)
        elif action is not None:
            action = torch.nan_to_num(
                action,
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )
            raw_action = _inverse_tanh(action)
        elif deterministic:
            raw_action = mean
            action = torch.tanh(raw_action)
        else:
            raw_action = base_dist.rsample()
            action = torch.tanh(raw_action)

        action = torch.nan_to_num(
            action,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        log_prob = _squashed_gaussian_log_prob(base_dist, raw_action)
        if deterministic:
            entropy = torch.zeros_like(log_prob)
        else:
            entropy_raw_action = base_dist.rsample()
            entropy = -_squashed_gaussian_log_prob(
                base_dist,
                entropy_raw_action,
            )
        if self.centralized_critic_enabled:
            joint_state = self.central_state_encoder(nodes).mean(
                dim=-2,
                keepdim=True,
            )
            joint_state = joint_state.expand(*encoded.shape[:-1], -1)
            value = self.central_critic(
                torch.cat((encoded, joint_state), dim=-1)
            ).squeeze(-1)
        else:
            value = self.critic(encoded).squeeze(-1)
        value = torch.nan_to_num(value, nan=0.0)
        return action, raw_action, log_prob, entropy, value

    def get_action_and_value(
        self,
        nodes,
        adjacency,
        action=None,
        deterministic: bool = False,
        raw_action=None,
    ):
        action, _, log_prob, entropy, value = self._action_and_value(
            nodes,
            adjacency,
            action=action,
            deterministic=deterministic,
            raw_action=raw_action,
        )
        return action, log_prob, entropy, value

    def sample_action_and_value(self, nodes, adjacency):
        """Sample a bounded action while exposing its pre-squash latent."""
        return self._action_and_value(nodes, adjacency)


class RolloutBuffer:
    """Stores collected experience for PPO update."""

    def __init__(self) -> None:
        self.observations: list[torch.Tensor] = []
        self.adjacencies: list[torch.Tensor | None] = []
        self.actions: list[torch.Tensor] = []
        self.raw_actions: list[torch.Tensor | None] = []
        self.rewards: list[torch.Tensor] = []
        self.dones: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []

    def add(
        self,
        obs,
        action,
        reward,
        done,
        log_prob,
        value,
        adjacency=None,
        raw_action=None,
    ):
        self.observations.append(obs.detach())
        self.adjacencies.append(
            adjacency.detach() if adjacency is not None else None
        )
        self.actions.append(action.detach())
        self.raw_actions.append(
            raw_action.detach() if raw_action is not None else None
        )
        self.rewards.append(reward.detach())
        self.dones.append(done.detach())
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())

    def clear(self) -> None:
        self.observations.clear()
        self.adjacencies.clear()
        self.actions.clear()
        self.raw_actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def compute_returns_and_advantages(
        self,
        last_value,
        gamma,
        gae_lambda,
        average_agents: bool = True,
    ):
        advantages = []
        gae = torch.zeros_like(self.rewards[0])
        values_plus = self.values + [last_value]
        for step in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[step]
                + gamma * values_plus[step + 1] * (1.0 - self.dones[step])
                - values_plus[step]
            )
            gae = delta + gamma * gae_lambda * (1.0 - self.dones[step]) * gae
            advantages.insert(0, gae)
        advantages_t = torch.stack(advantages)
        values_t = torch.stack(self.values)
        if average_agents:
            advantages_t = advantages_t.mean(dim=-1)
            returns_t = advantages_t + values_t.mean(dim=-1)
        else:
            returns_t = advantages_t + values_t
        return returns_t, advantages_t


class MAPPOTrainer:
    """Multi-Agent PPO trainer with graph attention."""

    def __init__(
        self,
        policy: MOGATMAPPOPolicy,
        config: MAPPOConfig,
        device: str = "cpu",
    ) -> None:
        self.policy = policy
        self.config = config
        self.device = torch.device(device)
        self.policy.action_std = config.action_std
        self.policy.centralized_critic_enabled = config.centralized_critic
        self.policy.to(self.device)
        critic_parameters = (
            [
                *policy.central_state_encoder.parameters(),
                *policy.central_critic.parameters(),
            ]
            if config.centralized_critic
            else list(policy.critic.parameters())
        )
        self.optimizer = torch.optim.Adam(
            [
                {"params": policy.encoder.parameters(), "lr": config.actor_lr},
                {"params": policy.actor.parameters(), "lr": config.actor_lr},
                {"params": critic_parameters, "lr": config.critic_lr},
            ]
        )
        self.buffer = RolloutBuffer()

    def describe(self) -> dict[str, float | int | str]:
        return {
            "action_distribution": ACTION_DISTRIBUTION,
            "gamma": self.config.gamma,
            "gae_lambda": self.config.gae_lambda,
            "clip_ratio": self.config.clip_ratio,
            "actor_lr": self.config.actor_lr,
            "critic_lr": self.config.critic_lr,
            "entropy_coef": self.config.entropy_coef,
            "rollout_steps": self.config.rollout_steps,
            "ppo_epochs": self.config.ppo_epochs,
            "mini_batch_size": self.config.mini_batch_size,
            "max_grad_norm": self.config.max_grad_norm,
            "value_coef": self.config.value_coef,
            "action_std": self.config.action_std,
            "use_graph_attention": self.config.use_graph_attention,
            "independent_agents": self.config.independent_agents,
            "centralized_critic": self.config.centralized_critic,
            "team_advantage_mix": self.config.team_advantage_mix,
        }

    def update(self, last_obs, last_adjacency):
        cfg = self.config
        with torch.no_grad():
            _, _, _, last_value = self.policy.get_action_and_value(
                last_obs.to(self.device),
                last_adjacency.to(self.device),
                deterministic=True,
            )
        last_value = last_value.detach().cpu()
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value.detach().cpu(),
            cfg.gamma,
            cfg.gae_lambda,
            average_agents=False,
        )
        returns = torch.nan_to_num(returns, nan=0.0)
        advantages = torch.nan_to_num(advantages, nan=0.0)
        if not cfg.independent_agents and cfg.team_advantage_mix > 0.0:
            team_mix = max(0.0, min(1.0, cfg.team_advantage_mix))
            team_advantage = advantages.mean(dim=-1, keepdim=True)
            advantages = (
                (1.0 - team_mix) * advantages
                + team_mix * team_advantage
            )
        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1e-8)

        obs_t = torch.nan_to_num(
            torch.stack(self.buffer.observations), nan=0.0
        )
        act_t = torch.nan_to_num(torch.stack(self.buffer.actions), nan=0.0)
        has_raw_actions = (
            len(self.buffer.raw_actions) == len(self.buffer.actions)
            and all(
                raw_action is not None
                for raw_action in self.buffer.raw_actions
            )
        )
        raw_act_t = (
            torch.nan_to_num(
                torch.stack(self.buffer.raw_actions),
                nan=0.0,
            )
            if has_raw_actions
            else None
        )
        old_logp_t = torch.nan_to_num(
            torch.stack(self.buffer.log_probs), nan=0.0
        )
        has_rollout_adjacency = (
            not cfg.independent_agents
            and len(self.buffer.adjacencies) == len(self.buffer.observations)
            and all(adjacency is not None for adjacency in self.buffer.adjacencies)
        )
        adjacency_t = (
            torch.nan_to_num(
                torch.stack(self.buffer.adjacencies), nan=0.0
            )
            if has_rollout_adjacency else None
        )
        if cfg.independent_agents:
            obs_batch = obs_t.reshape(-1, 1, obs_t.shape[-1]).to(self.device)
            act_batch = act_t.reshape(-1, 1, act_t.shape[-1]).to(self.device)
            raw_act_batch = (
                raw_act_t.reshape(-1, 1, raw_act_t.shape[-1]).to(self.device)
                if raw_act_t is not None
                else None
            )
            old_logp_batch = old_logp_t.reshape(-1).to(self.device)
            ret_batch = returns.reshape(-1).to(self.device)
            adv_batch = advantages.reshape(-1).to(self.device)
        else:
            obs_batch = obs_t.to(self.device)
            act_batch = act_t.to(self.device)
            raw_act_batch = (
                raw_act_t.to(self.device)
                if raw_act_t is not None
                else None
            )
            old_logp_batch = old_logp_t.to(self.device)
            ret_batch = returns.to(self.device)
            adv_batch = advantages.to(self.device)
        batch_size = obs_batch.shape[0]
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(cfg.ppo_epochs):
            for start in range(0, batch_size, cfg.mini_batch_size):
                end = start + cfg.mini_batch_size
                mb_obs = obs_batch[start:end]
                mb_act = act_batch[start:end]
                mb_raw_act = (
                    raw_act_batch[start:end]
                    if raw_act_batch is not None
                    else None
                )
                mb_old_logp = old_logp_batch[start:end]
                mb_ret = ret_batch[start:end]
                mb_adv = adv_batch[start:end]

                if has_rollout_adjacency and adjacency_t is not None:
                    adjacency_b = adjacency_t[start:end].to(self.device)
                elif cfg.independent_agents:
                    adjacency = torch.ones(1, 1, device=self.device)
                    batch_sz = mb_obs.shape[0]
                    adjacency_b = adjacency.unsqueeze(0).expand(
                        batch_sz, -1, -1
                    )
                else:
                    n_robots = mb_obs.shape[1]
                    if cfg.use_graph_attention:
                        adjacency = torch.ones(
                            n_robots, n_robots, device=self.device
                        )
                    else:
                        adjacency = torch.eye(n_robots, device=self.device)
                    batch_sz = mb_obs.shape[0]
                    adjacency_b = adjacency.unsqueeze(0).expand(
                        batch_sz, -1, -1
                    )

                _, new_logp, entropy, new_value = (
                    self.policy.get_action_and_value(
                        mb_obs,
                        adjacency_b,
                        mb_act,
                        raw_action=mb_raw_act,
                    )
                )
                if cfg.independent_agents:
                    new_logp = new_logp.reshape(-1)
                    entropy = entropy.reshape(-1)
                    new_value = new_value.reshape(-1)

                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio
                    )
                    * mb_adv
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_bonus = entropy.mean()
                value_loss = F.mse_loss(new_value, mb_ret)
                loss = (
                    policy_loss
                    - cfg.entropy_coef * entropy_bonus
                    + cfg.value_coef * value_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_bonus.item()
                num_updates += 1

        self.buffer.clear()
        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
        }
