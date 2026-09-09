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

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardWeights:
    goal: float = 10.0
    progress: float = 1.0
    collision: float = 10.0
    unsafe_distance: float = 2.0
    control_effort: float = 0.2
    waiting: float = 1.0
    smoothness: float = 0.08
    time: float = 0.0
    clearance: float = 0.0
    robot_clearance: float = 0.0
    pedestrian_clearance: float = 0.0
    safety_intervention: float = 0.0
    speed: float = 0.0


def compute_reward(
    progress: float,
    reached_goal: bool,
    collision: bool,
    unsafe_distance: bool,
    control_effort: float,
    waiting: bool,
    weights: RewardWeights,
    smoothness: float = 0.0,
    clearance_penalty: float = 0.0,
    robot_clearance_penalty: float = 0.0,
    pedestrian_clearance_penalty: float = 0.0,
    safety_intervention: float = 0.0,
    speed_reward: float = 0.0,
) -> float:
    reward = weights.progress * progress
    reward += weights.speed * speed_reward
    reward -= weights.time
    if reached_goal:
        reward += weights.goal
    if collision:
        reward -= weights.collision
    if unsafe_distance:
        reward -= weights.unsafe_distance
    if waiting:
        reward -= weights.waiting
    reward -= weights.control_effort * control_effort
    reward -= weights.smoothness * smoothness
    reward -= weights.clearance * clearance_penalty
    reward -= weights.robot_clearance * robot_clearance_penalty
    reward -= weights.pedestrian_clearance * pedestrian_clearance_penalty
    reward -= weights.safety_intervention * safety_intervention
    return reward
