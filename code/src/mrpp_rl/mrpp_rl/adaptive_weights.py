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

from mrpp_rl.reward import RewardWeights


@dataclass(frozen=True)
class SceneContext:
    local_robot_density: float
    nearest_obstacle_distance: float
    nearest_robot_distance: float = 999.0
    nearest_pedestrian_distance: float = 999.0


def adaptive_reward_weights(context: SceneContext) -> RewardWeights:
    density = max(0.0, min(1.0, context.local_robot_density))
    obstacle_risk = max(0.0, min(1.0, (1.05 - context.nearest_obstacle_distance) / 0.60))
    robot_risk = max(0.0, min(1.0, (1.05 - context.nearest_robot_distance) / 0.55))
    pedestrian_risk = max(
        0.0,
        min(1.0, (1.45 - context.nearest_pedestrian_distance) / 0.75),
    )
    safety_scale = 1.0 + 0.55 * density + 0.75 * obstacle_risk + robot_risk + pedestrian_risk
    return RewardWeights(
        goal=16.0,
        progress=2.8,
        collision=12.0 * safety_scale,
        unsafe_distance=1.2 * safety_scale,
        control_effort=0.05,
        waiting=0.12 + 0.30 * density,
        smoothness=0.07 + 0.025 * safety_scale,
        time=0.010 + 0.010 * density,
        clearance=0.22 * safety_scale,
        robot_clearance=0.55 * (1.0 + 0.5 * density + robot_risk),
        pedestrian_clearance=1.25 * (1.0 + pedestrian_risk),
        safety_intervention=0.22 * safety_scale,
        speed=0.35,
    )
