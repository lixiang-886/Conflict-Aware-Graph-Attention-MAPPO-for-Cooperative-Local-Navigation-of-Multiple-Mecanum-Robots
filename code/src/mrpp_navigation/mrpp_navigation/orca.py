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

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class AgentState:
    x: float
    y: float
    vx: float
    vy: float


@dataclass(frozen=True)
class VelocityCommand2D:
    vx: float
    vy: float


def reciprocal_avoidance_velocity(
    agent: AgentState,
    neighbors: list[AgentState],
    preferred: VelocityCommand2D,
    safety_radius: float = 0.5,
    gain: float = 0.8,
    max_speed: float = 0.6,
    time_horizon: float = 2.0,
) -> VelocityCommand2D:
    vx = preferred.vx
    vy = preferred.vy
    for neighbor in neighbors:
        dx = agent.x - neighbor.x
        dy = agent.y - neighbor.y
        distance = hypot(dx, dy)
        if 1e-6 < distance < safety_radius:
            scale = gain * (safety_radius - distance) / safety_radius
            vx += scale * dx / distance
            vy += scale * dy / distance
        predicted = _predicted_collision_adjustment(
            agent,
            neighbor,
            VelocityCommand2D(vx=vx, vy=vy),
            safety_radius=safety_radius,
            gain=gain,
            time_horizon=time_horizon,
        )
        vx += predicted.vx
        vy += predicted.vy
    speed = hypot(vx, vy)
    if speed > max_speed:
        vx = vx / speed * max_speed
        vy = vy / speed * max_speed
    return VelocityCommand2D(vx=vx, vy=vy)


def _predicted_collision_adjustment(
    agent: AgentState,
    neighbor: AgentState,
    planned: VelocityCommand2D,
    safety_radius: float,
    gain: float,
    time_horizon: float,
) -> VelocityCommand2D:
    if time_horizon <= 0.0:
        return VelocityCommand2D(vx=0.0, vy=0.0)
    rel_px = neighbor.x - agent.x
    rel_py = neighbor.y - agent.y
    rel_vx = neighbor.vx - planned.vx
    rel_vy = neighbor.vy - planned.vy
    rel_speed_sq = rel_vx * rel_vx + rel_vy * rel_vy
    if rel_speed_sq <= 1e-9:
        return VelocityCommand2D(vx=0.0, vy=0.0)
    time_to_closest = -(
        rel_px * rel_vx + rel_py * rel_vy
    ) / rel_speed_sq
    if time_to_closest <= 0.0 or time_to_closest > time_horizon:
        return VelocityCommand2D(vx=0.0, vy=0.0)

    closest_x = rel_px + rel_vx * time_to_closest
    closest_y = rel_py + rel_vy * time_to_closest
    closest_distance = hypot(closest_x, closest_y)
    predicted_radius = safety_radius * 1.15
    if closest_distance >= predicted_radius:
        return VelocityCommand2D(vx=0.0, vy=0.0)

    if closest_distance > 1e-6:
        away_x = -closest_x / closest_distance
        away_y = -closest_y / closest_distance
    else:
        current_distance = hypot(rel_px, rel_py)
        if current_distance <= 1e-6:
            away_x, away_y = -rel_vx, -rel_vy
            away_norm = hypot(away_x, away_y) or 1.0
            away_x /= away_norm
            away_y /= away_norm
        else:
            away_x = -rel_px / current_distance
            away_y = -rel_py / current_distance

    urgency = 1.0 - time_to_closest / time_horizon
    clearance = (predicted_radius - closest_distance) / predicted_radius
    scale = gain * urgency * clearance
    return VelocityCommand2D(vx=scale * away_x, vy=scale * away_y)
