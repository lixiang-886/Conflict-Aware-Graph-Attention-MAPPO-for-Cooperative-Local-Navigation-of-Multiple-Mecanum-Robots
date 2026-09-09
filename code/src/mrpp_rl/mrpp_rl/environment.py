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
from math import atan2, cos, hypot, sin

from mrpp_rl.lidar import DEFAULT_RANGE_MAX, LaserSectors


@dataclass(frozen=True)
class RobotState:
    name: str
    x: float
    y: float
    yaw: float
    v: float = 0.0
    vy: float = 0.0
    omega: float = 0.0


@dataclass(frozen=True)
class GoalState:
    x: float
    y: float
    yaw: float = 0.0


@dataclass(frozen=True)
class ObservationVector:
    distance_to_goal: float
    heading_error: float
    nearest_robot_distance: float
    speed: float
    angular_speed: float
    min_laser_range: float = 999.0
    laser_front: float = DEFAULT_RANGE_MAX
    laser_front_left: float = DEFAULT_RANGE_MAX
    laser_left: float = DEFAULT_RANGE_MAX
    laser_rear_left: float = DEFAULT_RANGE_MAX
    laser_rear: float = DEFAULT_RANGE_MAX
    laser_rear_right: float = DEFAULT_RANGE_MAX
    laser_right: float = DEFAULT_RANGE_MAX
    laser_front_right: float = DEFAULT_RANGE_MAX
    laser_range_max: float = DEFAULT_RANGE_MAX
    nearest_ped_distance: float = 999.0
    nearest_ped_dx: float = 0.0
    nearest_ped_dy: float = 0.0


def build_observation(
    robot: RobotState,
    goal: GoalState,
    neighbors: list[RobotState],
    min_laser: float = 999.0,
    laser_sectors: LaserSectors | None = None,
    laser_range_max: float = DEFAULT_RANGE_MAX,
    pedestrians: list[tuple[float, float]] | None = None,
) -> ObservationVector:
    dx = goal.x - robot.x
    dy = goal.y - robot.y
    target_heading = atan2(dy, dx)
    raw_heading_error = target_heading - robot.yaw
    heading_error = atan2(sin(raw_heading_error), cos(raw_heading_error))
    nearest = min(
        (hypot(robot.x - n.x, robot.y - n.y) for n in neighbors),
        default=999.0,
    )

    ped_dist = 999.0
    ped_dx = 0.0
    ped_dy = 0.0
    if pedestrians:
        for px, py in pedestrians:
            d = hypot(robot.x - px, robot.y - py)
            if d < ped_dist:
                ped_dist = d
                world_dx = px - robot.x
                world_dy = py - robot.y
                ped_dx = cos(robot.yaw) * world_dx + sin(robot.yaw) * world_dy
                ped_dy = -sin(robot.yaw) * world_dx + cos(robot.yaw) * world_dy

    sectors = laser_sectors or LaserSectors(
        front=min_laser,
        front_left=min_laser,
        left=min_laser,
        rear_left=min_laser,
        rear=min_laser,
        rear_right=min_laser,
        right=min_laser,
        front_right=min_laser,
    )
    return ObservationVector(
        distance_to_goal=hypot(dx, dy),
        heading_error=heading_error,
        nearest_robot_distance=nearest,
        speed=hypot(robot.v, robot.vy),
        angular_speed=robot.omega,
        min_laser_range=min_laser,
        laser_front=sectors.front,
        laser_front_left=sectors.front_left,
        laser_left=sectors.left,
        laser_rear_left=sectors.rear_left,
        laser_rear=sectors.rear,
        laser_rear_right=sectors.rear_right,
        laser_right=sectors.right,
        laser_front_right=sectors.front_right,
        laser_range_max=laser_range_max,
        nearest_ped_distance=ped_dist,
        nearest_ped_dx=ped_dx,
        nearest_ped_dy=ped_dy,
    )


def observation_laser_kwargs(laser_data) -> dict[str, object]:
    if laser_data is None:
        return {
            "min_laser": 999.0,
            "laser_sectors": None,
            "laser_range_max": DEFAULT_RANGE_MAX,
        }
    sectors = getattr(laser_data, "sectors", None)
    if not isinstance(sectors, LaserSectors):
        sectors = None
    return {
        "min_laser": float(getattr(laser_data, "min_range", 999.0)),
        "laser_sectors": sectors,
        "laser_range_max": float(
            getattr(laser_data, "range_max", DEFAULT_RANGE_MAX)
        ),
    }
