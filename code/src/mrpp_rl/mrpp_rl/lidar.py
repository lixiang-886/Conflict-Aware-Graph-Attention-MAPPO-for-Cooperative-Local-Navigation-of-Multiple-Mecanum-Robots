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
import math


SECTOR_NAMES = (
    "front",
    "front_left",
    "left",
    "rear_left",
    "rear",
    "rear_right",
    "right",
    "front_right",
)

SECTOR_CENTERS = {
    "front": 0.0,
    "front_left": math.pi / 4.0,
    "left": math.pi / 2.0,
    "rear_left": 3.0 * math.pi / 4.0,
    "rear": math.pi,
    "rear_right": -3.0 * math.pi / 4.0,
    "right": -math.pi / 2.0,
    "front_right": -math.pi / 4.0,
}


@dataclass(frozen=True)
class LaserSectors:
    front: float = 4.5
    front_left: float = 4.5
    left: float = 4.5
    rear_left: float = 4.5
    rear: float = 4.5
    rear_right: float = 4.5
    right: float = 4.5
    front_right: float = 4.5

    def as_tuple(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in SECTOR_NAMES)

    def minimum(self) -> float:
        return min(self.as_tuple())


@dataclass(frozen=True)
class LaserObservation:
    ranges: tuple[float, ...]
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float
    sectors: LaserSectors
    minimum_distance: float
    stamp_sec: float | None = None
    message_stamp_sec: float | None = None
    valid_sample_count: int = 0

    @property
    def min_range(self) -> float:
        return self.minimum_distance


@dataclass(frozen=True)
class LaserSafetyConfig:
    sector_statistic: str = "percentile"
    percentile: float = 10.0
    stale_timeout_sec: float = 0.5
    stale_or_missing_policy: str = "stop"
    emergency_stop_distance: float = 0.25
    stop_distance: float = 0.45
    slow_distance: float = 1.00
    emergency_max_angular_speed: float = 0.40
    max_angular_speed: float = 2.0
    stop_rotation_at_emergency: bool = False


@dataclass(frozen=True)
class LaserSafetyResult:
    command: tuple[float, float, float]
    intervened: bool
    emergency_stop: bool
    minimum_clearance: float
    motion_clearance: float
    translation_scale: float
    angular_limited: bool
    reason: str
    scan_stale: bool


DEFAULT_RANGE_MIN = 0.10
DEFAULT_RANGE_MAX = 4.50
DEFAULT_SAFETY_CONFIG = LaserSafetyConfig()


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def laser_scan_to_observation(
    ranges,
    angle_min: float,
    angle_increment: float,
    range_min: float = DEFAULT_RANGE_MIN,
    range_max: float = DEFAULT_RANGE_MAX,
    stamp_sec: float | None = None,
    message_stamp_sec: float | None = None,
    sector_statistic: str = "percentile",
    percentile: float = 10.0,
) -> LaserObservation:
    processed: list[float] = []
    valid_values: list[float] = []
    sector_values = {name: [] for name in SECTOR_NAMES}
    for index, raw_range in enumerate(tuple(ranges or ())):
        value = _sanitize_range(raw_range, range_min, range_max)
        processed.append(value)
        if not math.isfinite(value):
            continue
        valid_values.append(value)
        angle = wrap_angle(float(angle_min) + index * float(angle_increment))
        for name, center in SECTOR_CENTERS.items():
            if abs(wrap_angle(angle - center)) <= math.pi / 8.0:
                sector_values[name].append(value)

    sectors = LaserSectors(
        **{
            name: _reduce_sector(
                sector_values[name],
                range_max,
                sector_statistic,
                percentile,
            )
            for name in SECTOR_NAMES
        }
    )
    minimum = min(valid_values) if valid_values else float(range_max)
    return LaserObservation(
        ranges=tuple(processed),
        angle_min=float(angle_min),
        angle_increment=float(angle_increment),
        range_min=float(range_min),
        range_max=float(range_max),
        sectors=sectors,
        minimum_distance=minimum,
        stamp_sec=stamp_sec,
        message_stamp_sec=message_stamp_sec,
        valid_sample_count=len(valid_values),
    )


def laser_scan_to_sectors(
    ranges,
    angle_min: float,
    angle_increment: float,
    range_min: float = DEFAULT_RANGE_MIN,
    range_max: float = DEFAULT_RANGE_MAX,
    sector_statistic: str = "percentile",
    percentile: float = 10.0,
) -> LaserSectors:
    return laser_scan_to_observation(
        ranges,
        angle_min,
        angle_increment,
        range_min=range_min,
        range_max=range_max,
        sector_statistic=sector_statistic,
        percentile=percentile,
    ).sectors


def observation_from_sectors(
    sectors: LaserSectors,
    range_min: float = DEFAULT_RANGE_MIN,
    range_max: float = DEFAULT_RANGE_MAX,
    stamp_sec: float | None = None,
    message_stamp_sec: float | None = None,
) -> LaserObservation:
    values = sectors.as_tuple()
    valid = [value for value in values if math.isfinite(value)]
    return LaserObservation(
        ranges=values,
        angle_min=0.0,
        angle_increment=math.pi / 4.0,
        range_min=range_min,
        range_max=range_max,
        sectors=sectors,
        minimum_distance=min(valid) if valid else float(range_max),
        stamp_sec=stamp_sec,
        message_stamp_sec=message_stamp_sec,
        valid_sample_count=len(valid),
    )


def laser_observation_from_data(laser_data) -> LaserObservation | None:
    if laser_data is None:
        return None
    if isinstance(laser_data, LaserObservation):
        return laser_data
    sectors = getattr(laser_data, "sectors", None)
    if isinstance(sectors, LaserSectors):
        return observation_from_sectors(
            sectors,
            range_min=float(getattr(laser_data, "range_min", DEFAULT_RANGE_MIN)),
            range_max=float(getattr(laser_data, "range_max", DEFAULT_RANGE_MAX)),
            stamp_sec=getattr(laser_data, "stamp_sec", None),
            message_stamp_sec=getattr(laser_data, "message_stamp_sec", None),
        )
    return laser_scan_to_observation(
        getattr(laser_data, "ranges", ()),
        float(getattr(laser_data, "angle_min", 0.0)),
        float(getattr(laser_data, "angle_increment", 0.0)),
        range_min=float(getattr(laser_data, "range_min", DEFAULT_RANGE_MIN)),
        range_max=float(getattr(laser_data, "range_max", DEFAULT_RANGE_MAX)),
        stamp_sec=getattr(laser_data, "stamp_sec", None),
        message_stamp_sec=getattr(laser_data, "message_stamp_sec", None),
    )


def apply_laser_safety_filter(
    command: tuple[float, float, float],
    laser_observation: LaserObservation | LaserSectors | None,
    config: LaserSafetyConfig = DEFAULT_SAFETY_CONFIG,
    current_time_sec: float | None = None,
) -> LaserSafetyResult:
    if isinstance(laser_observation, LaserSectors):
        observation = observation_from_sectors(laser_observation)
    else:
        observation = laser_observation
    raw_vx, raw_vy, raw_omega = command
    if _scan_is_missing_or_stale(observation, config, current_time_sec):
        return _missing_scan_result(command, observation, config)

    assert observation is not None
    limiting_sector, motion_clearance = _motion_limiting_sector(
        raw_vx,
        raw_vy,
        observation.sectors,
    )
    clearance_scale = _clearance_scale(motion_clearance, config)
    safe_vx, safe_vy = _scale_obstacle_normal_component(
        raw_vx,
        raw_vy,
        limiting_sector,
        clearance_scale,
    )
    projected_from_obstacle = False
    uniformly_scaled = (
        raw_vx * clearance_scale,
        raw_vy * clearance_scale,
    )
    if (
        abs(safe_vx - uniformly_scaled[0]) > 1e-9
        or abs(safe_vy - uniformly_scaled[1]) > 1e-9
    ):
        projected_from_obstacle = True
    if math.hypot(safe_vx, safe_vy) <= 1e-9 and clearance_scale <= 1e-9:
        projected_vx, projected_vy = _project_velocity_away_from_closest_obstacle(
            raw_vx,
            raw_vy,
            observation.sectors,
            config,
        )
        if math.hypot(projected_vx, projected_vy) > 1e-9:
            safe_vx = projected_vx
            safe_vy = projected_vy
            projected_from_obstacle = True
    raw_speed = max(math.hypot(raw_vx, raw_vy), 1e-9)
    translation_scale = min(math.hypot(safe_vx, safe_vy) / raw_speed, 1.0)

    allowed_omega = _allowed_angular_speed(observation.minimum_distance, config)
    safe_omega = clamp(raw_omega, -allowed_omega, allowed_omega)
    angular_limited = abs(safe_omega - raw_omega) > 1e-9
    emergency_stop = (
        math.hypot(raw_vx, raw_vy) > 1e-9
        and motion_clearance <= config.emergency_stop_distance
        and math.hypot(safe_vx, safe_vy) <= 1e-9
    )
    intervened = (
        abs(safe_vx - raw_vx) > 1e-9
        or abs(safe_vy - raw_vy) > 1e-9
        or angular_limited
    )
    reasons: list[str] = []
    if translation_scale < 1.0:
        reasons.append("motion_clearance")
    if projected_from_obstacle:
        reasons.append("obstacle_projection")
    if angular_limited:
        reasons.append("angular_limit")
    return LaserSafetyResult(
        command=(safe_vx, safe_vy, safe_omega),
        intervened=intervened,
        emergency_stop=emergency_stop,
        minimum_clearance=observation.minimum_distance,
        motion_clearance=motion_clearance,
        translation_scale=translation_scale,
        angular_limited=angular_limited,
        reason=",".join(reasons) if reasons else "ok",
        scan_stale=False,
    )


def apply_safety_filter_to_commands_with_results(
    commands: dict[str, tuple[float, float, float]],
    laser_data: dict[str, object],
    config: LaserSafetyConfig = DEFAULT_SAFETY_CONFIG,
    current_time_sec: float | None = None,
) -> dict[str, LaserSafetyResult]:
    return {
        name: apply_laser_safety_filter(
            command,
            laser_observation_from_data(laser_data.get(name)),
            config=config,
            current_time_sec=current_time_sec,
        )
        for name, command in commands.items()
    }


def apply_safety_filter_to_commands(
    commands: dict[str, tuple[float, float, float]],
    laser_data: dict[str, object],
    config: LaserSafetyConfig = DEFAULT_SAFETY_CONFIG,
    current_time_sec: float | None = None,
) -> dict[str, tuple[float, float, float]]:
    results = apply_safety_filter_to_commands_with_results(
        commands,
        laser_data,
        config=config,
        current_time_sec=current_time_sec,
    )
    return {name: result.command for name, result in results.items()}


def _sanitize_range(raw_range, range_min: float, range_max: float) -> float:
    try:
        value = float(raw_range)
    except (TypeError, ValueError):
        return math.nan
    if math.isnan(value) or value == -math.inf:
        return math.nan
    if value == math.inf:
        return float(range_max)
    if value < range_min:
        return math.nan
    if value > range_max:
        return float(range_max)
    return value


def _reduce_sector(
    values: list[float],
    range_max: float,
    statistic: str,
    percentile: float,
) -> float:
    if not values:
        return float(range_max)
    ordered = sorted(values)
    if statistic == "minimum":
        return ordered[0]
    if statistic == "mean_of_k_smallest":
        k = max(1, int(math.ceil(len(ordered) * max(percentile, 1.0) / 100.0)))
        return sum(ordered[:k]) / k
    if statistic != "percentile":
        raise ValueError(f"Unsupported lidar sector statistic: {statistic}")
    index = int((len(ordered) - 1) * clamp(percentile, 0.0, 100.0) / 100.0)
    return ordered[index]


def _scan_is_missing_or_stale(
    observation: LaserObservation | None,
    config: LaserSafetyConfig,
    current_time_sec: float | None,
) -> bool:
    if observation is None or observation.valid_sample_count <= 0:
        return True
    if (
        current_time_sec is not None
        and observation.stamp_sec is not None
        and current_time_sec - observation.stamp_sec > config.stale_timeout_sec
    ):
        return True
    return False


def _missing_scan_result(
    command: tuple[float, float, float],
    observation: LaserObservation | None,
    config: LaserSafetyConfig,
) -> LaserSafetyResult:
    raw_vx, raw_vy, raw_omega = command
    policy = config.stale_or_missing_policy
    if policy == "allow":
        translation_scale = 1.0
    elif policy == "limit":
        translation_scale = 0.25
    elif policy == "stop":
        translation_scale = 0.0
    else:
        raise ValueError(f"Unsupported stale lidar policy: {policy}")
    safe_vx = raw_vx * translation_scale
    safe_vy = raw_vy * translation_scale
    allowed_omega = (
        config.max_angular_speed
        if policy == "allow" else config.emergency_max_angular_speed
    )
    safe_omega = clamp(raw_omega, -allowed_omega, allowed_omega)
    minimum = observation.minimum_distance if observation is not None else 0.0
    intervened = (
        abs(safe_vx - raw_vx) > 1e-9
        or abs(safe_vy - raw_vy) > 1e-9
        or abs(safe_omega - raw_omega) > 1e-9
    )
    return LaserSafetyResult(
        command=(safe_vx, safe_vy, safe_omega),
        intervened=intervened,
        emergency_stop=False,
        minimum_clearance=minimum,
        motion_clearance=minimum,
        translation_scale=translation_scale,
        angular_limited=abs(safe_omega - raw_omega) > 1e-9,
        reason="stale_scan",
        scan_stale=True,
    )


def _motion_limiting_sector(
    vx: float,
    vy: float,
    sectors: LaserSectors,
) -> tuple[str | None, float]:
    speed = math.hypot(vx, vy)
    if speed <= 1e-9:
        closest = min(SECTOR_NAMES, key=lambda name: float(getattr(sectors, name)))
        return closest, float(getattr(sectors, closest))
    motion_angle = wrap_angle(math.atan2(vy, vx))
    nearest = min(
        SECTOR_NAMES,
        key=lambda name: abs(wrap_angle(motion_angle - SECTOR_CENTERS[name])),
    )
    index = SECTOR_NAMES.index(nearest)
    neighbors = (
        SECTOR_NAMES[(index - 1) % len(SECTOR_NAMES)],
        nearest,
        SECTOR_NAMES[(index + 1) % len(SECTOR_NAMES)],
    )
    limiting = min(neighbors, key=lambda name: float(getattr(sectors, name)))
    return limiting, float(getattr(sectors, limiting))


def _motion_clearance(vx: float, vy: float, sectors: LaserSectors) -> float:
    return _motion_limiting_sector(vx, vy, sectors)[1]


def _scale_obstacle_normal_component(
    vx: float,
    vy: float,
    limiting_sector: str | None,
    scale: float,
) -> tuple[float, float]:
    if limiting_sector is None or scale >= 1.0:
        return vx, vy
    obstacle_angle = SECTOR_CENTERS[limiting_sector]
    obstacle_x = math.cos(obstacle_angle)
    obstacle_y = math.sin(obstacle_angle)
    toward_obstacle = vx * obstacle_x + vy * obstacle_y
    if toward_obstacle <= 1e-9:
        return vx, vy
    reduction = toward_obstacle * (1.0 - clamp(scale, 0.0, 1.0))
    safe_vx = vx - reduction * obstacle_x
    safe_vy = vy - reduction * obstacle_y
    return (
        0.0 if abs(safe_vx) <= 1e-12 else safe_vx,
        0.0 if abs(safe_vy) <= 1e-12 else safe_vy,
    )


def _project_velocity_away_from_closest_obstacle(
    vx: float,
    vy: float,
    sectors: LaserSectors,
    config: LaserSafetyConfig,
) -> tuple[float, float]:
    speed = math.hypot(vx, vy)
    if speed <= 1e-9:
        return 0.0, 0.0
    closest = min(SECTOR_NAMES, key=lambda name: float(getattr(sectors, name)))
    clearance = float(getattr(sectors, closest))
    if not math.isfinite(clearance) or clearance > config.stop_distance:
        return vx, vy

    obstacle_angle = SECTOR_CENTERS[closest]
    obstacle_x = math.cos(obstacle_angle)
    obstacle_y = math.sin(obstacle_angle)
    toward_obstacle = vx * obstacle_x + vy * obstacle_y
    if toward_obstacle < -1e-9:
        return vx, vy
    if clearance <= config.emergency_stop_distance:
        return 0.0, 0.0
    if toward_obstacle <= 1e-9:
        return vx, vy
    return (
        vx - toward_obstacle * obstacle_x,
        vy - toward_obstacle * obstacle_y,
    )


def _clearance_scale(distance: float, config: LaserSafetyConfig) -> float:
    if not math.isfinite(distance):
        return 1.0
    if distance <= config.emergency_stop_distance:
        return 0.0
    if distance <= config.stop_distance:
        return 0.0
    if distance >= config.slow_distance:
        return 1.0
    return clamp(
        (distance - config.stop_distance)
        / (config.slow_distance - config.stop_distance),
        0.0,
        1.0,
    )


def _allowed_angular_speed(distance: float, config: LaserSafetyConfig) -> float:
    if not math.isfinite(distance) or distance >= config.slow_distance:
        return config.max_angular_speed
    if distance <= config.emergency_stop_distance and config.stop_rotation_at_emergency:
        return 0.0
    if distance <= config.stop_distance:
        return config.emergency_max_angular_speed
    ratio = (
        (distance - config.stop_distance)
        / (config.slow_distance - config.stop_distance)
    )
    return (
        config.emergency_max_angular_speed
        + clamp(ratio, 0.0, 1.0)
        * (config.max_angular_speed - config.emergency_max_angular_speed)
    )
