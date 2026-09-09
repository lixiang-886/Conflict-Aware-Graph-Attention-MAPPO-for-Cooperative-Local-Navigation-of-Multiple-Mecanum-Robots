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

from dataclasses import asdict
from typing import Mapping

from mrpp_rl.lidar import DEFAULT_RANGE_MAX, DEFAULT_SAFETY_CONFIG
from mrpp_rl.lidar import SECTOR_NAMES, LaserSafetyConfig


OBSERVATION_SCHEMA_VERSION = "mrpp-observation-v3-body-relative-pedestrian"
OBS_DIM = 16
LIDAR_SECTOR_COUNT = len(SECTOR_NAMES)


def observation_metadata(
    lidar_normalization_range: float = DEFAULT_RANGE_MAX,
    laser_safety_config: LaserSafetyConfig = DEFAULT_SAFETY_CONFIG,
) -> dict[str, object]:
    return {
        "obs_dim": OBS_DIM,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "lidar_sector_count": LIDAR_SECTOR_COUNT,
        "lidar_sector_names": list(SECTOR_NAMES),
        "lidar_normalization_range": float(lidar_normalization_range),
        "laser_safety_config": asdict(laser_safety_config),
    }


def checkpoint_observation_fields() -> dict[str, object]:
    metadata = observation_metadata()
    return {
        "obs_dim": metadata["obs_dim"],
        "observation_schema_version": metadata["observation_schema_version"],
        "lidar_sector_count": metadata["lidar_sector_count"],
        "lidar_normalization_range": metadata["lidar_normalization_range"],
        "laser_safety_config": metadata["laser_safety_config"],
        "observation": metadata,
    }


def validate_checkpoint_observation(
    checkpoint: Mapping[str, object],
    expected_obs_dim: int = OBS_DIM,
    expected_schema_version: str = OBSERVATION_SCHEMA_VERSION,
    expected_lidar_sector_count: int = LIDAR_SECTOR_COUNT,
) -> None:
    raw = checkpoint.get("observation")
    metadata = raw if isinstance(raw, Mapping) else checkpoint

    obs_dim = metadata.get("obs_dim")
    if obs_dim != expected_obs_dim:
        raise ValueError(
            "Checkpoint observation dimension mismatch: "
            f"checkpoint={obs_dim}, expected={expected_obs_dim}. "
            "Old 9D min-laser checkpoints are not compatible with the "
            "16D lidar-sector observation schema."
        )

    schema_version = metadata.get("observation_schema_version")
    if schema_version != expected_schema_version:
        raise ValueError(
            "Checkpoint observation schema mismatch: "
            f"checkpoint={schema_version}, expected={expected_schema_version}."
        )

    sector_count = metadata.get("lidar_sector_count")
    if sector_count != expected_lidar_sector_count:
        raise ValueError(
            "Checkpoint lidar sector count mismatch: "
            f"checkpoint={sector_count}, expected={expected_lidar_sector_count}."
        )
