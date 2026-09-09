import pytest

from mrpp_rl.observation_schema import LIDAR_SECTOR_COUNT, OBS_DIM
from mrpp_rl.observation_schema import OBSERVATION_SCHEMA_VERSION
from mrpp_rl.observation_schema import checkpoint_observation_fields
from mrpp_rl.observation_schema import observation_metadata
from mrpp_rl.observation_schema import validate_checkpoint_observation


def test_observation_metadata_describes_lidar_sector_schema() -> None:
    metadata = observation_metadata()

    assert metadata["obs_dim"] == 16
    assert metadata["observation_schema_version"] == OBSERVATION_SCHEMA_VERSION
    assert metadata["lidar_sector_count"] == LIDAR_SECTOR_COUNT
    assert len(metadata["lidar_sector_names"]) == 8
    assert metadata["lidar_normalization_range"] == 4.5
    assert "laser_safety_config" in metadata
    assert OBS_DIM == 16


def test_checkpoint_observation_fields_validate() -> None:
    checkpoint = checkpoint_observation_fields()

    validate_checkpoint_observation(checkpoint)


def test_old_min_laser_checkpoint_is_rejected() -> None:
    checkpoint = {
        "obs_dim": 9,
        "observation_schema_version": "legacy_min_laser_observation",
        "lidar_sector_count": 0,
    }

    with pytest.raises(ValueError, match="Old 9D"):
        validate_checkpoint_observation(checkpoint)
