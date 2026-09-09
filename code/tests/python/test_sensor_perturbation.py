from dataclasses import dataclass
from pathlib import Path

from mrpp_rl.environment import RobotState
from mrpp_rl.lidar import laser_scan_to_observation
from mrpp_rl.sensor_perturbation import SensorPerturbationConfig
from mrpp_rl.sensor_perturbation import SensorPerturbationPipeline
from mrpp_rl.sensor_perturbation import load_sensor_perturbation_config


@dataclass(frozen=True)
class _Pedestrian:
    name: str
    x: float
    y: float
    vx: float
    vy: float


def _laser(value: float = 2.0):
    return laser_scan_to_observation(
        [value] * 16,
        angle_min=0.0,
        angle_increment=0.2,
        stamp_sec=1.0,
    )


def test_load_sensor_perturbation_yaml() -> None:
    config = load_sensor_perturbation_config(
        Path("src/mrpp_experiments/config/sensors/moderate.yaml")
    )

    assert config.profile == "moderate"
    assert config.enabled
    assert config.position_noise_std == 0.05
    assert config.latency_steps == 2
    assert config.dropout_probability == 0.05


def test_sensor_perturbation_is_repeatable_for_a_fixed_seed() -> None:
    config = SensorPerturbationConfig(
        enabled=True,
        position_noise_std=0.05,
        velocity_noise_std=0.02,
        lidar_range_noise_std=0.01,
        seed=17,
    )
    states = {"robot_1": RobotState("robot_1", 1.0, 2.0, 0.0, 0.3, 0.1)}
    pedestrians = {"ped_1": _Pedestrian("ped_1", 0.0, 1.0, 0.2, 0.0)}
    lasers = {"robot_1": _laser()}

    first = SensorPerturbationPipeline(config).apply(states, pedestrians, lasers)
    second = SensorPerturbationPipeline(config).apply(states, pedestrians, lasers)

    assert first == second
    assert first.robot_states["robot_1"] != states["robot_1"]


def test_latency_and_stream_dropout_use_timestamped_sample_and_hold() -> None:
    latency = SensorPerturbationPipeline(
        SensorPerturbationConfig(enabled=True, latency_steps=1, seed=3)
    )
    empty_pedestrians = {}
    lasers = {"robot_1": _laser()}
    first_state = {"robot_1": RobotState("robot_1", 0.0, 0.0, 0.0)}
    second_state = {"robot_1": RobotState("robot_1", 1.0, 0.0, 0.0)}

    latency.apply(first_state, empty_pedestrians, lasers)
    delayed = latency.apply(second_state, empty_pedestrians, lasers)

    assert delayed.robot_states["robot_1"].x == 0.0

    dropout = SensorPerturbationPipeline(
        SensorPerturbationConfig(
            enabled=True,
            dropout_probability=1.0,
            seed=5,
        )
    )
    first = dropout.apply(first_state, empty_pedestrians, lasers)
    held = dropout.apply(second_state, empty_pedestrians, lasers)

    assert first.robot_states["robot_1"].x == 0.0
    assert held.robot_states["robot_1"].x == 0.0
    assert dropout.stats.robot_state_dropouts == 2
    assert dropout.stats.lidar_scan_dropouts == 2


def test_lidar_sample_dropout_can_produce_a_fail_safe_missing_scan() -> None:
    pipeline = SensorPerturbationPipeline(
        SensorPerturbationConfig(
            enabled=True,
            lidar_sample_dropout_probability=1.0,
            seed=11,
        )
    )

    snapshot = pipeline.apply({}, {}, {"robot_1": _laser()})

    assert snapshot.laser_data["robot_1"].valid_sample_count == 0
    assert pipeline.stats.lidar_sample_dropouts == 16
