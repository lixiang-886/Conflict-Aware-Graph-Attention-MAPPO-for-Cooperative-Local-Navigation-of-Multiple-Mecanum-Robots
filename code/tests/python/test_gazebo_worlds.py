from pathlib import Path

FORMAL_WORLDS = {
    "static_clutter",
    "pedestrian_dynamic",
    "mixed_complex",
    "unseen_dense_crossing",
}


def test_worlds_load_sensors_system() -> None:
    world_dir = Path("src/mrpp_gazebo/worlds")
    for world in world_dir.glob("*.sdf"):
        content = world.read_text(encoding="utf-8")
        assert "ignition-gazebo-sensors-system" in content
        assert "ignition::gazebo::systems::Sensors" in content


def test_only_formal_worlds_are_kept() -> None:
    world_dir = Path("src/mrpp_gazebo/worlds")
    world_names = {path.stem for path in world_dir.glob("*.sdf")}

    assert world_names == FORMAL_WORLDS
