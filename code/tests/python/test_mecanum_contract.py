from pathlib import Path


def test_formal_robot_model_is_mecanum_not_diff_drive() -> None:
    content = Path("src/mrpp_gazebo/models/mecanum_robot/model.sdf").read_text(
        encoding="utf-8"
    )

    assert "MecanumDrive" in content
    assert "DiffDrive" not in content
    for joint in (
        "front_left_wheel_joint",
        "front_right_wheel_joint",
        "back_left_wheel_joint",
        "back_right_wheel_joint",
    ):
        assert joint in content


def test_visualization_xacro_has_no_diff_drive_plugin() -> None:
    content = Path("src/mrpp_description/urdf/mrpp_robot.urdf.xacro").read_text(
        encoding="utf-8"
    )

    assert "DiffDrive" not in content
    assert "left_wheel_joint" not in content
    assert "right_wheel_joint" not in content
    assert 'name="front_left_wheel"' in content
    assert 'name="back_right_wheel"' in content
