from pathlib import Path

from mrpp_gazebo.spawn_robots import RobotSpawn, load_robot_spawns, render_robot_sdf
from mrpp_gazebo.spawn_robots import spawn_robot


def test_load_robot_spawns(tmp_path: Path) -> None:
    config = tmp_path / "robots.yaml"
    config.write_text(
        "robots:\n"
        "  - name: robot_1\n"
        "    x: 1.0\n"
        "    y: 2.0\n"
        "    yaw: 0.5\n",
        encoding="utf-8",
    )

    robots = load_robot_spawns(config)

    assert len(robots) == 1
    assert robots[0].name == "robot_1"
    assert robots[0].x == 1.0
    assert robots[0].y == 2.0
    assert robots[0].yaw == 0.5


def test_render_robot_sdf_namespaces_topics(tmp_path: Path) -> None:
    model = tmp_path / "model.sdf"
    model.write_text(
        '<sdf><model name="mecanum_robot">'
        "<link name=\"base_scan\">"
        "<sensor><topic>scan</topic></sensor>"
        "</link>"
        "<plugin>"
        "<topic>cmd_vel</topic>"
        "<odom_topic>odom</odom_topic>"
        "<topic>joint_states</topic>"
        "<tf_topic>/tf</tf_topic>"
        "</plugin>"
        "</model></sdf>",
        encoding="utf-8",
    )

    content = render_robot_sdf(
        RobotSpawn(name="robot_2", x=1.0, y=2.0, yaw=0.5),
        model,
    )

    assert 'name="robot_2"' in content
    assert "<topic>/model/robot_2/cmd_vel</topic>" in content
    assert "<odom_topic>/model/robot_2/odom</odom_topic>" in content
    assert "<topic>/model/robot_2/scan</topic>" in content


def test_spawn_robot_passes_sdf_by_string(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "model.sdf"
    model.write_text(
        '<sdf><model name="turtlebot3_burger"/></sdf>',
        encoding="utf-8",
    )
    calls = []

    def fake_run(cmd, check):
        calls.append((cmd, check))

    monkeypatch.setattr("mrpp_gazebo.spawn_robots.subprocess.run", fake_run)

    spawn_robot(RobotSpawn(name="robot_1", x=1.0, y=2.0, yaw=0.5), model)

    cmd, check = calls[0]
    assert check is True
    assert "-string" in cmd
    assert "-file" not in cmd
    assert 'name="robot_1"' in cmd[cmd.index("-string") + 1]


def test_render_robot_sdf_replaces_mecanum_color_placeholders(tmp_path: Path) -> None:
    model = tmp_path / "model.sdf"
    model.write_text(
        '<sdf><model name="mecanum_robot">'
        "<material><diffuse>PLACEHOLDER_COLOR</diffuse></material>"
        "<material><diffuse>PLACEHOLDER_ACCENT_COLOR</diffuse></material>"
        "</model></sdf>",
        encoding="utf-8",
    )

    content = render_robot_sdf(
        RobotSpawn(name="robot_3", x=0.0, y=0.0, yaw=0.0),
        model,
    )

    assert 'name="robot_3"' in content
    assert "PLACEHOLDER_COLOR" not in content
    assert "0.1 0.3 0.9 1.0" in content
    assert "0.45 0.62 1.0 1.0" in content
