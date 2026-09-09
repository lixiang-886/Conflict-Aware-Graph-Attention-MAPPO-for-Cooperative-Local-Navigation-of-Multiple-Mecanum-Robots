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
from pathlib import Path
import re
import subprocess
import sys


@dataclass(frozen=True)
class RobotSpawn:
    name: str
    x: float
    y: float
    yaw: float


def load_robot_spawns(path: Path) -> list[RobotSpawn]:
    robots: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line == "robots:":
            continue
        if line.startswith("- "):
            if current is not None:
                robots.append(current)
            current = {}
            line = line[2:].strip()
        if ":" in line and current is not None:
            key, value = line.split(":", 1)
            current[key.strip()] = value.strip()
    if current is not None:
        robots.append(current)
    return [
        RobotSpawn(
            name=item["name"],
            x=float(item["x"]),
            y=float(item["y"]),
            yaw=float(item["yaw"]),
        )
        for item in robots
    ]


ROBOT_COLORS = {
    "robot_1": "0.8 0.1 0.1 1.0",
    "robot_2": "0.1 0.7 0.1 1.0",
    "robot_3": "0.1 0.3 0.9 1.0",
    "robot_4": "0.9 0.8 0.1 1.0",
    "robot_5": "0.8 0.4 0.0 1.0",
    "robot_6": "0.6 0.1 0.6 1.0",
}

ROBOT_ACCENT_COLORS = {
    "robot_1": "1.0 0.45 0.45 1.0",
    "robot_2": "0.45 1.0 0.45 1.0",
    "robot_3": "0.45 0.62 1.0 1.0",
    "robot_4": "1.0 0.92 0.35 1.0",
    "robot_5": "1.0 0.62 0.25 1.0",
    "robot_6": "0.92 0.42 1.0 1.0",
}


def _rename_model(content: str, robot_name: str) -> str:
    return re.sub(
        r'<model\s+name=(["\'])([^"\']+)\1',
        f'<model name="{robot_name}"',
        content,
        count=1,
    )


def render_robot_sdf(robot: RobotSpawn, model_path: Path) -> str:
    ns = f"/model/{robot.name}"
    content = model_path.read_text(encoding="utf-8")
    content = _rename_model(content, robot.name)
    content = content.replace("PLACEHOLDER_NAME", robot.name)
    content = content.replace(
        "PLACEHOLDER_POSE",
        f"{robot.x} {robot.y} 0.2 0 0 {robot.yaw}",
    )
    content = content.replace("PLACEHOLDER_NS", ns)
    content = content.replace(
        "<topic>cmd_vel</topic>", f"<topic>{ns}/cmd_vel</topic>"
    )
    content = content.replace(
        "<odom_topic>odom</odom_topic>", f"<odom_topic>{ns}/odom</odom_topic>"
    )
    content = content.replace(
        "<topic>joint_states</topic>", f"<topic>{ns}/joint_states</topic>"
    )
    content = content.replace("<topic>scan</topic>", f"<topic>{ns}/scan</topic>")
    content = content.replace("<topic>~/scan</topic>", f"<topic>{ns}/scan</topic>")
    content = content.replace("<topic>imu</topic>", f"<topic>{ns}/imu</topic>")
    content = content.replace("<tf_topic>/tf</tf_topic>", f"<tf_topic>{ns}/tf</tf_topic>")
    color = ROBOT_COLORS.get(robot.name, "0.8 0.2 0.2 1.0")
    accent = ROBOT_ACCENT_COLORS.get(robot.name, "1.0 0.45 0.45 1.0")
    content = content.replace("PLACEHOLDER_COLOR", color)
    content = content.replace("PLACEHOLDER_ACCENT_COLOR", accent)
    content = content.replace("0.8 0.2 0.2 1.0", color)
    content = content.replace("1.0 0.45 0.45 1.0", accent)
    content = content.replace(
        "<diffuse>0.3 0.3 0.3 1.0</diffuse>",
        f"<diffuse>{color}</diffuse>",
    )
    return content


def spawn_robot(robot: RobotSpawn, model_path: Path) -> None:
    content = render_robot_sdf(robot, model_path)
    cmd = [
        "ros2",
        "run",
        "ros_gz_sim",
        "create",
        "-name",
        robot.name,
        "-string",
        content,
        "-x",
        str(robot.x),
        "-y",
        str(robot.y),
        "-z",
        "0.2",
        "-Y",
        str(robot.yaw),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: spawn_robots.py <robots.yaml> <robot.sdf-or-urdf>")
        return 2
    config_path = Path(sys.argv[1])
    model_path = Path(sys.argv[2])
    for robot in load_robot_spawns(config_path):
        spawn_robot(robot, model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
