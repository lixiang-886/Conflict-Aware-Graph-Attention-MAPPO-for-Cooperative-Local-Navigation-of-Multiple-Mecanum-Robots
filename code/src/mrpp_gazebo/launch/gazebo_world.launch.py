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

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    package_dir = Path(get_package_share_directory("mrpp_gazebo"))
    default_world = str(package_dir / "worlds" / "static_clutter.sdf")
    world = LaunchConfiguration("world")

    return LaunchDescription(
        [
            DeclareLaunchArgument("world", default_value=default_world),
            ExecuteProcess(cmd=["ign", "gazebo", world], output="screen"),
        ]
    )
