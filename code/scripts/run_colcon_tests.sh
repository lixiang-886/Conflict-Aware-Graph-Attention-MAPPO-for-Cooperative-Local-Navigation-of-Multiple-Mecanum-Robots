#!/usr/bin/env bash
set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-humble}"
source scripts/source_setup_file.sh "/opt/ros/${ROS_DISTRO}/setup.bash"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
colcon build --symlink-install
source scripts/source_setup_file.sh install/setup.bash
colcon test
colcon test-result --verbose
