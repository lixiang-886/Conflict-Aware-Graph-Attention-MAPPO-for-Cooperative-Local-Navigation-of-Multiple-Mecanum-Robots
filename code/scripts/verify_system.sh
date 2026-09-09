#!/usr/bin/env bash
set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-humble}"
if [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
  source scripts/source_setup_file.sh "/opt/ros/${ROS_DISTRO}/setup.bash"
fi

echo "== Ubuntu =="
lsb_release -a
if ! lsb_release -rs | grep -q "^22.04"; then
  echo "Warning: this project is currently targeted at Ubuntu 22.04.x LTS for the available server."
fi

echo "== ROS =="
if ! command -v ros2 >/dev/null 2>&1; then
  echo "ros2 command not found"
  exit 1
fi
ros2 --version || true

echo "== Gazebo =="
if command -v ign >/dev/null 2>&1; then
  ign gazebo --versions || ign gazebo --version || ign --versions || true
elif command -v gz >/dev/null 2>&1; then
  gz sim --versions || gz --versions || true
else
  echo "Neither ign nor gz command was found"
  exit 1
fi

echo "== Python =="
python3 --version

echo "System verification completed."
