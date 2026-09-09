#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ROS_DISTRO="${ROS_DISTRO:-humble}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

source scripts/source_setup_file.sh "/opt/ros/${ROS_DISTRO}/setup.bash"
source scripts/source_setup_file.sh install/setup.bash

"${PYTHON_BIN}" scripts/gazebo_live_smoke.py "$@"
