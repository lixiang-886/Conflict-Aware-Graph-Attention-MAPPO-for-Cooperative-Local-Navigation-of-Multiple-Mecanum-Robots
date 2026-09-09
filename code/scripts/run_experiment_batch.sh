#!/usr/bin/env bash
set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-humble}"
source scripts/source_setup_file.sh "/opt/ros/${ROS_DISTRO}/setup.bash"
source scripts/source_setup_file.sh install/setup.bash

bash scripts/make_experiment_plan.sh

python3 -m mrpp_experiments.run_batch \
  --plan results/raw/experiment_plan.csv \
  --output-dir results/raw/batch \
  "$@"
