#!/usr/bin/env bash
set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-humble}"
INCLUDE_ABLATIONS="${INCLUDE_ABLATIONS:-0}"
source scripts/source_setup_file.sh "/opt/ros/${ROS_DISTRO}/setup.bash"
source scripts/source_setup_file.sh install/setup.bash

include_ablation_args=()
if [[ "${INCLUDE_ABLATIONS}" == "1" ]]; then
  include_ablation_args+=(--include-ablations)
fi

python3 -m mrpp_experiments.experiment_matrix \
  --scenario-dir src/mrpp_experiments/config/scenarios \
  --scenario static_clutter \
  --scenario pedestrian_dynamic \
  --scenario mixed_complex \
  --output-root results/raw \
  --plan-output results/raw/experiment_plan.csv \
  --method-registry src/mrpp_experiments/config/methods/paper_methods.csv \
  "${include_ablation_args[@]}" \
  --seed 0 \
  --seed 1 \
  --seed 2 \
  --seed 3 \
  --seed 4
