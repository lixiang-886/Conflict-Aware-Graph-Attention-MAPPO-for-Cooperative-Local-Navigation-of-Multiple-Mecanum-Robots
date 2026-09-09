#!/usr/bin/env bash
set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-humble}"
SEED="${SEED:-0}"
EPISODES="${EPISODES:-8}"
MAX_STEPS="${MAX_STEPS:-}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
DEVICE="${DEVICE:-auto}"
SIM_STEP_MODE="${SIM_STEP_MODE:-realtime}"
GAZEBO_UPDATE_RATE="${GAZEBO_UPDATE_RATE:-}"
TRAIN_TIMEOUT_SEC="${TRAIN_TIMEOUT_SEC:-7200}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-10}"
VALIDATION_LOG_INTERVAL="${VALIDATION_LOG_INTERVAL:-0}"
MIN_SUCCESS_FOR_UPDATE="${MIN_SUCCESS_FOR_UPDATE:-}"
MAX_COLLISIONS_FOR_UPDATE="${MAX_COLLISIONS_FOR_UPDATE:-}"
SCENARIOS="${SCENARIOS:-static_clutter pedestrian_dynamic mixed_complex}"
ALGORITHMS="${ALGORITHMS:-ippo mappo maddpg mo_gat_mappo}"
OUTPUT_ROOT="${OUTPUT_ROOT:-models/checkpoints/independent_learning_seed${SEED}}"
TRAINING_ROOT="${TRAINING_ROOT:-results/training/independent_learning_seed${SEED}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${TRAINING_ROOT}/tensorboard}"

source scripts/source_setup_file.sh "/opt/ros/${ROS_DISTRO}/setup.bash"
source scripts/source_setup_file.sh install/setup.bash

mkdir -p "${OUTPUT_ROOT}" "${TRAINING_ROOT}" "${TENSORBOARD_DIR}"

for scenario_name in ${SCENARIOS}; do
  scenario="src/mrpp_experiments/config/scenarios/${scenario_name}.yaml"
  for algorithm in ${ALGORITHMS}; do
    echo "=== independent training ${algorithm}: ${scenario_name} seed=${SEED} ==="
    output_dir="${OUTPUT_ROOT}/${scenario_name}/${algorithm}"
    training_dir="${TRAINING_ROOT}/${scenario_name}/${algorithm}"
    mkdir -p "${output_dir}" "${training_dir}"

    max_step_args=()
    if [[ -n "${MAX_STEPS}" ]]; then
      max_step_args+=(--max-steps "${MAX_STEPS}")
    fi
    sim_step_args=(--sim-step-mode "${SIM_STEP_MODE}")
    if [[ -n "${GAZEBO_UPDATE_RATE}" ]]; then
      sim_step_args+=(--gazebo-update-rate "${GAZEBO_UPDATE_RATE}")
    fi
    quality_gate_args=()
    if [[ -n "${MIN_SUCCESS_FOR_UPDATE}" ]]; then
      quality_gate_args+=(--min-success-for-update "${MIN_SUCCESS_FOR_UPDATE}")
    fi
    if [[ -n "${MAX_COLLISIONS_FOR_UPDATE}" ]]; then
      quality_gate_args+=(--max-collisions-for-update "${MAX_COLLISIONS_FOR_UPDATE}")
    fi

    timeout --kill-after=30s "${TRAIN_TIMEOUT_SEC}" \
      ros2 run mrpp_rl train_mo_gat_mappo \
      --scenario "${scenario}" \
      --algorithm "${algorithm}" \
      --episodes "${EPISODES}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --log-interval "${LOG_INTERVAL}" \
      --validation-interval "${VALIDATION_INTERVAL}" \
      --validation-log-interval "${VALIDATION_LOG_INTERVAL}" \
      "${sim_step_args[@]}" \
      "${quality_gate_args[@]}" \
      --output "${output_dir}/${algorithm}.pt" \
      --training-log "${training_dir}/training_log.csv" \
      --tensorboard-dir "${TENSORBOARD_DIR}/${scenario_name}/${algorithm}" \
      "${max_step_args[@]}"

    if [[ -f "${output_dir}/${algorithm}_best.pt" ]]; then
      cp "${output_dir}/${algorithm}_best.pt" \
        "${OUTPUT_ROOT}/${scenario_name}/${algorithm}_best.pt"
    fi
  done
done

echo "Independent learning training batch completed."
