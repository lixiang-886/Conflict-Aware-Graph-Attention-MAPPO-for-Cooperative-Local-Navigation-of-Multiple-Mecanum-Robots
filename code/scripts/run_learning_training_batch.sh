#!/usr/bin/env bash
set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-humble}"
SCENARIO="${SCENARIO:-src/mrpp_experiments/config/scenarios/mixed_complex.yaml}"
EPISODES="${EPISODES:-10}"
MAX_STEPS="${MAX_STEPS:-}"
SEED="${SEED:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
DEVICE="${DEVICE:-auto}"
SIM_STEP_MODE="${SIM_STEP_MODE:-realtime}"
GAZEBO_UPDATE_RATE="${GAZEBO_UPDATE_RATE:-}"
TRAIN_TIMEOUT_SEC="${TRAIN_TIMEOUT_SEC:-7200}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-10}"
VALIDATION_LOG_INTERVAL="${VALIDATION_LOG_INTERVAL:-0}"
MIN_SUCCESS_FOR_UPDATE="${MIN_SUCCESS_FOR_UPDATE:-}"
MAX_COLLISIONS_FOR_UPDATE="${MAX_COLLISIONS_FOR_UPDATE:-}"
ALGORITHMS="${ALGORITHMS:-ippo mappo maddpg mo_gat_mappo}"
OUTPUT_DIR="${OUTPUT_DIR:-models/checkpoints/long_mixed_complex_seed${SEED}}"
TRAINING_DIR="${TRAINING_DIR:-results/training/long_mixed_complex_seed${SEED}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${TRAINING_DIR}/tensorboard}"

source scripts/source_setup_file.sh "/opt/ros/${ROS_DISTRO}/setup.bash"
source scripts/source_setup_file.sh install/setup.bash

mkdir -p "${OUTPUT_DIR}" "${TRAINING_DIR}" "${TENSORBOARD_DIR}"

for algorithm in ${ALGORITHMS}; do
  echo "=== training ${algorithm} ==="
  algorithm_output_dir="${OUTPUT_DIR}/${algorithm}"
  algorithm_training_dir="${TRAINING_DIR}/${algorithm}"
  mkdir -p "${algorithm_output_dir}" "${algorithm_training_dir}"
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
    --scenario "${SCENARIO}" \
    --algorithm "${algorithm}" \
    --episodes "${EPISODES}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --log-interval "${LOG_INTERVAL}" \
    --validation-interval "${VALIDATION_INTERVAL}" \
    --validation-log-interval "${VALIDATION_LOG_INTERVAL}" \
    "${sim_step_args[@]}" \
    "${quality_gate_args[@]}" \
    --output "${algorithm_output_dir}/${algorithm}.pt" \
    --training-log "${algorithm_training_dir}/training_log.csv" \
    --tensorboard-dir "${TENSORBOARD_DIR}/${algorithm}" \
    "${max_step_args[@]}"
  cp "${algorithm_output_dir}/${algorithm}.pt" "${OUTPUT_DIR}/${algorithm}.pt"
  if [[ -f "${algorithm_output_dir}/${algorithm}_best.pt" ]]; then
    cp "${algorithm_output_dir}/${algorithm}_best.pt" "${OUTPUT_DIR}/${algorithm}_best.pt"
  fi
done

echo "Learning training batch completed."
