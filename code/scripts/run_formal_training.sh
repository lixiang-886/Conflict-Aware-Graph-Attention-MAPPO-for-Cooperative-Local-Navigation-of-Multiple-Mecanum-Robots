#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ROS_DISTRO="${ROS_DISTRO:-humble}"
SEEDS="${SEEDS:-0 1 2}"
SCENARIOS="${SCENARIOS:-static_clutter pedestrian_dynamic mixed_complex}"
ALGORITHMS="${ALGORITHMS:-ippo mappo maddpg mo_gat_mappo}"
EPISODES="${EPISODES:-30}"
MAX_STEPS="${MAX_STEPS:-}"
LOG_INTERVAL="${LOG_INTERVAL:-150}"
DEVICE="${DEVICE:-cuda}"
SIM_STEP_MODE="${SIM_STEP_MODE:-realtime}"
GAZEBO_UPDATE_RATE="${GAZEBO_UPDATE_RATE:-}"
TRAIN_TIMEOUT_SEC="${TRAIN_TIMEOUT_SEC:-14400}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-5}"
VALIDATION_LOG_INTERVAL="${VALIDATION_LOG_INTERVAL:-0}"
VALIDATION_ROLLBACK="${VALIDATION_ROLLBACK:-0}"
MIN_SUCCESS_FOR_UPDATE="${MIN_SUCCESS_FOR_UPDATE:-}"
MAX_COLLISIONS_FOR_UPDATE="${MAX_COLLISIONS_FOR_UPDATE:-}"
DISABLE_ORCA_PRIOR="${DISABLE_ORCA_PRIOR:-0}"
DISABLE_SAFETY_FILTER="${DISABLE_SAFETY_FILTER:-0}"
VARIANT_LABEL="${VARIANT_LABEL:-}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORMAL_DOMAIN_BASE="${FORMAL_DOMAIN_BASE:-50}"
FORMAL_SCENARIO_DOMAIN_STRIDE="${FORMAL_SCENARIO_DOMAIN_STRIDE:-10}"
FORMAL_SEED_DOMAIN_STRIDE="${FORMAL_SEED_DOMAIN_STRIDE:-40}"
MAX_ROS_DOMAIN_ID="${MAX_ROS_DOMAIN_ID:-232}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-models/checkpoints}"
TRAINING_BASE="${TRAINING_BASE:-results/training}"

source scripts/source_setup_file.sh "/opt/ros/${ROS_DISTRO}/setup.bash"
source scripts/source_setup_file.sh install/setup.bash

domain_for_job() {
  local seed="$1"
  local scenario_index="$2"
  local algorithm_index="$3"
  echo $((FORMAL_DOMAIN_BASE + seed * FORMAL_SEED_DOMAIN_STRIDE + scenario_index * FORMAL_SCENARIO_DOMAIN_STRIDE + algorithm_index))
}

scenario_index=0
for scenario_name in ${SCENARIOS}; do
  scenario="src/mrpp_experiments/config/scenarios/${scenario_name}.yaml"
  if [[ ! -f "${scenario}" ]]; then
    echo "Missing scenario: ${scenario}" >&2
    exit 1
  fi
  algorithm_index=0
  for algorithm in ${ALGORITHMS}; do
    for seed in ${SEEDS}; do
      checkpoint_root="${CHECKPOINT_BASE}/formal_stage1_${algorithm}_seed${seed}"
      training_root="${TRAINING_BASE}/formal_stage1_${algorithm}_seed${seed}"
      output_dir="${checkpoint_root}/${scenario_name}/${algorithm}"
      training_dir="${training_root}/${scenario_name}/${algorithm}"
      tensorboard_dir="${training_root}/tensorboard/${scenario_name}/${algorithm}"
      best_checkpoint="${output_dir}/${algorithm}_best.pt"

      if [[ "${FORCE_TRAIN}" != "1" && -f "${best_checkpoint}" ]]; then
        echo "=== skip existing ${algorithm}: ${scenario_name} seed=${seed} ==="
        continue
      fi

      mkdir -p "${output_dir}" "${training_dir}" "${tensorboard_dir}"
      export ROS_DOMAIN_ID
      ROS_DOMAIN_ID="$(domain_for_job "${seed}" "${scenario_index}" "${algorithm_index}")"
      if (( ROS_DOMAIN_ID < 0 || ROS_DOMAIN_ID > MAX_ROS_DOMAIN_ID )); then
        echo "Invalid ROS_DOMAIN_ID=${ROS_DOMAIN_ID}; adjust FORMAL_DOMAIN_BASE or domain strides to keep it in 0..${MAX_ROS_DOMAIN_ID}." >&2
        exit 2
      fi

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
      validation_args=()
      if [[ "${VALIDATION_ROLLBACK}" == "0" ]]; then
        validation_args+=(--no-validation-rollback)
      fi
      control_args=()
      if [[ "${DISABLE_ORCA_PRIOR}" == "1" ]]; then
        control_args+=(--disable-orca-prior)
      fi
      if [[ "${DISABLE_SAFETY_FILTER}" == "1" ]]; then
        control_args+=(--disable-safety-filter)
      fi
      if [[ -n "${VARIANT_LABEL}" ]]; then
        control_args+=(--variant-label "${VARIANT_LABEL}")
      fi

      echo "=== formal training ${algorithm}: ${scenario_name} seed=${seed} domain=${ROS_DOMAIN_ID} ==="
      timeout --kill-after=30s "${TRAIN_TIMEOUT_SEC}" \
        ros2 run mrpp_rl train_mo_gat_mappo \
        --scenario "${scenario}" \
        --algorithm "${algorithm}" \
        --episodes "${EPISODES}" \
        --seed "${seed}" \
        --device "${DEVICE}" \
        --log-interval "${LOG_INTERVAL}" \
        --validation-interval "${VALIDATION_INTERVAL}" \
        --validation-log-interval "${VALIDATION_LOG_INTERVAL}" \
        "${validation_args[@]}" \
        "${sim_step_args[@]}" \
        "${quality_gate_args[@]}" \
        "${control_args[@]}" \
        --output "${output_dir}/${algorithm}.pt" \
        --training-log "${training_dir}/training_log.csv" \
        --tensorboard-dir "${tensorboard_dir}" \
        "${max_step_args[@]}"

      if [[ -f "${best_checkpoint}" ]]; then
        cp "${best_checkpoint}" "${checkpoint_root}/${scenario_name}/${algorithm}_best.pt"
      fi
    done
    algorithm_index=$((algorithm_index + 1))
  done
  scenario_index=$((scenario_index + 1))
done

echo "Formal training batch completed."
