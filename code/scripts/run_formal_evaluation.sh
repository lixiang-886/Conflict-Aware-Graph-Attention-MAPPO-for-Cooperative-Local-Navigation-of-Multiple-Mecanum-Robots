#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ROS_DISTRO="${ROS_DISTRO:-humble}"
SEEDS="${SEEDS:-0 1 2}"
SCENARIOS="${SCENARIOS:-static_clutter pedestrian_dynamic mixed_complex}"
ALGORITHMS="${ALGORITHMS:-ippo mappo maddpg mo_gat_mappo}"
DEVICE="${DEVICE:-cuda}"
SIM_STEP_MODE="${SIM_STEP_MODE:-realtime}"
GAZEBO_UPDATE_RATE="${GAZEBO_UPDATE_RATE:-}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
EVAL_TIMEOUT_SEC="${EVAL_TIMEOUT_SEC:-2400}"
FORCE_EVAL="${FORCE_EVAL:-0}"
SKIP_MISSING_CHECKPOINTS="${SKIP_MISSING_CHECKPOINTS:-1}"
CLEAN_STALE_SIM_PROCESSES="${CLEAN_STALE_SIM_PROCESSES:-1}"
SIM_CLEANUP_SLEEP_SEC="${SIM_CLEANUP_SLEEP_SEC:-1}"
DISABLE_ORCA_PRIOR="${DISABLE_ORCA_PRIOR:-0}"
DISABLE_SAFETY_FILTER="${DISABLE_SAFETY_FILTER:-0}"
VARIANT_LABEL="${VARIANT_LABEL:-}"
FORMAL_DOMAIN_BASE="${FORMAL_DOMAIN_BASE:-90}"
FORMAL_SCENARIO_DOMAIN_STRIDE="${FORMAL_SCENARIO_DOMAIN_STRIDE:-10}"
FORMAL_SEED_DOMAIN_STRIDE="${FORMAL_SEED_DOMAIN_STRIDE:-40}"
MAX_ROS_DOMAIN_ID="${MAX_ROS_DOMAIN_ID:-232}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-models/checkpoints}"
EVAL_BASE="${EVAL_BASE:-results/eval}"

source scripts/source_setup_file.sh "/opt/ros/${ROS_DISTRO}/setup.bash"
source scripts/source_setup_file.sh install/setup.bash

cleanup_stale_sim_processes() {
  if [[ "${CLEAN_STALE_SIM_PROCESSES}" != "1" ]]; then
    return
  fi
  pkill -f '[i]gn gazebo' || true
  pkill -f '[r]os_gz_bridge' || true
  pkill -f '[p]arameter_bridge' || true
  sleep "${SIM_CLEANUP_SLEEP_SEC}"
}

trap cleanup_stale_sim_processes EXIT

domain_for_job() {
  local seed="$1"
  local scenario_index="$2"
  local algorithm_index="$3"
  echo $((FORMAL_DOMAIN_BASE + seed * FORMAL_SEED_DOMAIN_STRIDE + scenario_index * FORMAL_SCENARIO_DOMAIN_STRIDE + algorithm_index))
}

checkpoint_for() {
  local algorithm="$1"
  local scenario="$2"
  local seed="$3"
  local candidates=(
    "${CHECKPOINT_BASE}/formal_stage2_${algorithm}_seed${seed}/${scenario}/${algorithm}/${algorithm}_best.pt"
    "${CHECKPOINT_BASE}/formal_stage2_${algorithm}_seed${seed}/${scenario}/${algorithm}_best.pt"
    "${CHECKPOINT_BASE}/formal_stage1_${algorithm}_seed${seed}/${scenario}/${algorithm}/${algorithm}_best.pt"
    "${CHECKPOINT_BASE}/formal_stage1_${algorithm}_seed${seed}/${scenario}/${algorithm}_best.pt"
    "${CHECKPOINT_BASE}/formal_stage1_seed${seed}/${scenario}/${algorithm}/${algorithm}_best.pt"
    "${CHECKPOINT_BASE}/formal_stage1_seed${seed}/${scenario}/${algorithm}_best.pt"
    "${CHECKPOINT_BASE}/independent_learning_seed${seed}/${scenario}/${algorithm}/${algorithm}_best.pt"
    "${CHECKPOINT_BASE}/independent_learning_seed${seed}/${scenario}/${algorithm}_best.pt"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

merge_seed_algorithm() {
  local algorithm="$1"
  local seed="$2"
  local root="${EVAL_BASE}/formal_stage1_${algorithm}_seed${seed}"
  local merge_args=()
  local count=0
  for scenario_name in ${SCENARIOS}; do
    local output="${root}/${scenario_name}/eval_seed${seed}.csv"
    if [[ -f "${output}" ]]; then
      merge_args+=(--input "${output}")
      count=$((count + 1))
    fi
  done
  if [[ "${count}" -gt 0 ]]; then
    ros2 run mrpp_experiments merge_mrpp_results \
      "${merge_args[@]}" \
      --output "${root}/all_metrics.csv"
    python3 scripts/normalize_result_config_columns.py "${root}/all_metrics.csv"
  fi
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
      root="${EVAL_BASE}/formal_stage1_${algorithm}_seed${seed}"
      outdir="${root}/${scenario_name}"
      output="${outdir}/eval_seed${seed}.csv"
      trajectory="${outdir}/eval_seed${seed}_trajectories.csv"

      if [[ "${FORCE_EVAL}" != "1" && -f "${output}" ]]; then
        echo "=== skip existing eval ${algorithm}: ${scenario_name} seed=${seed} ==="
        continue
      fi

      checkpoint_args=()
      if [[ "${algorithm}" == "ippo" || "${algorithm}" == "mappo" || "${algorithm}" == "maddpg" || "${algorithm}" == "mo_gat_mappo" || "${algorithm}" == "ablation_no_graph_attention" || "${algorithm}" == "ablation_fixed_reward" || "${algorithm}" == "ablation_no_safety_distance" ]]; then
        if checkpoint="$(checkpoint_for "${algorithm}" "${scenario_name}" "${seed}")"; then
          checkpoint_args=(--checkpoint "${checkpoint}")
        else
          message="missing checkpoint for ${algorithm}: ${scenario_name} seed=${seed}"
          if [[ "${SKIP_MISSING_CHECKPOINTS}" == "1" ]]; then
            echo "=== skip ${message} ==="
            continue
          fi
          echo "${message}" >&2
          exit 1
        fi
      fi

      mkdir -p "${outdir}"
      export ROS_DOMAIN_ID
      ROS_DOMAIN_ID="$(domain_for_job "${seed}" "${scenario_index}" "${algorithm_index}")"
      if (( ROS_DOMAIN_ID < 0 || ROS_DOMAIN_ID > MAX_ROS_DOMAIN_ID )); then
        echo "Invalid ROS_DOMAIN_ID=${ROS_DOMAIN_ID}; adjust FORMAL_DOMAIN_BASE or domain strides to keep it in 0..${MAX_ROS_DOMAIN_ID}." >&2
        exit 2
      fi

      sim_step_args=(--sim-step-mode "${SIM_STEP_MODE}")
      if [[ -n "${GAZEBO_UPDATE_RATE}" ]]; then
        sim_step_args+=(--gazebo-update-rate "${GAZEBO_UPDATE_RATE}")
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

      echo "=== formal eval ${algorithm}: ${scenario_name} seed=${seed} domain=${ROS_DOMAIN_ID} ==="
      cleanup_stale_sim_processes
      timeout --kill-after=30s "${EVAL_TIMEOUT_SEC}" \
        ros2 run mrpp_rl evaluate_mrpp \
        --scenario "${scenario}" \
        --algorithm "${algorithm}" \
        "${checkpoint_args[@]}" \
        --seed "${seed}" \
        --device "${DEVICE}" \
        --log-interval "${LOG_INTERVAL}" \
        "${sim_step_args[@]}" \
        "${control_args[@]}" \
        --output "${output}" \
        --trajectory-output "${trajectory}" \
        > "${outdir}/eval_seed${seed}.log" 2>&1
      cleanup_stale_sim_processes
    done
    algorithm_index=$((algorithm_index + 1))
  done
  scenario_index=$((scenario_index + 1))
done

for algorithm in ${ALGORITHMS}; do
  for seed in ${SEEDS}; do
    merge_seed_algorithm "${algorithm}" "${seed}"
  done
done

echo "Formal evaluation batch completed."
