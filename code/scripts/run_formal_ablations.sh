#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ROS_DISTRO="${ROS_DISTRO:-humble}"
SEEDS="${SEEDS:-0 1 2}"
SCENARIOS="${SCENARIOS:-mixed_complex}"
ABLATION_VARIANTS="${ABLATION_VARIANTS:-full no_gat fixed_reward no_gate no_coordinator}"
EPISODES="${EPISODES:-30}"
MAX_STEPS="${MAX_STEPS:-}"
DEVICE="${DEVICE:-cuda}"
SIM_STEP_MODE="${SIM_STEP_MODE:-realtime}"
GAZEBO_UPDATE_RATE="${GAZEBO_UPDATE_RATE:-}"
LOG_INTERVAL="${LOG_INTERVAL:-150}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-5}"
VALIDATION_LOG_INTERVAL="${VALIDATION_LOG_INTERVAL:-0}"
TRAIN_TIMEOUT_SEC="${TRAIN_TIMEOUT_SEC:-14400}"
EVAL_TIMEOUT_SEC="${EVAL_TIMEOUT_SEC:-2400}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_TABLES="${RUN_TABLES:-1}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
SKIP_MISSING_CHECKPOINTS="${SKIP_MISSING_CHECKPOINTS:-0}"
FORMAL_DOMAIN_BASE="${FORMAL_DOMAIN_BASE:-20}"
FORMAL_SCENARIO_DOMAIN_STRIDE="${FORMAL_SCENARIO_DOMAIN_STRIDE:-10}"
FORMAL_SEED_DOMAIN_STRIDE="${FORMAL_SEED_DOMAIN_STRIDE:-60}"
FORMAL_VARIANT_DOMAIN_STRIDE="${FORMAL_VARIANT_DOMAIN_STRIDE:-1}"
MAX_ROS_DOMAIN_ID="${MAX_ROS_DOMAIN_ID:-232}"
RUN_LABEL="${RUN_LABEL:-formal_v62_focused_ablation_20260711}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-models/checkpoints/${RUN_LABEL}}"
TRAINING_BASE="${TRAINING_BASE:-results/training/${RUN_LABEL}}"
EVAL_BASE="${EVAL_BASE:-results/eval/${RUN_LABEL}/runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/eval/${RUN_LABEL}}"
CLEAN_STALE_SIM_PROCESSES="${CLEAN_STALE_SIM_PROCESSES:-1}"
SIM_CLEANUP_SLEEP_SEC="${SIM_CLEANUP_SLEEP_SEC:-1}"

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
  local variant_index="$3"
  echo $((FORMAL_DOMAIN_BASE + seed * FORMAL_SEED_DOMAIN_STRIDE + scenario_index * FORMAL_SCENARIO_DOMAIN_STRIDE + variant_index * FORMAL_VARIANT_DOMAIN_STRIDE))
}

configure_variant() {
  local variant="$1"
  VARIANT_LABEL=""
  VARIANT_ALGORITHM=""
  VARIANT_DISABLE_ORCA=0
  VARIANT_DISABLE_SAFETY=0
  case "${variant}" in
    full)
      VARIANT_LABEL="full_mo_gat_mappo"
      VARIANT_ALGORITHM="mo_gat_mappo"
      ;;
    no_gat|no_graph_attention)
      VARIANT_LABEL="ablation_no_graph_attention"
      VARIANT_ALGORITHM="ablation_no_graph_attention"
      ;;
    fixed_reward)
      VARIANT_LABEL="ablation_fixed_reward"
      VARIANT_ALGORITHM="ablation_fixed_reward"
      ;;
    no_gate)
      VARIANT_LABEL="ablation_no_gate"
      VARIANT_ALGORITHM="ablation_no_gate"
      ;;
    no_coordinator)
      VARIANT_LABEL="ablation_no_coordinator"
      VARIANT_ALGORITHM="ablation_no_coordinator"
      ;;
    *)
      echo "Unknown ablation variant: ${variant}" >&2
      exit 2
      ;;
  esac
}

checkpoint_for_variant() {
  local variant="$1"
  local algorithm="$2"
  local scenario="$3"
  local seed="$4"
  local candidates=(
    "${CHECKPOINT_BASE}/formal_ablation_${variant}_seed${seed}/${scenario}/${algorithm}/${algorithm}_best.pt"
    "${CHECKPOINT_BASE}/formal_ablation_${variant}_seed${seed}/${scenario}/${algorithm}_best.pt"
  )
  if [[ "${variant}" == "full" ]]; then
    candidates+=(
      "${CHECKPOINT_BASE}/formal_stage1_${algorithm}_seed${seed}/${scenario}/${algorithm}/${algorithm}_best.pt"
      "${CHECKPOINT_BASE}/formal_stage1_${algorithm}_seed${seed}/${scenario}/${algorithm}_best.pt"
      "${CHECKPOINT_BASE}/formal_stage1_seed${seed}/${scenario}/${algorithm}/${algorithm}_best.pt"
      "${CHECKPOINT_BASE}/formal_stage1_seed${seed}/${scenario}/${algorithm}_best.pt"
      "${CHECKPOINT_BASE}/formal_stage2_${algorithm}_seed${seed}/${scenario}/${algorithm}/${algorithm}_best.pt"
      "${CHECKPOINT_BASE}/formal_stage2_${algorithm}_seed${seed}/${scenario}/${algorithm}_best.pt"
    )
  fi
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

control_args_for_variant() {
  CONTROL_ARGS=(--variant-label "${VARIANT_LABEL}")
  if [[ "${VARIANT_DISABLE_ORCA}" == "1" ]]; then
    CONTROL_ARGS+=(--disable-orca-prior)
  fi
  if [[ "${VARIANT_DISABLE_SAFETY}" == "1" ]]; then
    CONTROL_ARGS+=(--disable-safety-filter)
  fi
}

validate_domain() {
  if (( ROS_DOMAIN_ID < 0 || ROS_DOMAIN_ID > MAX_ROS_DOMAIN_ID )); then
    echo "Invalid ROS_DOMAIN_ID=${ROS_DOMAIN_ID}; adjust FORMAL_DOMAIN_BASE or domain strides to keep it in 0..${MAX_ROS_DOMAIN_ID}." >&2
    exit 2
  fi
}

training_row_count() {
  local training_log="$1"
  if [[ ! -f "${training_log}" ]]; then
    echo 0
    return
  fi
  awk 'END { print (NR > 0 ? NR - 1 : 0) }' "${training_log}"
}

training_is_complete() {
  local best_checkpoint="$1"
  local final_checkpoint="$2"
  local config_snapshot="$3"
  local training_log="$4"
  [[ -f "${best_checkpoint}" \
    && -f "${final_checkpoint}" \
    && -f "${config_snapshot}" \
    && "$(training_row_count "${training_log}")" -eq "${EPISODES}" ]]
}

variant_index=0
for variant in ${ABLATION_VARIANTS}; do
  configure_variant "${variant}"
  control_args_for_variant
  scenario_index=0
  for scenario_name in ${SCENARIOS}; do
    scenario="src/mrpp_experiments/config/scenarios/${scenario_name}.yaml"
    if [[ ! -f "${scenario}" ]]; then
      echo "Missing scenario: ${scenario}" >&2
      exit 1
    fi
    for seed in ${SEEDS}; do
      export ROS_DOMAIN_ID
      ROS_DOMAIN_ID="$(domain_for_job "${seed}" "${scenario_index}" "${variant_index}")"
      validate_domain

      checkpoint_root="${CHECKPOINT_BASE}/formal_ablation_${variant}_seed${seed}"
      training_root="${TRAINING_BASE}/formal_ablation_${variant}_seed${seed}"
      eval_root="${EVAL_BASE}/formal_ablation_${variant}_seed${seed}"
      output_dir="${checkpoint_root}/${scenario_name}/${VARIANT_ALGORITHM}"
      training_dir="${training_root}/${scenario_name}/${VARIANT_ALGORITHM}"
      tensorboard_dir="${training_root}/tensorboard/${scenario_name}/${VARIANT_ALGORITHM}"
      eval_dir="${eval_root}/${scenario_name}"
      best_checkpoint="${output_dir}/${VARIANT_ALGORITHM}_best.pt"
      final_checkpoint="${output_dir}/${VARIANT_ALGORITHM}.pt"
      config_snapshot="${output_dir}/config_snapshot.json"
      training_log="${training_dir}/training_log.csv"

      max_step_args=()
      if [[ -n "${MAX_STEPS}" ]]; then
        max_step_args+=(--max-steps "${MAX_STEPS}")
      fi
      sim_step_args=(--sim-step-mode "${SIM_STEP_MODE}")
      if [[ -n "${GAZEBO_UPDATE_RATE}" ]]; then
        sim_step_args+=(--gazebo-update-rate "${GAZEBO_UPDATE_RATE}")
      fi

      if [[ "${RUN_TRAIN}" == "1" ]]; then
        if [[ "${FORCE_TRAIN}" != "1" ]] && training_is_complete \
          "${best_checkpoint}" "${final_checkpoint}" "${config_snapshot}" "${training_log}"; then
          echo "=== skip existing ablation train ${variant}: ${scenario_name} seed=${seed} ==="
        else
          if [[ -d "${output_dir}" || -d "${training_dir}" || -d "${tensorboard_dir}" ]]; then
            echo "=== clear incomplete ablation train ${variant}: ${scenario_name} seed=${seed} ==="
            rm -rf "${output_dir}" "${training_dir}" "${tensorboard_dir}"
          fi
          mkdir -p "${output_dir}" "${training_dir}" "${tensorboard_dir}"
          echo "=== ablation train ${variant}: ${scenario_name} seed=${seed} domain=${ROS_DOMAIN_ID} ==="
          cleanup_stale_sim_processes
          timeout --kill-after=30s "${TRAIN_TIMEOUT_SEC}" \
            ros2 run mrpp_rl train_mo_gat_mappo \
            --scenario "${scenario}" \
            --algorithm "${VARIANT_ALGORITHM}" \
            --episodes "${EPISODES}" \
            --seed "${seed}" \
            --device "${DEVICE}" \
            --log-interval "${LOG_INTERVAL}" \
            --validation-interval "${VALIDATION_INTERVAL}" \
            --validation-log-interval "${VALIDATION_LOG_INTERVAL}" \
            --no-validation-rollback \
            --initial-validation-diagnostic-only \
            "${sim_step_args[@]}" \
            "${CONTROL_ARGS[@]}" \
            "${max_step_args[@]}" \
            --output "${output_dir}/${VARIANT_ALGORITHM}.pt" \
            --training-log "${training_log}" \
            --tensorboard-dir "${tensorboard_dir}" \
            > "${training_dir}/train.log" 2>&1
          cleanup_stale_sim_processes
        fi
      fi

      if [[ "${RUN_EVAL}" == "1" ]]; then
        mkdir -p "${eval_dir}"
        output="${eval_dir}/eval_seed${seed}.csv"
        trajectory="${eval_dir}/eval_seed${seed}_trajectories.csv"
        if [[ "${FORCE_EVAL}" != "1" && -f "${output}" ]]; then
          echo "=== skip existing ablation eval ${variant}: ${scenario_name} seed=${seed} ==="
          continue
        fi
        if checkpoint="$(checkpoint_for_variant "${variant}" "${VARIANT_ALGORITHM}" "${scenario_name}" "${seed}")"; then
          checkpoint_args=(--checkpoint "${checkpoint}")
        else
          message="missing checkpoint for ablation ${variant}: ${scenario_name} seed=${seed}"
          if [[ "${SKIP_MISSING_CHECKPOINTS}" == "1" ]]; then
            echo "=== skip ${message} ==="
            continue
          fi
          echo "${message}" >&2
          exit 1
        fi
        echo "=== ablation eval ${variant}: ${scenario_name} seed=${seed} domain=${ROS_DOMAIN_ID} ==="
        cleanup_stale_sim_processes
        timeout --kill-after=30s "${EVAL_TIMEOUT_SEC}" \
          ros2 run mrpp_rl evaluate_mrpp \
          --scenario "${scenario}" \
          --algorithm "${VARIANT_ALGORITHM}" \
          "${checkpoint_args[@]}" \
          --seed "${seed}" \
          --device "${DEVICE}" \
          --log-interval "${LOG_INTERVAL}" \
          "${sim_step_args[@]}" \
          "${CONTROL_ARGS[@]}" \
          "${max_step_args[@]}" \
          --output "${output}" \
          --trajectory-output "${trajectory}" \
          > "${eval_dir}/eval_seed${seed}.log" 2>&1
        cleanup_stale_sim_processes
      fi
    done
    scenario_index=$((scenario_index + 1))
  done
  variant_index=$((variant_index + 1))
done

if [[ "${RUN_EVAL}" == "1" && "${RUN_TABLES}" == "1" ]]; then
  mkdir -p "${OUTPUT_ROOT}"
  merge_args=()
  input_count=0
  for variant in ${ABLATION_VARIANTS}; do
    for seed in ${SEEDS}; do
      root="${EVAL_BASE}/formal_ablation_${variant}_seed${seed}"
      seed_merge_args=()
      seed_count=0
      for scenario_name in ${SCENARIOS}; do
        output="${root}/${scenario_name}/eval_seed${seed}.csv"
        if [[ -f "${output}" ]]; then
          seed_merge_args+=(--input "${output}")
          merge_args+=(--input "${output}")
          seed_count=$((seed_count + 1))
          input_count=$((input_count + 1))
        fi
      done
      if [[ "${seed_count}" -gt 0 ]]; then
        ros2 run mrpp_experiments merge_mrpp_results \
          "${seed_merge_args[@]}" \
          --output "${root}/all_metrics.csv"
      fi
    done
  done
  if [[ "${input_count}" -gt 0 ]]; then
    ros2 run mrpp_experiments merge_mrpp_results \
      "${merge_args[@]}" \
      --output "${OUTPUT_ROOT}/all_metrics.csv"
    ros2 run mrpp_paper plot_mrpp_metrics \
      --input "${OUTPUT_ROOT}/all_metrics.csv" \
      --output "${OUTPUT_ROOT}/metric_summary.csv"
    ros2 run mrpp_paper mrpp_statistical_tests \
      --input "${OUTPUT_ROOT}/all_metrics.csv" \
      --output "${OUTPUT_ROOT}/statistical_tests.csv" \
      --metric average_completion_time \
      --metric makespan \
      --metric average_waiting_time \
      --metric average_path_efficiency \
      --metric average_turning_angle_per_meter \
      --metric min_robot_robot_distance \
      --metric intervention_time_ratio
    python3 scripts/export_formal_ablation_table.py \
      --all-metrics "${OUTPUT_ROOT}/all_metrics.csv" \
      --output-root "${OUTPUT_ROOT}/paper_assets"
    python3 scripts/finalize_focused_ablation.py \
      --run-root "${OUTPUT_ROOT}" \
      --checkpoint-root "${CHECKPOINT_BASE}" \
      --training-root "${TRAINING_BASE}"
  fi
  cat > "${OUTPUT_ROOT}/README.md" <<EOF
# Formal Ablation Comparison

Generated by \`scripts/run_formal_ablations.sh\`.

- Variants: \`${ABLATION_VARIANTS}\`
- Scenarios: \`${SCENARIOS}\`
- Combined metrics: \`${OUTPUT_ROOT}/all_metrics.csv\`
- Mean/std summary: \`${OUTPUT_ROOT}/metric_summary.csv\`
- Paired statistical tests: \`${OUTPUT_ROOT}/statistical_tests.csv\`
- Paper assets: \`${OUTPUT_ROOT}/paper_assets\`
- Protocol manifest: \`${OUTPUT_ROOT}/protocol_manifest.json\`
- Validation report: \`${OUTPUT_ROOT}/VALIDATION.md\`

Each result row records \`variant\`, \`orca_prior_enabled\`, and
\`safety_filter_enabled\` so full and ablated runs cannot be mixed silently.
EOF
fi

echo "Formal ablation batch completed."
