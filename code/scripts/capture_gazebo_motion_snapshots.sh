#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ROS_DISTRO="${ROS_DISTRO:-humble}"
SCENARIO="${SCENARIO:-${1:-mixed_complex}}"
ALGORITHM="${ALGORITHM:-mo_gat_mappo}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/eval/final_selected_20260704/gazebo_snapshots}"
EVAL_TIMEOUT_SEC="${EVAL_TIMEOUT_SEC:-900}"
WINDOW_PATTERN="${WINDOW_PATTERN:-Gazebo|Ignition}"
CLEANUP_OLD_PROCESSES="${CLEANUP_OLD_PROCESSES:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-25}"
TOPDOWN_GUI_CONFIG="${TOPDOWN_GUI_CONFIG:-${ROOT_DIR}/scripts/gazebo_topdown_gui.config}"
INITIAL_SETTLE_SEC="${INITIAL_SETTLE_SEC:-0.3}"
MANUAL_START="${MANUAL_START:-0}"
MANUAL_START_FILE="${MANUAL_START_FILE:-}"
USE_GUI_CONFIG="${USE_GUI_CONFIG:-1}"
FINAL_HOLD_FILE="${FINAL_HOLD_FILE:-}"

case "${SCENARIO}" in
  static_clutter)
    DEFAULT_SEED=2
    DEFAULT_STEPS="initial 50 150 250 350 400 450"
    ;;
  pedestrian_dynamic)
    DEFAULT_SEED=2
    DEFAULT_STEPS="initial 25 75 125 175 225 275"
    ;;
  mixed_complex)
    DEFAULT_SEED=0
    DEFAULT_STEPS="initial 100 250 400 550 700 850"
    ;;
  *)
    echo "Unknown scenario: ${SCENARIO}" >&2
    exit 1
    ;;
esac

SEED="${SEED:-${2:-${DEFAULT_SEED}}}"
SNAPSHOT_STEPS="${SNAPSHOT_STEPS:-${DEFAULT_STEPS}}"
SCENARIO_FILE="src/mrpp_experiments/config/scenarios/${SCENARIO}.yaml"
OUTDIR="${OUTPUT_ROOT}/${SCENARIO}_seed${SEED}"
LOG_FILE="${OUTDIR}/evaluate_gui.log"
RESULT_CSV="${OUTDIR}/eval_seed${SEED}.csv"
TRAJ_CSV="${OUTDIR}/eval_seed${SEED}_trajectories.csv"

source scripts/source_setup_file.sh "/opt/ros/${ROS_DISTRO}/setup.bash"
source scripts/source_setup_file.sh install/setup.bash

checkpoint_candidates=(
  "models/checkpoints/mixed_mogat_seed0_safegate_20260704/formal_stage1_${ALGORITHM}_seed${SEED}/${SCENARIO}/${ALGORITHM}/${ALGORITHM}_best.pt"
  "models/checkpoints/targeted_retrain_current_20260703/formal_stage1_${ALGORITHM}_seed${SEED}/${SCENARIO}/${ALGORITHM}/${ALGORITHM}_best.pt"
  "models/checkpoints/pedestrian_dynamic_multiseed_selectionfix_20260703/formal_stage1_${ALGORITHM}_seed${SEED}/${SCENARIO}/${ALGORITHM}/${ALGORITHM}_best.pt"
  "models/checkpoints/formal_stage2_${ALGORITHM}_seed${SEED}/${SCENARIO}/${ALGORITHM}/${ALGORITHM}_best.pt"
  "models/checkpoints/formal_stage2_${ALGORITHM}_seed${SEED}/${SCENARIO}/${ALGORITHM}_best.pt"
  "models/checkpoints/formal_stage1_${ALGORITHM}_seed${SEED}/${SCENARIO}/${ALGORITHM}/${ALGORITHM}_best.pt"
  "models/checkpoints/formal_stage1_${ALGORITHM}_seed${SEED}/${SCENARIO}/${ALGORITHM}_best.pt"
  "models/checkpoints/formal_stage1_seed${SEED}/${SCENARIO}/${ALGORITHM}/${ALGORITHM}_best.pt"
  "models/checkpoints/formal_stage1_seed${SEED}/${SCENARIO}/${ALGORITHM}_best.pt"
)

CHECKPOINT="${CHECKPOINT:-}"
if [[ -z "${CHECKPOINT}" ]]; then
  for candidate in "${checkpoint_candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      CHECKPOINT="${candidate}"
      break
    fi
  done
fi

if [[ -z "${CHECKPOINT}" ]]; then
  echo "No checkpoint found for ${ALGORITHM} ${SCENARIO} seed=${SEED}" >&2
  exit 1
fi

mkdir -p "${OUTDIR}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-228}"
if [[ "${USE_GUI_CONFIG}" == "1" ]]; then
  export MRPP_GAZEBO_GUI_CONFIG="${MRPP_GAZEBO_GUI_CONFIG:-${TOPDOWN_GUI_CONFIG}}"
else
  unset MRPP_GAZEBO_GUI_CONFIG
fi

if [[ "${CLEANUP_OLD_PROCESSES}" == "1" ]]; then
  pkill -9 -f "ros_gz_bridge" 2>/dev/null || true
  pkill -9 -f "parameter_bridge" 2>/dev/null || true
  pkill -9 -f "ign gazebo" 2>/dev/null || true
  sleep 2
fi

echo "Scenario: ${SCENARIO}"
echo "Seed: ${SEED}"
echo "Checkpoint: ${CHECKPOINT}"
echo "ROS_DOMAIN_ID: ${ROS_DOMAIN_ID}"
echo "GUI config: ${MRPP_GAZEBO_GUI_CONFIG:-<default interactive Gazebo GUI>}"
echo "Snapshots: ${SNAPSHOT_STEPS}"
echo "Output: ${OUTDIR}"
echo "Manual start: ${MANUAL_START}"

eval_pid=""
initial_gazebo_pid=""

cleanup_processes() {
  if [[ -n "${eval_pid}" ]] && kill -0 "${eval_pid}" 2>/dev/null; then
    kill "${eval_pid}" 2>/dev/null || true
    wait "${eval_pid}" 2>/dev/null || true
  fi
  if [[ -n "${initial_gazebo_pid}" ]] && kill -0 "${initial_gazebo_pid}" 2>/dev/null; then
    kill "${initial_gazebo_pid}" 2>/dev/null || true
    wait "${initial_gazebo_pid}" 2>/dev/null || true
  fi
}
trap cleanup_processes INT TERM

has_initial_snapshot=0
has_final_snapshot=0
runtime_steps=()
for step in ${SNAPSHOT_STEPS}; do
  if [[ "${step}" == "initial" || "${step}" == "0" ]]; then
    has_initial_snapshot=1
  elif [[ "${step}" == "final" ]]; then
    has_final_snapshot=1
  else
    runtime_steps+=("${step}")
  fi
done

write_robot_config() {
  /usr/bin/python3 - "${SCENARIO_FILE}" "${OUTDIR}/tmp_robot_config.yaml" <<'PY'
from pathlib import Path
import sys
import yaml

scenario_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
lines = ["robots:"]
for robot in scenario["robots"]:
    start = robot["start"]
    lines.append(f"  - name: {robot['name']}")
    lines.append(f"    x: {start['x']}")
    lines.append(f"    y: {start['y']}")
    lines.append(f"    yaw: {start['yaw']}")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

capture_initial_snapshot() {
  local gazebo_pkg
  local world_file
  local robot_model
  local ped_model
  local init_log
  local output

  gazebo_pkg="$(ros2 pkg prefix mrpp_gazebo)/share/mrpp_gazebo"
  world_file="${gazebo_pkg}/worlds/${SCENARIO}.sdf"
  robot_model="${gazebo_pkg}/models/mecanum_robot/model.sdf"
  ped_model="${gazebo_pkg}/models/pedestrian_fuel/model.sdf"
  init_log="${OUTDIR}/initial_gazebo.log"
  output="${OUTDIR}/${SCENARIO}_seed${SEED}_initial.png"

  echo "Capturing initial top-down scene..."
  write_robot_config
  export IGN_GAZEBO_RESOURCE_PATH="${gazebo_pkg}/models:${IGN_GAZEBO_RESOURCE_PATH:-}"
  export GZ_SIM_RESOURCE_PATH="${gazebo_pkg}/models:${GZ_SIM_RESOURCE_PATH:-}"
  ign gazebo "${world_file}" \
    --gui-config "${MRPP_GAZEBO_GUI_CONFIG}" \
    -v 2 \
    > "${init_log}" 2>&1 &
  initial_gazebo_pid=$!
  sleep 8
  if ! kill -0 "${initial_gazebo_pid}" 2>/dev/null; then
    echo "Initial Gazebo process exited unexpectedly." >&2
    cat "${init_log}" >&2 || true
    exit 1
  fi

  ros2 run mrpp_gazebo spawn_robots \
    "${OUTDIR}/tmp_robot_config.yaml" \
    "${robot_model}"

  if [[ "${SCENARIO}" == "pedestrian_dynamic" || "${SCENARIO}" == "mixed_complex" ]]; then
    ros2 run mrpp_gazebo pedestrian_controller \
      --scenario "${SCENARIO}" \
      --sdf "${ped_model}"
  fi

  if [[ "${INITIAL_SETTLE_SEC}" != "0" && "${INITIAL_SETTLE_SEC}" != "0.0" ]]; then
    ign service \
      -s "/world/${SCENARIO}/control" \
      --reqtype ignition.msgs.WorldControl \
      --reptype ignition.msgs.Boolean \
      --timeout 2000 \
      --req "pause: false" >/dev/null 2>&1 || true
    sleep "${INITIAL_SETTLE_SEC}"
    ign service \
      -s "/world/${SCENARIO}/control" \
      --reqtype ignition.msgs.WorldControl \
      --reptype ignition.msgs.Boolean \
      --timeout 2000 \
      --req "pause: true" >/dev/null 2>&1 || true
  fi

  sleep 3
  python3 scripts/capture_gazebo_window.py \
    --output "${output}" \
    --title-pattern "${WINDOW_PATTERN}" \
    --timeout 20 \
    --frame

  kill "${initial_gazebo_pid}" 2>/dev/null || true
  wait "${initial_gazebo_pid}" 2>/dev/null || true
  initial_gazebo_pid=""
  sleep 2
}

if [[ "${has_initial_snapshot}" == "1" ]]; then
  capture_initial_snapshot
  pkill -9 -f "ign gazebo" 2>/dev/null || true
  sleep 2
fi

if [[ "${#runtime_steps[@]}" -eq 0 && "${has_final_snapshot}" == "0" ]]; then
  echo "Initial snapshot capture complete:"
  find "${OUTDIR}" -maxdepth 1 -type f -name '*_initial.png' -printf '  %p\n' | sort
  exit 0
fi

eval_args=(
  ros2 run mrpp_rl evaluate_mrpp
  --scenario "${SCENARIO_FILE}"
  --algorithm "${ALGORITHM}"
  --checkpoint "${CHECKPOINT}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --gui
  --sim-step-mode realtime
  --log-interval "${LOG_INTERVAL}"
  --output "${RESULT_CSV}"
  --trajectory-output "${TRAJ_CSV}"
)

if [[ "${MANUAL_START}" == "1" ]]; then
  if [[ -z "${MANUAL_START_FILE}" ]]; then
    MANUAL_START_FILE="${OUTDIR}/manual_start.signal"
  fi
  rm -f "${MANUAL_START_FILE}"
  eval_args+=(--manual-start-file "${MANUAL_START_FILE}")
fi

if [[ "${has_final_snapshot}" == "1" ]]; then
  if [[ -z "${FINAL_HOLD_FILE}" ]]; then
    FINAL_HOLD_FILE="${OUTDIR}/final_hold.signal"
  fi
  rm -f "${FINAL_HOLD_FILE}"
  eval_args+=(--final-hold-file "${FINAL_HOLD_FILE}")
fi

timeout --kill-after=30s "${EVAL_TIMEOUT_SEC}" \
  "${eval_args[@]}" \
  > "${LOG_FILE}" 2>&1 &

eval_pid=$!

if [[ "${MANUAL_START}" == "1" ]]; then
  ready_pattern="Manual GUI adjustment ready."
  echo "Waiting until Gazebo is ready for manual view adjustment..."
  while kill -0 "${eval_pid}" 2>/dev/null; do
    if grep -q "${ready_pattern}" "${LOG_FILE}" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if ! grep -q "${ready_pattern}" "${LOG_FILE}" 2>/dev/null; then
    echo "Evaluator exited before manual adjustment became ready." >&2
    tail -120 "${LOG_FILE}" >&2 || true
    wait "${eval_pid}" || true
    exit 1
  fi
  echo "Gazebo is ready. Adjust the GUI view now, then press Enter here to start evaluation and automatic screenshots."
  read -r
  touch "${MANUAL_START_FILE}"
fi

wait_for_step() {
  local step="$1"
  local pattern="eval step=${step}/"
  while kill -0 "${eval_pid}" 2>/dev/null; do
    if grep -q "${pattern}" "${LOG_FILE}" 2>/dev/null; then
      return 0
    fi
    if grep -q "Final GUI hold ready" "${LOG_FILE}" 2>/dev/null; then
      return 1
    fi
    sleep 0.1
  done
  grep -q "${pattern}" "${LOG_FILE}" 2>/dev/null
}

capture_gazebo_png() {
  local output="$1"
  local attempt
  for attempt in 1 2 3; do
    if python3 scripts/capture_gazebo_window.py \
      --output "${output}" \
      --title-pattern "${WINDOW_PATTERN}" \
      --timeout 20 \
      --frame; then
      return 0
    fi
    sleep 1
  done
  return 1
}

for step in "${runtime_steps[@]}"; do
  if wait_for_step "${step}"; then
    output="${OUTDIR}/${SCENARIO}_seed${SEED}_step${step}.png"
    if ! capture_gazebo_png "${output}"; then
      echo "Warning: failed to capture step ${step}" >&2
    fi
  else
    if grep -q "Final GUI hold ready" "${LOG_FILE}" 2>/dev/null; then
      break
    fi
    echo "Skipping step ${step}; evaluator finished before this log point." >&2
  fi
done

if [[ "${has_final_snapshot}" == "1" ]]; then
  final_pattern="Final GUI hold ready"
  while kill -0 "${eval_pid}" 2>/dev/null; do
    if grep -q "${final_pattern}" "${LOG_FILE}" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
  if grep -q "${final_pattern}" "${LOG_FILE}" 2>/dev/null; then
    output="${OUTDIR}/${SCENARIO}_seed${SEED}_final.png"
    capture_status=0
    capture_gazebo_png "${output}" || capture_status=$?
    touch "${FINAL_HOLD_FILE}"
    if [[ "${capture_status}" -ne 0 ]]; then
      echo "Failed to capture final arrival state." >&2
      wait "${eval_pid}" || true
      exit "${capture_status}"
    fi
  else
    echo "Evaluator exited before the final GUI hold." >&2
    wait "${eval_pid}" || true
    exit 1
  fi
fi

wait "${eval_pid}"

echo "Snapshot capture complete:"
find "${OUTDIR}" -maxdepth 1 -type f -name '*.png' -printf '  %p\n' | sort
