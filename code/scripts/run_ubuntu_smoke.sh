#!/usr/bin/env bash
set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-humble}"
source scripts/source_setup_file.sh "/opt/ros/${ROS_DISTRO}/setup.bash"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

python3 -m pip install --user -r requirements-ubuntu.txt
colcon build --symlink-install
source scripts/source_setup_file.sh install/setup.bash
colcon test
colcon test-result --verbose

PYTHONPATH=src/mrpp_gazebo:src/mrpp_experiments:src/mrpp_navigation:src/mrpp_rl:src/mrpp_paper \
python3 -m pytest tests/python -q

bash scripts/make_experiment_plan.sh
test -f results/raw/experiment_plan.csv
python3 -m mrpp_experiments.run_batch \
  --plan results/raw/experiment_plan.csv \
  --output-dir results/raw/batch \
  --manifest-only
test -f results/raw/batch/batch_manifest.txt
bash scripts/run_data_pipeline_dry_run.sh
test -f results/raw/all_metrics_dry_run.csv
test -f results/paper/metric_summary_dry_run.csv

echo "Ubuntu smoke checks completed."
