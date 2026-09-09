#!/usr/bin/env bash
set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-humble}"
source scripts/source_setup_file.sh "/opt/ros/${ROS_DISTRO}/setup.bash"
source scripts/source_setup_file.sh install/setup.bash

bash scripts/make_experiment_plan.sh

python3 -m mrpp_experiments.dry_run_results \
  --plan results/raw/experiment_plan.csv \
  --output results/raw/all_metrics_dry_run.csv

python3 -m mrpp_paper.plot_metrics \
  --input results/raw/all_metrics_dry_run.csv \
  --output results/paper/metric_summary_dry_run.csv

python3 -m mrpp_paper.statistical_tests \
  --input results/raw/all_metrics_dry_run.csv \
  --output results/paper/statistical_tests_dry_run.csv

echo "Dry-run data pipeline completed."
