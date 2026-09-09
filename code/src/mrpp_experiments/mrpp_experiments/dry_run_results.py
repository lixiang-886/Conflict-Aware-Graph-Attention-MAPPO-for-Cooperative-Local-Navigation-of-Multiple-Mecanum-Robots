# Copyright 2026 lixiang-886
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
from math import hypot
from pathlib import Path

from mrpp_experiments.experiment_matrix import read_experiment_plan
from mrpp_experiments.results import EpisodeResult, write_results_csv
from mrpp_experiments.scenario import Scenario, load_scenario

DRY_RUN_NOTE = "Pipeline dry-run row; replace with Gazebo output before paper analysis."


def build_dry_run_results(plan_path: Path) -> list[EpisodeResult]:
    rows: list[EpisodeResult] = []
    for job in read_experiment_plan(plan_path):
        scenario = load_scenario(job.scenario_path)
        rows.append(_dry_run_result(scenario, algorithm=job.algorithm, seed=job.seed))
    return rows


def write_dry_run_results(plan_path: Path, output: Path) -> None:
    write_results_csv(output, build_dry_run_results(plan_path))


def _dry_run_result(scenario: Scenario, algorithm: str, seed: int) -> EpisodeResult:
    direct_lengths = [
        hypot(robot.goal.x - robot.start.x, robot.goal.y - robot.start.y)
        for robot in scenario.robots
    ]
    average_path_length = sum(direct_lengths) / len(direct_lengths)
    completion_times = [length / 0.35 for length in direct_lengths]
    flowtime = sum(completion_times)
    makespan = max(completion_times)
    return EpisodeResult(
        scenario=scenario.name,
        algorithm=algorithm,
        seed=seed,
        robot_count=len(scenario.robots),
        success_rate=0.0,
        collision_count=0,
        deadlock_count=0,
        average_path_length=average_path_length,
        average_completion_time=flowtime / len(completion_times),
        makespan=makespan,
        flowtime=flowtime,
        average_waiting_time=0.0,
        data_source="dry_run",
        notes=DRY_RUN_NOTE,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate canonical dry-run result CSV from an experiment plan."
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    write_dry_run_results(plan_path=Path(args.plan), output=Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
