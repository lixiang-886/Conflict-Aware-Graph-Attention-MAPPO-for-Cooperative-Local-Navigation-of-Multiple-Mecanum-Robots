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

from mrpp_experiments.results import EpisodeResult, write_results_csv
from mrpp_experiments.scenario import Scenario, load_scenario

DEFAULT_BASELINES = (
    "priority_astar",
    "ippo",
    "mappo",
    "maddpg",
    "mo_gat_mappo",
)


def build_template_results(
    scenario_path: Path, algorithms: list[str], seed: int
) -> list[EpisodeResult]:
    scenario = load_scenario(scenario_path)
    return [
        _build_template_result(scenario, algorithm=algorithm, seed=seed)
        for algorithm in algorithms
    ]


def write_template_results(
    scenario_paths: list[Path],
    output: Path,
    algorithms: list[str] | None = None,
    seed: int = 0,
) -> None:
    selected_algorithms = algorithms or list(DEFAULT_BASELINES)
    rows: list[EpisodeResult] = []
    for scenario_path in scenario_paths:
        rows.extend(build_template_results(scenario_path, selected_algorithms, seed))
    write_results_csv(output, rows)


def _build_template_result(scenario: Scenario, algorithm: str, seed: int) -> EpisodeResult:
    direct_lengths = [
        hypot(robot.goal.x - robot.start.x, robot.goal.y - robot.start.y)
        for robot in scenario.robots
    ]
    average_path_length = sum(direct_lengths) / len(direct_lengths)
    flowtime = sum(length / 0.35 for length in direct_lengths)
    makespan = max(length / 0.35 for length in direct_lengths)
    return EpisodeResult(
        scenario=scenario.name,
        algorithm=algorithm,
        seed=seed,
        robot_count=len(scenario.robots),
        success_rate=0.0,
        collision_count=0,
        deadlock_count=0,
        average_path_length=average_path_length,
        average_completion_time=flowtime / len(scenario.robots),
        makespan=makespan,
        flowtime=flowtime,
        average_waiting_time=0.0,
        data_source="template",
        notes="Template row for CSV/schema smoke tests; replace with Gazebo output.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create paper-result CSV template rows for scenarios."
    )
    parser.add_argument("--scenario-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--algorithm", action="append", dest="algorithms")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    scenario_paths = sorted(Path(args.scenario_dir).glob("*.yaml"))
    write_template_results(
        scenario_paths=scenario_paths,
        output=Path(args.output),
        algorithms=args.algorithms,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
