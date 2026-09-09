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
import csv
from dataclasses import dataclass, fields
from pathlib import Path

from mrpp_experiments.method_registry import load_method_registry, select_method_names


@dataclass(frozen=True)
class ExperimentJob:
    scenario: str
    scenario_path: Path
    algorithm: str
    seed: int
    output_csv: Path


EXPERIMENT_PLAN_FIELDNAMES = tuple(field.name for field in fields(ExperimentJob))


def resolve_algorithm_names(
    cli_algorithms: list[str] | None,
    method_registry: Path | None,
    include_ablations: bool,
) -> list[str]:
    if cli_algorithms:
        return cli_algorithms
    if method_registry:
        registry = load_method_registry(method_registry)
        return select_method_names(registry, include_ablations=include_ablations)
    raise ValueError("Provide at least one --algorithm or a --method-registry file.")


def build_experiment_jobs(
    scenario_dir: Path,
    algorithms: list[str],
    seeds: list[int],
    output_root: Path,
    scenario_names: list[str] | None = None,
) -> list[ExperimentJob]:
    if scenario_names:
        scenario_paths = [scenario_dir / f"{name}.yaml" for name in scenario_names]
        missing = [path for path in scenario_paths if not path.exists()]
        if missing:
            missing_names = ", ".join(path.stem for path in missing)
            raise FileNotFoundError(f"Missing scenario file(s): {missing_names}")
    else:
        scenario_paths = sorted(scenario_dir.glob("*.yaml"))
    jobs: list[ExperimentJob] = []
    for scenario_path in scenario_paths:
        scenario = scenario_path.stem
        for algorithm in algorithms:
            for seed in seeds:
                jobs.append(
                    ExperimentJob(
                        scenario=scenario,
                        scenario_path=scenario_path,
                        algorithm=algorithm,
                        seed=seed,
                        output_csv=output_root / algorithm / f"{scenario}_seed{seed}.csv",
                    )
                )
    return jobs


def write_experiment_plan(path: Path, jobs: list[ExperimentJob]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPERIMENT_PLAN_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "scenario": job.scenario,
                    "scenario_path": str(job.scenario_path),
                    "algorithm": job.algorithm,
                    "seed": job.seed,
                    "output_csv": str(job.output_csv),
                }
            )


def read_experiment_plan(path: Path) -> list[ExperimentJob]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            ExperimentJob(
                scenario=str(row["scenario"]),
                scenario_path=Path(row["scenario_path"]),
                algorithm=str(row["algorithm"]),
                seed=int(row["seed"]),
                output_csv=Path(row["output_csv"]),
            )
            for row in reader
        ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a scenario x algorithm x seed experiment plan."
    )
    parser.add_argument("--scenario-dir", required=True)
    parser.add_argument("--output-root", default="results/raw")
    parser.add_argument("--plan-output", required=True)
    parser.add_argument("--algorithm", action="append", dest="algorithms")
    parser.add_argument("--method-registry")
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help="Scenario name to include. Can be repeated; defaults to all YAML files.",
    )
    parser.add_argument("--seed", action="append", dest="seeds", type=int, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    algorithms = resolve_algorithm_names(
        cli_algorithms=args.algorithms,
        method_registry=Path(args.method_registry) if args.method_registry else None,
        include_ablations=args.include_ablations,
    )
    jobs = build_experiment_jobs(
        scenario_dir=Path(args.scenario_dir),
        algorithms=algorithms,
        seeds=args.seeds,
        output_root=Path(args.output_root),
        scenario_names=args.scenarios,
    )
    write_experiment_plan(Path(args.plan_output), jobs)
    print(f"wrote {len(jobs)} jobs to {args.plan_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
