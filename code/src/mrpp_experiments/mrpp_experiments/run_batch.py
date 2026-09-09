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
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from mrpp_experiments.experiment_matrix import ExperimentJob, read_experiment_plan
from mrpp_experiments.merge_results import merge_result_csv_files

CHECKPOINT_ALGORITHMS = {
    "ippo",
    "maddpg",
    "mappo",
    "mo_gat_mappo",
    "ablation_no_graph_attention",
    "ablation_fixed_reward",
    "ablation_no_safety_distance",
}
STATUS_FIELDNAMES = (
    "scenario",
    "algorithm",
    "seed",
    "status",
    "output_csv",
    "duration_s",
    "message",
)


@dataclass(frozen=True)
class BatchResult:
    job: ExperimentJob
    status: str
    duration_s: float
    message: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a batch of multi-robot path-planning experiments."
    )
    parser.add_argument("--scenario-dir")
    parser.add_argument("--algorithm")
    parser.add_argument("--plan")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--checkpoint-dir", default="models/checkpoints")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional per-job step cap passed to evaluate_mrpp for smoke runs.",
    )
    parser.add_argument("--timeout-sec", type=float, default=3600.0)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--scenario", action="append", dest="scenarios")
    parser.add_argument("--seed", action="append", dest="seeds", type=int)
    parser.add_argument("--merge-output", default=None)
    return parser


def create_batch_manifest(plan_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "batch_manifest.txt"
    lines = [f"plan={plan_path}"]
    for job in read_experiment_plan(plan_path):
        lines.append(
            "RUN "
            f"scenario={job.scenario} "
            f"algorithm={job.algorithm} "
            f"seed={job.seed} "
            f"scenario_path={job.scenario_path} "
            f"output_csv={job.output_csv}"
        )
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def filter_jobs(
    jobs: list[ExperimentJob],
    scenarios: list[str] | None = None,
    algorithm: str | None = None,
    seeds: list[int] | None = None,
) -> list[ExperimentJob]:
    selected = jobs
    if scenarios:
        selected = [job for job in selected if job.scenario in set(scenarios)]
    if algorithm:
        selected = [job for job in selected if job.algorithm == algorithm]
    if seeds:
        selected = [job for job in selected if job.seed in set(seeds)]
    return selected


def checkpoint_for_job(job: ExperimentJob, checkpoint_dir: Path) -> Path:
    for candidate in checkpoint_candidates_for_job(job, checkpoint_dir):
        if candidate.exists():
            return candidate
    return checkpoint_candidates_for_job(job, checkpoint_dir)[0]


def checkpoint_candidates_for_job(
    job: ExperimentJob,
    checkpoint_dir: Path,
) -> list[Path]:
    seed_dir = checkpoint_dir / f"independent_learning_seed{job.seed}"
    return [
        seed_dir / job.scenario / f"{job.algorithm}_best.pt",
        seed_dir / job.scenario / job.algorithm / f"{job.algorithm}_best.pt",
        seed_dir / job.scenario / job.algorithm / f"{job.algorithm}.pt",
        checkpoint_dir / job.scenario / f"{job.algorithm}_best.pt",
        checkpoint_dir / job.scenario / job.algorithm / f"{job.algorithm}_best.pt",
        checkpoint_dir / job.scenario / job.algorithm / f"{job.algorithm}.pt",
        checkpoint_dir / f"{job.scenario}_{job.algorithm}_best.pt",
        checkpoint_dir / f"{job.algorithm}_best.pt",
        checkpoint_dir / f"{job.algorithm}.pt",
    ]


def command_for_job(
    job: ExperimentJob,
    checkpoint_dir: Path,
    max_steps: int | None = None,
) -> list[str]:
    command = [
        "ros2",
        "run",
        "mrpp_rl",
        "evaluate_mrpp",
        "--scenario",
        str(job.scenario_path),
        "--algorithm",
        job.algorithm,
        "--seed",
        str(job.seed),
        "--output",
        str(job.output_csv),
    ]
    if job.algorithm in CHECKPOINT_ALGORITHMS:
        command.extend(["--checkpoint", str(checkpoint_for_job(job, checkpoint_dir))])
    if max_steps is not None:
        command.extend(["--max-steps", str(max_steps)])
    return command


def run_batch_jobs(
    jobs: list[ExperimentJob],
    output_dir: Path,
    checkpoint_dir: Path,
    timeout_sec: float,
    max_steps: int | None = None,
    max_workers: int = 1,
    force: bool = False,
) -> list[BatchResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if max_workers <= 1:
        return [
            _run_one_job(
                job,
                output_dir,
                checkpoint_dir,
                timeout_sec,
                force,
                max_steps=max_steps,
            )
            for job in jobs
        ]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(
            executor.map(
                lambda job: _run_one_job(
                    job,
                    output_dir,
                    checkpoint_dir,
                    timeout_sec,
                    force,
                    max_steps=max_steps,
                ),
                jobs,
            )
        )


def _run_one_job(
    job: ExperimentJob,
    output_dir: Path,
    checkpoint_dir: Path,
    timeout_sec: float,
    force: bool,
    max_steps: int | None = None,
) -> BatchResult:
    start = time.time()
    run_dir = output_dir / job.algorithm / f"{job.scenario}_seed{job.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    if job.output_csv.exists() and not force:
        return BatchResult(job, "skipped_existing", 0.0, "output exists")
    if job.algorithm in CHECKPOINT_ALGORITHMS:
        checkpoint = checkpoint_for_job(job, checkpoint_dir)
        if not checkpoint.exists():
            candidates = ", ".join(
                str(path) for path in checkpoint_candidates_for_job(job, checkpoint_dir)
            )
            return BatchResult(
                job,
                "failed",
                0.0,
                f"missing checkpoint; tried: {candidates}",
            )
    _cleanup_old_processes()
    command = command_for_job(job, checkpoint_dir, max_steps=max_steps)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        try:
            completed = subprocess.run(
                command,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            _cleanup_old_processes()
            return BatchResult(job, "failed", time.time() - start, "timeout")
    _cleanup_old_processes()
    duration = time.time() - start
    if completed.returncode != 0:
        return BatchResult(
            job,
            "failed",
            duration,
            f"exit_code={completed.returncode}",
        )
    if not job.output_csv.exists():
        return BatchResult(job, "failed", duration, "output csv missing")
    return BatchResult(job, "success", duration)


def _cleanup_old_processes() -> None:
    for pattern in ("ros_gz_bridge", "parameter_bridge", "ign gazebo"):
        subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)


def write_status_csv(path: Path, rows: list[BatchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "scenario": row.job.scenario,
                    "algorithm": row.job.algorithm,
                    "seed": row.job.seed,
                    "status": row.status,
                    "output_csv": row.job.output_csv,
                    "duration_s": f"{row.duration_s:.3f}",
                    "message": row.message,
                }
            )


def merge_successful_outputs(rows: list[BatchResult], output: Path) -> None:
    paths = [
        row.job.output_csv
        for row in rows
        if row.status in {"success", "skipped_existing"} and row.job.output_csv.exists()
    ]
    if paths:
        merge_result_csv_files(paths, output)


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    if args.plan:
        plan_path = Path(args.plan)
        create_batch_manifest(plan_path, output_dir)
        jobs = filter_jobs(
            read_experiment_plan(plan_path),
            scenarios=args.scenarios,
            algorithm=args.algorithm,
            seeds=args.seeds,
        )
        if args.manifest_only:
            return 0
        results = run_batch_jobs(
            jobs,
            output_dir=output_dir,
            checkpoint_dir=Path(args.checkpoint_dir),
            timeout_sec=args.timeout_sec,
            max_steps=args.max_steps,
            max_workers=args.max_workers,
            force=args.force,
        )
        write_status_csv(output_dir / "batch_status.csv", results)
        merge_output = (
            Path(args.merge_output)
            if args.merge_output else output_dir.parent / "all_metrics.csv"
        )
        merge_successful_outputs(results, merge_output)
        failed = [row for row in results if row.status == "failed"]
        return 1 if failed else 0

    if not args.scenario_dir or not args.algorithm:
        raise SystemExit(
            "--scenario-dir and --algorithm are required when --plan is not provided"
        )

    scenario_dir = Path(args.scenario_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = sorted(scenario_dir.glob("*.yaml"))
    manifest = output_dir / "batch_manifest.txt"
    lines = [f"algorithm={args.algorithm}"]
    for scenario in scenarios:
        line = f"RUN scenario={scenario.name} algorithm={args.algorithm}"
        print(line)
        lines.append(line)
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
