#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


SCENARIO = "mixed_complex"
EXPECTED_VARIANTS = {
    "full_mo_gat_mappo": ("full", "mo_gat_mappo"),
    "ablation_no_graph_attention": ("no_gat", "ablation_no_graph_attention"),
    "ablation_fixed_reward": ("fixed_reward", "ablation_fixed_reward"),
    "ablation_no_gate": ("no_gate", "ablation_no_gate"),
    "ablation_no_coordinator": ("no_coordinator", "ablation_no_coordinator"),
}
CRITICAL_METRICS = (
    "success_rate",
    "collision_count",
    "average_completion_time",
    "makespan",
    "average_waiting_time",
    "average_path_efficiency",
    "average_turning_angle_per_meter",
    "min_robot_robot_distance",
    "robot_robot_near_miss_count",
    "robot_pedestrian_near_miss_count",
    "intervention_time_ratio",
)
SOURCE_FILES = (
    "src/mrpp_rl/mrpp_rl/train.py",
    "src/mrpp_rl/mrpp_rl/evaluate.py",
    "src/mrpp_rl/mrpp_rl/algorithm_config.py",
    "src/mrpp_rl/mrpp_rl/control.py",
    "src/mrpp_rl/mrpp_rl/mappo.py",
    "src/mrpp_rl/mrpp_rl/reward.py",
    "src/mrpp_rl/mrpp_rl/adaptive_weights.py",
    "src/mrpp_rl/mrpp_rl/waypoints.py",
    "src/mrpp_experiments/mrpp_experiments/metrics.py",
    "src/mrpp_experiments/mrpp_experiments/results.py",
    "src/mrpp_experiments/config/scenarios/mixed_complex.yaml",
    "src/mrpp_gazebo/worlds/mixed_complex.sdf",
    "src/mrpp_gazebo/models/mecanum_robot/model.sdf",
    "scripts/run_formal_ablations.sh",
    "scripts/export_formal_ablation_table.py",
    "scripts/finalize_focused_ablation.py",
)
def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str], *, binary: bool = False) -> str | bytes:
    return subprocess.check_output(command, text=not binary)


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def as_float(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric field {key!r}: {row.get(key)!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite numeric field {key!r}: {value}")
    return value


def expected_paths(
    run_root: Path,
    checkpoint_root: Path,
    training_root: Path,
    variant: str,
    seed: int,
) -> dict[str, Path]:
    token, algorithm = EXPECTED_VARIANTS[variant]
    run_name = f"formal_ablation_{token}_seed{seed}"
    checkpoint_dir = checkpoint_root / run_name / SCENARIO / algorithm
    training_dir = training_root / run_name / SCENARIO / algorithm
    evaluation_dir = run_root / "runs" / run_name / SCENARIO
    return {
        "checkpoint": checkpoint_dir / f"{algorithm}_best.pt",
        "training_config": checkpoint_dir / "config_snapshot.json",
        "environment": checkpoint_dir / "environment.json",
        "training_log": training_dir / "training_log.csv",
        "evaluation": evaluation_dir / f"eval_seed{seed}.csv",
        "evaluation_config": evaluation_dir / f"eval_seed{seed}_config.json",
        "trajectory": evaluation_dir / f"eval_seed{seed}_trajectories.csv",
        "evaluation_log": evaluation_dir / f"eval_seed{seed}.log",
    }


def validate_training_config(
    config: dict[str, object],
    variant: str,
    algorithm: str,
    seed: int,
    episodes: int,
) -> list[str]:
    errors: list[str] = []
    expected = {
        "scenario": SCENARIO,
        "variant": variant,
        "seed": seed,
        "episodes": episodes,
        "dt": 0.1,
        "device": "cuda",
        "sim_step_mode": "realtime",
        "resume_checkpoint": None,
        "validation_interval": 5,
        "validation_rollback": False,
        "initial_validation_diagnostic_only": True,
        "orca_prior_enabled": True,
        "safety_filter_enabled": True,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            errors.append(
                f"training config {variant} seed={seed}: {key}="
                f"{config.get(key)!r}, expected {value!r}"
            )
    algorithm_config = config.get("algorithm")
    if not isinstance(algorithm_config, dict) or algorithm_config.get("name") != algorithm:
        errors.append(
            f"training config {variant} seed={seed}: algorithm mismatch"
        )
    return errors


def validate_evaluation_config(
    config: dict[str, object],
    variant: str,
    algorithm: str,
    seed: int,
    checkpoint: Path,
) -> list[str]:
    errors: list[str] = []
    expected = {
        "scenario": SCENARIO,
        "variant": variant,
        "algorithm": algorithm,
        "seed": seed,
        "dt": 0.1,
        "device": "cuda",
        "sim_step_mode": "realtime",
        "orca_prior_enabled": True,
        "safety_filter_enabled": True,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            errors.append(
                f"evaluation config {variant} seed={seed}: {key}="
                f"{config.get(key)!r}, expected {value!r}"
            )
    if Path(str(config.get("checkpoint", ""))) != checkpoint:
        errors.append(
            f"evaluation config {variant} seed={seed}: checkpoint path mismatch"
        )
    return errors


def build_report_table(rows: list[dict[str, str]]) -> list[str]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    lines = [
        "| Variant | Seeds | Success mean | Collisions | RR near | RP near |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for variant in EXPECTED_VARIANTS:
        values = sorted(grouped[variant], key=lambda row: int(row["seed"]))
        seeds = ", ".join(row["seed"] for row in values)
        success = sum(as_float(row, "success_rate") for row in values) / len(values)
        collisions = sum(as_float(row, "collision_count") for row in values)
        rr_near = sum(as_float(row, "robot_robot_near_miss_count") for row in values)
        rp_near = sum(as_float(row, "robot_pedestrian_near_miss_count") for row in values)
        lines.append(
            f"| `{variant}` | {seeds} | {success:.3f} | "
            f"{collisions:.0f} | {rr_near:.0f} | {rp_near:.0f} |"
        )
    return lines


def build_hash_table(
    title: str,
    hashes: dict[str, str],
    *,
    relative_to: Path | None = None,
) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Artifact | SHA-256 |",
        "| --- | --- |",
    ]
    for artifact, digest in sorted(hashes.items()):
        display_path = Path(artifact)
        if relative_to is not None:
            try:
                display_path = display_path.relative_to(relative_to)
            except ValueError:
                pass
        lines.append(f"| `{display_path}` | `{digest}` |")
    return lines


def validate_and_finalize(
    run_root: Path,
    checkpoint_root: Path,
    training_root: Path,
    seeds: tuple[int, ...],
    episodes: int,
) -> None:
    metrics_path = run_root / "all_metrics.csv"
    rows = read_csv(metrics_path)
    errors: list[str] = []
    warnings: list[str] = []
    expected_keys = {
        (variant, seed)
        for variant in EXPECTED_VARIANTS
        for seed in seeds
    }
    observed_keys: set[tuple[str, int]] = set()
    metric_fingerprints: dict[str, set[tuple[float, ...]]] = defaultdict(set)

    for row in rows:
        variant = row.get("variant", "")
        try:
            seed = int(row.get("seed", ""))
        except ValueError:
            errors.append(f"invalid seed in row: {row.get('seed')!r}")
            continue
        key = (variant, seed)
        if key in observed_keys:
            errors.append(f"duplicate result row: {variant} seed={seed}")
        observed_keys.add(key)
        if variant not in EXPECTED_VARIANTS:
            errors.append(f"unexpected variant: {variant!r}")
            continue
        _token, algorithm = EXPECTED_VARIANTS[variant]
        if row.get("scenario") != SCENARIO:
            errors.append(f"{variant} seed={seed}: unexpected scenario")
        if row.get("algorithm") != algorithm:
            errors.append(f"{variant} seed={seed}: algorithm mismatch")
        if row.get("data_source") != "gazebo":
            errors.append(f"{variant} seed={seed}: data_source is not gazebo")
        if not as_bool(row.get("orca_prior_enabled")):
            errors.append(f"{variant} seed={seed}: ORCA prior disabled")
        if not as_bool(row.get("safety_filter_enabled")):
            errors.append(f"{variant} seed={seed}: safety filter disabled")
        try:
            fingerprint = tuple(as_float(row, metric) for metric in CRITICAL_METRICS)
        except ValueError as exc:
            errors.append(f"{variant} seed={seed}: {exc}")
        else:
            metric_fingerprints[variant].add(fingerprint)

    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        errors.append(f"matrix mismatch: missing={missing}, extra={extra}")
    if len(rows) != len(expected_keys):
        errors.append(f"row count {len(rows)} does not match {len(expected_keys)}")
    for variant, fingerprints in metric_fingerprints.items():
        if len(fingerprints) == 1:
            errors.append(f"all seed-level critical metrics are identical for {variant}")

    required_outputs = (
        metrics_path,
        run_root / "metric_summary.csv",
        run_root / "statistical_tests.csv",
        run_root / "paper_assets/focused_ablation_table.tex",
        run_root / "paper_assets/focused_ablation_table.md",
    )
    for path in required_outputs:
        if not path.is_file():
            errors.append(f"missing required output: {path}")

    checkpoints: dict[str, str] = {}
    configs: dict[str, str] = {}
    trajectories: dict[str, str] = {}
    checkpoint_hashes: set[str] = set()
    trajectory_hashes: set[str] = set()
    for variant, (_token, algorithm) in EXPECTED_VARIANTS.items():
        for seed in seeds:
            paths = expected_paths(
                run_root,
                checkpoint_root,
                training_root,
                variant,
                seed,
            )
            for label, path in paths.items():
                if not path.is_file():
                    errors.append(f"missing {label}: {path}")
            if any(not path.is_file() for path in paths.values()):
                continue
            training_config = json.loads(
                paths["training_config"].read_text(encoding="utf-8")
            )
            evaluation_config = json.loads(
                paths["evaluation_config"].read_text(encoding="utf-8")
            )
            errors.extend(
                validate_training_config(
                    training_config,
                    variant,
                    algorithm,
                    seed,
                    episodes,
                )
            )
            errors.extend(
                validate_evaluation_config(
                    evaluation_config,
                    variant,
                    algorithm,
                    seed,
                    paths["checkpoint"],
                )
            )
            training_rows = read_csv(paths["training_log"])
            if len(training_rows) != episodes:
                errors.append(
                    f"training log {variant} seed={seed}: "
                    f"{len(training_rows)} rows, expected {episodes}"
                )
            checkpoint_hash = sha256(paths["checkpoint"])
            trajectory_hash = sha256(paths["trajectory"])
            checkpoints[str(paths["checkpoint"])] = checkpoint_hash
            configs[str(paths["training_config"])] = sha256(paths["training_config"])
            configs[str(paths["evaluation_config"])] = sha256(
                paths["evaluation_config"]
            )
            trajectories[str(paths["trajectory"])] = trajectory_hash
            checkpoint_hashes.add(checkpoint_hash)
            trajectory_hashes.add(trajectory_hash)

    if len(checkpoint_hashes) != len(expected_keys):
        errors.append("checkpoint hashes are not unique across all variant/seed jobs")
    if len(trajectory_hashes) != len(expected_keys):
        errors.append("trajectory hashes are not unique across all variant/seed jobs")

    if errors:
        message = "Focused-ablation validation failed:\n- " + "\n- ".join(errors)
        raise SystemExit(message)

    source_hashes = {
        path: sha256(Path(path))
        for path in SOURCE_FILES
        if Path(path).is_file()
    }
    result_hashes = {
        str(path.relative_to(run_root)): sha256(path)
        for path in required_outputs
    }
    git_status = command_output(["git", "status", "--porcelain=v1"])
    assert isinstance(git_status, str)
    git_status_lines = git_status.splitlines()
    provenance_outputs = (
        ":(glob,exclude)results/eval/**/protocol_manifest.json",
        ":(glob,exclude)results/eval/**/VALIDATION.md",
    )
    tracked_diff_command = ["git", "diff", "--binary", "HEAD", "--", "."]
    tracked_diff_command.extend(provenance_outputs)
    tracked_diff = command_output(tracked_diff_command, binary=True)
    assert isinstance(tracked_diff, bytes)
    commit = command_output(["git", "rev-parse", "HEAD"])
    assert isinstance(commit, str)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit.strip(),
        "git_worktree_dirty": bool(git_status_lines),
        "git_status": git_status_lines,
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "tracked_diff_excludes": list(provenance_outputs),
        "protocol": {
            "scenario": SCENARIO,
            "variants": list(EXPECTED_VARIANTS),
            "seeds": list(seeds),
            "training_episodes": episodes,
            "validation_interval": 5,
            "validation_rollback": False,
            "initial_validation_diagnostic_only": True,
            "device": "cuda",
            "sim_step_mode": "realtime",
            "dt_seconds": 0.1,
            "shared_orca_prior": True,
            "shared_safety_filter": True,
            "data_source": "gazebo",
        },
        "matrix_coverage": {
            "expected": len(expected_keys),
            "observed": len(rows),
            "complete": True,
        },
        "result_sha256": result_hashes,
        "source_sha256": source_hashes,
        "checkpoint_sha256": checkpoints,
        "config_sha256": configs,
        "trajectory_sha256": trajectories,
    }
    (run_root / "protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = [
        "# Focused Ablation Validation",
        "",
        "Status: **PASS**",
        "",
        "## Scope",
        "",
        f"- Scenario: `{SCENARIO}`",
        f"- Variants: {', '.join(f'`{name}`' for name in EXPECTED_VARIANTS)}",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- Matrix coverage: {len(rows)}/{len(expected_keys)}",
        f"- Training budget: {episodes} episodes per independently initialized job",
        "- Deterministic evaluation: real-time Gazebo, CUDA inference, dt = 0.1 s",
        "- Shared layers: A* waypoint guide, ORCA-style prior, and final safety filter",
        "- Validation rollback: disabled",
        "- Initial validation: diagnostic only; excluded from best-checkpoint selection",
        "",
        "## Safety Outcomes",
        "",
        *build_report_table(rows),
        "",
        "Safety outcomes are reported as observations, not formal guarantees.",
        "",
        "## Integrity Checks",
        "",
        "- All 15 scenario/variant/seed rows are present exactly once.",
        "- Every result row is marked `data_source=gazebo`.",
        "- Every training and evaluation config uses CUDA, real-time stepping,",
        "  the shared ORCA prior, and the shared final safety filter.",
        f"- All {len(checkpoint_hashes)} checkpoint hashes are unique.",
        f"- All {len(trajectory_hashes)} trajectory hashes are unique.",
        f"- Git commit: `{manifest['git_commit']}`",
        f"- Dirty worktree recorded: `{str(bool(git_status_lines)).lower()}`",
        f"- Tracked source/artifact diff SHA-256: `{manifest['tracked_diff_sha256']}`",
        "- Generated protocol manifests and validation reports under `results/eval`",
        "  are excluded from that diff hash to avoid cross-reference.",
        "",
        "## Provenance Hashes",
        "",
        *build_hash_table("Result Artifacts", result_hashes),
        "",
        *build_hash_table("Critical Sources", source_hashes),
        "",
        *build_hash_table(
            "Best Checkpoints",
            checkpoints,
            relative_to=checkpoint_root,
        ),
        "",
        "Evaluation-config and trajectory hashes are recorded in",
        "`protocol_manifest.json`.",
        "",
        "### Recorded Worktree Status",
        "",
        "```text",
        *(git_status_lines or ["(clean)"]),
        "```",
        "",
        "The v62 directory is focused component-ablation evidence only. It does",
        "not replace the v60 four-method comparison or the v63 supplementary audit.",
    ]
    if warnings:
        report.extend(["", "## Warnings", "", *[f"- {item}" for item in warnings]])
    (run_root / "VALIDATION.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and freeze the formal focused-ablation evidence."
    )
    parser.add_argument(
        "--run-root",
        default="results/eval/formal_v62_focused_ablation_20260711",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="models/checkpoints/formal_v62_focused_ablation_20260711",
    )
    parser.add_argument(
        "--training-root",
        default="results/training/formal_v62_focused_ablation_20260711",
    )
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--episodes", type=int, default=30)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    seeds = tuple(args.seeds) if args.seeds else (0, 1, 2)
    validate_and_finalize(
        Path(args.run_root),
        Path(args.checkpoint_root),
        Path(args.training_root),
        seeds,
        args.episodes,
    )
    print(f"Focused ablation validated at {args.run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
