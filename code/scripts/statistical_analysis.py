#!/usr/bin/env python3
"""Compute descriptive seed-paired differences and bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import statistics


METRIC_DIRECTIONS = {
    "success_rate": "higher",
    "collision_count": "lower",
    "average_completion_time": "lower",
    "makespan": "lower",
    "penalized_average_completion_time": "lower",
    "penalized_makespan": "lower",
    "average_waiting_time": "lower",
    "average_path_efficiency": "higher",
    "min_robot_robot_distance": "higher",
    "min_robot_pedestrian_distance": "higher",
    "robot_robot_near_miss_count": "lower",
    "robot_pedestrian_near_miss_count": "lower",
    "intervention_time_ratio": "lower",
    "mean_policy_inference_time_ms": "lower",
    "p95_policy_inference_time_ms": "lower",
    "mean_control_step_time_ms": "lower",
}
LEARNING_ONLY_METRICS = {
    "mean_policy_inference_time_ms",
    "p95_policy_inference_time_ms",
}
CONTEXT_FIELDS = (
    "scenario",
    "sensor_profile",
    "safety_sensor_profile",
    "randomization_profile",
    "variant",
)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _bootstrap_ci(values: list[float], seed: int, draws: int = 10_000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = [
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(draws)
    ]
    return _percentile(means, 0.025), _percentile(means, 0.975)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proposed", default="sensors_mo_gat_mappo")
    parser.add_argument("--expected-paired-units", type=int, default=3)
    args = parser.parse_args()
    if args.expected_paired_units <= 0:
        parser.error("--expected-paired-units must be positive")

    with args.input.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {
        (
            row["algorithm"],
            *(row[field] for field in CONTEXT_FIELDS),
            int(row["analysis_seed"]),
        ): row
        for row in rows
    }
    if len(indexed) != len(rows):
        raise SystemExit(
            "Duplicate seed-summary rows exist for the same method and experiment "
            "context. Re-run aggregation before statistical analysis."
        )
    methods = sorted({row["algorithm"] for row in rows} - {args.proposed})
    contexts = sorted({tuple(row[field] for field in CONTEXT_FIELDS) for row in rows})
    output = []
    for context in contexts:
        context_values = dict(zip(CONTEXT_FIELDS, context))
        proposed_rows = {
            seed: row
            for (method, *row_context, seed), row in indexed.items()
            if method == args.proposed and tuple(row_context) == context
        }
        if not proposed_rows:
            raise SystemExit(
                f"Missing proposed method {args.proposed!r} for context {context}"
            )
        for comparator in methods:
            comparator_rows = {
                seed: row
                for (method, *row_context, seed), row in indexed.items()
                if method == comparator and tuple(row_context) == context
            }
            if not comparator_rows:
                continue
            if set(comparator_rows) != set(proposed_rows):
                raise SystemExit(
                    f"Unpaired analysis seeds for context {context}: "
                    f"{comparator}={sorted(comparator_rows)}, "
                    f"{args.proposed}={sorted(proposed_rows)}"
                )
            paired_seeds = sorted(set(proposed_rows).intersection(comparator_rows))
            if len(paired_seeds) != args.expected_paired_units:
                raise SystemExit(
                    f"Expected {args.expected_paired_units} paired analysis units "
                    f"for context {context} and comparator {comparator}; "
                    f"found {len(paired_seeds)}"
                )
            for metric, direction in METRIC_DIRECTIONS.items():
                if comparator == "orca" and metric in LEARNING_ONLY_METRICS:
                    continue
                differences = [
                    float(proposed_rows[seed][metric])
                    - float(comparator_rows[seed][metric])
                    for seed in paired_seeds
                ]
                if not differences:
                    continue
                low, high = _bootstrap_ci(
                    differences,
                    seed=sum(
                        ord(char)
                        for char in f"{context}:{comparator}:{metric}"
                    ),
                )
                mean_difference = statistics.fmean(differences)
                output.append(
                    {
                        **context_values,
                        "proposed": args.proposed,
                        "comparator": comparator,
                        "metric": metric,
                        "direction": direction,
                        "paired_analysis_units": len(differences),
                        "mean_paired_difference_proposed_minus_comparator": mean_difference,
                        "bootstrap_95_ci_low": low,
                        "bootstrap_95_ci_high": high,
                        "proposed_descriptively_better": int(
                            mean_difference > 0.0
                            if direction == "higher" else mean_difference < 0.0
                        ),
                        "paired_differences": ";".join(
                            f"{value:.8g}" for value in differences
                        ),
                        "inferential_claim": "descriptive_only",
                    }
                )
    if not output:
        raise SystemExit(
            "No paired analysis units were found for the proposed and comparator "
            "methods."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"Wrote {len(output)} descriptive paired comparisons: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
