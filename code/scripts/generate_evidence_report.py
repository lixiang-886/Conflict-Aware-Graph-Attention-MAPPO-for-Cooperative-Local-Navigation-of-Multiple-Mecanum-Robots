#!/usr/bin/env python3
"""Generate machine-traceable facts for the Sensors manuscript narrative."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


PROPOSED = "sensors_mo_gat_mappo"
PROTOCOL_VARIANT = "sensors-matched-v1"
METHOD_LABELS = {
    "orca": "Classical",
    "matched_mappo": "Matched MAPPO",
    "gat_mappo": "GAT-MAPPO",
    PROPOSED: "MO-GAT-MAPPO",
}
SCENARIO_LABELS = {
    "static_clutter": "Static clutter",
    "pedestrian_dynamic": "Dynamic pedestrians",
    "mixed_complex": "Mixed complex",
}
SENSOR_PROFILES = ("nominal", "mild", "moderate", "severe")
ROBUSTNESS_METHODS = ("orca", "matched_mappo", PROPOSED)
MATCHED_LEARNING_METHODS = ("matched_mappo", "gat_mappo", PROPOSED)
STATIC_NOT_APPLICABLE_METRICS = {
    "min_robot_pedestrian_distance",
    "robot_pedestrian_near_miss_count",
    "robot_pedestrian_filter_intervention_ratio",
}
LEARNING_ONLY_METRICS = {
    "mean_policy_inference_time_ms",
    "p95_policy_inference_time_ms",
}
PRIMARY_OUTCOME_METRICS = {
    "success_rate",
    "collision_count",
    "penalized_average_completion_time",
    "penalized_makespan",
    "average_waiting_time",
    "min_robot_robot_distance",
    "min_robot_pedestrian_distance",
    "robot_robot_near_miss_count",
    "robot_pedestrian_near_miss_count",
    "intervention_time_ratio",
}
METRIC_DIRECTIONS = {
    "success_rate": "higher",
    "collision_count": "lower",
    "penalized_average_completion_time": "lower",
    "penalized_makespan": "lower",
    "average_waiting_time": "lower",
    "average_path_length": "lower",
    "average_path_efficiency": "higher",
    "min_robot_robot_distance": "higher",
    "min_robot_pedestrian_distance": "higher",
    "robot_robot_near_miss_count": "lower",
    "robot_pedestrian_near_miss_count": "lower",
    "intervention_time_ratio": "lower",
    "lidar_intervention_ratio": "lower",
    "robot_robot_filter_intervention_ratio": "lower",
    "robot_pedestrian_filter_intervention_ratio": "lower",
    "stale_sensor_stop_ratio": "lower",
    "mean_normalized_intervention_magnitude": "lower",
    "mean_policy_inference_time_ms": "lower",
    "p95_policy_inference_time_ms": "lower",
    "mean_control_step_time_ms": "lower",
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(row: dict[str, str], metric: str, suffix: str) -> float:
    return float(row[f"{metric}_{suffix}"])


def _best_comparator(
    rows: list[dict[str, str]],
    metric: str,
    direction: str,
) -> dict[str, str]:
    comparators = [row for row in rows if row["algorithm"] != PROPOSED]
    if metric in LEARNING_ONLY_METRICS:
        comparators = [row for row in comparators if row["algorithm"] != "orca"]
    key = lambda row: _number(row, metric, "mean")
    return (
        max(comparators, key=key)
        if direction == "higher"
        else min(comparators, key=key)
    )


def _relative_improvement(
    proposed: float,
    comparator: float,
    direction: str,
) -> float | None:
    denominator = abs(comparator)
    if denominator <= 1e-12:
        return None
    signed_gain = (
        proposed - comparator
        if direction == "higher"
        else comparator - proposed
    )
    return 100.0 * signed_gain / denominator


def _scenario_facts(rows: list[dict[str, str]], scenario: str) -> dict[str, object]:
    selected = [row for row in rows if row["scenario"] == scenario]
    if len(selected) != len(METHOD_LABELS):
        raise ValueError(
            f"Expected {len(METHOD_LABELS)} main rows for {scenario}; "
            f"found {len(selected)}"
        )
    for row in selected:
        if (
            row.get("sensor_profile") != "nominal"
            or row.get("safety_sensor_profile") != "nominal"
            or row.get("randomization_profile") != "randomized_v1"
            or row.get("variant") != PROTOCOL_VARIANT
        ):
            raise ValueError(f"Invalid main-comparison context: {row}")
    indexed = {row["algorithm"]: row for row in selected}
    expected = set(METHOD_LABELS)
    if set(indexed) != expected:
        raise ValueError(
            f"Incomplete main grid for {scenario}: "
            f"expected={sorted(expected)}, found={sorted(indexed)}"
        )
    proposed_row = indexed[PROPOSED]
    metrics: dict[str, object] = {}
    wins = 0
    ties = 0
    losses = 0
    for metric, direction in METRIC_DIRECTIONS.items():
        if (
            scenario == "static_clutter"
            and metric in STATIC_NOT_APPLICABLE_METRICS
        ):
            continue
        comparator_row = _best_comparator(selected, metric, direction)
        proposed_value = _number(proposed_row, metric, "mean")
        comparator_value = _number(comparator_row, metric, "mean")
        difference = proposed_value - comparator_value
        favorable_difference = (
            difference if direction == "higher" else -difference
        )
        if abs(favorable_difference) <= 1e-12:
            outcome = "tie"
            if metric in PRIMARY_OUTCOME_METRICS:
                ties += 1
        elif favorable_difference > 0.0:
            outcome = "win"
            if metric in PRIMARY_OUTCOME_METRICS:
                wins += 1
        else:
            outcome = "loss"
            if metric in PRIMARY_OUTCOME_METRICS:
                losses += 1
        metrics[metric] = {
            "direction": direction,
            "proposed_mean": proposed_value,
            "proposed_std": _number(proposed_row, metric, "std"),
            "best_comparator": comparator_row["algorithm"],
            "comparator_mean": comparator_value,
            "comparator_std": _number(comparator_row, metric, "std"),
            "proposed_minus_comparator": difference,
            "relative_improvement_percent": _relative_improvement(
                proposed_value,
                comparator_value,
                direction,
            ),
            "outcome": outcome,
        }
    return {
        "metrics": metrics,
        "win_count": wins,
        "tie_count": ties,
        "loss_count": losses,
        "primary_metric_count": wins + ties + losses,
    }


def _robustness_facts(rows: list[dict[str, str]]) -> dict[str, object]:
    expected = {
        (scenario, method, profile)
        for scenario in SCENARIO_LABELS
        for method in ROBUSTNESS_METHODS
        for profile in SENSOR_PROFILES
    }
    indexed = {
        (row["scenario"], row["algorithm"], row["sensor_profile"]): row
        for row in rows
    }
    if len(indexed) != len(rows):
        raise ValueError("Duplicate policy-only robustness summary rows")
    found = set(indexed)
    if found != expected:
        raise ValueError(
            "Incomplete policy-only robustness grid: "
            f"missing={sorted(expected - found)}, "
            f"unexpected={sorted(found - expected)}"
        )
    for row in rows:
        if (
            row.get("safety_sensor_profile") != "nominal"
            or row.get("randomization_profile") != "randomized_v1"
            or row.get("variant") != PROTOCOL_VARIANT
        ):
            raise ValueError(f"Invalid policy-only robustness context: {row}")
    output: dict[str, object] = {}
    for scenario in SCENARIO_LABELS:
        scenario_output = {}
        for method in ROBUSTNESS_METHODS:
            nominal = indexed.get((scenario, method, "nominal"))
            severe = indexed.get((scenario, method, "severe"))
            if nominal is None or severe is None:
                raise ValueError(
                    f"Missing nominal/severe robustness row for "
                    f"{scenario}/{method}"
                )
            scenario_output[method] = {
                metric: {
                    "nominal_mean": _number(nominal, metric, "mean"),
                    "severe_mean": _number(severe, metric, "mean"),
                    "severe_minus_nominal": (
                        _number(severe, metric, "mean")
                        - _number(nominal, metric, "mean")
                    ),
                }
                for metric in (
                    "success_rate",
                    "collision_count",
                    "penalized_average_completion_time",
                    "min_robot_robot_distance",
                    "min_robot_pedestrian_distance",
                    "intervention_time_ratio",
                    "stale_sensor_stop_ratio",
                )
                if not (
                    metric == "min_robot_pedestrian_distance"
                    and scenario == "static_clutter"
                )
            }
        output[scenario] = scenario_output
    return output


def _joint_stress_facts(rows: list[dict[str, str]]) -> dict[str, object]:
    expected_count = len(SCENARIO_LABELS) * len(ROBUSTNESS_METHODS)
    if len(rows) != expected_count or any(
        row.get("sensor_profile") != "severe"
        or row.get("safety_sensor_profile") != "severe"
        or row.get("randomization_profile") != "randomized_v1"
        or row.get("variant") != PROTOCOL_VARIANT
        for row in rows
    ):
        raise ValueError(
            "Joint-stress summary must contain exactly the severe paired grid"
        )
    output = {}
    for row in rows:
        if (
            row.get("sensor_profile") != "severe"
            or row.get("safety_sensor_profile") != "severe"
        ):
            continue
        key = f"{row['scenario']}/{row['algorithm']}"
        if key in output:
            raise ValueError(f"Duplicate joint-stress summary row: {key}")
        output[key] = {
            metric: {
                "mean": _number(row, metric, "mean"),
                "std": _number(row, metric, "std"),
            }
            for metric in (
                "success_rate",
                "collision_count",
                "penalized_average_completion_time",
                "intervention_time_ratio",
                "stale_sensor_stop_ratio",
            )
        }
    expected_keys = {
        f"{scenario}/{method}"
        for scenario in SCENARIO_LABELS
        for method in ROBUSTNESS_METHODS
    }
    if set(output) != expected_keys:
        raise ValueError(
            "Incomplete severe joint-stress grid: "
            f"missing={sorted(expected_keys - set(output))}, "
            f"unexpected={sorted(set(output) - expected_keys)}"
        )
    return output


def _configuration_audit(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    indexed = {row.get("method", ""): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(MATCHED_LEARNING_METHODS):
        raise ValueError(
            "Configuration audit must contain exactly the three matched "
            "learning methods"
        )
    for method, row in indexed.items():
        if (
            row.get("reward") != "adaptive"
            or row.get("critic") != "centralized"
            or row.get("training_mode") != "fast"
            or row.get("training_episodes") != "200"
            or row.get("checkpoint_selection")
            != "deterministic_validation_lexicographic"
        ):
            raise ValueError(f"Invalid matched configuration row: {method}")
    return indexed


def _markdown(report: dict[str, object]) -> str:
    lines = [
        "# Sensors Evidence Report",
        "",
        "This file is generated from processed CSVs. It is a narrative-writing aid,",
        "not an additional statistical analysis.",
        "",
        "## Main Matched Comparison",
        "",
    ]
    main = report["main"]
    assert isinstance(main, dict)
    for scenario, label in SCENARIO_LABELS.items():
        facts = main[scenario]
        assert isinstance(facts, dict)
        lines.append(
            f"- {label}: {facts['win_count']} wins, {facts['tie_count']} ties, "
            f"{facts['loss_count']} losses against the best comparator for each "
            "pre-registered primary outcome metric."
        )
        metrics = facts["metrics"]
        assert isinstance(metrics, dict)
        for metric in (
            "success_rate",
            "penalized_average_completion_time",
            "penalized_makespan",
            "min_robot_robot_distance",
            "intervention_time_ratio",
        ):
            values = metrics[metric]
            assert isinstance(values, dict)
            improvement = values["relative_improvement_percent"]
            improvement_text = (
                "undefined from zero comparator"
                if improvement is None
                else f"{float(improvement):+.2f}%"
            )
            lines.append(
                f"  - {metric}: proposed={float(values['proposed_mean']):.6g}, "
                f"best comparator={values['best_comparator']} "
                f"({float(values['comparator_mean']):.6g}), "
                f"favorable relative change={improvement_text}, "
                f"outcome={values['outcome']}."
            )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Analysis units are training seeds or paired Classical evaluation blocks.",
            "- Bootstrap intervals are descriptive with three units; do not claim significance.",
            "- Policy-only perturbation keeps safety inputs nominal.",
            "- Joint severe stress perturbs policy and safety inputs and is reported separately.",
            "- Zero collisions under the shared safety layer are not a formal guarantee.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-summary", type=Path, required=True)
    parser.add_argument("--robustness-summary", type=Path, required=True)
    parser.add_argument("--joint-stress-summary", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--config-audit", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    main_rows = _read(args.main_summary)
    robustness_rows = _read(args.robustness_summary)
    joint_rows = _read(args.joint_stress_summary)
    statistics_rows = _read(args.statistics)
    config_audit_rows = _read(args.config_audit)
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if (
        source_manifest.get("schema") != "sensors-source-provenance-v1"
        or not isinstance(source_manifest.get("source_tree_sha256"), str)
        or len(source_manifest["source_tree_sha256"]) != 64
    ):
        raise ValueError("Invalid Sensors source-provenance manifest")
    if not statistics_rows or any(
        row.get("inferential_claim") != "descriptive_only"
        for row in statistics_rows
    ):
        raise ValueError(
            "Paired-statistics input must be non-empty and descriptive-only"
        )
    report = {
        "schema": "sensors-evidence-report-v1",
        "sources": {
            str(path): _sha256(path)
            for path in (
                args.main_summary,
                args.robustness_summary,
                args.joint_stress_summary,
                args.statistics,
                args.config_audit,
                args.source_manifest,
            )
        },
        "source_provenance": {
            "source_tree_sha256": source_manifest["source_tree_sha256"],
            "git_head": source_manifest.get("git_head"),
            "git_branch": source_manifest.get("git_branch"),
        },
        "main": {
            scenario: _scenario_facts(main_rows, scenario)
            for scenario in SCENARIO_LABELS
        },
        "robustness_nominal_to_severe": _robustness_facts(robustness_rows),
        "joint_severe_policy_and_safety": _joint_stress_facts(joint_rows),
        "configuration_audit": _configuration_audit(config_audit_rows),
        "paired_descriptive_statistics": statistics_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(_markdown(report), encoding="utf-8")
    print(f"Generated evidence report: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
