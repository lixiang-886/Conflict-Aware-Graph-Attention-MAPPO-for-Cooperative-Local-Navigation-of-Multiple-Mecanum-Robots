#!/usr/bin/env python3
"""Generate traceable LaTeX tables from Sensors aggregate CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METHOD_LABELS = {
    "orca": "Classical local controller",
    "matched_mappo": "Matched MAPPO",
    "gat_mappo": "GAT-MAPPO",
    "sensors_mo_gat_mappo": "MO-GAT-MAPPO",
}
SCENARIO_LABELS = {
    "static_clutter": "Static clutter",
    "pedestrian_dynamic": "Dynamic pedestrians",
    "mixed_complex": "Mixed complex",
}
MAIN_METRICS = (
    ("success_rate", "Success (\\%)", "higher", 1),
    ("penalized_average_completion_time", "Pen. avg. time", "lower", 2),
    ("penalized_makespan", "Pen. makespan", "lower", 2),
    ("average_waiting_time", "Waiting", "lower", 2),
    ("min_robot_robot_distance", "Min RR", "higher", 2),
    ("min_robot_pedestrian_distance", "Min RP", "higher", 2),
    ("intervention_time_ratio", "Int. ratio", "lower", 3),
)
METHOD_ORDER = tuple(METHOD_LABELS)
PROFILE_LABELS = {
    "nominal": "Nominal",
    "mild": "Mild",
    "moderate": "Moderate",
    "severe": "Severe",
}
ROBUSTNESS_METHODS = ("orca", "matched_mappo", "sensors_mo_gat_mappo")
SAFETY_OUTCOME_METRICS = (
    ("collision_count", "Collision", "lower", 2),
    ("robot_robot_near_miss_count", "RR near", "lower", 2),
    ("robot_pedestrian_near_miss_count", "RP near", "lower", 2),
    ("min_robot_robot_distance", "Min RR", "higher", 3),
    ("min_robot_pedestrian_distance", "Min RP", "higher", 3),
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _format(mean: float, std: float, decimals: int, bold: bool) -> str:
    value = f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"
    return f"\\textbf{{{value}}}" if bold else value


def _best(rows: list[dict[str, str]], metric: str, direction: str) -> float:
    values = [float(row[f"{metric}_mean"]) for row in rows]
    return max(values) if direction == "higher" else min(values)


def _require_grid(
    rows: list[dict[str, str]],
    *,
    methods: tuple[str, ...],
    profiles: tuple[str, ...] | None = None,
) -> None:
    profile_values = profiles if profiles is not None else (None,)
    for scenario in SCENARIO_LABELS:
        for profile in profile_values:
            for method in methods:
                matches = [
                    row
                    for row in rows
                    if row.get("scenario") == scenario
                    and row.get("algorithm") == method
                    and (
                        profile is None
                        or row.get("sensor_profile") == profile
                    )
                ]
                context = (scenario, profile, method)
                if len(matches) != 1:
                    raise ValueError(
                        f"Expected exactly one summary row for {context}; "
                        f"found {len(matches)}"
                    )


def main_table(rows: list[dict[str, str]]) -> str:
    methods = list(METHOD_ORDER)
    _require_grid(rows, methods=METHOD_ORDER)
    lines = [
        "\\begin{table*}[t]",
        "\\caption{Main matched comparison.}",
        "\\label{tab:sensors-main-comparison}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{ll" + "r" * len(MAIN_METRICS) + "}",
        "\\toprule",
        "Scenario & Method & "
        + " & ".join(label for _metric, label, _direction, _decimals in MAIN_METRICS)
        + " \\\\",
        "\\midrule",
    ]
    for scenario in SCENARIO_LABELS:
        scenario_rows = [
            row
            for row in rows
            if row["scenario"] == scenario and row["algorithm"] in methods
        ]
        indexed = {row["algorithm"]: row for row in scenario_rows}
        best = {
            metric: _best(scenario_rows, metric, direction)
            for metric, _label, direction, _decimals in MAIN_METRICS
        }
        discriminative = {
            metric: (
                max(float(row[f"{metric}_mean"]) for row in scenario_rows)
                - min(float(row[f"{metric}_mean"]) for row in scenario_rows)
                > 1e-12
            )
            for metric, _label, _direction, _decimals in MAIN_METRICS
        }
        for index, method in enumerate(methods):
            if method not in indexed:
                continue
            row = indexed[method]
            cells = []
            for metric, _label, _direction, decimals in MAIN_METRICS:
                if metric == "min_robot_pedestrian_distance" and scenario == "static_clutter":
                    cells.append("--")
                    continue
                mean = float(row[f"{metric}_mean"])
                std = float(row[f"{metric}_std"])
                display_mean = mean * 100.0 if metric == "success_rate" else mean
                display_std = std * 100.0 if metric == "success_rate" else std
                cells.append(
                    _format(
                        display_mean,
                        display_std,
                        decimals,
                        discriminative[metric]
                        and abs(mean - best[metric]) <= 1e-12,
                    )
                )
            scenario_cell = SCENARIO_LABELS[scenario] if index == 0 else ""
            lines.append(
                f"{scenario_cell} & {METHOD_LABELS[method]} & "
                + " & ".join(cells)
                + " \\\\"
            )
        lines.append("\\midrule" if scenario != "mixed_complex" else "\\bottomrule")
    lines.extend(["\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def safety_table(rows: list[dict[str, str]]) -> str:
    metrics = (
        ("intervention_time_ratio", "Total"),
        ("lidar_intervention_ratio", "LiDAR"),
        ("robot_robot_filter_intervention_ratio", "RR"),
        ("robot_pedestrian_filter_intervention_ratio", "RP"),
        ("stale_sensor_stop_ratio", "Stale stop"),
        ("mean_normalized_intervention_magnitude", "Mean change"),
        ("max_normalized_intervention_magnitude", "Max change"),
    )
    _require_grid(rows, methods=METHOD_ORDER)
    lines = [
        "\\begin{table*}[t]",
        "\\caption{Safety-filter intervention by module.}",
        "\\label{tab:sensors-safety-modules}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{ll" + "r" * len(metrics) + "}",
        "\\toprule",
        "Scenario & Method & " + " & ".join(label for _metric, label in metrics) + " \\\\",
        "\\midrule",
    ]
    indexed = {(row["scenario"], row["algorithm"]): row for row in rows}
    for scenario in SCENARIO_LABELS:
        for method in METHOD_ORDER:
            row = indexed.get((scenario, method))
            if row is None:
                continue
            values = []
            for metric, _label in metrics:
                if (
                    scenario == "static_clutter"
                    and metric == "robot_pedestrian_filter_intervention_ratio"
                ):
                    values.append("--")
                    continue
                values.append(
                    _format(
                        float(row[f"{metric}_mean"]),
                        float(row[f"{metric}_std"]),
                        3,
                        False,
                    )
                )
            lines.append(
                f"{SCENARIO_LABELS[scenario]} & "
                f"{METHOD_LABELS[method]} & "
                + " & ".join(values)
                + " \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def safety_outcome_table(rows: list[dict[str, str]]) -> str:
    _require_grid(rows, methods=METHOD_ORDER)
    indexed = {(row["scenario"], row["algorithm"]): row for row in rows}
    lines = [
        "\\begin{table*}[t]",
        "\\caption{Safety outcomes.}",
        "\\label{tab:sensors-safety-outcomes}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{ll" + "r" * len(SAFETY_OUTCOME_METRICS) + "}",
        "\\toprule",
        "Scenario & Method & "
        + " & ".join(
            label
            for _metric, label, _direction, _decimals in SAFETY_OUTCOME_METRICS
        )
        + " \\\\",
        "\\midrule",
    ]
    for scenario in SCENARIO_LABELS:
        scenario_rows = [indexed[(scenario, method)] for method in METHOD_ORDER]
        best = {
            metric: _best(scenario_rows, metric, direction)
            for metric, _label, direction, _decimals in SAFETY_OUTCOME_METRICS
        }
        discriminative = {
            metric: (
                max(float(row[f"{metric}_mean"]) for row in scenario_rows)
                - min(float(row[f"{metric}_mean"]) for row in scenario_rows)
                > 1e-12
            )
            for metric, _label, _direction, _decimals in SAFETY_OUTCOME_METRICS
        }
        for method_index, method in enumerate(METHOD_ORDER):
            row = indexed[(scenario, method)]
            values = []
            for metric, _label, _direction, decimals in SAFETY_OUTCOME_METRICS:
                if scenario == "static_clutter" and metric in {
                    "robot_pedestrian_near_miss_count",
                    "min_robot_pedestrian_distance",
                }:
                    values.append("--")
                    continue
                mean = float(row[f"{metric}_mean"])
                values.append(
                    _format(
                        mean,
                        float(row[f"{metric}_std"]),
                        decimals,
                        discriminative[metric]
                        and abs(mean - best[metric]) <= 1e-12,
                    )
                )
            scenario_cell = (
                SCENARIO_LABELS[scenario] if method_index == 0 else ""
            )
            lines.append(
                f"{scenario_cell} & {METHOD_LABELS[method]} & "
                + " & ".join(values)
                + " \\\\"
            )
        lines.append(
            "\\midrule" if scenario != "mixed_complex" else "\\bottomrule"
        )
    lines.extend(["\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def robustness_table(rows: list[dict[str, str]]) -> str:
    metrics = (
        ("success_rate", "Success (\\%)", 1),
        ("collision_count", "Collision", 2),
        ("penalized_average_completion_time", "Pen. avg. time", 2),
        ("min_robot_robot_distance", "Min RR", 2),
        ("min_robot_pedestrian_distance", "Min RP", 2),
        ("intervention_time_ratio", "Intervention", 3),
        ("stale_sensor_stop_ratio", "Stale stop", 3),
    )
    _require_grid(
        rows,
        methods=ROBUSTNESS_METHODS,
        profiles=tuple(PROFILE_LABELS),
    )
    indexed = {
        (row["scenario"], row["sensor_profile"], row["algorithm"]): row
        for row in rows
    }
    lines = [
        "\\begin{table*}[t]",
        "\\caption{Performance under sensor perturbations.}",
        "\\label{tab:sensors-robustness}",
        "\\centering",
        "\\scriptsize",
        "\\begin{tabular}{lll" + "r" * len(metrics) + "}",
        "\\toprule",
        "Scenario & Perturbation & Method & "
        + " & ".join(label for _metric, label, _decimals in metrics)
        + " \\\\ ",
        "\\midrule",
    ]
    for scenario in SCENARIO_LABELS:
        for profile in PROFILE_LABELS:
            for method in ROBUSTNESS_METHODS:
                row = indexed.get((scenario, profile, method))
                if row is None:
                    continue
                values = []
                for metric, _label, decimals in metrics:
                    if metric == "min_robot_pedestrian_distance" and scenario == "static_clutter":
                        values.append("--")
                    else:
                        values.append(
                            _format(
                                float(row[f"{metric}_mean"])
                                * (100.0 if metric == "success_rate" else 1.0),
                                float(row[f"{metric}_std"])
                                * (100.0 if metric == "success_rate" else 1.0),
                                decimals,
                                False,
                            )
                        )
                lines.append(
                    f"{SCENARIO_LABELS[scenario]} & {PROFILE_LABELS[profile]} & "
                    f"{METHOD_LABELS[method]} & "
                    + " & ".join(values)
                    + " \\\\"
                )
        if scenario != "mixed_complex":
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def safety_input_stress_table(rows: list[dict[str, str]]) -> str:
    metrics = (
        ("success_rate", "Success (\\%)", 1),
        ("collision_count", "Collision", 2),
        ("penalized_average_completion_time", "Pen. avg. time", 2),
        ("min_robot_robot_distance", "Min RR", 2),
        ("min_robot_pedestrian_distance", "Min RP", 2),
        ("intervention_time_ratio", "Intervention", 3),
        ("stale_sensor_stop_ratio", "Stale stop", 3),
    )
    indexed = {}
    for row in rows:
        if (
            row.get("sensor_profile") == "severe"
            and row.get("safety_sensor_profile") == "severe"
        ):
            key = (row["scenario"], row["algorithm"])
            if key in indexed:
                raise ValueError(f"Duplicate severe policy-and-safety row: {key}")
            indexed[key] = row
    expected = {
        (scenario, method)
        for scenario in SCENARIO_LABELS
        for method in ROBUSTNESS_METHODS
    }
    missing = sorted(expected - set(indexed))
    if missing:
        raise ValueError(f"Missing severe policy-and-safety rows: {missing}")

    lines = [
        "\\begin{table*}[t]",
        "\\caption{Severe joint policy-and-safety sensor stress test.}",
        "\\label{tab:sensors-joint-stress}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{ll" + "r" * len(metrics) + "}",
        "\\toprule",
        "Scenario & Method & "
        + " & ".join(label for _metric, label, _decimals in metrics)
        + " \\\\",
        "\\midrule",
    ]
    for scenario in SCENARIO_LABELS:
        for method in ROBUSTNESS_METHODS:
            row = indexed[(scenario, method)]
            values = []
            for metric, _label, decimals in metrics:
                if metric == "min_robot_pedestrian_distance" and scenario == "static_clutter":
                    values.append("--")
                else:
                    values.append(
                        _format(
                            float(row[f"{metric}_mean"])
                            * (100.0 if metric == "success_rate" else 1.0),
                            float(row[f"{metric}_std"])
                            * (100.0 if metric == "success_rate" else 1.0),
                            decimals,
                            False,
                        )
                    )
            lines.append(
                f"{SCENARIO_LABELS[scenario]} & {METHOD_LABELS[method]} & "
                + " & ".join(values)
                + " \\\\"
            )
        if scenario != "mixed_complex":
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def computation_table(rows: list[dict[str, str]]) -> str:
    """Summarize online policy and complete control-step timing."""
    _require_grid(rows, methods=METHOD_ORDER)
    indexed = {(row["scenario"], row["algorithm"]): row for row in rows}
    lines = [
        "\\begin{table}[t]",
        "\\caption{Online computation.}",
        "\\label{tab:sensors-computation}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Scenario & Method & Mean policy (ms) & P95 policy (ms) & Mean loop (ms) \\\\",
        "\\midrule",
    ]
    for scenario in SCENARIO_LABELS:
        for method in METHOD_ORDER:
            row = indexed[(scenario, method)]
            if method == "orca":
                mean_policy = "--"
                p95_policy = "--"
            else:
                mean_policy = _format(
                    float(row["mean_policy_inference_time_ms_mean"]),
                    float(row["mean_policy_inference_time_ms_std"]),
                    3,
                    False,
                )
                p95_policy = _format(
                    float(row["p95_policy_inference_time_ms_mean"]),
                    float(row["p95_policy_inference_time_ms_std"]),
                    3,
                    False,
                )
            mean_loop = _format(
                float(row["mean_control_step_time_ms_mean"]),
                float(row["mean_control_step_time_ms_std"]),
                3,
                False,
            )
            lines.append(
                f"{SCENARIO_LABELS[scenario]} & {METHOD_LABELS[method]} & "
                f"{mean_policy} & {p95_policy} & {mean_loop} \\\\"
            )
        if scenario != "mixed_complex":
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def configuration_table(rows: list[dict[str, str]]) -> str:
    indexed = {row["method"]: row for row in rows}
    missing = [method for method in METHOD_ORDER[1:] if method not in indexed]
    if missing:
        raise ValueError(
            "Missing matched configuration audit rows: " + ", ".join(missing)
        )
    lines = [
        "\\begin{table*}[t]",
        "\\caption{Configuration audit for the matched learning methods.}",
        "\\label{tab:sensors-config-audit}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lccccccr}",
        "\\toprule",
        "Method & Reward & Graph & Critic & Credit & Coordinator & Gate & Action std \\\\ ",
        "\\midrule",
    ]
    for method in ("matched_mappo", "gat_mappo", "sensors_mo_gat_mappo"):
        row = indexed.get(method)
        if row is None:
            continue
        lines.append(
            f"{METHOD_LABELS[method]} & {row['reward']} & {row['graph']} & "
            f"{row['critic']} & {row['credit']} & {row['coordinator']} & "
            f"{row['gate']} & {float(row['action_std']):.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--robustness-summary", type=Path, default=None)
    parser.add_argument("--safety-stress-summary", type=Path, default=None)
    parser.add_argument("--config-audit", type=Path, default=None)
    args = parser.parse_args()
    rows = _read(args.summary)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "main_comparison_table.tex").write_text(
        main_table(rows), encoding="utf-8"
    )
    (args.output_root / "safety_intervention_table.tex").write_text(
        safety_table(rows), encoding="utf-8"
    )
    (args.output_root / "safety_outcome_table.tex").write_text(
        safety_outcome_table(rows), encoding="utf-8"
    )
    (args.output_root / "computational_cost_table.tex").write_text(
        computation_table(rows), encoding="utf-8"
    )
    if args.robustness_summary is not None:
        (args.output_root / "sensor_robustness_table.tex").write_text(
            robustness_table(_read(args.robustness_summary)),
            encoding="utf-8",
        )
    if args.safety_stress_summary is not None:
        (args.output_root / "joint_sensor_stress_table.tex").write_text(
            safety_input_stress_table(_read(args.safety_stress_summary)),
            encoding="utf-8",
        )
    if args.config_audit is not None:
        (args.output_root / "configuration_audit_table.tex").write_text(
            configuration_table(_read(args.config_audit)),
            encoding="utf-8",
        )
    print(f"Generated LaTeX tables in {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
