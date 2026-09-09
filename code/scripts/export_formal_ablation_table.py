#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


SCENARIO_ORDER = ("mixed_complex",)
SCENARIO_LABELS = {
    "mixed_complex": "Mixed complex",
}
VARIANT_ORDER = (
    "full_mo_gat_mappo",
    "ablation_no_graph_attention",
    "ablation_fixed_reward",
    "ablation_no_gate",
    "ablation_no_coordinator",
)
VARIANT_LABELS = {
    "full_mo_gat_mappo": "Full MO-GAT-MAPPO",
    "ablation_no_graph_attention": "w/o GAT",
    "ablation_fixed_reward": "w/o adaptive reward",
    "ablation_no_gate": "w/o interaction gate",
    "ablation_no_coordinator": "w/o conflict coordinator",
}
PRIMARY_METRICS = {
    "average_completion_time": ("lower", 2),
    "makespan": ("lower", 2),
    "average_waiting_time": ("lower", 2),
    "min_robot_robot_distance": ("higher", 3),
    "intervention_time_ratio": ("lower", 3),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def stats(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def format_stat(mean_value: float | None, std_value: float | None, digits: int, scale: float = 1.0) -> str:
    if mean_value is None:
        return "N/A"
    std_value = 0.0 if std_value is None else std_value
    return f"{mean_value * scale:.{digits}f} +/- {std_value * scale:.{digits}f}"


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def stat_mean(value: str) -> float | None:
    if value == "N/A":
        return None
    return float(value.split("+/-", maxsplit=1)[0].strip())


def primary_best_flags(rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    flags: set[tuple[str, str]] = set()
    for metric, (direction, digits) in PRIMARY_METRICS.items():
        candidates = [
            (row["variant"], round(value, digits))
            for row in rows
            if (value := stat_mean(row[metric])) is not None
        ]
        if not candidates:
            continue
        values = [value for _variant, value in candidates]
        best = max(values) if direction == "higher" else min(values)
        flags.update(
            (variant, metric)
            for variant, value in candidates
            if value == best
        )
    return flags


def display_stat(
    row: dict[str, str],
    metric: str,
    best_flags: set[tuple[str, str]],
    *,
    latex: bool,
) -> str:
    value = row[metric]
    if latex:
        value = value.replace("+/-", r"$\pm$")
    if (row["variant"], metric) not in best_flags:
        return value
    return rf"\textbf{{{value}}}" if latex else f"**{value}**"


def summarize_group(rows: list[dict[str, str]]) -> dict[str, float | None]:
    successful_rows = [row for row in rows if as_float(row, "success_rate") > 0.0]
    success_mean, success_std = stats([as_float(row, "success_rate") for row in rows])
    collision_mean, collision_std = stats([as_float(row, "collision_count") for row in rows])
    time_mean, time_std = stats([as_float(row, "average_completion_time") for row in successful_rows])
    makespan_mean, makespan_std = stats([as_float(row, "makespan") for row in successful_rows])
    wait_mean, wait_std = stats([as_float(row, "average_waiting_time") for row in rows])
    path_mean, path_std = stats([as_float(row, "average_path_length") for row in rows])
    efficiency_mean, efficiency_std = stats([as_float(row, "average_path_efficiency") for row in rows])
    turning_mean, turning_std = stats(
        [as_float(row, "average_turning_angle_per_meter") for row in rows]
    )
    sharp_mean, sharp_std = stats([as_float(row, "total_sharp_turn_count") for row in rows])
    rr_min_mean, rr_min_std = stats([as_float(row, "min_robot_robot_distance") for row in rows])
    rp_min_mean, rp_min_std = stats([as_float(row, "min_robot_pedestrian_distance") for row in rows])
    near_mean, near_std = stats(
        [
            as_float(row, "robot_robot_near_miss_count")
            + as_float(row, "robot_pedestrian_near_miss_count")
            for row in rows
        ]
    )
    intervention_mean, intervention_std = stats([as_float(row, "intervention_time_ratio") for row in rows])
    intervention_count_mean, intervention_count_std = stats(
        [as_float(row, "safety_filter_intervention_count") for row in rows]
    )
    intervention_magnitude_mean, intervention_magnitude_std = stats(
        [as_float(row, "mean_intervention_magnitude") for row in rows]
    )
    intervention_max_mean, intervention_max_std = stats(
        [as_float(row, "max_intervention_magnitude") for row in rows]
    )
    return {
        "success_mean": success_mean,
        "success_std": success_std,
        "collision_mean": collision_mean,
        "collision_std": collision_std,
        "time_mean": time_mean,
        "time_std": time_std,
        "makespan_mean": makespan_mean,
        "makespan_std": makespan_std,
        "wait_mean": wait_mean,
        "wait_std": wait_std,
        "path_mean": path_mean,
        "path_std": path_std,
        "efficiency_mean": efficiency_mean,
        "efficiency_std": efficiency_std,
        "turning_mean": turning_mean,
        "turning_std": turning_std,
        "sharp_mean": sharp_mean,
        "sharp_std": sharp_std,
        "rr_min_mean": rr_min_mean,
        "rr_min_std": rr_min_std,
        "rp_min_mean": rp_min_mean,
        "rp_min_std": rp_min_std,
        "near_mean": near_mean,
        "near_std": near_std,
        "intervention_mean": intervention_mean,
        "intervention_std": intervention_std,
        "intervention_count_mean": intervention_count_mean,
        "intervention_count_std": intervention_count_std,
        "intervention_magnitude_mean": intervention_magnitude_mean,
        "intervention_magnitude_std": intervention_magnitude_std,
        "intervention_max_mean": intervention_max_mean,
        "intervention_max_std": intervention_max_std,
    }


def build_summary_rows(all_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        variant = row.get("variant") or row.get("algorithm", "")
        if variant in VARIANT_ORDER:
            grouped[(row["scenario"], variant)].append(row)

    output_rows: list[dict[str, str]] = []
    for scenario in SCENARIO_ORDER:
        for variant in VARIANT_ORDER:
            rows = grouped.get((scenario, variant), [])
            if not rows:
                continue
            summary = summarize_group(rows)
            output_rows.append(
                {
                    "scenario": scenario,
                    "variant": variant,
                    "count": str(len(rows)),
                    "success_rate": format_stat(summary["success_mean"], summary["success_std"], 3),
                    "success_percent": format_stat(summary["success_mean"], summary["success_std"], 1, 100.0),
                    "collision_count": format_stat(summary["collision_mean"], summary["collision_std"], 2),
                    "average_completion_time": format_stat(summary["time_mean"], summary["time_std"], 2),
                    "makespan": format_stat(summary["makespan_mean"], summary["makespan_std"], 2),
                    "average_waiting_time": format_stat(summary["wait_mean"], summary["wait_std"], 2),
                    "average_path_length": format_stat(summary["path_mean"], summary["path_std"], 2),
                    "average_path_efficiency": format_stat(summary["efficiency_mean"], summary["efficiency_std"], 3),
                    "average_turning_angle_per_meter": format_stat(summary["turning_mean"], summary["turning_std"], 3),
                    "total_sharp_turn_count": format_stat(summary["sharp_mean"], summary["sharp_std"], 1),
                    "min_robot_robot_distance": format_stat(summary["rr_min_mean"], summary["rr_min_std"], 3),
                    "min_robot_pedestrian_distance": format_stat(summary["rp_min_mean"], summary["rp_min_std"], 3),
                    "total_near_miss_count": format_stat(summary["near_mean"], summary["near_std"], 1),
                    "intervention_time_ratio": format_stat(summary["intervention_mean"], summary["intervention_std"], 3),
                    "safety_filter_intervention_count": format_stat(summary["intervention_count_mean"], summary["intervention_count_std"], 1),
                    "mean_intervention_magnitude": format_stat(summary["intervention_magnitude_mean"], summary["intervention_magnitude_std"], 3),
                    "max_intervention_magnitude": format_stat(summary["intervention_max_mean"], summary["intervention_max_std"], 3),
                }
            )
    return output_rows


def markdown_table(rows: list[dict[str, str]]) -> str:
    best_flags = primary_best_flags(rows)
    headers = [
        "Scenario",
        "Variant",
        "Succ. (%)",
        "Coll.",
        "Avg. (s)",
        "Make. (s)",
        "Wait (s)",
        "Eff.",
        "Turn (rad/m)",
        "Min RR (m)",
        "Near",
        "Int. ratio",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        cells = [
            SCENARIO_LABELS.get(row["scenario"], row["scenario"]),
            VARIANT_LABELS.get(row["variant"], row["variant"]),
            row["success_percent"],
            row["collision_count"],
            display_stat(
                row, "average_completion_time", best_flags, latex=False
            ),
            display_stat(row, "makespan", best_flags, latex=False),
            display_stat(row, "average_waiting_time", best_flags, latex=False),
            row["average_path_efficiency"],
            row["average_turning_angle_per_meter"],
            display_stat(row, "min_robot_robot_distance", best_flags, latex=False),
            row["total_near_miss_count"],
            display_stat(row, "intervention_time_ratio", best_flags, latex=False),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def latex_table(rows: list[dict[str, str]]) -> str:
    best_flags = primary_best_flags(rows)
    lines = [
        r"\begin{table}[H]",
        r"\caption{Component ablation.}",
        r"\label{tab:component-ablation}",
        r"\begin{adjustwidth}{-\extralength}{0cm}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular*}{\fulllength}{@{\extracolsep{\fill}}lrrrrrrrrr@{}}",
        r"\toprule",
        r"Variant & Succ. (\%) & Avg. (s) & Make. (s) & Wait (s) & Eff. & Turn (rad/m) & Min RR (m) & Near & Int. ratio \\",
        r"\midrule",
    ]
    for row in rows:
        success = row["success_percent"].replace("+/-", r"$\pm$")
        time = display_stat(
            row, "average_completion_time", best_flags, latex=True
        )
        makespan = display_stat(row, "makespan", best_flags, latex=True)
        wait = display_stat(row, "average_waiting_time", best_flags, latex=True)
        efficiency = row["average_path_efficiency"].replace("+/-", r"$\pm$")
        turning = row["average_turning_angle_per_meter"].replace("+/-", r"$\pm$")
        min_rr = display_stat(
            row, "min_robot_robot_distance", best_flags, latex=True
        )
        near_total = row["total_near_miss_count"].replace("+/-", r"$\pm$")
        intervention = display_stat(
            row, "intervention_time_ratio", best_flags, latex=True
        )
        cells = [
            latex_escape(VARIANT_LABELS.get(row["variant"], row["variant"])),
            success,
            time,
            makespan,
            wait,
            efficiency,
            turning,
            min_rr,
            near_total,
            intervention,
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular*}",
            r"\end{adjustwidth}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export paper-ready formal ablation tables.")
    parser.add_argument(
        "--all-metrics",
        default="results/eval/formal_ablation_comparison/all_metrics.csv",
        help="Merged formal ablation metric rows.",
    )
    parser.add_argument(
        "--output-root",
        default="results/eval/formal_ablation_comparison/paper_assets",
        help="Directory for generated paper assets.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    all_rows = read_csv(Path(args.all_metrics))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows = build_summary_rows(all_rows)
    if not summary_rows:
        raise SystemExit("No formal ablation rows found.")
    fieldnames = list(summary_rows[0].keys())
    write_csv(output_root / "ablation_summary.csv", summary_rows, fieldnames)
    markdown = (
        "# Formal Ablation Study\n\n"
        "Values are mean +/- standard deviation over seeds 0, 1, and 2 where shown.\n\n"
        + markdown_table(summary_rows)
        + "\n"
    )
    (output_root / "ablation_study_table.md").write_text(markdown, encoding="utf-8")
    (output_root / "focused_ablation_table.md").write_text(markdown, encoding="utf-8")
    latex = latex_table(summary_rows) + "\n"
    (output_root / "ablation_study_table.tex").write_text(latex, encoding="utf-8")
    (output_root / "focused_ablation_table.tex").write_text(
        latex,
        encoding="utf-8",
    )
    print(f"Formal ablation paper assets written to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
