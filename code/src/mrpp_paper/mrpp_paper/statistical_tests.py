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
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

DEFAULT_METRICS = (
    "success_rate",
    "collision_count",
    "average_path_length",
    "average_completion_time",
)
FIELDNAMES = (
    "scenario",
    "metric",
    "algorithm_a",
    "algorithm_b",
    "test_name",
    "statistic",
    "p_value",
    "adjusted_p_value",
    "effect_size",
    "significant",
    "notes",
)


def build_statistical_tests(
    csv_path: Path,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    alpha: float = 0.05,
) -> list[dict[str, str]]:
    rows = _read_gazebo_rows(csv_path)
    grouped: dict[tuple[str, str], dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        seed = int(row["seed"])
        scenario = row["scenario"]
        algorithm = row["algorithm"]
        for metric in metrics:
            if metric in row and _is_float(row[metric]):
                grouped[(scenario, metric)][algorithm][seed] = float(row[metric])

    results: list[dict[str, str]] = []
    pending_indices: list[int] = []
    for (scenario, metric), by_algorithm in sorted(grouped.items()):
        for algorithm_a, algorithm_b in combinations(sorted(by_algorithm), 2):
            paired = _paired_values(by_algorithm[algorithm_a], by_algorithm[algorithm_b])
            if len(paired) < 2:
                results.append(
                    _result_row(
                        scenario,
                        metric,
                        algorithm_a,
                        algorithm_b,
                        statistic=0.0,
                        p_value=1.0,
                        effect_size=0.0,
                        notes="insufficient_pairs",
                    )
                )
                continue
            values_a = [pair[0] for pair in paired]
            values_b = [pair[1] for pair in paired]
            statistic, p_value = _wilcoxon_signed_rank(values_a, values_b)
            effect = _cliffs_delta(values_a, values_b)
            pending_indices.append(len(results))
            results.append(
                _result_row(
                    scenario,
                    metric,
                    algorithm_a,
                    algorithm_b,
                    statistic=statistic,
                    p_value=p_value,
                    effect_size=effect,
                    notes=f"paired_seeds={len(paired)}",
                )
            )

    _apply_holm(results, pending_indices, alpha=alpha)
    return results


def _read_gazebo_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader if row.get("data_source") == "gazebo"]


def _paired_values(
    values_a: dict[int, float],
    values_b: dict[int, float],
) -> list[tuple[float, float]]:
    seeds = sorted(set(values_a) & set(values_b))
    return [(values_a[seed], values_b[seed]) for seed in seeds]


def _wilcoxon_signed_rank(
    values_a: list[float],
    values_b: list[float],
) -> tuple[float, float]:
    diffs = [a - b for a, b in zip(values_a, values_b) if abs(a - b) > 1e-12]
    if not diffs:
        return 0.0, 1.0
    abs_ranked = sorted((abs(diff), diff) for diff in diffs)
    ranks: list[tuple[float, float]] = []
    index = 0
    while index < len(abs_ranked):
        end = index + 1
        while end < len(abs_ranked) and abs_ranked[end][0] == abs_ranked[index][0]:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        for _, diff in abs_ranked[index:end]:
            ranks.append((avg_rank, diff))
        index = end
    w_pos = sum(rank for rank, diff in ranks if diff > 0)
    w_neg = sum(rank for rank, diff in ranks if diff < 0)
    statistic = min(w_pos, w_neg)
    n = len(ranks)
    mean = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    if variance <= 0:
        return statistic, 1.0
    z = (statistic - mean) / math.sqrt(variance)
    p_value = min(1.0, 2.0 * _normal_cdf(z))
    return statistic, p_value


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _cliffs_delta(values_a: list[float], values_b: list[float]) -> float:
    greater = 0
    lower = 0
    for value_a in values_a:
        for value_b in values_b:
            if value_a > value_b:
                greater += 1
            elif value_a < value_b:
                lower += 1
    total = len(values_a) * len(values_b)
    return (greater - lower) / total if total else 0.0


def _result_row(
    scenario: str,
    metric: str,
    algorithm_a: str,
    algorithm_b: str,
    statistic: float,
    p_value: float,
    effect_size: float,
    notes: str,
) -> dict[str, str]:
    return {
        "scenario": scenario,
        "metric": metric,
        "algorithm_a": algorithm_a,
        "algorithm_b": algorithm_b,
        "test_name": "paired_wilcoxon_signed_rank",
        "statistic": f"{statistic:.8g}",
        "p_value": f"{p_value:.8g}",
        "adjusted_p_value": "",
        "effect_size": f"{effect_size:.8g}",
        "significant": "false",
        "notes": notes,
    }


def _apply_holm(
    rows: list[dict[str, str]],
    indices: list[int],
    alpha: float,
) -> None:
    ordered = sorted(indices, key=lambda index: float(rows[index]["p_value"]))
    m = len(ordered)
    prev_adjusted = 0.0
    for rank, index in enumerate(ordered):
        raw = float(rows[index]["p_value"])
        adjusted = min(1.0, max(prev_adjusted, (m - rank) * raw))
        prev_adjusted = adjusted
        rows[index]["adjusted_p_value"] = f"{adjusted:.8g}"
        rows[index]["significant"] = str(adjusted < alpha).lower()
    for index, row in enumerate(rows):
        if index not in indices:
            row["adjusted_p_value"] = row["p_value"]


def _is_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def write_statistical_tests(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paper statistical tests.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/paper/statistical_tests.csv")
    parser.add_argument("--metric", action="append", dest="metrics")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metrics = tuple(args.metrics) if args.metrics else DEFAULT_METRICS
    rows = build_statistical_tests(Path(args.input), metrics=metrics)
    write_statistical_tests(rows, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
