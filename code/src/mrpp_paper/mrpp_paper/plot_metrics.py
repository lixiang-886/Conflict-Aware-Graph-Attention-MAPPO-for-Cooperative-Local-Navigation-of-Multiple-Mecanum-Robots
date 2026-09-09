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

import argparse
import csv
import math
from collections import defaultdict
from statistics import median
from pathlib import Path


def summarize_metrics(csv_path: Path) -> list[dict[str, float | str]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            method_label = row.get("variant", "") or row["algorithm"]
            groups[(row["scenario"], method_label)].append(row)

    summaries: list[dict[str, float | str]] = []
    for (scenario, algorithm), rows in sorted(groups.items()):
        numeric_keys = [
            key
            for key in rows[0]
            if key not in {"scenario", "algorithm", "variant", "seed"}
            and all(_is_float(row[key]) for row in rows)
        ]
        summary: dict[str, float | str] = {
            "scenario": scenario,
            "algorithm": algorithm,
            "count": len(rows),
        }
        for key in numeric_keys:
            values = [float(row[key]) for row in rows]
            stats = _summary_stats(values)
            summary[key] = stats["mean"]
            for stat_name, stat_value in stats.items():
                summary[f"{key}_{stat_name}"] = stat_value
        summaries.append(summary)
    return summaries


def _summary_stats(values: list[float]) -> dict[str, float]:
    count = len(values)
    mean = sum(values) / count
    if count > 1:
        variance = sum((value - mean) ** 2 for value in values) / (count - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    sem = std / math.sqrt(count) if count else 0.0
    ci = 1.96 * sem
    return {
        "count": float(count),
        "mean": mean,
        "std": std,
        "sem": sem,
        "median": float(median(values)),
        "min": min(values),
        "max": max(values),
        "ci95_low": mean - ci,
        "ci95_high": mean + ci,
    }


def _is_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def write_summary(rows: list[dict[str, float | str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate paper metric summaries.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = summarize_metrics(Path(args.input))
    write_summary(summary, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
