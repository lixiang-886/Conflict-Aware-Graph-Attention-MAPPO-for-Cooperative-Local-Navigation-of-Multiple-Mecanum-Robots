#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


CONFIG_FIELDS = ("variant", "orca_prior_enabled", "safety_filter_enabled")
DEFAULTS = {
    "variant": "",
    "orca_prior_enabled": "1",
    "safety_filter_enabled": "1",
}


def target_fieldnames(existing: list[str]) -> list[str]:
    fields = list(existing)
    for field in CONFIG_FIELDS:
        if field in fields:
            continue
        if "data_source" in fields:
            fields.insert(fields.index("data_source"), field)
        elif "notes" in fields:
            fields.insert(fields.index("notes"), field)
        elif "schema_version" in fields:
            fields.insert(fields.index("schema_version"), field)
        else:
            fields.append(field)
    return fields


def normalize_csv(path: Path) -> bool:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        original_fields = list(reader.fieldnames)
        rows = list(reader)

    output_fields = target_fieldnames(original_fields)
    changed = output_fields != original_fields
    for row in rows:
        for field, default in DEFAULTS.items():
            if field not in row:
                row[field] = default
                changed = True
            elif field != "variant" and row[field] == "":
                row[field] = default
                changed = True

    if not changed:
        return False

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return True


def iter_inputs(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("all_metrics.csv")))
            files.extend(
                sorted(
                    child
                    for child in path.rglob("eval_seed*.csv")
                    if not child.name.endswith("_trajectories.csv")
                )
            )
        else:
            if not path.name.endswith("_trajectories.csv"):
                files.append(path)
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Add missing variant/orca_prior_enabled/safety_filter_enabled columns "
            "to legacy result CSVs. Existing non-empty values are preserved."
        )
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true", help="Fail if any file would change.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    changed_files: list[Path] = []
    for path in iter_inputs(args.paths):
        if not path.exists() or path.suffix != ".csv":
            continue
        if args.check:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise ValueError(f"{path} has no CSV header")
                rows = list(reader)
                fields = list(reader.fieldnames)
            would_change = target_fieldnames(fields) != fields or any(
                field not in row or (field != "variant" and row[field] == "")
                for row in rows
                for field in CONFIG_FIELDS
            )
            if would_change:
                changed_files.append(path)
        elif normalize_csv(path):
            changed_files.append(path)

    for path in changed_files:
        print(path)
    if args.check and changed_files:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
