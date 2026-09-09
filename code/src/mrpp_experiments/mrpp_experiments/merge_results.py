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
from pathlib import Path

from mrpp_experiments.results import EpisodeResult, read_results_csv, write_results_csv


def merge_result_csv_files(paths: list[Path], output: Path) -> None:
    rows: list[EpisodeResult] = []
    for path in paths:
        rows.extend(read_results_csv(path))
    write_results_csv(output, rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge canonical experiment result CSV files.")
    parser.add_argument("--input", action="append", dest="inputs", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    merge_result_csv_files([Path(path) for path in args.inputs], Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
