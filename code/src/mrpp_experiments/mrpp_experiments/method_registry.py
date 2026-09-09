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

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentMethod:
    name: str
    role: str
    description: str


def load_method_registry(path: Path) -> list[ExperimentMethod]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            ExperimentMethod(
                name=str(row["name"]),
                role=str(row["role"]),
                description=str(row["description"]),
            )
            for row in reader
        ]


def select_method_names(methods: list[ExperimentMethod], include_ablations: bool) -> list[str]:
    selected: list[str] = []
    for method in methods:
        if method.role not in ("baseline", "proposed") and not include_ablations:
            continue
        selected.append(method.name)
    return selected
