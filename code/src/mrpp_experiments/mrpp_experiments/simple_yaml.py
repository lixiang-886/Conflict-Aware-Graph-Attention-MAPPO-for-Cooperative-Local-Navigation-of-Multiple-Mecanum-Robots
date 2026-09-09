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


def parse_inline_mapping(text: str) -> dict[str, float]:
    stripped = text.strip().removeprefix("{").removesuffix("}")
    result: dict[str, float] = {}
    for item in stripped.split(","):
        key, value = item.split(":", 1)
        result[key.strip()] = float(value.strip())
    return result


def parse_scalar(value: str) -> str | int | float:
    value = value.strip()
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
