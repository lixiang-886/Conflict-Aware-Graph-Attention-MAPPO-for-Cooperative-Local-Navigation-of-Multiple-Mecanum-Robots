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

from heapq import heappop, heappush
from math import inf
from typing import Iterable

GridPoint = tuple[int, int]


def _neighbors(point: GridPoint, width: int, height: int) -> Iterable[GridPoint]:
    x, y = point
    for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
        if 0 <= nx < width and 0 <= ny < height:
            yield nx, ny


def _heuristic(a: GridPoint, b: GridPoint) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _reconstruct(came_from: dict[GridPoint, GridPoint], current: GridPoint) -> list[GridPoint]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return list(reversed(path))


def astar_grid(
    start: GridPoint,
    goal: GridPoint,
    obstacles: set[GridPoint],
    width: int,
    height: int,
) -> list[GridPoint]:
    open_set: list[tuple[float, GridPoint]] = []
    heappush(open_set, (0.0, start))
    came_from: dict[GridPoint, GridPoint] = {}
    g_score: dict[GridPoint, float] = {start: 0.0}

    while open_set:
        _, current = heappop(open_set)
        if current == goal:
            return _reconstruct(came_from, current)

        for neighbor in _neighbors(current, width, height):
            if neighbor in obstacles:
                continue
            tentative = g_score[current] + 1.0
            if tentative < g_score.get(neighbor, inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                f_score = tentative + _heuristic(neighbor, goal)
                heappush(open_set, (f_score, neighbor))

    return []
