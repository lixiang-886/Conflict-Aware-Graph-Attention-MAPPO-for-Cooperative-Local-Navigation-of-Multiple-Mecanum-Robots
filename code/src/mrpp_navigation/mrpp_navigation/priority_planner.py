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

from mrpp_navigation.astar import GridPoint, astar_grid


def priority_plan(
    starts: list[GridPoint],
    goals: list[GridPoint],
    obstacles: set[GridPoint],
    width: int,
    height: int,
) -> list[list[GridPoint]]:
    reserved = set(obstacles)
    paths: list[list[GridPoint]] = []
    for start, goal in zip(starts, goals):
        path = astar_grid(
            start=start, goal=goal,
            obstacles=reserved - {start, goal},
            width=width, height=height,
        )
        paths.append(path)
        reserved.update(path)
    return paths
