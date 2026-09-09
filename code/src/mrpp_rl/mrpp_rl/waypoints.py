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

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from mrpp_experiments.scenario import Scenario
from mrpp_navigation.astar import GridPoint, astar_grid
from mrpp_rl.environment import GoalState, RobotState

GRID_RESOLUTION = 0.25
OBSTACLE_INFLATION = 0.65
WORLD_OBSTACLE_INFLATION = {
    "static_clutter": 0.75,
}
WAYPOINT_REACHED_DIST = 0.55
LOOKAHEAD_DISTANCE = 0.8
MINIMUM_LOOKAHEAD = 0.4
MAXIMUM_LOOKAHEAD = 1.2
LOOKAHEAD_SPEED_GAIN = 0.6
FINAL_GOAL_SWITCH_DISTANCE = 1.0
MAX_PROJECTION_ADVANCE = 2.0
LINE_OF_SIGHT_CLEARANCE_CELLS = 1
DEFAULT_MARGIN = 1.0
WAYPOINT_ENABLED_WORLDS = {
    "static_clutter",
    "pedestrian_dynamic",
    "mixed_complex",
}
DIRECT_PATH_WORLDS = {
    "pedestrian_dynamic",
}


@dataclass(frozen=True)
class BoxObstacle:
    x: float
    y: float
    yaw: float
    sx: float
    sy: float


@dataclass(frozen=True)
class CylinderObstacle:
    x: float
    y: float
    radius: float


Obstacle = BoxObstacle | CylinderObstacle

MIXED_COMPLEX_TASK_KEEP_OUTS = {
    "robot_1": (
        BoxObstacle(x=-2.5, y=4.4, yaw=0.0, sx=4.8, sy=1.2),
    ),
    "robot_2": (
        # Keep robot_2 out of the upper pedestrian lane so local avoidance does
        # not push it into the inflated static obstacle at (-3, 5).
        BoxObstacle(x=-2.5, y=4.4, yaw=0.0, sx=4.8, sy=1.2),
    ),
    "robot_3": (
        # robot_3 otherwise clips the lower edge of the same pedestrian lane
        # while moving from the upper-right corner toward the lower-left goal.
        BoxObstacle(x=-2.5, y=4.4, yaw=0.0, sx=4.8, sy=1.2),
    ),
}


@dataclass(frozen=True)
class PathTrackingState:
    local_goal: GoalState
    progress: float
    remaining_distance: float
    cross_track_error: float
    tangent_yaw: float
    nearest_path_index: int
    progress_distance: float


@dataclass(frozen=True)
class WorldGrid:
    min_x: float
    min_y: float
    resolution: float
    width: int
    height: int
    obstacles: frozenset[GridPoint]

    def to_grid(self, x: float, y: float) -> GridPoint:
        gx = round((x - self.min_x) / self.resolution)
        gy = round((y - self.min_y) / self.resolution)
        return (
            min(max(int(gx), 0), self.width - 1),
            min(max(int(gy), 0), self.height - 1),
        )

    def to_world(self, point: GridPoint) -> tuple[float, float]:
        return (
            self.min_x + point[0] * self.resolution,
            self.min_y + point[1] * self.resolution,
        )

    def nearest_free(self, point: GridPoint) -> GridPoint:
        if point not in self.obstacles:
            return point
        queue = deque([point])
        visited = {point}
        while queue:
            x, y = queue.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                neighbor = (nx, ny)
                if (
                    nx < 0 or ny < 0
                    or nx >= self.width or ny >= self.height
                    or neighbor in visited
                ):
                    continue
                if neighbor not in self.obstacles:
                    return neighbor
                visited.add(neighbor)
                queue.append(neighbor)
        return point


class WaypointManager:
    def __init__(
        self,
        paths: dict[str, list[GoalState]],
        reached_dist: float = WAYPOINT_REACHED_DIST,
        lookahead_distance: float = LOOKAHEAD_DISTANCE,
        minimum_lookahead: float = MINIMUM_LOOKAHEAD,
        maximum_lookahead: float = MAXIMUM_LOOKAHEAD,
        lookahead_speed_gain: float = LOOKAHEAD_SPEED_GAIN,
        final_goal_switch_distance: float = FINAL_GOAL_SWITCH_DISTANCE,
        max_projection_advance: float = MAX_PROJECTION_ADVANCE,
    ) -> None:
        self._paths = {name: list(path) for name, path in paths.items()}
        self._indices = {name: 0 for name in self._paths}
        self._progress_distances = {name: 0.0 for name in self._paths}
        self._cumulative = {
            name: _cumulative_distances(path) for name, path in self._paths.items()
        }
        self._reached_dist = reached_dist
        self._lookahead_distance = lookahead_distance
        self._minimum_lookahead = minimum_lookahead
        self._maximum_lookahead = maximum_lookahead
        self._lookahead_speed_gain = lookahead_speed_gain
        self._final_goal_switch_distance = final_goal_switch_distance
        self._max_projection_advance = max_projection_advance

    def reset(self) -> None:
        for name in self._indices:
            self._indices[name] = 0
            self._progress_distances[name] = 0.0

    def current_goal(
        self,
        name: str,
        state: RobotState,
        final_goal: GoalState,
    ) -> GoalState:
        return self.tracking_state(name, state, final_goal).local_goal

    def tracking_state(
        self,
        name: str,
        state: RobotState,
        final_goal: GoalState,
    ) -> PathTrackingState:
        path = self._paths.get(name)
        if not path:
            distance = math.hypot(state.x - final_goal.x, state.y - final_goal.y)
            tangent_yaw = math.atan2(final_goal.y - state.y, final_goal.x - state.x)
            return PathTrackingState(
                local_goal=final_goal,
                progress=0.0,
                remaining_distance=distance,
                cross_track_error=0.0,
                tangent_yaw=tangent_yaw,
                nearest_path_index=0,
                progress_distance=0.0,
            )
        if len(path) == 1:
            distance = math.hypot(state.x - path[0].x, state.y - path[0].y)
            return PathTrackingState(
                local_goal=path[0],
                progress=1.0 if distance <= self._final_goal_switch_distance else 0.0,
                remaining_distance=distance,
                cross_track_error=0.0,
                tangent_yaw=path[0].yaw,
                nearest_path_index=0,
                progress_distance=0.0,
            )

        cumulative = self._cumulative[name]
        total_length = cumulative[-1]
        projection = _project_onto_path(
            path,
            cumulative,
            state,
            min_progress=self._progress_distances.get(name, 0.0),
            max_progress=(
                self._progress_distances.get(name, 0.0)
                + self._max_projection_advance
            ),
        )
        projected_progress = max(
            self._progress_distances.get(name, 0.0),
            projection.progress_distance,
        )
        projected_progress = min(projected_progress, total_length)
        self._progress_distances[name] = projected_progress
        self._indices[name] = projection.segment_index

        final_distance = math.hypot(state.x - final_goal.x, state.y - final_goal.y)
        if final_distance <= self._final_goal_switch_distance:
            local_goal = final_goal
        else:
            lookahead = self._adaptive_lookahead(state)
            local_goal = _point_at_progress(
                path,
                cumulative,
                min(projected_progress + lookahead, total_length),
                final_goal,
            )
        progress = projected_progress / total_length if total_length > 1e-6 else 1.0
        return PathTrackingState(
            local_goal=local_goal,
            progress=progress,
            remaining_distance=max(total_length - projected_progress, 0.0),
            cross_track_error=projection.cross_track_error,
            tangent_yaw=projection.tangent_yaw,
            nearest_path_index=projection.segment_index,
            progress_distance=projected_progress,
        )

    def tracking_states(
        self,
        states: dict[str, RobotState],
        final_goals: dict[str, GoalState],
    ) -> dict[str, PathTrackingState]:
        return {
            name: self.tracking_state(name, states[name], goal)
            for name, goal in final_goals.items()
            if name in states
        }

    def current_goals(
        self,
        states: dict[str, RobotState],
        final_goals: dict[str, GoalState],
    ) -> dict[str, GoalState]:
        return {
            name: self.current_goal(name, states[name], goal)
            for name, goal in final_goals.items()
            if name in states
        }

    def path_for(self, name: str) -> list[GoalState]:
        return list(self._paths.get(name, []))

    def _adaptive_lookahead(self, state: RobotState) -> float:
        speed = math.hypot(state.v, state.vy)
        lookahead = self._lookahead_distance + self._lookahead_speed_gain * speed
        return max(
            self._minimum_lookahead,
            min(self._maximum_lookahead, lookahead),
        )


@dataclass(frozen=True)
class _PathProjection:
    progress_distance: float
    cross_track_error: float
    tangent_yaw: float
    segment_index: int


def _cumulative_distances(path: list[GoalState]) -> list[float]:
    if not path:
        return [0.0]
    distances = [0.0]
    for previous, current in zip(path, path[1:]):
        distances.append(
            distances[-1] + math.hypot(current.x - previous.x, current.y - previous.y)
        )
    return distances


def densify_goal_path(
    path: list[GoalState],
    max_segment_length: float,
) -> list[GoalState]:
    if len(path) <= 1 or max_segment_length <= 0.0:
        return list(path)

    dense = [path[0]]
    for start, end in zip(path, path[1:]):
        dx = end.x - start.x
        dy = end.y - start.y
        distance = math.hypot(dx, dy)
        steps = max(1, math.ceil(distance / max_segment_length))
        segment_yaw = math.atan2(dy, dx) if distance > 1e-9 else end.yaw
        for index in range(1, steps + 1):
            if index == steps:
                dense.append(end)
                continue
            ratio = index / steps
            dense.append(
                GoalState(
                    x=start.x + ratio * dx,
                    y=start.y + ratio * dy,
                    yaw=segment_yaw,
                )
            )
    return dense


def _project_onto_path(
    path: list[GoalState],
    cumulative: list[float],
    state: RobotState,
    min_progress: float = 0.0,
    max_progress: float | None = None,
) -> _PathProjection:
    best: _PathProjection | None = None
    for index, (start, end) in enumerate(zip(path, path[1:])):
        dx = end.x - start.x
        dy = end.y - start.y
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-12:
            continue
        raw_ratio = ((state.x - start.x) * dx + (state.y - start.y) * dy) / length_sq
        ratio = max(0.0, min(1.0, raw_ratio))
        projected_x = start.x + ratio * dx
        projected_y = start.y + ratio * dy
        progress = cumulative[index] + math.sqrt(length_sq) * ratio
        if progress + 1e-6 < min_progress:
            continue
        if max_progress is not None and progress - 1e-6 > max_progress:
            continue
        cross_track = math.hypot(state.x - projected_x, state.y - projected_y)
        candidate = _PathProjection(
            progress_distance=progress,
            cross_track_error=cross_track,
            tangent_yaw=math.atan2(dy, dx),
            segment_index=index,
        )
        if best is None or candidate.cross_track_error < best.cross_track_error:
            best = candidate

    if best is not None:
        return best

    progress = min(max(min_progress, 0.0), cumulative[-1])
    point = _point_at_progress(path, cumulative, progress, path[-1])
    index = max(0, min(len(path) - 2, _segment_index_for_progress(cumulative, progress)))
    next_point = path[min(index + 1, len(path) - 1)]
    return _PathProjection(
        progress_distance=progress,
        cross_track_error=math.hypot(state.x - point.x, state.y - point.y),
        tangent_yaw=math.atan2(next_point.y - point.y, next_point.x - point.x),
        segment_index=index,
    )


def _point_at_progress(
    path: list[GoalState],
    cumulative: list[float],
    progress: float,
    final_goal: GoalState,
) -> GoalState:
    if not path:
        return final_goal
    if len(path) == 1 or progress >= cumulative[-1]:
        return final_goal
    index = _segment_index_for_progress(cumulative, progress)
    start = path[index]
    end = path[index + 1]
    segment_length = cumulative[index + 1] - cumulative[index]
    if segment_length <= 1e-9:
        return GoalState(x=end.x, y=end.y, yaw=end.yaw)
    ratio = (progress - cumulative[index]) / segment_length
    x = start.x + ratio * (end.x - start.x)
    y = start.y + ratio * (end.y - start.y)
    yaw = math.atan2(end.y - start.y, end.x - start.x)
    return GoalState(x=x, y=y, yaw=yaw)


def _segment_index_for_progress(cumulative: list[float], progress: float) -> int:
    for index in range(len(cumulative) - 1):
        if progress <= cumulative[index + 1]:
            return index
    return max(0, len(cumulative) - 2)


def build_waypoint_manager(
    scenario: Scenario,
    world_sdf: Path,
    resolution: float = GRID_RESOLUTION,
    inflation: float = OBSTACLE_INFLATION,
    simplify: bool = True,
    max_segment_length: float | None = None,
    manager_kwargs: dict[str, float] | None = None,
) -> WaypointManager:
    grid = build_world_grid(
        world_sdf=world_sdf,
        scenario=scenario,
        resolution=resolution,
        inflation=inflation,
    )
    paths: dict[str, list[GoalState]] = {}
    for task in scenario.robots:
        start = grid.nearest_free(grid.to_grid(task.start.x, task.start.y))
        goal = grid.nearest_free(grid.to_grid(task.goal.x, task.goal.y))
        task_obstacles = set(grid.obstacles)
        task_obstacles.update(
            _task_keep_out_cells(
                scenario.world,
                task.name,
                grid,
            )
        )
        grid_path = astar_grid(
            start=start,
            goal=goal,
            obstacles=task_obstacles - {start, goal},
            width=grid.width,
            height=grid.height,
        )
        if not grid_path:
            raise RuntimeError(
                f"No A* path found for {task.name} in scenario {scenario.name}."
            )
        simple_path = simplify_grid_path(grid_path, grid) if simplify else grid_path
        waypoints = []
        for point in simple_path:
            x, y = grid.to_world(point)
            waypoints.append(GoalState(x=x, y=y, yaw=task.goal.yaw))
        if not waypoints:
            waypoints.append(GoalState(x=task.goal.x, y=task.goal.y, yaw=task.goal.yaw))
        else:
            start_x, start_y = grid.to_world(start)
            waypoints[0] = GoalState(
                x=start_x,
                y=start_y,
                yaw=task.start.yaw,
            )
            waypoints[-1] = GoalState(
                x=task.goal.x,
                y=task.goal.y,
                yaw=task.goal.yaw,
            )
        if max_segment_length is not None:
            waypoints = densify_goal_path(waypoints, max_segment_length)
        paths[task.name] = waypoints
    return WaypointManager(paths, **(manager_kwargs or {}))


def build_scenario_waypoint_manager(
    scenario: Scenario,
    world_sdf: Path,
) -> WaypointManager | None:
    if scenario.world not in WAYPOINT_ENABLED_WORLDS:
        return None
    if scenario.world in DIRECT_PATH_WORLDS:
        return build_direct_waypoint_manager(scenario)
    manager_kwargs = {}
    simplify = True
    max_segment_length = None
    if scenario.world == "static_clutter":
        manager_kwargs = {
            "lookahead_distance": 1.10,
            "minimum_lookahead": 0.60,
            "maximum_lookahead": 1.60,
            "lookahead_speed_gain": 0.80,
            "final_goal_switch_distance": 1.00,
            "max_projection_advance": 2.00,
        }
    elif scenario.world == "mixed_complex":
        manager_kwargs = {
            "lookahead_distance": 0.80,
            "minimum_lookahead": 0.40,
            "maximum_lookahead": 1.15,
            "lookahead_speed_gain": 0.55,
            "final_goal_switch_distance": 1.00,
            "max_projection_advance": 1.50,
        }
        max_segment_length = 1.00
    return build_waypoint_manager(
        scenario,
        world_sdf,
        inflation=WORLD_OBSTACLE_INFLATION.get(
            scenario.world,
            OBSTACLE_INFLATION,
        ),
        simplify=simplify,
        max_segment_length=max_segment_length,
        manager_kwargs=manager_kwargs,
    )


def build_direct_waypoint_manager(scenario: Scenario) -> WaypointManager:
    paths = {}
    for task in scenario.robots:
        waypoints = [
            GoalState(x=task.start.x, y=task.start.y, yaw=task.start.yaw),
            GoalState(x=task.goal.x, y=task.goal.y, yaw=task.goal.yaw),
        ]
        paths[task.name] = waypoints
    return WaypointManager(paths)


def _task_keep_out_cells(
    world_name: str,
    robot_name: str,
    grid: WorldGrid,
) -> set[GridPoint]:
    if world_name != "mixed_complex":
        return set()
    keep_outs = MIXED_COMPLEX_TASK_KEEP_OUTS.get(robot_name, ())
    cells: set[GridPoint] = set()
    for ix in range(grid.width):
        x = grid.min_x + ix * grid.resolution
        for iy in range(grid.height):
            y = grid.min_y + iy * grid.resolution
            if any(_contains(obstacle, x, y, 0.0) for obstacle in keep_outs):
                cells.add((ix, iy))
    return cells


def build_world_grid(
    world_sdf: Path,
    scenario: Scenario,
    resolution: float = GRID_RESOLUTION,
    inflation: float = OBSTACLE_INFLATION,
) -> WorldGrid:
    obstacles = parse_static_obstacles(world_sdf)
    min_x, max_x, min_y, max_y = _grid_bounds(world_sdf, scenario, obstacles)
    width = int(math.ceil((max_x - min_x) / resolution)) + 1
    height = int(math.ceil((max_y - min_y) / resolution)) + 1
    occupied: set[GridPoint] = set()
    for ix in range(width):
        x = min_x + ix * resolution
        for iy in range(height):
            y = min_y + iy * resolution
            if any(_contains(obstacle, x, y, inflation) for obstacle in obstacles):
                occupied.add((ix, iy))
    return WorldGrid(
        min_x=min_x,
        min_y=min_y,
        resolution=resolution,
        width=width,
        height=height,
        obstacles=frozenset(occupied),
    )


def parse_static_obstacles(world_sdf: Path) -> list[Obstacle]:
    root = ET.fromstring(world_sdf.read_text(encoding="utf-8"))
    obstacles: list[Obstacle] = []
    for model in root.findall(".//model"):
        name = model.get("name", "")
        if name == "ground_plane":
            continue
        if _text(model.find("static")) != "true":
            continue
        pose = _pose_values(model.find("pose"))
        for collision in model.findall(".//collision"):
            geometry = collision.find("geometry")
            if geometry is None:
                continue
            box = geometry.find("box")
            cylinder = geometry.find("cylinder")
            if box is not None:
                size = _float_list(_text(box.find("size")))
                if len(size) >= 2:
                    obstacles.append(
                        BoxObstacle(
                            x=pose[0],
                            y=pose[1],
                            yaw=pose[5],
                            sx=size[0],
                            sy=size[1],
                        )
                    )
            elif cylinder is not None:
                radius_text = _text(cylinder.find("radius"))
                if radius_text:
                    obstacles.append(
                        CylinderObstacle(
                            x=pose[0],
                            y=pose[1],
                            radius=float(radius_text),
                        )
                    )
    return obstacles


def simplify_grid_path(path: list[GridPoint], grid: WorldGrid) -> list[GridPoint]:
    if len(path) <= 2:
        return path
    simplified = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        candidate = len(path) - 1
        while candidate > anchor + 1:
            if _line_is_free(path[anchor], path[candidate], grid):
                break
            candidate -= 1
        simplified.append(path[candidate])
        anchor = candidate
    return _drop_collinear_grid_points(simplified)


def _drop_collinear_grid_points(path: list[GridPoint]) -> list[GridPoint]:
    if len(path) <= 2:
        return path
    compact = [path[0]]
    for index in range(1, len(path) - 1):
        previous = compact[-1]
        current = path[index]
        following = path[index + 1]
        dx1 = current[0] - previous[0]
        dy1 = current[1] - previous[1]
        dx2 = following[0] - current[0]
        dy2 = following[1] - current[1]
        if dx1 * dy2 == dy1 * dx2:
            continue
        compact.append(current)
    compact.append(path[-1])
    return compact


def _grid_bounds(
    world_sdf: Path,
    scenario: Scenario,
    obstacles: list[Obstacle],
) -> tuple[float, float, float, float]:
    wall_bounds = _inner_wall_bounds(obstacles)
    if wall_bounds is not None:
        return wall_bounds

    ground_bounds = _ground_plane_bounds(world_sdf)
    if ground_bounds is not None:
        return ground_bounds

    xs = [pose.start.x for pose in scenario.robots] + [pose.goal.x for pose in scenario.robots]
    ys = [pose.start.y for pose in scenario.robots] + [pose.goal.y for pose in scenario.robots]
    for obstacle in obstacles:
        xs.append(obstacle.x)
        ys.append(obstacle.y)
    return (
        min(xs) - DEFAULT_MARGIN,
        max(xs) + DEFAULT_MARGIN,
        min(ys) - DEFAULT_MARGIN,
        max(ys) + DEFAULT_MARGIN,
    )


def _inner_wall_bounds(obstacles: list[Obstacle]) -> tuple[float, float, float, float] | None:
    min_x_candidates: list[float] = []
    max_x_candidates: list[float] = []
    min_y_candidates: list[float] = []
    max_y_candidates: list[float] = []
    for obstacle in obstacles:
        if not isinstance(obstacle, BoxObstacle):
            continue
        if obstacle.sx < 0.4 and obstacle.sy > 2.0:
            if obstacle.x > 0:
                max_x_candidates.append(obstacle.x - obstacle.sx / 2.0)
            elif obstacle.x < 0:
                min_x_candidates.append(obstacle.x + obstacle.sx / 2.0)
        if obstacle.sy < 0.4 and obstacle.sx > 2.0:
            if obstacle.y > 0:
                max_y_candidates.append(obstacle.y - obstacle.sy / 2.0)
            elif obstacle.y < 0:
                min_y_candidates.append(obstacle.y + obstacle.sy / 2.0)
    if not (
        min_x_candidates
        and max_x_candidates
        and min_y_candidates
        and max_y_candidates
    ):
        return None
    return (
        float(min(min_x_candidates)),
        float(max(max_x_candidates)),
        float(min(min_y_candidates)),
        float(max(max_y_candidates)),
    )


def _ground_plane_bounds(world_sdf: Path) -> tuple[float, float, float, float] | None:
    root = ET.fromstring(world_sdf.read_text(encoding="utf-8"))
    for model in root.findall(".//model"):
        if model.get("name") != "ground_plane":
            continue
        size_text = _text(model.find(".//plane/size"))
        size = _float_list(size_text)
        if len(size) >= 2:
            return (-size[0] / 2.0, size[0] / 2.0, -size[1] / 2.0, size[1] / 2.0)
    return None


def _contains(obstacle: Obstacle, x: float, y: float, inflation: float) -> bool:
    if isinstance(obstacle, CylinderObstacle):
        return math.hypot(x - obstacle.x, y - obstacle.y) <= obstacle.radius + inflation

    dx = x - obstacle.x
    dy = y - obstacle.y
    cos_yaw = math.cos(-obstacle.yaw)
    sin_yaw = math.sin(-obstacle.yaw)
    local_x = cos_yaw * dx - sin_yaw * dy
    local_y = sin_yaw * dx + cos_yaw * dy
    return (
        abs(local_x) <= obstacle.sx / 2.0 + inflation
        and abs(local_y) <= obstacle.sy / 2.0 + inflation
    )


def _line_is_free(
    start: GridPoint,
    end: GridPoint,
    grid: WorldGrid,
    clearance_cells: int = LINE_OF_SIGHT_CLEARANCE_CELLS,
) -> bool:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    steps = max(abs(dx), abs(dy), 1) * 4
    previous = start
    for index in range(steps + 1):
        ratio = index / steps
        point = (
            round(start[0] + dx * ratio),
            round(start[1] + dy * ratio),
        )
        if _near_obstacle(point, grid, clearance_cells):
            return False
        if (
            point != previous
            and point[0] != previous[0]
            and point[1] != previous[1]
        ):
            if (
                _near_obstacle((previous[0], point[1]), grid, clearance_cells)
                or _near_obstacle((point[0], previous[1]), grid, clearance_cells)
            ):
                return False
        previous = point
    return True


def _near_obstacle(point: GridPoint, grid: WorldGrid, clearance_cells: int) -> bool:
    x, y = point
    for dx in range(-clearance_cells, clearance_cells + 1):
        for dy in range(-clearance_cells, clearance_cells + 1):
            neighbor = (x + dx, y + dy)
            if neighbor in grid.obstacles:
                return True
    return False


def _pose_values(element: ET.Element | None) -> tuple[float, float, float, float, float, float]:
    values = _float_list(_text(element))
    values = values + [0.0] * (6 - len(values))
    return tuple(values[:6])  # type: ignore[return-value]


def _text(element: ET.Element | None) -> str:
    if element is None or element.text is None:
        return ""
    return element.text.strip()


def _float_list(text: str) -> list[float]:
    return [float(part) for part in text.split() if part]
