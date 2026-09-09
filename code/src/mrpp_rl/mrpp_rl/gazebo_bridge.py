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

import math
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage

from mrpp_rl.environment import GoalState, RobotState
from mrpp_rl.lidar import DEFAULT_RANGE_MAX, DEFAULT_RANGE_MIN
from mrpp_rl.lidar import LaserObservation, laser_scan_to_observation

try:
    from ros_gz_interfaces.msg import WorldStatistics
    from ros_gz_interfaces.srv import ControlWorld
except ModuleNotFoundError:  # pragma: no cover - only absent in light unit stubs.
    WorldStatistics = None
    ControlWorld = None

ROBOT_RADIUS = 0.18
PEDESTRIAN_RADIUS = 0.35
OBSTACLE_COLLISION_RANGE = 0.18
UNSEEN_PEDESTRIAN_XY = 999.0


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


LaserData = LaserObservation


def _stamp_sec_from_scan(msg: LaserScan) -> float | None:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return None
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    if sec == 0 and nanosec == 0:
        return None
    return sec + nanosec * 1e-9


def _stamp_sec_from_header(header) -> float | None:
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    if sec == 0 and nanosec == 0:
        return None
    return sec + nanosec * 1e-9


@dataclass
class PedestrianState:
    name: str
    x: float
    y: float
    vx: float
    vy: float


class GazeboBridge(Node):
    """
    ROS2 node that bridges Gazebo simulation and RL training.

    Subscribes to each robot's odom and scan topics, publishes cmd_vel.
    Also tracks pedestrian positions for collision avoidance.
    """

    def __init__(
        self,
        robot_names: list[str],
        pedestrian_names: list[str] | None = None,
        world_name: str = "pedestrian_dynamic",
    ) -> None:
        super().__init__("gazebo_rl_bridge")
        self._robot_names = list(robot_names)
        self._pedestrian_names = list(pedestrian_names or [])
        self._world_name = world_name
        self._collision_radius = ROBOT_RADIUS * 2.0
        self._obstacle_threshold = 0.5

        self.states: dict[str, RobotState] = {}
        self.goals: dict[str, GoalState] = {}
        self.laser: dict[str, LaserObservation] = {}
        self.pedestrians: dict[str, PedestrianState] = {}
        self._pedestrian_configs: dict[str, object] = {}
        self._scripted_pedestrians = False
        self._static_obstacles: list[object] | None = None
        self._odom_seen = {name: False for name in self._robot_names}
        self._pose_seen = {name: False for name in self._robot_names}
        self._scan_seen = {name: False for name in self._robot_names}
        self._odom_updates = {name: 0 for name in self._robot_names}
        self._scan_updates = {name: 0 for name in self._robot_names}
        self._ped_seen = {name: False for name in self._pedestrian_names}
        self._stats_seen = False
        self._sim_time_sec = 0.0
        self._sim_iterations = 0
        self._world_paused = False
        self._raw_odom = {
            name: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            for name in self._robot_names
        }
        self._odom_reference = {
            name: (0.0, 0.0, 0.0) for name in self._robot_names
        }
        self._world_origins = {
            name: (0.0, 0.0, 0.0) for name in self._robot_names
        }
        self._last_pose_sample: dict[str, tuple[float, float, float, float]] = {}

        for name in self._robot_names:
            self.states[name] = RobotState(name=name, x=0.0, y=0.0, yaw=0.0)
            self.goals[name] = GoalState(x=0.0, y=0.0)
            self.laser[name] = laser_scan_to_observation(
                [],
                angle_min=0.0,
                angle_increment=0.0,
                range_min=DEFAULT_RANGE_MIN,
                range_max=DEFAULT_RANGE_MAX,
            )
            self.create_subscription(
                Odometry, f"/{name}/odom", self._make_odom_cb(name), 10
            )
            self.create_subscription(
                LaserScan, f"/{name}/scan", self._make_scan_cb(name), 10
            )
        self.create_subscription(
            TFMessage, "/gazebo/dynamic_pose", self._dynamic_pose_cb, 10
        )
        if WorldStatistics is not None:
            self.create_subscription(
                WorldStatistics,
                f"/world/{world_name}/stats",
                self._stats_cb,
                10,
            )
        self._world_control_client = (
            self.create_client(ControlWorld, f"/world/{world_name}/control")
            if ControlWorld is not None else None
        )

        for name in self._pedestrian_names:
            self.pedestrians[name] = PedestrianState(
                name=name,
                x=UNSEEN_PEDESTRIAN_XY,
                y=UNSEEN_PEDESTRIAN_XY,
                vx=0.0,
                vy=0.0,
            )
            self.create_subscription(
                Odometry, f"/{name}/odom", self._make_ped_cb(name), 10
            )

        self._cmd_pubs = {
            name: self.create_publisher(Twist, f"/{name}/cmd_vel", 10)
            for name in self._robot_names
        }

    def set_start_positions(
        self, start_positions: dict[str, tuple[float, float, float]]
    ) -> None:
        if not hasattr(self, "_pose_seen"):
            self._pose_seen = {}
        if not hasattr(self, "_last_pose_sample"):
            self._last_pose_sample = {}
        for name, pose in start_positions.items():
            if name not in self._raw_odom:
                continue
            self._world_origins[name] = pose
            raw_x, raw_y, raw_yaw, _v, _vy, _omega = self._raw_odom[name]
            self._odom_reference[name] = (raw_x, raw_y, raw_yaw)
            self._last_pose_sample.pop(name, None)
            self._pose_seen[name] = False
            x, y, yaw = pose
            self.states[name] = RobotState(
                name=name,
                x=x,
                y=y,
                yaw=yaw,
                v=0.0,
                vy=0.0,
                omega=0.0,
            )

    def _update_state_from_raw(self, name: str) -> None:
        raw_x, raw_y, raw_yaw, v, vy, omega = self._raw_odom[name]
        ref_x, ref_y, ref_yaw = self._odom_reference[name]
        origin_x, origin_y, origin_yaw = self._world_origins[name]
        dx = raw_x - ref_x
        dy = raw_y - ref_y
        dyaw = _wrap_angle(raw_yaw - ref_yaw)
        cos_yaw = math.cos(origin_yaw)
        sin_yaw = math.sin(origin_yaw)
        world_x = origin_x + cos_yaw * dx - sin_yaw * dy
        world_y = origin_y + sin_yaw * dx + cos_yaw * dy
        world_yaw = _wrap_angle(origin_yaw + dyaw)
        self.states[name] = RobotState(
            name=name, x=world_x, y=world_y, yaw=world_yaw,
            v=v, vy=vy, omega=omega
        )

    def _make_odom_cb(self, name: str):
        def callback(msg: Odometry) -> None:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny, cosy)
            v = msg.twist.twist.linear.x
            vy = msg.twist.twist.linear.y
            omega = msg.twist.twist.angular.z
            self._raw_odom[name] = (p.x, p.y, yaw, v, vy, omega)
            if self._pose_seen.get(name, False):
                current = self.states.get(
                    name,
                    RobotState(name=name, x=0.0, y=0.0, yaw=0.0),
                )
                self.states[name] = RobotState(
                    name=name,
                    x=current.x,
                    y=current.y,
                    yaw=current.yaw,
                    v=v,
                    vy=vy,
                    omega=omega,
                )
            else:
                self._update_state_from_raw(name)
            self._odom_seen[name] = True
            if hasattr(self, "_odom_updates"):
                self._odom_updates[name] = self._odom_updates.get(name, 0) + 1

        return callback

    def _dynamic_pose_cb(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            name = self._robot_name_from_frame(transform.child_frame_id)
            if name is None:
                continue
            stamp_sec = _stamp_sec_from_header(getattr(transform, "header", None))
            if stamp_sec is None:
                stamp_sec = (
                    self._sim_time_sec
                    if getattr(self, "_sim_time_sec", 0.0) > 0.0
                    else self.get_clock().now().nanoseconds * 1e-9
                )
            p = transform.transform.translation
            q = transform.transform.rotation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny, cosy)

            v = 0.0
            vy = 0.0
            omega = 0.0
            previous = self._last_pose_sample.get(name)
            if previous is not None:
                prev_t, prev_x, prev_y, prev_yaw = previous
                dt = stamp_sec - prev_t
                if dt > 1e-4:
                    world_vx = (p.x - prev_x) / dt
                    world_vy = (p.y - prev_y) / dt
                    dyaw = _wrap_angle(yaw - prev_yaw)
                    v = math.cos(yaw) * world_vx + math.sin(yaw) * world_vy
                    vy = -math.sin(yaw) * world_vx + math.cos(yaw) * world_vy
                    omega = dyaw / dt
                else:
                    current = self.states.get(name)
                    if current is not None:
                        v = current.v
                        vy = current.vy
                        omega = current.omega
            self._last_pose_sample[name] = (stamp_sec, p.x, p.y, yaw)
            self.states[name] = RobotState(
                name=name, x=p.x, y=p.y, yaw=yaw, v=v, vy=vy, omega=omega
            )
            self._pose_seen[name] = True
            self._odom_seen[name] = True
            if hasattr(self, "_odom_updates"):
                self._odom_updates[name] = self._odom_updates.get(name, 0) + 1

    def _robot_name_from_frame(self, frame_id: str) -> str | None:
        if frame_id in self._robot_names:
            return frame_id
        return None

    def _make_scan_cb(self, name: str):
        def callback(msg: LaserScan) -> None:
            message_stamp_sec = _stamp_sec_from_scan(msg)
            receive_stamp_sec = self.get_clock().now().nanoseconds * 1e-9
            self.laser[name] = laser_scan_to_observation(
                msg.ranges,
                angle_min=msg.angle_min,
                angle_increment=msg.angle_increment,
                range_min=msg.range_min,
                range_max=msg.range_max,
                stamp_sec=receive_stamp_sec,
                message_stamp_sec=message_stamp_sec,
            )
            self._scan_seen[name] = True
            if hasattr(self, "_scan_updates"):
                self._scan_updates[name] = self._scan_updates.get(name, 0) + 1

        return callback

    def _stats_cb(self, msg) -> None:
        sim_time = getattr(msg, "sim_time")
        self._sim_time_sec = int(sim_time.sec) + int(sim_time.nanosec) * 1e-9
        self._sim_iterations = int(getattr(msg, "iterations", 0))
        self._world_paused = bool(getattr(msg, "paused", False))
        self._stats_seen = True

    def _make_ped_cb(self, name: str):
        def callback(msg: Odometry) -> None:
            if (
                getattr(self, "_scripted_pedestrians", False)
                and name in self._pedestrian_configs
            ):
                self._ped_seen[name] = True
                return
            p = msg.pose.pose.position
            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.linear.y
            self.pedestrians[name] = PedestrianState(
                name=name, x=p.x, y=p.y, vx=vx, vy=vy
            )
            self._ped_seen[name] = True

        return callback

    def set_pedestrian_configs(
        self,
        configs: list[object],
        scripted: bool = True,
    ) -> None:
        self._pedestrian_configs = {
            str(getattr(config, "name")): config
            for config in configs
        }
        self._scripted_pedestrians = bool(scripted)
        self.update_pedestrian_fallbacks(0.0)

    def set_static_obstacles(self, obstacles: list[object]) -> None:
        self._static_obstacles = list(obstacles)

    def update_pedestrian_fallbacks(self, elapsed_time: float) -> None:
        if not self._pedestrian_configs:
            return
        try:
            from mrpp_gazebo.pedestrian_controller import pedestrian_pose_at
        except ModuleNotFoundError:
            return
        for name, cfg in self._pedestrian_configs.items():
            if (
                not getattr(self, "_scripted_pedestrians", False)
                and self._ped_seen.get(name, False)
            ):
                continue
            x, y, _yaw, vx, vy = pedestrian_pose_at(cfg, elapsed_time)
            self.pedestrians[name] = PedestrianState(
                name=name, x=x, y=y, vx=vx, vy=vy
            )

    def reset_pedestrians(self, elapsed_time: float = 0.0) -> None:
        for name in self._ped_seen:
            self._ped_seen[name] = False
        self.update_pedestrian_fallbacks(elapsed_time)

    def set_goals(self, goals: dict[str, GoalState]) -> None:
        self.goals.update(goals)

    def get_robot_states(self) -> dict[str, RobotState]:
        return dict(self.states)

    def get_pedestrian_states(self) -> dict[str, PedestrianState]:
        return dict(self.pedestrians)

    def get_laser_data(self) -> dict[str, LaserObservation]:
        return dict(self.laser)

    def current_time_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def current_sim_time_sec(self) -> float:
        return self._sim_time_sec

    def message_counters(self) -> dict[str, dict[str, int]]:
        return {
            "odom": dict(self._odom_updates),
            "scan": dict(self._scan_updates),
        }

    def wait_until_stats_ready(self, timeout_sec: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            self.spin_once(timeout_sec=0.05)
            if self._stats_seen:
                return
        raise RuntimeError(
            "Gazebo world statistics did not become ready. "
            "Check the /world/<name>/stats bridge."
        )

    def _require_world_control(self):
        if self._world_control_client is None:
            raise RuntimeError(
                "Gazebo world control service is unavailable. "
                "Check ros_gz_interfaces and the /world/<name>/control bridge."
            )
        return self._world_control_client

    def wait_for_world_control(self, timeout_sec: float = 5.0) -> None:
        client = self._require_world_control()
        if not client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError(
                "Gazebo world control service did not become ready. "
                "Check the /world/<name>/control bridge."
            )

    def set_world_paused(self, paused: bool, timeout_sec: float = 2.0) -> None:
        self.wait_for_world_control(timeout_sec=timeout_sec)
        req = ControlWorld.Request()
        req.world_control.pause = bool(paused)
        future = self._world_control_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done() or not future.result() or not future.result().success:
            state = "pause" if paused else "resume"
            raise RuntimeError(f"Gazebo world {state} request failed.")
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            self.spin_once(timeout_sec=0.01)
            if self._world_paused == bool(paused):
                return
        state = "paused" if paused else "running"
        raise RuntimeError(f"Gazebo world did not report {state} state.")

    def pause_world(self, timeout_sec: float = 2.0) -> None:
        self.set_world_paused(True, timeout_sec=timeout_sec)

    def resume_world(self, timeout_sec: float = 2.0) -> None:
        self.set_world_paused(False, timeout_sec=timeout_sec)

    def reset_world(self, timeout_sec: float = 5.0) -> bool:
        if self._world_control_client is None:
            return False
        if not self._world_control_client.wait_for_service(timeout_sec=timeout_sec):
            return False
        req = ControlWorld.Request()
        req.world_control.reset.time_only = True
        req.world_control.pause = bool(self._world_paused)
        future = self._world_control_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done() or not future.result() or not future.result().success:
            raise RuntimeError("Gazebo world reset request failed.")
        self._last_pose_sample.clear()
        for name in self._ped_seen:
            self._ped_seen[name] = False
        self.update_pedestrian_fallbacks(0.0)
        return True

    def run_until_sim_time(self, target_time_sec: float, timeout_sec: float = 2.0) -> None:
        self.wait_for_world_control(timeout_sec=timeout_sec)
        deadline = time.monotonic() + timeout_sec
        request_accepted = False
        while time.monotonic() < deadline:
            if self._sim_time_sec + 1e-9 >= target_time_sec and self._world_paused:
                return
            req = ControlWorld.Request()
            sec = int(target_time_sec)
            nsec = int(round((target_time_sec - sec) * 1_000_000_000))
            if nsec >= 1_000_000_000:
                sec += 1
                nsec -= 1_000_000_000
            req.world_control.run_to_sim_time.sec = sec
            req.world_control.run_to_sim_time.nanosec = nsec
            future = self._world_control_client.call_async(req)
            remaining = max(0.05, min(0.5, deadline - time.monotonic()))
            rclpy.spin_until_future_complete(self, future, timeout_sec=remaining)
            if future.done() and future.result() and future.result().success:
                request_accepted = True
                break
            self.spin_once(timeout_sec=0.02)
        if not request_accepted:
            raise RuntimeError("Gazebo run_to_sim_time request failed.")

        while time.monotonic() < deadline:
            self.spin_once(timeout_sec=0.001)
            if self._sim_time_sec + 1e-9 >= target_time_sec and self._world_paused:
                return
        raise RuntimeError(
            "Gazebo did not reach requested simulation time "
            f"{target_time_sec:.3f}s."
        )

    def wait_for_message_updates(
        self,
        previous: dict[str, dict[str, int]],
        timeout_sec: float = 1.0,
        require_scan: bool = True,
    ) -> None:
        deadline = time.monotonic() + timeout_sec
        previous_odom = previous.get("odom", {})
        previous_scan = previous.get("scan", {})
        while time.monotonic() < deadline:
            self.spin_once(timeout_sec=0.001)
            odom_ready = all(
                self._odom_updates.get(name, 0) > previous_odom.get(name, -1)
                for name in self._robot_names
            )
            scan_ready = True
            if require_scan:
                scan_ready = all(
                    self._scan_updates.get(name, 0) > previous_scan.get(name, -1)
                    for name in self._robot_names
                )
            if odom_ready and scan_ready:
                return
        raise RuntimeError(
            "Gazebo bridge did not receive fresh odom/scan after fast step."
        )

    def spin_once(self, timeout_sec: float = 0.0) -> None:
        rclpy.spin_once(self, timeout_sec=timeout_sec)

    def spin_some(self, max_callbacks: int = 8, timeout_sec: float = 0.0) -> None:
        for _ in range(max(1, max_callbacks)):
            rclpy.spin_once(self, timeout_sec=timeout_sec)

    def wait_until_ready(
        self,
        timeout_sec: float = 15.0,
        require_scan: bool = True,
    ) -> None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            self.spin_once(timeout_sec=0.1)
            odom_ready = all(self._odom_seen.values())
            scan_ready = all(self._scan_seen.values()) if require_scan else True
            if odom_ready and scan_ready:
                return

        missing_odom = [
            name for name, seen in self._odom_seen.items() if not seen
        ]
        missing_scan = [
            name for name, seen in self._scan_seen.items() if not seen
        ]
        details = []
        if missing_odom:
            details.append(f"missing odom: {', '.join(missing_odom)}")
        if require_scan and missing_scan:
            details.append(f"missing scan: {', '.join(missing_scan)}")
        raise RuntimeError(
            "Gazebo bridge did not become ready within "
            f"{timeout_sec:.1f}s ({'; '.join(details)}). "
            "Check ros_gz_bridge and Gazebo topic names."
        )

    def send_commands(
        self,
        commands: dict[str, tuple[float, float] | tuple[float, float, float]],
    ) -> None:
        for name, command in commands.items():
            if name in self._cmd_pubs:
                if len(command) == 2:
                    v, omega = command
                    vy = 0.0
                else:
                    v, vy, omega = command
                msg = Twist()
                msg.linear.x = float(v)
                msg.linear.y = float(vy)
                msg.angular.z = float(omega)
                self._cmd_pubs[name].publish(msg)

    def stop_all(self) -> None:
        self.send_commands({name: (0.0, 0.0, 0.0) for name in self._robot_names})

    def reset(self, start_positions: dict[str, tuple[float, float, float]]) -> None:
        """Reset robot positions using Gazebo set_pose service."""
        self.stop_all()
        import time
        time.sleep(0.5)

        for name, (x, y, yaw) in start_positions.items():
            qw = math.cos(yaw / 2)
            qz = math.sin(yaw / 2)
            req = (
                f'name: "{name}" '
                f"position {{ x: {x} y: {y} z: 0.2 }} "
                f"orientation {{ w: {qw} z: {qz} }}"
            )
            try:
                subprocess.run(
                    [
                        "ign", "service", "-s",
                        f"/world/{self._world_name}/set_pose",
                        "--reqtype", "ignition.msgs.Pose",
                        "--reptype", "ignition.msgs.Boolean",
                        "--timeout", "3000",
                        "--req", req,
                    ],
                    capture_output=True,
                    timeout=5,
                )
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                pass
            if name in self.states:
                self.states[name] = RobotState(
                    name=name, x=x, y=y, yaw=yaw, v=0.0, vy=0.0, omega=0.0
                )
                self._odom_seen[name] = True

        time.sleep(1.0)
        for _ in range(5):
            self.spin_once(timeout_sec=0.05)
        self.set_start_positions(start_positions)
        self.reset_pedestrians(0.0)
        self.stop_all()
        self.spin_once(timeout_sec=0.0)

    def get_collisions(self) -> dict[str, bool]:
        categories = self.get_collision_categories()
        return {name: category is not None for name, category in categories.items()}

    def get_collision_categories(self) -> dict[str, str | None]:
        result: dict[str, str | None] = {}
        for name in self._robot_names:
            state = self.states[name]
            category: str | None = None

            for other_name, other_state in self.states.items():
                if other_name == name:
                    continue
                dist = math.hypot(
                    state.x - other_state.x, state.y - other_state.y
                )
                if dist < self._collision_radius:
                    category = "robot_robot"
                    break

            if category is None:
                for ped in self.pedestrians.values():
                    dist = math.hypot(state.x - ped.x, state.y - ped.y)
                    if dist < ROBOT_RADIUS + PEDESTRIAN_RADIUS:
                        category = "robot_pedestrian"
                        break

            if category is None:
                if self._static_obstacles is not None:
                    if _collides_with_static_obstacle(state, self._static_obstacles):
                        category = "robot_obstacle"
                else:
                    laser = self.laser.get(name)
                    if laser and laser.min_range < OBSTACLE_COLLISION_RANGE:
                        category = "robot_obstacle"

            result[name] = category
        return result

    def get_unsafe_distances(self, threshold: float = 0.6) -> dict[str, bool]:
        result = {}
        for name in self._robot_names:
            state = self.states[name]
            unsafe = False

            for other_name, other_state in self.states.items():
                if other_name == name:
                    continue
                dist = math.hypot(
                    state.x - other_state.x, state.y - other_state.y
                )
                if dist < threshold:
                    unsafe = True
                    break

            if not unsafe:
                for ped in self.pedestrians.values():
                    dist = math.hypot(state.x - ped.x, state.y - ped.y)
                    if dist < threshold:
                        unsafe = True
                        break

            if not unsafe:
                laser = self.laser.get(name)
                if laser and laser.min_range < threshold:
                    unsafe = True

            result[name] = unsafe
        return result

    def get_neighbor_distances(self) -> dict[str, float]:
        result = {}
        for name in self._robot_names:
            state = self.states[name]
            min_dist = float("inf")
            for other_name, other_state in self.states.items():
                if other_name == name:
                    continue
                dist = math.hypot(
                    state.x - other_state.x, state.y - other_state.y
                )
                min_dist = min(min_dist, dist)
            for ped in self.pedestrians.values():
                dist = math.hypot(state.x - ped.x, state.y - ped.y)
                min_dist = min(min_dist, dist)
            result[name] = min_dist if min_dist != float("inf") else 999.0
        return result

    def get_min_laser_range(self) -> dict[str, float]:
        return {name: ld.min_range for name, ld in self.laser.items()}


def _collides_with_static_obstacle(
    state: RobotState,
    obstacles: list[object],
) -> bool:
    return any(
        _contains_robot_center_with_radius(obstacle, state.x, state.y, ROBOT_RADIUS)
        for obstacle in obstacles
    )


def _contains_robot_center_with_radius(
    obstacle: object,
    x: float,
    y: float,
    radius: float,
) -> bool:
    obstacle_radius = getattr(obstacle, "radius", None)
    if obstacle_radius is not None:
        return (
            math.hypot(
                x - float(getattr(obstacle, "x")),
                y - float(getattr(obstacle, "y")),
            )
            <= float(obstacle_radius) + radius
        )

    sx = getattr(obstacle, "sx", None)
    sy = getattr(obstacle, "sy", None)
    if sx is None or sy is None:
        return False
    dx = x - float(getattr(obstacle, "x"))
    dy = y - float(getattr(obstacle, "y"))
    yaw = float(getattr(obstacle, "yaw", 0.0))
    cos_yaw = math.cos(-yaw)
    sin_yaw = math.sin(-yaw)
    local_x = cos_yaw * dx - sin_yaw * dy
    local_y = sin_yaw * dx + cos_yaw * dy
    half_x = float(sx) / 2.0
    half_y = float(sy) / 2.0
    outside_x = max(abs(local_x) - half_x, 0.0)
    outside_y = max(abs(local_y) - half_y, 0.0)
    return outside_x * outside_x + outside_y * outside_y <= radius * radius


def build_robot_config_yaml(
    scenario_path: Path, output_path: Path
) -> list[str]:
    from mrpp_experiments.scenario import load_scenario

    scenario = load_scenario(scenario_path)
    return build_robot_config_from_scenario(scenario, output_path)


def build_robot_config_from_scenario(scenario, output_path: Path) -> list[str]:
    lines = ["robots:"]
    names = []
    for robot in scenario.robots:
        lines.append(f"  - name: {robot.name}")
        lines.append(f"    x: {robot.start.x}")
        lines.append(f"    y: {robot.start.y}")
        lines.append(f"    yaw: {robot.start.yaw}")
        names.append(robot.name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return names


def launch_gazebo(
    world_sdf: Path,
    gui: bool = False,
    update_rate: float | None = None,
    run_on_start: bool = True,
) -> subprocess.Popen:
    if gui:
        cmd = ["ign", "gazebo", str(world_sdf)]
        gui_config = os.environ.get("MRPP_GAZEBO_GUI_CONFIG")
        if gui_config:
            cmd.extend(["--gui-config", gui_config])
        if run_on_start:
            cmd.append("-r")
        cmd.extend(["-v", "2"])
    else:
        cmd = [
            "ign",
            "gazebo",
            "-s",
            "--headless-rendering",
            str(world_sdf),
        ]
        if run_on_start:
            cmd.append("-r")
        cmd.extend(["-v", "0"])
    if update_rate is not None:
        cmd.extend(["-z", f"{update_rate:g}"])
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def spawn_robots(
    robot_config: Path, robot_model: Path, delay: float = 3.0
) -> subprocess.Popen:
    import time

    time.sleep(delay)
    return subprocess.Popen(
        [
            "ros2", "run", "mrpp_gazebo", "spawn_robots",
            str(robot_config), str(robot_model),
        ],
    )


def launch_topic_bridges(
    robot_names: list[str],
    pedestrian_names: list[str] | None = None,
    world_name: str | None = None,
) -> list[subprocess.Popen]:
    bridges = []

    if world_name:
        pose_bridge = _launch_bridge_process(
            [
                "ros2", "run", "ros_gz_bridge", "parameter_bridge",
                f"/world/{world_name}/dynamic_pose/info@tf2_msgs/msg/TFMessage"
                f"[ignition.msgs.Pose_V",
                "--ros-args", "-r",
                f"/world/{world_name}/dynamic_pose/info:=/gazebo/dynamic_pose",
            ],
        )
        bridges.append(pose_bridge)

        stats_bridge = _launch_bridge_process(
            [
                "ros2", "run", "ros_gz_bridge", "parameter_bridge",
                f"/world/{world_name}/stats@ros_gz_interfaces/msg/WorldStatistics"
                f"[ignition.msgs.WorldStatistics",
            ],
        )
        bridges.append(stats_bridge)

        control_bridge = _launch_bridge_process(
            [
                "ros2", "run", "ros_gz_bridge", "parameter_bridge",
                f"/world/{world_name}/control@ros_gz_interfaces/srv/ControlWorld",
            ],
        )
        bridges.append(control_bridge)

    for name in robot_names:
        odom_bridge = _launch_bridge_process(
            [
                "ros2", "run", "ros_gz_bridge", "parameter_bridge",
                f"/model/{name}/odom@nav_msgs/msg/Odometry"
                f"[ignition.msgs.Odometry",
                "--ros-args", "-r",
                f"/model/{name}/odom:=/{name}/odom",
            ],
        )
        bridges.append(odom_bridge)

        cmd_bridge = _launch_bridge_process(
            [
                "ros2", "run", "ros_gz_bridge", "parameter_bridge",
                f"/model/{name}/cmd_vel@geometry_msgs/msg/Twist"
                f"]ignition.msgs.Twist",
                "--ros-args", "-r",
                f"/model/{name}/cmd_vel:=/{name}/cmd_vel",
            ],
        )
        bridges.append(cmd_bridge)

        scan_bridge = _launch_bridge_process(
            [
                "ros2", "run", "ros_gz_bridge", "parameter_bridge",
                f"/model/{name}/scan@sensor_msgs/msg/LaserScan"
                f"[ignition.msgs.LaserScan",
                "--ros-args", "-r",
                f"/model/{name}/scan:=/{name}/scan",
            ],
        )
        bridges.append(scan_bridge)

    # Bridge pedestrian odom topics
    for name in (pedestrian_names or []):
        ped_odom_bridge = _launch_bridge_process(
            [
                "ros2", "run", "ros_gz_bridge", "parameter_bridge",
                f"/model/{name}/odom@nav_msgs/msg/Odometry"
                f"[ignition.msgs.Odometry",
                "--ros-args", "-r",
                f"/model/{name}/odom:=/{name}/odom",
            ],
        )
        bridges.append(ped_odom_bridge)

    return bridges


def _launch_bridge_process(command: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def stop_processes(
    processes: list[subprocess.Popen],
    timeout_sec: float = 2.0,
) -> None:
    """Stop managed wrappers and every child in their process groups."""
    for proc in processes:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (AttributeError, ProcessLookupError, PermissionError):
            if proc.poll() is None:
                try:
                    proc.terminate()
                except (AttributeError, ProcessLookupError):
                    pass
    deadline = time.monotonic() + max(timeout_sec, 0.0)
    for proc in processes:
        remaining = max(0.0, deadline - time.monotonic())
        if proc.poll() is None:
            try:
                proc.wait(timeout=remaining)
            except (AttributeError, subprocess.TimeoutExpired):
                pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (AttributeError, ProcessLookupError, PermissionError):
            if proc.poll() is None:
                try:
                    proc.kill()
                except (AttributeError, ProcessLookupError):
                    pass


def ensure_processes_running(
    processes: list[subprocess.Popen],
    label: str,
) -> None:
    failed = []
    for index, proc in enumerate(processes):
        code = proc.poll()
        if code is not None:
            failed.append(f"#{index} exit={code}")
    if failed:
        raise RuntimeError(
            f"{label} process exited early: {', '.join(failed)}"
        )
