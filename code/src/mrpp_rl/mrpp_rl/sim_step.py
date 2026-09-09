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

import time
from typing import Literal


SimulationStepMode = Literal["realtime", "fast"]


class SimulationStepper:
    def __init__(
        self,
        bridge,
        dt: float,
        mode: SimulationStepMode = "realtime",
        fast_timeout_sec: float = 2.0,
        fast_startup_timeout_sec: float = 20.0,
    ) -> None:
        if mode not in ("realtime", "fast"):
            raise ValueError(f"Unsupported simulation step mode: {mode}")
        self.bridge = bridge
        self.dt = float(dt)
        self.mode = mode
        self.fast_timeout_sec = float(fast_timeout_sec)
        self.fast_startup_timeout_sec = float(fast_startup_timeout_sec)

    def initialize(self) -> None:
        if self.mode != "fast":
            return
        self.bridge.wait_until_stats_ready(timeout_sec=self.fast_startup_timeout_sec)
        self.bridge.wait_for_world_control(timeout_sec=self.fast_startup_timeout_sec)

    def before_reset(self) -> None:
        if self.mode != "fast":
            return
        self.bridge.pause_world(timeout_sec=self.fast_timeout_sec)

    def resume(self) -> None:
        if self.mode != "fast":
            return
        self.bridge.resume_world(timeout_sec=self.fast_timeout_sec)

    def synchronize_after_reset(self, resume: bool = True) -> None:
        if self.mode != "fast":
            return
        self.bridge.pause_world(timeout_sec=self.fast_timeout_sec)
        previous = self.bridge.message_counters()
        target_time = self.bridge.current_sim_time_sec() + self.dt
        self.bridge.run_until_sim_time(target_time, timeout_sec=self.fast_timeout_sec)
        self.bridge.wait_for_message_updates(
            previous,
            timeout_sec=self.fast_timeout_sec,
            require_scan=True,
        )
        if resume:
            self.bridge.resume_world(timeout_sec=self.fast_timeout_sec)

    def advance(self, elapsed_time_sec: float) -> None:
        if self.mode == "realtime":
            time.sleep(self.dt)
            self.bridge.update_pedestrian_fallbacks(elapsed_time_sec + self.dt)
            self.bridge.spin_once(timeout_sec=0.0)
            return

        previous = self.bridge.message_counters()
        target_time = self.bridge.current_sim_time_sec() + self.dt
        deadline = time.monotonic() + self.fast_timeout_sec
        while time.monotonic() < deadline:
            self.bridge.spin_once(timeout_sec=0.001)
            if self.bridge.current_sim_time_sec() + 1e-9 >= target_time:
                break
        else:
            raise RuntimeError(
                "Gazebo fast simulation did not advance to the next step."
            )
        self.bridge.update_pedestrian_fallbacks(elapsed_time_sec + self.dt)
        self.bridge.wait_for_message_updates(
            previous,
            timeout_sec=self.fast_timeout_sec,
            require_scan=True,
        )
