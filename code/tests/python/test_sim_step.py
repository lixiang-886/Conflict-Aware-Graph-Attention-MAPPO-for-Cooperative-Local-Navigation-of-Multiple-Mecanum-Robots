from __future__ import annotations

from mrpp_rl.sim_step import SimulationStepper


class FakeBridge:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.sim_time = 0.0

    def pause_world(self, timeout_sec: float) -> None:
        del timeout_sec
        self.calls.append("pause")

    def resume_world(self, timeout_sec: float) -> None:
        del timeout_sec
        self.calls.append("resume")

    def wait_until_stats_ready(self, timeout_sec: float) -> None:
        self.calls.append(f"stats:{timeout_sec:.1f}")

    def wait_for_world_control(self, timeout_sec: float) -> None:
        self.calls.append(f"control:{timeout_sec:.1f}")

    def message_counters(self) -> dict[str, dict[str, int]]:
        self.calls.append("counters")
        return {"odom": {"robot_1": 0}, "scan": {"robot_1": 0}}

    def current_sim_time_sec(self) -> float:
        return self.sim_time

    def run_until_sim_time(self, target_time_sec: float, timeout_sec: float) -> None:
        del timeout_sec
        assert self.calls[-1] == "counters"
        self.calls.append("run_until")
        self.sim_time = target_time_sec

    def wait_for_message_updates(
        self,
        previous: dict[str, dict[str, int]],
        timeout_sec: float,
        require_scan: bool = True,
    ) -> None:
        del previous, timeout_sec, require_scan
        self.calls.append("wait_updates")

    def update_pedestrian_fallbacks(self, elapsed_time_sec: float) -> None:
        self.calls.append(f"pedestrians:{elapsed_time_sec:.1f}")


def test_fast_synchronize_pauses_before_run_to_time() -> None:
    bridge = FakeBridge()
    stepper = SimulationStepper(bridge, dt=0.1, mode="fast")

    stepper.synchronize_after_reset(resume=False)

    assert bridge.calls == ["pause", "counters", "run_until", "wait_updates"]


def test_fast_initialize_uses_startup_timeout() -> None:
    bridge = FakeBridge()
    stepper = SimulationStepper(
        bridge,
        dt=0.1,
        mode="fast",
        fast_startup_timeout_sec=12.0,
    )

    stepper.initialize()

    assert bridge.calls == ["stats:12.0", "control:12.0"]


def test_fast_synchronize_resumes_when_requested() -> None:
    bridge = FakeBridge()
    stepper = SimulationStepper(bridge, dt=0.1, mode="fast")

    stepper.synchronize_after_reset(resume=True)

    assert bridge.calls == [
        "pause",
        "counters",
        "run_until",
        "wait_updates",
        "resume",
    ]
