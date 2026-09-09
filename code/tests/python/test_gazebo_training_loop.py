from __future__ import annotations

import importlib
import math
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch


class _Scalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _Tensor:
    def __init__(self, data):
        self.data = data
        self.shape = self._shape(data)

    def _shape(self, data):
        if isinstance(data, list) and data:
            return (len(data),) + self._shape(data[0])
        if isinstance(data, list):
            return (0,)
        return ()

    def unsqueeze(self, dim):
        if dim != 0:
            raise NotImplementedError
        return _Tensor([self.data])

    def squeeze(self, dim):
        if dim != 0:
            raise NotImplementedError
        if isinstance(self.data, list) and len(self.data) == 1:
            return _Tensor(self.data[0])
        return self

    def __getitem__(self, key):
        value = self.data
        if isinstance(key, tuple):
            for item in key:
                value = value[item]
        else:
            value = value[key]
        if isinstance(value, list):
            return _Tensor(value)
        return _Scalar(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, device):
        return self


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _install_torch_stub(monkeypatch):
    torch = types.ModuleType("torch")
    torch.float32 = object()
    torch.tensor = lambda data, dtype=None: _Tensor(list(data))
    torch.stack = lambda tensors: _Tensor([tensor.data for tensor in tensors])
    torch.nan_to_num = lambda tensor, nan=0.0: tensor

    def ones(*shape):
        if len(shape) == 2:
            rows, cols = shape
            return _Tensor([[1.0 for _ in range(cols)] for _ in range(rows)])
        if len(shape) == 3:
            batch, rows, cols = shape
            return _Tensor(
                [
                    [[1.0 for _ in range(cols)] for _ in range(rows)]
                    for _ in range(batch)
                ]
            )
        raise NotImplementedError

    def zeros(*shape):
        if len(shape) == 2:
            rows, cols = shape
            return _Tensor([[0.0 for _ in range(cols)] for _ in range(rows)])
        if len(shape) == 3:
            batch, rows, cols = shape
            return _Tensor(
                [
                    [[0.0 for _ in range(cols)] for _ in range(rows)]
                    for _ in range(batch)
                ]
            )
        raise NotImplementedError

    torch.ones = ones
    torch.zeros = zeros
    torch.no_grad = lambda: _NoGrad()
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


class _MonkeyPatch:
    def __init__(self):
        self._patches = []

    def setattr(self, target, name, value):
        patcher = patch.object(target, name, value)
        patcher.start()
        self._patches.append(patcher)

    def setitem(self, mapping, key, value):
        patcher = patch.dict(mapping, {key: value})
        patcher.start()
        self._patches.append(patcher)

    def delitem(self, mapping, key, raising=True):
        if key not in mapping:
            if raising:
                raise KeyError(key)
            return
        original = mapping[key]

        class _DeletePatch:
            def start(self_inner):
                del mapping[key]

            def stop(self_inner):
                mapping[key] = original

        patcher = _DeletePatch()
        patcher.start()
        self._patches.append(patcher)

    def undo(self):
        for patcher in reversed(self._patches):
            patcher.stop()


def _install_ros_stubs(monkeypatch):
    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda: None
    rclpy.shutdown = lambda: None
    rclpy.spin_once = lambda node, timeout_sec=0.0: None

    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = object

    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msgs_msg.Twist = type("Twist", (), {})

    nav_msgs = types.ModuleType("nav_msgs")
    nav_msgs_msg = types.ModuleType("nav_msgs.msg")
    nav_msgs_msg.Odometry = type("Odometry", (), {})

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.LaserScan = type("LaserScan", (), {})

    tf2_msgs = types.ModuleType("tf2_msgs")
    tf2_msgs_msg = types.ModuleType("tf2_msgs.msg")
    tf2_msgs_msg.TFMessage = type("TFMessage", (), {})

    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.node", rclpy_node)
    monkeypatch.setitem(sys.modules, "geometry_msgs", geometry_msgs)
    monkeypatch.setitem(sys.modules, "geometry_msgs.msg", geometry_msgs_msg)
    monkeypatch.setitem(sys.modules, "nav_msgs", nav_msgs)
    monkeypatch.setitem(sys.modules, "nav_msgs.msg", nav_msgs_msg)
    monkeypatch.setitem(sys.modules, "sensor_msgs", sensor_msgs)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_msgs_msg)
    monkeypatch.setitem(sys.modules, "tf2_msgs", tf2_msgs)
    monkeypatch.setitem(sys.modules, "tf2_msgs.msg", tf2_msgs_msg)


def _safe_laser_data(robot_names):
    lidar = importlib.import_module("mrpp_rl.lidar")
    return {
        name: lidar.laser_scan_to_observation(
            [4.5],
            angle_min=0.0,
            angle_increment=1.0,
            range_min=0.1,
            range_max=4.5,
        )
        for name in robot_names
    }


def _install_mappo_stub(monkeypatch):
    mappo = types.ModuleType("mrpp_rl.mappo")
    mappo.MAPPOConfig = type("MAPPOConfig", (), {})
    mappo.MAPPOTrainer = type("MAPPOTrainer", (), {})
    mappo.MOGATMAPPOPolicy = type("MOGATMAPPOPolicy", (), {})
    monkeypatch.setitem(sys.modules, "mrpp_rl.mappo", mappo)
    maddpg = types.ModuleType("mrpp_rl.maddpg")
    maddpg.MADDPGConfig = type("MADDPGConfig", (), {})
    maddpg.MADDPGPolicy = type("MADDPGPolicy", (), {})
    maddpg.MADDPGTrainer = type("MADDPGTrainer", (), {})
    monkeypatch.setitem(sys.modules, "mrpp_rl.maddpg", maddpg)


class _ValidationMetrics:
    def __init__(
        self,
        *,
        success_count=4,
        collision_count=0,
        timeout_failure_count=0,
        deadlock_failure_count=0,
        robot_pedestrian_near_miss_count=0,
        robot_robot_near_miss_count=0,
        min_robot_pedestrian_distance=1.0,
        min_robot_robot_distance=0.8,
        waiting_time=1.0,
        makespan=35.0,
        path_length=13.0,
        path_efficiency=0.9,
        completion_time=33.0,
        deadlock_count=0,
    ):
        self.successes = {
            f"robot_{index}": index <= success_count for index in range(1, 5)
        }
        self.collision_count = collision_count
        self.timeout_failure_count = timeout_failure_count
        self.deadlock_failure_count = deadlock_failure_count
        self.robot_pedestrian_near_miss_count = robot_pedestrian_near_miss_count
        self.robot_robot_near_miss_count = robot_robot_near_miss_count
        self.min_robot_pedestrian_distance = min_robot_pedestrian_distance
        self.min_robot_robot_distance = min_robot_robot_distance
        self.deadlock_count = deadlock_count
        self._waiting_time = waiting_time
        self._makespan = makespan
        self._path_length = path_length
        self._path_efficiency = path_efficiency
        self._completion_time = completion_time

    def success_rate(self):
        return sum(1 for value in self.successes.values() if value) / len(
            self.successes
        )

    def average_path_length(self):
        return self._path_length

    def average_path_efficiency(self):
        return self._path_efficiency

    def average_completion_time(self):
        return self._completion_time

    def makespan(self):
        return self._makespan

    def average_waiting_time(self):
        return self._waiting_time


def _import_train_with_stubs(monkeypatch):
    _install_torch_stub(monkeypatch)
    _install_ros_stubs(monkeypatch)
    _install_mappo_stub(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.train", raising=False)
    return importlib.import_module("mrpp_rl.train")


def test_validation_selection_prefers_safer_checkpoint_before_faster_one():
    monkeypatch = _MonkeyPatch()
    try:
        train = _import_train_with_stubs(monkeypatch)

        safer = _ValidationMetrics(
            robot_pedestrian_near_miss_count=0,
            min_robot_pedestrian_distance=1.0,
            waiting_time=1.5,
            makespan=34.0,
            path_length=13.5,
        )
        faster_but_close_to_pedestrians = _ValidationMetrics(
            robot_pedestrian_near_miss_count=8,
            min_robot_pedestrian_distance=0.63,
            waiting_time=0.5,
            makespan=31.0,
            path_length=12.8,
        )

        assert train._validation_selection_score(safer, 4) > train._validation_selection_score(
            faster_but_close_to_pedestrians,
            4,
        )
    finally:
        monkeypatch.undo()


def test_validation_selection_prefers_faster_safe_checkpoint_before_extra_clearance():
    monkeypatch = _MonkeyPatch()
    try:
        train = _import_train_with_stubs(monkeypatch)

        efficient = _ValidationMetrics(
            min_robot_pedestrian_distance=0.78,
            min_robot_robot_distance=0.70,
            completion_time=31.0,
            makespan=33.0,
            waiting_time=0.5,
            path_length=13.0,
        )
        overly_conservative = _ValidationMetrics(
            min_robot_pedestrian_distance=1.25,
            min_robot_robot_distance=0.95,
            completion_time=34.0,
            makespan=36.0,
            waiting_time=2.0,
            path_length=13.8,
        )

        assert train._validation_selection_score(
            efficient, 4
        ) > train._validation_selection_score(overly_conservative, 4)
    finally:
        monkeypatch.undo()


def test_rollout_selection_prefers_faster_shorter_checkpoint_when_safety_ties():
    monkeypatch = _MonkeyPatch()
    try:
        train = _import_train_with_stubs(monkeypatch)

        slow = train._rollout_selection_score(
            4,
            0,
            0,
            0,
            {
                "average_waiting_time": 3.0,
                "average_completion_time": 40.0,
                "average_path_length": 16.0,
            },
        )
        fast = train._rollout_selection_score(
            4,
            0,
            0,
            0,
            {
                "average_waiting_time": 1.0,
                "average_completion_time": 32.0,
                "average_path_length": 14.0,
            },
        )

        assert fast > slow
    finally:
        monkeypatch.undo()


def test_rollout_selection_keeps_success_and_collision_before_speed():
    monkeypatch = _MonkeyPatch()
    try:
        train = _import_train_with_stubs(monkeypatch)

        complete_but_slow = train._rollout_selection_score(
            4,
            0,
            0,
            0,
            {
                "average_waiting_time": 4.0,
                "average_completion_time": 50.0,
                "average_path_length": 20.0,
            },
        )
        incomplete_fast = train._rollout_selection_score(
            3,
            0,
            1,
            0,
            {
                "average_waiting_time": 0.0,
                "average_completion_time": 20.0,
                "average_path_length": 10.0,
            },
        )
        safe_but_slow = train._rollout_selection_score(
            4,
            0,
            0,
            0,
            {
                "average_waiting_time": 4.0,
                "average_completion_time": 50.0,
                "average_path_length": 20.0,
            },
        )
        collided_fast = train._rollout_selection_score(
            4,
            1,
            0,
            0,
            {
                "average_waiting_time": 0.0,
                "average_completion_time": 20.0,
                "average_path_length": 10.0,
            },
        )

        assert complete_but_slow > incomplete_fast
        assert safe_but_slow > collided_fast
    finally:
        monkeypatch.undo()


def test_selection_summary_records_safety_and_efficiency_fields():
    monkeypatch = _MonkeyPatch()
    try:
        train = _import_train_with_stubs(monkeypatch)
        metrics = _ValidationMetrics(
            robot_pedestrian_near_miss_count=2,
            robot_robot_near_miss_count=1,
            min_robot_pedestrian_distance=0.72,
            min_robot_robot_distance=0.58,
            waiting_time=4.0,
            makespan=40.0,
            path_length=14.0,
            path_efficiency=0.84,
            completion_time=38.0,
        )

        summary = train._selection_summary(metrics, 4, "unit")

        assert summary["source"] == "unit"
        assert summary["success_count"] == 4
        assert summary["robot_pedestrian_near_miss_count"] == 2
        assert summary["robot_robot_near_miss_count"] == 1
        assert summary["min_robot_pedestrian_distance"] == 0.72
        assert summary["average_waiting_time"] == 4.0
        assert summary["average_path_efficiency"] == 0.84
    finally:
        monkeypatch.undo()


def test_train_episode_pumps_ros_callbacks_each_step():
    monkeypatch = _MonkeyPatch()
    torch = _install_torch_stub(monkeypatch)
    _install_ros_stubs(monkeypatch)
    _install_mappo_stub(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.train", raising=False)
    train = importlib.import_module("mrpp_rl.train")
    environment = importlib.import_module("mrpp_rl.environment")
    monkeypatch.setattr(train.time, "sleep", lambda _: None)

    robot_name = "robot_1"
    state = environment.RobotState(
        name=robot_name,
        x=0.0,
        y=0.0,
        yaw=0.0,
        v=0.0,
        omega=0.0,
    )

    class FakeBridge:
        def __init__(self):
            self.spin_count = 0

        def spin_once(self, timeout_sec=0.0):
            self.spin_count += 1

        def update_pedestrian_fallbacks(self, elapsed_time):
            self.pedestrian_time = elapsed_time

        def get_robot_states(self):
            return {robot_name: state}

        def get_neighbor_distances(self):
            return {robot_name: 999.0}

        def get_laser_data(self):
            return _safe_laser_data([robot_name])

        def get_pedestrian_states(self):
            return {}

        def send_commands(self, commands):
            self.commands = commands

        def get_collisions(self):
            return {robot_name: False}

        def get_unsafe_distances(self):
            return {robot_name: False}

        def stop_all(self):
            pass

    class FakePolicy:
        def get_action_and_value(self, obs, adjacency, action=None):
            del adjacency, action
            batch, robots = obs.shape[0], obs.shape[1]
            return (
                torch.ones(batch, robots, 3),
                torch.zeros(batch, robots),
                torch.zeros(batch, robots),
                torch.zeros(batch, robots),
            )

    class FakeBuffer:
        def __init__(self):
            self.observations = []

        def add(self, obs, action, reward, done, log_prob, value, adjacency=None):
            del action, reward, done, log_prob, value, adjacency
            self.observations.append(obs)

    class FakeTrainer:
        def __init__(self):
            self.config = SimpleNamespace(rollout_steps=100)
            self.policy = FakePolicy()
            self.buffer = FakeBuffer()

        def update(self, last_obs, last_adj):
            del last_obs, last_adj
            self.buffer.observations.clear()
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    bridge = FakeBridge()
    goals = {robot_name: train.GoalState(x=1.0, y=0.0)}
    scenario = SimpleNamespace(max_steps=1, dt=0.1)

    try:
        train.train_episode(
            bridge=bridge,
            trainer=FakeTrainer(),
            scenario=scenario,
            robot_names=[robot_name],
            goals=goals,
        )
    finally:
        monkeypatch.undo()

    assert bridge.spin_count > 0


def test_train_episode_marks_collided_robot_failed_and_stops_it_next_step():
    monkeypatch = _MonkeyPatch()
    torch = _install_torch_stub(monkeypatch)
    _install_ros_stubs(monkeypatch)
    _install_mappo_stub(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.train", raising=False)
    train = importlib.import_module("mrpp_rl.train")
    environment = importlib.import_module("mrpp_rl.environment")
    monkeypatch.setattr(train.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        train,
        "residual_avoidance_command",
        lambda *args, **kwargs: (0.2, 0.1, 0.0),
    )

    robot_names = ["robot_1", "robot_2"]
    states = {
        "robot_1": environment.RobotState(
            name="robot_1", x=0.0, y=0.0, yaw=0.0, v=0.2, omega=0.1
        ),
        "robot_2": environment.RobotState(
            name="robot_2", x=2.0, y=0.0, yaw=0.0, v=0.2, omega=0.1
        ),
    }

    class FakeBridge:
        def __init__(self):
            self.commands = []
            self.stopped = False

        def spin_once(self, timeout_sec=0.0):
            pass

        def update_pedestrian_fallbacks(self, elapsed_time):
            self.pedestrian_time = elapsed_time

        def get_robot_states(self):
            return states

        def get_neighbor_distances(self):
            return {name: 999.0 for name in robot_names}

        def get_laser_data(self):
            return _safe_laser_data(robot_names)

        def get_pedestrian_states(self):
            return {}

        def send_commands(self, commands):
            self.commands.append(dict(commands))

        def get_collisions(self):
            return {"robot_1": True, "robot_2": False}

        def get_unsafe_distances(self):
            return {name: False for name in robot_names}

        def stop_all(self):
            self.stopped = True

    class FakePolicy:
        def get_action_and_value(self, obs, adjacency, action=None):
            del adjacency, action
            batch, robots = obs.shape[0], obs.shape[1]
            return (
                torch.ones(batch, robots, 3),
                torch.zeros(batch, robots),
                torch.zeros(batch, robots),
                torch.zeros(batch, robots),
            )

    class FakeBuffer:
        def __init__(self):
            self.observations = []

        def add(self, obs, action, reward, done, log_prob, value, adjacency=None):
            del action, reward, done, log_prob, value, adjacency
            self.observations.append(obs)

    class FakeTrainer:
        def __init__(self):
            self.config = SimpleNamespace(rollout_steps=100)
            self.policy = FakePolicy()
            self.buffer = FakeBuffer()

        def update(self, last_obs, last_adj):
            del last_obs, last_adj
            self.buffer.observations.clear()
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    bridge = FakeBridge()
    goals = {
        "robot_1": train.GoalState(x=10.0, y=0.0),
        "robot_2": train.GoalState(x=10.0, y=0.0),
    }
    scenario = SimpleNamespace(max_steps=2, dt=0.1)

    try:
        collisions, _deadlocks, reached, failed = train.train_episode(
            bridge=bridge,
            trainer=FakeTrainer(),
            scenario=scenario,
            robot_names=robot_names,
            goals=goals,
        )
    finally:
        monkeypatch.undo()

    assert collisions["robot_1"] == 1
    assert failed["robot_1"]
    assert not reached["robot_1"]
    assert bridge.commands[1]["robot_1"] == (0.0, 0.0, 0.0)
    assert bridge.commands[1]["robot_2"] == (0.2, 0.1, 0.0)
    assert bridge.stopped


def test_train_episode_rewards_progress_toward_active_waypoint():
    monkeypatch = _MonkeyPatch()
    torch = _install_torch_stub(monkeypatch)
    _install_ros_stubs(monkeypatch)
    _install_mappo_stub(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.train", raising=False)
    train = importlib.import_module("mrpp_rl.train")
    environment = importlib.import_module("mrpp_rl.environment")
    monkeypatch.setattr(train.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        train,
        "residual_avoidance_command",
        lambda *args, **kwargs: (0.0, 0.2, 0.0),
    )

    robot_name = "robot_1"
    before = environment.RobotState(
        name=robot_name,
        x=0.0,
        y=0.0,
        yaw=0.0,
        v=0.2,
        vy=0.0,
        omega=0.0,
    )
    after = environment.RobotState(
        name=robot_name,
        x=0.0,
        y=0.4,
        yaw=0.0,
        v=0.2,
        vy=0.0,
        omega=0.0,
    )

    class FakeBridge:
        def __init__(self):
            self.command_sent = False
            self.stopped = False

        def spin_once(self, timeout_sec=0.0):
            pass

        def update_pedestrian_fallbacks(self, elapsed_time):
            self.pedestrian_time = elapsed_time

        def get_robot_states(self):
            return {robot_name: after if self.command_sent else before}

        def get_neighbor_distances(self):
            return {robot_name: 999.0}

        def get_laser_data(self):
            return _safe_laser_data([robot_name])

        def get_pedestrian_states(self):
            return {}

        def send_commands(self, commands):
            self.commands = commands
            self.command_sent = True

        def get_collisions(self):
            return {robot_name: False}

        def get_unsafe_distances(self):
            return {robot_name: False}

        def stop_all(self):
            self.stopped = True

    class FakePolicy:
        def get_action_and_value(self, obs, adjacency, action=None):
            del adjacency, action
            batch, robots = obs.shape[0], obs.shape[1]
            return (
                torch.ones(batch, robots, 3),
                torch.zeros(batch, robots),
                torch.zeros(batch, robots),
                torch.zeros(batch, robots),
            )

    class FakeBuffer:
        def __init__(self):
            self.observations = []
            self.rewards = []

        def add(self, obs, action, reward, done, log_prob, value, adjacency=None):
            del action, done, log_prob, value, adjacency
            self.observations.append(obs)
            self.rewards.append(reward)

    class FakeTrainer:
        def __init__(self):
            self.config = SimpleNamespace(rollout_steps=100)
            self.policy = FakePolicy()
            self.buffer = FakeBuffer()

        def update(self, last_obs, last_adj):
            del last_obs, last_adj
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    class FakeWaypointManager:
        def current_goals(self, states, final_goals):
            del states, final_goals
            return {robot_name: train.GoalState(x=0.0, y=1.0)}

    trainer = FakeTrainer()
    goals = {robot_name: train.GoalState(x=10.0, y=0.0)}
    scenario = SimpleNamespace(max_steps=1, dt=0.1)

    try:
        train.train_episode(
            bridge=FakeBridge(),
            trainer=trainer,
            scenario=scenario,
            robot_names=[robot_name],
            goals=goals,
            waypoint_manager=FakeWaypointManager(),
        )
    finally:
        monkeypatch.undo()

    assert trainer.buffer.rewards[0].data[0] > 0.0


def test_debug_max_steps_override_preserves_scenario_fields():
    monkeypatch = _MonkeyPatch()
    _install_torch_stub(monkeypatch)
    _install_ros_stubs(monkeypatch)
    _install_mappo_stub(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.train", raising=False)
    monkeypatch.delitem(sys.modules, "mrpp_rl.evaluate", raising=False)
    train = importlib.import_module("mrpp_rl.train")
    evaluate = importlib.import_module("mrpp_rl.evaluate")
    scenario_mod = importlib.import_module("mrpp_experiments.scenario")

    scenario = scenario_mod.Scenario(
        name="debug",
        world="static_clutter",
        robots=(),
        max_steps=1200,
        dt=0.1,
    )

    try:
        train_debug = train._override_max_steps(scenario, 25)
        eval_debug = evaluate._override_max_steps(scenario, 40)
    finally:
        monkeypatch.undo()

    assert train._override_max_steps(scenario, None) is scenario
    assert train_debug.name == "debug"
    assert train_debug.max_steps == 25
    assert eval_debug.world == "static_clutter"
    assert eval_debug.max_steps == 40
    assert scenario.max_steps == 1200


def test_training_log_schema_contains_paper_fields():
    monkeypatch = _MonkeyPatch()
    _install_torch_stub(monkeypatch)
    _install_ros_stubs(monkeypatch)
    _install_mappo_stub(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.train", raising=False)
    train = importlib.import_module("mrpp_rl.train")

    try:
        fields = train.TRAIN_LOG_FIELDNAMES
    finally:
        monkeypatch.undo()

    assert "episode_reward" in fields
    assert "success_rate" in fields
    assert "rollout_success_rate" in fields
    assert "rollout_collision_count" in fields
    assert "eval_success_rate" in fields
    assert "eval_collision_count" in fields
    assert "selection_source" in fields
    assert "policy_loss" in fields
    assert "checkpoint_path" in fields


def test_resolve_device_auto_uses_cuda_when_available():
    monkeypatch = _MonkeyPatch()
    torch = _install_torch_stub(monkeypatch)
    torch.cuda = type(
        "Cuda",
        (),
        {"is_available": staticmethod(lambda: True)},
    )
    _install_ros_stubs(monkeypatch)
    _install_mappo_stub(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.train", raising=False)
    train = importlib.import_module("mrpp_rl.train")

    try:
        device = train._resolve_device("auto")
    finally:
        monkeypatch.undo()

    assert device == "cuda"


def test_evaluate_resolve_device_auto_uses_cuda_when_available():
    monkeypatch = _MonkeyPatch()
    torch = _install_torch_stub(monkeypatch)
    torch.cuda = type(
        "Cuda",
        (),
        {"is_available": staticmethod(lambda: True)},
    )
    _install_ros_stubs(monkeypatch)
    _install_mappo_stub(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.evaluate", raising=False)
    evaluate = importlib.import_module("mrpp_rl.evaluate")

    try:
        device = evaluate._resolve_device("auto")
    finally:
        monkeypatch.undo()

    assert device == "cuda"


def test_load_training_checkpoint_restores_policy_and_optimizer():
    monkeypatch = _MonkeyPatch()
    torch = _install_torch_stub(monkeypatch)
    _install_ros_stubs(monkeypatch)
    _install_mappo_stub(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.train", raising=False)
    train = importlib.import_module("mrpp_rl.train")
    algorithm_config = importlib.import_module("mrpp_rl.algorithm_config")

    checkpoint = {
        "episode": 12,
        "policy_state_dict": {"weight": 1},
        "optimizer_state_dict": {"step": 3},
        "algorithm_config": algorithm_config.get_algorithm_config("mo_gat_mappo").to_dict(),
        **train.checkpoint_observation_fields(),
    }
    torch.load = lambda path, map_location=None: checkpoint

    class FakePolicy:
        def __init__(self):
            self.loaded = None

        def load_state_dict(self, state):
            self.loaded = state

    class FakeOptimizer:
        def __init__(self):
            self.loaded = None

        def load_state_dict(self, state):
            self.loaded = state

    class FakeTrainer:
        def __init__(self):
            self.device = "cpu"
            self.optimizer = FakeOptimizer()

    policy = FakePolicy()
    trainer = FakeTrainer()

    try:
        loaded = train._load_training_checkpoint(
            checkpoint_path=train.Path("checkpoint.pt"),
            algorithm="mo_gat_mappo",
            policy=policy,
            trainer=trainer,
        )
    finally:
        monkeypatch.undo()

    assert loaded is checkpoint
    assert policy.loaded == {"weight": 1}
    assert trainer.optimizer.loaded == {"step": 3}


def test_load_training_checkpoint_rejects_cross_scenario_resume():
    monkeypatch = _MonkeyPatch()
    torch = _install_torch_stub(monkeypatch)
    _install_ros_stubs(monkeypatch)
    _install_mappo_stub(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.train", raising=False)
    train = importlib.import_module("mrpp_rl.train")
    algorithm_config = importlib.import_module("mrpp_rl.algorithm_config")

    checkpoint = {
        "episode": 12,
        "scenario": "static_clutter",
        "policy_state_dict": {"weight": 1},
        "optimizer_state_dict": {"step": 3},
        "algorithm_config": algorithm_config.get_algorithm_config(
            "mo_gat_mappo"
        ).to_dict(),
        **train.checkpoint_observation_fields(),
    }
    torch.load = lambda path, map_location=None: checkpoint

    class FakePolicy:
        def load_state_dict(self, state):
            raise AssertionError("policy should not load mismatched checkpoint")

    class FakeTrainer:
        device = "cpu"

    try:
        try:
            train._load_training_checkpoint(
                checkpoint_path=train.Path("checkpoint.pt"),
                algorithm="mo_gat_mappo",
                policy=FakePolicy(),
                trainer=FakeTrainer(),
                scenario_name="pedestrian_dynamic",
            )
        except RuntimeError as exc:
            assert "Cross-scenario resume is disabled" in str(exc)
        else:
            raise AssertionError("expected cross-scenario checkpoint rejection")
    finally:
        monkeypatch.undo()


def test_evaluate_build_controller_dispatches_baselines_without_checkpoint():
    monkeypatch = _MonkeyPatch()
    _install_torch_stub(monkeypatch)
    _install_ros_stubs(monkeypatch)
    _install_mappo_stub(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.evaluate", raising=False)
    evaluate = importlib.import_module("mrpp_rl.evaluate")

    try:
        astar = evaluate.build_controller("priority_astar")
        orca = evaluate.build_controller("orca", world_name="static_clutter")
        shared_profile = importlib.import_module(
            "mrpp_rl.control"
        ).control_profile_for_algorithm(
            "sensors_mo_gat_mappo",
            "static_clutter",
        )
        try:
            evaluate.build_controller("mo_gat_mappo")
        except ValueError as exc:
            missing_checkpoint = str(exc)
        else:
            missing_checkpoint = ""
    finally:
        monkeypatch.undo()

    assert astar.algorithm_name == "priority_astar"
    assert orca.algorithm_name == "orca"
    assert orca.control_profile == shared_profile
    assert "--checkpoint is required" in missing_checkpoint


def test_evaluate_build_controller_loads_maddpg_policy_checkpoint():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return

    monkeypatch = _MonkeyPatch()
    _install_ros_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.evaluate", raising=False)
    evaluate = importlib.import_module("mrpp_rl.evaluate")
    algorithm_config = importlib.import_module("mrpp_rl.algorithm_config")
    observation_schema = importlib.import_module("mrpp_rl.observation_schema")
    maddpg = importlib.import_module("mrpp_rl.maddpg")

    policy = maddpg.MADDPGPolicy(node_dim=16, hidden_dim=8)
    checkpoint = {
        "policy_state_dict": policy.state_dict(),
        "algorithm_config": algorithm_config.get_algorithm_config("maddpg").to_dict(),
        **observation_schema.checkpoint_observation_fields(),
    }

    try:
        controller = evaluate.build_controller(
            "maddpg",
            checkpoint=checkpoint,
            hidden_dim=8,
        )
    finally:
        monkeypatch.undo()

    assert controller.algorithm_name == "maddpg"
    assert isinstance(controller.policy, maddpg.MADDPGPolicy)


def test_evaluate_episode_stops_when_all_robots_reach_goals():
    monkeypatch = _MonkeyPatch()
    torch = _install_torch_stub(monkeypatch)
    _install_ros_stubs(monkeypatch)
    _install_mappo_stub(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.evaluate", raising=False)
    evaluate = importlib.import_module("mrpp_rl.evaluate")
    environment = importlib.import_module("mrpp_rl.environment")
    sleep_calls = []
    monkeypatch.setattr(evaluate.time, "sleep", lambda dt: sleep_calls.append(dt))

    robot_name = "robot_1"
    state = environment.RobotState(
        name=robot_name,
        x=1.0,
        y=0.0,
        yaw=0.0,
        v=0.0,
        omega=0.0,
    )

    class FakeBridge:
        def __init__(self):
            self.commands = []
            self.stopped = False

        def update_pedestrian_fallbacks(self, elapsed_time):
            self.pedestrian_time = elapsed_time

        def spin_once(self, timeout_sec=0.0):
            pass

        def get_robot_states(self):
            return {robot_name: state}

        def get_laser_data(self):
            return _safe_laser_data([robot_name])

        def get_pedestrian_states(self):
            return {}

        def get_collisions(self):
            return {robot_name: False}

        def send_commands(self, commands):
            self.commands.append(commands)

        def stop_all(self):
            self.stopped = True

    class FakePolicy:
        def get_action_and_value(
            self,
            obs,
            adjacency,
            action=None,
            deterministic=False,
        ):
            del adjacency, action, deterministic
            batch, robots = obs.shape[0], obs.shape[1]
            return (
                torch.zeros(batch, robots, 3),
                torch.zeros(batch, robots),
                torch.zeros(batch, robots),
                torch.zeros(batch, robots),
            )

    bridge = FakeBridge()
    goals = {robot_name: evaluate.GoalState(x=1.0, y=0.0)}
    scenario = SimpleNamespace(name="unit", max_steps=10, dt=0.1)

    try:
        metrics = evaluate.evaluate_episode(
            bridge=bridge,
            policy=FakePolicy(),
            scenario=scenario,
            robot_names=[robot_name],
            goals=goals,
            seed=0,
        )
    finally:
        monkeypatch.undo()

    assert len(sleep_calls) == 1
    assert len(bridge.commands) == 1
    assert bridge.stopped
    assert metrics.success_rate() == 1.0
    assert metrics.completion_times[robot_name] == 0.1


def test_evaluate_episode_marks_collision_failed_and_ends_episode():
    monkeypatch = _MonkeyPatch()
    torch = _install_torch_stub(monkeypatch)
    _install_ros_stubs(monkeypatch)
    _install_mappo_stub(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.evaluate", raising=False)
    evaluate = importlib.import_module("mrpp_rl.evaluate")
    environment = importlib.import_module("mrpp_rl.environment")
    sleep_calls = []
    monkeypatch.setattr(evaluate.time, "sleep", lambda dt: sleep_calls.append(dt))
    monkeypatch.setattr(
        evaluate,
        "residual_avoidance_command",
        lambda *args, **kwargs: (0.2, 0.0, 0.0),
    )

    robot_name = "robot_1"
    state = environment.RobotState(
        name=robot_name,
        x=0.0,
        y=0.0,
        yaw=0.0,
        v=0.2,
        omega=0.0,
    )

    class FakeBridge:
        def __init__(self):
            self.commands = []
            self.stopped = False

        def update_pedestrian_fallbacks(self, elapsed_time):
            self.pedestrian_time = elapsed_time

        def spin_once(self, timeout_sec=0.0):
            pass

        def get_robot_states(self):
            return {robot_name: state}

        def get_laser_data(self):
            return _safe_laser_data([robot_name])

        def get_pedestrian_states(self):
            return {}

        def get_collisions(self):
            return {robot_name: True}

        def get_collision_categories(self):
            return {robot_name: "robot_pedestrian"}

        def send_commands(self, commands):
            self.commands.append(dict(commands))

        def stop_all(self):
            self.stopped = True

    class FakePolicy:
        def get_action_and_value(
            self,
            obs,
            adjacency,
            action=None,
            deterministic=False,
        ):
            del adjacency, action, deterministic
            batch, robots = obs.shape[0], obs.shape[1]
            return (
                torch.zeros(batch, robots, 3),
                torch.zeros(batch, robots),
                torch.zeros(batch, robots),
                torch.zeros(batch, robots),
            )

    bridge = FakeBridge()
    goals = {robot_name: evaluate.GoalState(x=1.0, y=0.0)}
    scenario = SimpleNamespace(name="unit", max_steps=10, dt=0.1)

    try:
        metrics = evaluate.evaluate_episode(
            bridge=bridge,
            policy=FakePolicy(),
            scenario=scenario,
            robot_names=[robot_name],
            goals=goals,
            seed=0,
        )
    finally:
        monkeypatch.undo()

    assert len(sleep_calls) == 1
    assert len(bridge.commands) == 1
    assert bridge.stopped
    assert metrics.success_rate() == 0.0
    assert metrics.collision_count == 1
    assert metrics.robot_pedestrian_collision_count == 1
    assert metrics.robot_robot_collision_count == 0
    assert metrics.robot_obstacle_collision_count == 0
    assert robot_name not in metrics.completion_times


def test_evaluate_episode_allows_deadlocked_robot_to_keep_trying_until_limit():
    monkeypatch = _MonkeyPatch()
    _install_torch_stub(monkeypatch)
    _install_ros_stubs(monkeypatch)
    _install_mappo_stub(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.evaluate", raising=False)
    evaluate = importlib.import_module("mrpp_rl.evaluate")
    environment = importlib.import_module("mrpp_rl.environment")
    monkeypatch.setattr(evaluate.time, "sleep", lambda dt: None)
    monkeypatch.setattr(evaluate, "DEADLOCK_TIMEOUT_SEC", 0.15)

    robot_name = "robot_1"
    state = environment.RobotState(
        name=robot_name,
        x=0.0,
        y=0.0,
        yaw=0.0,
        v=0.0,
        omega=0.0,
    )

    class FakeBridge:
        def __init__(self):
            self.commands = []
            self.stopped = False

        def update_pedestrian_fallbacks(self, elapsed_time):
            self.pedestrian_time = elapsed_time

        def spin_once(self, timeout_sec=0.0):
            pass

        def get_robot_states(self):
            return {robot_name: state}

        def get_laser_data(self):
            return _safe_laser_data([robot_name])

        def get_pedestrian_states(self):
            return {}

        def get_collisions(self):
            return {robot_name: False}

        def send_commands(self, commands):
            self.commands.append(dict(commands))

        def stop_all(self):
            self.stopped = True

    class FakeController:
        algorithm_name = "unit_controller"
        last_policy_inference_time_s = 0.0

        def compute_commands(
            self, robot_names, states, laser_data, pedestrian_states, active_goals
        ):
            del states, laser_data, pedestrian_states, active_goals
            return {name: (0.0, 0.0, 0.0) for name in robot_names}

        def close(self):
            pass

    bridge = FakeBridge()
    goals = {robot_name: evaluate.GoalState(x=1.0, y=0.0)}
    scenario = SimpleNamespace(name="unit", max_steps=4, dt=0.1)

    try:
        metrics = evaluate.evaluate_episode(
            bridge=bridge,
            controller=FakeController(),
            scenario=scenario,
            robot_names=[robot_name],
            goals=goals,
            seed=0,
        )
    finally:
        monkeypatch.undo()

    assert len(bridge.commands) == scenario.max_steps
    assert bridge.stopped
    assert metrics.success_rate() == 0.0
    assert metrics.deadlock_count == 1
    assert metrics.deadlock_failure_count == 1
    assert metrics.timeout_failure_count == 0


def test_evaluate_episode_can_disable_safety_filter():
    monkeypatch = _MonkeyPatch()
    _install_torch_stub(monkeypatch)
    _install_ros_stubs(monkeypatch)
    _install_mappo_stub(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.evaluate", raising=False)
    evaluate = importlib.import_module("mrpp_rl.evaluate")
    environment = importlib.import_module("mrpp_rl.environment")
    monkeypatch.setattr(evaluate.time, "sleep", lambda dt: None)

    def fail_safety_filter(*args, **kwargs):
        raise AssertionError("laser safety filter should be disabled")

    def fail_robot_proximity_filter(*args, **kwargs):
        raise AssertionError("robot proximity safety filter should be disabled")

        monkeypatch.setattr(
            evaluate,
            "apply_safety_filter_to_commands_with_results",
            fail_safety_filter,
        )
    monkeypatch.setattr(
        evaluate,
        "apply_robot_proximity_safety_filter",
        fail_robot_proximity_filter,
    )

    robot_name = "robot_1"
    state = environment.RobotState(
        name=robot_name,
        x=0.0,
        y=0.0,
        yaw=0.0,
        v=0.0,
        omega=0.0,
    )

    class FakeBridge:
        def __init__(self):
            self.commands = []
            self.stopped = False

        def update_pedestrian_fallbacks(self, elapsed_time):
            self.pedestrian_time = elapsed_time

        def spin_once(self, timeout_sec=0.0):
            pass

        def get_robot_states(self):
            return {robot_name: state}

        def get_laser_data(self):
            return _safe_laser_data([robot_name])

        def get_pedestrian_states(self):
            return {}

        def get_collisions(self):
            return {robot_name: False}

        def send_commands(self, commands):
            self.commands.append(dict(commands))

        def stop_all(self):
            self.stopped = True

    class FakeController:
        algorithm_name = "unit_controller"
        last_policy_inference_time_s = 0.0

        def compute_commands(
            self, robot_names, states, laser_data, pedestrian_states, active_goals
        ):
            del states, laser_data, pedestrian_states, active_goals
            return {name: (0.2, 0.0, 0.0) for name in robot_names}

        def close(self):
            pass

    bridge = FakeBridge()
    goals = {robot_name: evaluate.GoalState(x=2.0, y=0.0)}
    scenario = SimpleNamespace(
        name="unit",
        world="pedestrian_dynamic",
        max_steps=1,
        dt=0.1,
    )

    try:
        metrics = evaluate.evaluate_episode(
            bridge=bridge,
            controller=FakeController(),
            scenario=scenario,
            robot_names=[robot_name],
            goals=goals,
            seed=0,
            use_safety_filter=False,
        )
    finally:
        monkeypatch.undo()

    assert len(bridge.commands) == 1
    assert bridge.commands[0][robot_name] == (0.12, 0.0, 0.0)
    assert bridge.stopped
    assert metrics.collision_count == 0


def test_topic_bridges_use_robot_scoped_topics():
    monkeypatch = _MonkeyPatch()
    _install_ros_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.gazebo_bridge", raising=False)
    gazebo_bridge = importlib.import_module("mrpp_rl.gazebo_bridge")
    calls = []

    class FakeProcess:
        def poll(self):
            return None

    start_new_session_values = []

    def fake_popen(cmd, stdout=None, stderr=None, start_new_session=False):
        del stdout, stderr
        calls.append(cmd)
        start_new_session_values.append(start_new_session)
        return FakeProcess()

    monkeypatch.setattr(gazebo_bridge.subprocess, "Popen", fake_popen)

    try:
        gazebo_bridge.launch_topic_bridges(
            ["robot_1"], ["ped_1"], world_name="static_clutter"
        )
    finally:
        monkeypatch.undo()

    joined = "\n".join(" ".join(cmd) for cmd in calls)
    assert "/model/robot_1/cmd_vel:=/robot_1/cmd_vel" in joined
    assert "/model/robot_1/odom:=/robot_1/odom" in joined
    assert "/model/robot_1/scan:=/robot_1/scan" in joined
    assert "/model/ped_1/odom:=/ped_1/odom" in joined
    assert "/world/static_clutter/dynamic_pose/info:=/gazebo/dynamic_pose" in joined
    assert "/cmd_vel:=/cmd_vel" not in joined
    assert start_new_session_values
    assert all(start_new_session_values)


def test_launch_gazebo_uses_headless_server():
    monkeypatch = _MonkeyPatch()
    _install_ros_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.gazebo_bridge", raising=False)
    gazebo_bridge = importlib.import_module("mrpp_rl.gazebo_bridge")
    calls = []

    class FakeProcess:
        pass

    def fake_popen(cmd, stdout=None, stderr=None):
        del stdout, stderr
        calls.append(cmd)
        return FakeProcess()

    monkeypatch.setattr(gazebo_bridge.subprocess, "Popen", fake_popen)

    try:
        gazebo_bridge.launch_gazebo("/tmp/world.sdf")
    finally:
        monkeypatch.undo()

    assert calls == [
        [
            "ign",
            "gazebo",
            "-s",
            "--headless-rendering",
            "/tmp/world.sdf",
            "-r",
            "-v",
            "0",
        ]
    ]


def test_launch_gazebo_gui_mode_opens_full_gazebo():
    monkeypatch = _MonkeyPatch()
    _install_ros_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.gazebo_bridge", raising=False)
    gazebo_bridge = importlib.import_module("mrpp_rl.gazebo_bridge")
    calls = []

    class FakeProcess:
        pass

    def fake_popen(cmd, stdout=None, stderr=None):
        del stdout, stderr
        calls.append(cmd)
        return FakeProcess()

    monkeypatch.setattr(gazebo_bridge.subprocess, "Popen", fake_popen)

    try:
        gazebo_bridge.launch_gazebo("/tmp/world.sdf", gui=True)
    finally:
        monkeypatch.undo()

    assert calls == [["ign", "gazebo", "/tmp/world.sdf", "-r", "-v", "2"]]


def test_bridge_transforms_relative_odom_to_world_pose():
    monkeypatch = _MonkeyPatch()
    _install_ros_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.gazebo_bridge", raising=False)
    gazebo_bridge = importlib.import_module("mrpp_rl.gazebo_bridge")

    bridge = gazebo_bridge.GazeboBridge.__new__(gazebo_bridge.GazeboBridge)
    bridge._raw_odom = {"robot_1": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)}
    bridge._odom_reference = {"robot_1": (0.0, 0.0, 0.0)}
    bridge._world_origins = {"robot_1": (0.0, 0.0, 0.0)}
    bridge.states = {}

    try:
        bridge.set_start_positions({"robot_1": (2.0, 3.0, 1.57079632679)})
        bridge._raw_odom["robot_1"] = (1.0, 0.0, 0.0, 0.2, 0.0, 0.0)
        bridge._update_state_from_raw("robot_1")
        state = bridge.states["robot_1"]

        assert abs(state.x - 2.0) < 1e-6
        assert abs(state.y - 4.0) < 1e-6
        assert abs(state.yaw - 1.57079632679) < 1e-6

        bridge.set_start_positions({"robot_1": (-1.0, -2.0, 0.0)})
        state = bridge.states["robot_1"]
    finally:
        monkeypatch.undo()

    assert abs(state.x + 1.0) < 1e-6
    assert abs(state.y + 2.0) < 1e-6


def test_bridge_prefers_dynamic_pose_position_over_odom():
    monkeypatch = _MonkeyPatch()
    _install_ros_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.gazebo_bridge", raising=False)
    gazebo_bridge = importlib.import_module("mrpp_rl.gazebo_bridge")
    environment = importlib.import_module("mrpp_rl.environment")

    bridge = gazebo_bridge.GazeboBridge.__new__(gazebo_bridge.GazeboBridge)
    bridge._raw_odom = {"robot_1": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)}
    bridge._odom_reference = {"robot_1": (0.0, 0.0, 0.0)}
    bridge._world_origins = {"robot_1": (0.0, 0.0, 0.0)}
    bridge._pose_seen = {"robot_1": True}
    bridge._odom_seen = {"robot_1": False}
    bridge.states = {
        "robot_1": environment.RobotState(
            name="robot_1",
            x=2.0,
            y=3.0,
            yaw=0.4,
            v=0.0,
            vy=0.0,
            omega=0.0,
        )
    }
    odom = SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=99.0, y=88.0),
                orientation=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
            )
        ),
        twist=SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=0.5, y=0.2),
                angular=SimpleNamespace(z=0.1),
            )
        ),
    )

    try:
        bridge._make_odom_cb("robot_1")(odom)
        state = bridge.states["robot_1"]
    finally:
        monkeypatch.undo()

    assert state.x == 2.0
    assert state.y == 3.0
    assert state.yaw == 0.4
    assert state.v == 0.5
    assert state.vy == 0.2
    assert state.omega == 0.1


def test_dynamic_pose_velocity_uses_message_stamp_not_wall_clock():
    monkeypatch = _MonkeyPatch()
    _install_ros_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.gazebo_bridge", raising=False)
    gazebo_bridge = importlib.import_module("mrpp_rl.gazebo_bridge")
    environment = importlib.import_module("mrpp_rl.environment")

    bridge = gazebo_bridge.GazeboBridge.__new__(gazebo_bridge.GazeboBridge)
    bridge._robot_names = ["robot_1"]
    bridge._last_pose_sample = {}
    bridge._pose_seen = {"robot_1": False}
    bridge._odom_seen = {"robot_1": False}
    bridge._odom_updates = {"robot_1": 0}
    bridge._sim_time_sec = 0.0
    bridge.states = {
        "robot_1": environment.RobotState(name="robot_1", x=0.0, y=0.0, yaw=0.0)
    }
    bridge.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=999_000_000_000)
    )

    def transform(stamp_sec, x, yaw):
        return SimpleNamespace(
            child_frame_id="robot_1",
            header=SimpleNamespace(
                stamp=SimpleNamespace(
                    sec=int(stamp_sec),
                    nanosec=int((stamp_sec - int(stamp_sec)) * 1_000_000_000),
                )
            ),
            transform=SimpleNamespace(
                translation=SimpleNamespace(x=x, y=0.0),
                rotation=SimpleNamespace(
                    w=math.cos(yaw / 2.0),
                    x=0.0,
                    y=0.0,
                    z=math.sin(yaw / 2.0),
                ),
            ),
        )

    try:
        bridge._dynamic_pose_cb(SimpleNamespace(transforms=[transform(1.0, 0.0, 0.0)]))
        bridge._dynamic_pose_cb(SimpleNamespace(transforms=[transform(1.5, 0.3, 0.1)]))
        state = bridge.states["robot_1"]
    finally:
        monkeypatch.undo()

    assert abs(state.v - 0.6 * math.cos(0.1)) < 1e-6
    assert abs(state.vy + 0.6 * math.sin(0.1)) < 1e-6
    assert abs(state.omega - 0.2) < 1e-6


def test_bridge_static_geometry_disables_laser_obstacle_fallback():
    monkeypatch = _MonkeyPatch()
    _install_ros_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.gazebo_bridge", raising=False)
    gazebo_bridge = importlib.import_module("mrpp_rl.gazebo_bridge")
    environment = importlib.import_module("mrpp_rl.environment")

    bridge = gazebo_bridge.GazeboBridge.__new__(gazebo_bridge.GazeboBridge)
    bridge._robot_names = ["robot_1"]
    bridge._collision_radius = gazebo_bridge.ROBOT_RADIUS * 2.0
    bridge.states = {
        "robot_1": environment.RobotState(
            name="robot_1",
            x=2.0,
            y=2.0,
            yaw=0.0,
        )
    }
    bridge.pedestrians = {}
    bridge.laser = {
        "robot_1": gazebo_bridge.laser_scan_to_observation(
            [0.10],
            angle_min=0.0,
            angle_increment=1.0,
            range_min=0.1,
            range_max=4.5,
        )
    }

    try:
        bridge.set_static_obstacles([])
        categories = bridge.get_collision_categories()

        bridge._static_obstacles = None
        fallback_categories = bridge.get_collision_categories()
    finally:
        monkeypatch.undo()

    assert categories["robot_1"] is None
    assert fallback_categories["robot_1"] == "robot_obstacle"


def test_box_collision_uses_circle_rectangle_distance_at_corners():
    monkeypatch = _MonkeyPatch()
    _install_ros_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.gazebo_bridge", raising=False)
    gazebo_bridge = importlib.import_module("mrpp_rl.gazebo_bridge")

    box = SimpleNamespace(x=2.0, y=0.5, yaw=0.0, sx=0.4, sy=0.4)

    try:
        near_corner_clear = gazebo_bridge._contains_robot_center_with_radius(
            box,
            1.63,
            0.87,
            0.18,
        )
        near_corner_collision = gazebo_bridge._contains_robot_center_with_radius(
            box,
            1.75,
            0.75,
            0.18,
        )
    finally:
        monkeypatch.undo()

    assert near_corner_clear is False
    assert near_corner_collision is True


def test_bridge_scan_callback_uses_receive_time_for_runtime_staleness():
    monkeypatch = _MonkeyPatch()
    _install_ros_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.gazebo_bridge", raising=False)
    gazebo_bridge = importlib.import_module("mrpp_rl.gazebo_bridge")

    bridge = gazebo_bridge.GazeboBridge.__new__(gazebo_bridge.GazeboBridge)
    bridge.laser = {}
    bridge._scan_seen = {"robot_1": False}
    bridge.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=100_000_000_000)
    )
    scan = SimpleNamespace(
        ranges=[float("inf"), 0.2],
        angle_min=0.0,
        angle_increment=1.0,
        range_min=0.1,
        range_max=4.5,
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=3, nanosec=500_000_000),
        ),
    )

    try:
        bridge._make_scan_cb("robot_1")(scan)
    finally:
        monkeypatch.undo()

    observation = bridge.laser["robot_1"]
    assert observation.stamp_sec == 100.0
    assert observation.message_stamp_sec == 3.5
    assert observation.range_max == 4.5
    assert observation.ranges[0] == 4.5
    assert bridge._scan_seen["robot_1"]


def test_bridge_static_geometry_detects_box_collision():
    monkeypatch = _MonkeyPatch()
    _install_ros_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.gazebo_bridge", raising=False)
    gazebo_bridge = importlib.import_module("mrpp_rl.gazebo_bridge")
    environment = importlib.import_module("mrpp_rl.environment")

    bridge = gazebo_bridge.GazeboBridge.__new__(gazebo_bridge.GazeboBridge)
    bridge._robot_names = ["robot_1"]
    bridge._collision_radius = gazebo_bridge.ROBOT_RADIUS * 2.0
    bridge.states = {
        "robot_1": environment.RobotState(
            name="robot_1",
            x=0.65,
            y=0.0,
            yaw=0.0,
        )
    }
    bridge.pedestrians = {}
    bridge.laser = {}
    obstacle = SimpleNamespace(x=0.0, y=0.0, yaw=0.0, sx=1.0, sy=1.0)

    try:
        bridge.set_static_obstacles([obstacle])
        categories = bridge.get_collision_categories()
    finally:
        monkeypatch.undo()

    assert categories["robot_1"] == "robot_obstacle"


def test_bridge_scripted_pedestrians_ignore_actor_odom():
    monkeypatch = _MonkeyPatch()
    _install_ros_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.gazebo_bridge", raising=False)
    gazebo_bridge = importlib.import_module("mrpp_rl.gazebo_bridge")
    pedestrian_controller = importlib.import_module(
        "mrpp_gazebo.pedestrian_controller"
    )

    bridge = gazebo_bridge.GazeboBridge.__new__(gazebo_bridge.GazeboBridge)
    bridge.pedestrians = {}
    bridge._ped_seen = {"ped_1": False}
    bridge._pedestrian_configs = {}
    bridge._scripted_pedestrians = False
    cfg = pedestrian_controller.PedestrianConfig(
        name="ped_1",
        start_x=-5.0,
        start_y=3.0,
        yaw=0.0,
        speed=0.5,
        direction="east",
    )
    odom = SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=99.0, y=88.0),
            ),
        ),
        twist=SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=1.0, y=2.0),
            ),
        ),
    )

    try:
        bridge.set_pedestrian_configs([cfg], scripted=True)
        bridge._make_ped_cb("ped_1")(odom)
    finally:
        monkeypatch.undo()

    ped = bridge.pedestrians["ped_1"]
    assert ped.x == -5.0
    assert ped.y == 3.0
    assert bridge._ped_seen["ped_1"]


def test_bridge_scripted_pedestrians_advance_after_seen():
    monkeypatch = _MonkeyPatch()
    _install_ros_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "mrpp_rl.gazebo_bridge", raising=False)
    gazebo_bridge = importlib.import_module("mrpp_rl.gazebo_bridge")
    pedestrian_controller = importlib.import_module(
        "mrpp_gazebo.pedestrian_controller"
    )

    bridge = gazebo_bridge.GazeboBridge.__new__(gazebo_bridge.GazeboBridge)
    bridge.pedestrians = {}
    bridge._ped_seen = {"ped_1": True}
    bridge._pedestrian_configs = {}
    bridge._scripted_pedestrians = False
    cfg = pedestrian_controller.PedestrianConfig(
        name="ped_1",
        start_x=-5.0,
        start_y=3.0,
        yaw=0.0,
        speed=0.5,
        direction="east",
    )

    try:
        bridge.set_pedestrian_configs([cfg], scripted=True)
        bridge.update_pedestrian_fallbacks(10.0)
    finally:
        monkeypatch.undo()

    ped = bridge.pedestrians["ped_1"]
    assert ped.x == 0.0
    assert ped.y == 3.0
