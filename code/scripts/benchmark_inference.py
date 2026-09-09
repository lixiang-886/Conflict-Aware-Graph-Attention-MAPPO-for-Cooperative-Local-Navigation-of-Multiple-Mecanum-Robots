#!/usr/bin/env python3
"""Run a controlled sparse-vs-fully-connected graph inference microbenchmark.

This benchmark is intentionally independent of Gazebo. It uses synthetic robot
states and a fixed randomly initialized graph encoder to isolate graph-building
and graph-attention computation. Absolute timings therefore depend on the local
hardware/software stack and should not be interpreted as end-to-end navigation
latency.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/mrpp_rl"))

from mrpp_rl.control import conflict_aware_graph  # noqa: E402
from mrpp_rl.control import control_profile_for_algorithm  # noqa: E402
from mrpp_rl.environment import GoalState, RobotState  # noqa: E402
from mrpp_rl.graph_attention import GraphAttentionEncoder  # noqa: E402
from mrpp_rl.observation_schema import OBS_DIM  # noqa: E402


ROBOT_COUNTS = (4, 6, 8, 10, 16, 20)
GPU_MODEL = "NVIDIA GeForce RTX 4090"
METHOD_BY_MODE = {
    "cpa_conditioned": "CA-GAT-MAPPO",
    "fully_connected": "Fully-connected GAT",
}
FIELDNAMES = (
    "method",
    "robot_count",
    "target_iterations",
    "warmup_iterations",
    "timed_iterations",
    "possible_directed_nonself_edge_count",
    "active_edge_count_mean",
    "edge_density_mean",
    "graph_build_time_mean_ms",
    "graph_build_time_median_ms",
    "graph_build_time_p95_ms",
    "gat_forward_time_mean_ms",
    "gat_forward_time_median_ms",
    "gat_forward_time_p95_ms",
    "total_policy_inference_time_mean_ms",
    "total_policy_inference_time_median_ms",
    "total_policy_inference_time_p95_ms",
    "peak_GPU_memory_MB",
    "GPU_model",
)


def _synthetic_case(
    robot_count: int,
) -> tuple[list[str], dict[str, RobotState], dict[str, GoalState]]:
    names = [f"robot_{index + 1}" for index in range(robot_count)]
    states: dict[str, RobotState] = {}
    goals: dict[str, GoalState] = {}
    for index, name in enumerate(names):
        if index < 4:
            angle = 2.0 * math.pi * index / 4.0
            radius = 2.0
            goal_x = -radius * math.cos(angle)
            goal_y = -radius * math.sin(angle)
        else:
            outer_count = robot_count - 4
            angle = 2.0 * math.pi * (index - 4) / outer_count + 0.2
            radius = max(4.5, robot_count * 0.30)
            goal_x = (radius + 2.0) * math.cos(angle)
            goal_y = (radius + 2.0) * math.sin(angle)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        states[name] = RobotState(
            name=name,
            x=x,
            y=y,
            yaw=math.atan2(goal_y - y, goal_x - x),
        )
        goals[name] = GoalState(x=goal_x, y=goal_y)
    return names, states, goals


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _summary(values: list[float]) -> tuple[float, float, float]:
    return (
        statistics.fmean(values),
        statistics.median(values),
        _percentile(values, 0.95),
    )


def _time_graph_build(
    mode: str,
    names: list[str],
    states: dict[str, RobotState],
    goals: dict[str, GoalState],
    timed_iterations: int,
) -> tuple[list[list[float]], list[int], list[float]]:
    profile = control_profile_for_algorithm("mo_gat_mappo", "mixed_complex")
    timings: list[float] = []
    active_edge_counts: list[int] = []
    adjacency: list[list[float]] = []
    for _ in range(timed_iterations):
        start = time.perf_counter_ns()
        if mode == "fully_connected":
            adjacency = [[1.0 for _ in names] for _ in names]
            directed_active_edges = len(names) * (len(names) - 1)
        else:
            adjacency, diagnostics = conflict_aware_graph(
                names,
                states,
                goals,
                max_speed=0.6,
                profile=profile,
            )
            # conflict_aware_graph records one diagnostic per unordered pair but
            # activates both adjacency directions. Report directed edges to match
            # possible_directed_nonself_edge_count.
            directed_active_edges = 2 * sum(
                int(bool(item["edge_active"])) for item in diagnostics
            )
        active_edge_counts.append(directed_active_edges)
        timings.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return adjacency, active_edge_counts, timings


def _lightweight_action_head(features: torch.Tensor) -> torch.Tensor:
    """Small deterministic head used only to close the synthetic policy step."""
    return torch.tanh(features.mean(dim=1))


def _time_encoder_and_policy(
    encoder: GraphAttentionEncoder,
    nodes: torch.Tensor,
    adjacency: torch.Tensor,
    warmup: int,
    timed_iterations: int,
) -> tuple[list[float], list[float], float]:
    device = nodes.device
    with torch.inference_mode():
        for _ in range(warmup):
            features = encoder(nodes, adjacency)
            _lightweight_action_head(features)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        gat_timings: list[float] = []
        policy_timings: list[float] = []
        for _ in range(timed_iterations):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter_ns()
            encoder(nodes, adjacency)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            gat_timings.append((time.perf_counter_ns() - start) / 1_000_000.0)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter_ns()
            features = encoder(nodes, adjacency)
            _lightweight_action_head(features)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            policy_timings.append((time.perf_counter_ns() - start) / 1_000_000.0)

    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        if device.type == "cuda"
        else 0.0
    )
    return gat_timings, policy_timings, peak_mb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--node-dim", type=int, default=OBS_DIM)
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()

    if args.iterations < 1000:
        parser.error("--iterations must be at least 1000")
    if args.warmup <= 0:
        parser.error("--warmup must be positive")
    if args.warmup >= args.iterations:
        parser.error("--warmup must be smaller than --iterations")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    device = torch.device(args.device)
    torch.manual_seed(0)
    encoder = GraphAttentionEncoder(args.node_dim, args.hidden_dim).to(device).eval()
    timed_iterations = args.iterations - args.warmup
    rows: list[dict[str, object]] = []

    for robot_count in ROBOT_COUNTS:
        names, states, goals = _synthetic_case(robot_count)
        nodes = torch.randn(
            (1, robot_count, args.node_dim),
            generator=torch.Generator().manual_seed(robot_count),
        ).to(device)
        possible_edges = robot_count * (robot_count - 1)

        for mode in ("fully_connected", "cpa_conditioned"):
            adjacency_values, active_edge_counts, graph_times = _time_graph_build(
                mode,
                names,
                states,
                goals,
                timed_iterations,
            )
            adjacency = torch.tensor(
                [adjacency_values],
                dtype=torch.float32,
                device=device,
            )
            gat_times, policy_times, peak_mb = _time_encoder_and_policy(
                encoder,
                nodes,
                adjacency,
                args.warmup,
                timed_iterations,
            )

            graph_mean, graph_median, graph_p95 = _summary(graph_times)
            gat_mean, gat_median, gat_p95 = _summary(gat_times)
            policy_mean, policy_median, policy_p95 = _summary(policy_times)
            active_mean = statistics.fmean(active_edge_counts)

            rows.append(
                {
                    "method": METHOD_BY_MODE[mode],
                    "robot_count": robot_count,
                    "target_iterations": args.iterations,
                    "warmup_iterations": args.warmup,
                    "timed_iterations": timed_iterations,
                    "possible_directed_nonself_edge_count": possible_edges,
                    "active_edge_count_mean": active_mean,
                    "edge_density_mean": (
                        active_mean / possible_edges if possible_edges else 0.0
                    ),
                    "graph_build_time_mean_ms": graph_mean,
                    "graph_build_time_median_ms": graph_median,
                    "graph_build_time_p95_ms": graph_p95,
                    "gat_forward_time_mean_ms": gat_mean,
                    "gat_forward_time_median_ms": gat_median,
                    "gat_forward_time_p95_ms": gat_p95,
                    "total_policy_inference_time_mean_ms": graph_mean + policy_mean,
                    "total_policy_inference_time_median_ms": (
                        graph_median + policy_median
                    ),
                    "total_policy_inference_time_p95_ms": graph_p95 + policy_p95,
                    "peak_GPU_memory_MB": peak_mb,
                    "GPU_model": GPU_MODEL if device.type == "cuda" else "CPU",
                }
            )
            print(
                f"N={robot_count} mode={mode} edges={active_mean:.1f}/"
                f"{possible_edges} total={graph_mean + policy_mean:.4f} ms",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
