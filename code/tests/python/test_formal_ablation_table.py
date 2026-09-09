from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_exporter() -> ModuleType:
    path = Path("scripts/export_formal_ablation_table.py")
    spec = importlib.util.spec_from_file_location("export_formal_ablation_table", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ablation_export_ignores_zero_success_completion_times() -> None:
    exporter = load_exporter()
    rows = [
        {
            "scenario": "mixed_complex",
            "algorithm": "mo_gat_mappo",
            "variant": "ablation_no_gate",
            "success_rate": "0.0",
            "collision_count": "0",
            "average_completion_time": "0.0",
            "makespan": "0.0",
            "average_waiting_time": "140.0",
            "average_path_efficiency": "0.80",
            "min_robot_robot_distance": "0.54",
            "min_robot_pedestrian_distance": "0.83",
            "robot_robot_near_miss_count": "1000",
            "robot_pedestrian_near_miss_count": "0",
            "intervention_time_ratio": "0.10",
        },
        {
            "scenario": "mixed_complex",
            "algorithm": "mo_gat_mappo",
            "variant": "ablation_no_gate",
            "success_rate": "0.5",
            "collision_count": "1",
            "average_completion_time": "140.5",
            "makespan": "145.0",
            "average_waiting_time": "128.0",
            "average_path_efficiency": "0.75",
            "min_robot_robot_distance": "0.55",
            "min_robot_pedestrian_distance": "0.51",
            "robot_robot_near_miss_count": "0",
            "robot_pedestrian_near_miss_count": "8",
            "intervention_time_ratio": "0.20",
        },
    ]

    summary = exporter.build_summary_rows(rows)

    assert len(summary) == 1
    assert summary[0]["success_percent"] == "25.0 +/- 35.4"
    assert summary[0]["average_completion_time"] == "140.50 +/- 0.00"
    assert summary[0]["total_near_miss_count"] == "504.0 +/- 701.4"
    latex = exporter.latex_table(summary)
    assert r"\begin{table}[H]" in latex
    assert "w/o interaction gate" in latex
    assert "Turn (rad/m)" in latex
    assert "Min RR (m)" in latex
    assert "Near" in latex
    assert "Int. ratio" in latex
    assert r"\textbf{140.50 $\pm$ 0.00}" in latex
    assert r"\begin{adjustwidth}{-\extralength}{0cm}" in latex
    assert r"\footnotesize" in latex
    assert r"\scriptsize" not in latex
    assert r"\resizebox" not in latex
    assert r"\emph{Note:}" not in latex


def test_ablation_export_marks_all_zero_success_time_as_not_applicable() -> None:
    exporter = load_exporter()
    rows = [
        {
            "scenario": "mixed_complex",
            "algorithm": "mo_gat_mappo",
            "variant": "ablation_no_coordinator",
            "success_rate": "0.0",
            "collision_count": "3",
            "average_completion_time": "0.0",
            "makespan": "0.0",
            "average_waiting_time": "66.0",
            "average_path_efficiency": "0.0",
            "min_robot_robot_distance": "0.51",
            "min_robot_pedestrian_distance": "0.52",
            "robot_robot_near_miss_count": "62",
            "robot_pedestrian_near_miss_count": "98",
            "intervention_time_ratio": "0.4",
        }
    ]

    summary = exporter.build_summary_rows(rows)

    assert summary[0]["success_percent"] == "0.0 +/- 0.0"
    assert summary[0]["average_completion_time"] == "N/A"
