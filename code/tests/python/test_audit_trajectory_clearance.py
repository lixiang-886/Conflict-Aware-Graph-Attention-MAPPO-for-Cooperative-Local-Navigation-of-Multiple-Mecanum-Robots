from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_audit_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "audit_trajectory_clearance.py"
    spec = importlib.util.spec_from_file_location("audit_trajectory_clearance", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_audit_trajectory_clearance_reports_speed_and_clearance(tmp_path: Path) -> None:
    module = _load_audit_module()
    csv_path = tmp_path / "trajectory.csv"
    csv_path.write_text(
        "\n".join(
            [
                "scenario,algorithm,seed,robot_name,point_index,t,x,y,v,omega,success,completion_time,schema_version",
                "static_clutter,unit,1,robot_1,0,0.0,0.0,0.0,0,0,1,1.0,v1",
                "static_clutter,unit,1,robot_1,1,1.0,1.0,0.0,0,0,1,1.0,v1",
                "static_clutter,unit,1,robot_2,0,0.0,0.5,0.0,0,0,1,1.0,v1",
                "static_clutter,unit,1,robot_2,1,1.0,1.5,0.0,0,0,1,1.0,v1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = module.audit_trajectory_csv(
        csv_path,
        robot_robot_near_miss_threshold=0.7,
    )

    assert summary["scenario"] == "static_clutter"
    assert summary["robot_count"] == 2
    assert summary["mean_segment_speed_mps"] == 1.0
    assert summary["min_robot_robot_distance_m"] == 0.5
    assert summary["robot_robot_near_miss_count"] == 2
