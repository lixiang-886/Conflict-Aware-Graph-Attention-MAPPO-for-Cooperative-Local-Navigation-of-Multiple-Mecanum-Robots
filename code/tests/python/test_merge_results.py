from pathlib import Path

from mrpp_experiments.merge_results import merge_result_csv_files
from mrpp_experiments.results import EpisodeResult, read_results_csv, write_results_csv


def test_merge_result_csv_files_keeps_row_order(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    output = tmp_path / "all_metrics.csv"

    write_results_csv(
        first,
        [
            EpisodeResult(
                scenario="static_clutter",
                algorithm="priority_astar",
                seed=0,
                robot_count=4,
                success_rate=0.0,
                collision_count=0,
                deadlock_count=0,
                average_path_length=1.0,
                average_completion_time=1.0,
                makespan=1.0,
                flowtime=4.0,
                average_waiting_time=0.0,
                data_source="dry_run",
            )
        ],
    )
    write_results_csv(
        second,
        [
            EpisodeResult(
                scenario="static_clutter",
                algorithm="mo_gat_mappo",
                seed=0,
                robot_count=4,
                success_rate=0.0,
                collision_count=0,
                deadlock_count=0,
                average_path_length=1.0,
                average_completion_time=1.0,
                makespan=1.0,
                flowtime=4.0,
                average_waiting_time=0.0,
                data_source="dry_run",
            )
        ],
    )

    merge_result_csv_files([first, second], output)
    rows = read_results_csv(output)

    assert [row.algorithm for row in rows] == ["priority_astar", "mo_gat_mappo"]
