from pathlib import Path

from mrpp_paper.plot_metrics import summarize_metrics


def test_summarize_metrics(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "scenario,algorithm,success_rate,collision_count\n"
        "static_clutter,mappo,0.5,2\n"
        "static_clutter,mappo,1.0,0\n",
        encoding="utf-8",
    )

    summary = summarize_metrics(csv_path)

    assert summary[0]["scenario"] == "static_clutter"
    assert summary[0]["algorithm"] == "mappo"
    assert summary[0]["success_rate"] == 0.75
    assert summary[0]["success_rate_count"] == 2.0
    assert summary[0]["success_rate_std"] > 0.0
    assert summary[0]["success_rate_ci95_low"] < 0.75
    assert summary[0]["success_rate_ci95_high"] > 0.75
    assert summary[0]["collision_count"] == 1.0


def test_summarize_metrics_ignores_non_numeric_result_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "scenario,algorithm,success_rate,collision_count,data_source,notes,schema_version\n"
        "static_clutter,mo_gat_mappo,0.5,2,dry_run,not evidence,mrpp-results-v1\n"
        "static_clutter,mo_gat_mappo,1.0,0,dry_run,not evidence,mrpp-results-v1\n",
        encoding="utf-8",
    )

    summary = summarize_metrics(csv_path)

    assert summary[0]["success_rate"] == 0.75
    assert summary[0]["collision_count"] == 1.0
    assert "data_source" not in summary[0]


def test_summarize_metrics_does_not_average_seed_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "scenario,algorithm,seed,robot_count,success_rate\n"
        "static_clutter,mo_gat_mappo,0,4,0.5\n"
        "static_clutter,mo_gat_mappo,1,4,1.0\n",
        encoding="utf-8",
    )

    summary = summarize_metrics(csv_path)

    assert "seed" not in summary[0]
    assert summary[0]["robot_count"] == 4.0


def test_summarize_metrics_groups_ablation_rows_by_variant(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "scenario,algorithm,variant,seed,success_rate,orca_prior_enabled,safety_filter_enabled\n"
        "mixed_complex,mo_gat_mappo,full_mo_gat_mappo,0,1.0,1,1\n"
        "mixed_complex,mo_gat_mappo,ablation_no_orca_prior,0,0.5,0,1\n",
        encoding="utf-8",
    )

    summary = summarize_metrics(csv_path)

    assert [row["algorithm"] for row in summary] == [
        "ablation_no_orca_prior",
        "full_mo_gat_mappo",
    ]
    assert summary[0]["orca_prior_enabled"] == 0.0
    assert summary[1]["orca_prior_enabled"] == 1.0


def test_statistical_tests_generate_pairwise_rows(tmp_path: Path) -> None:
    from mrpp_paper.statistical_tests import build_statistical_tests

    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "scenario,algorithm,seed,success_rate,data_source\n"
        "static_clutter,orca,0,0.5,gazebo\n"
        "static_clutter,orca,1,0.5,gazebo\n"
        "static_clutter,mo_gat_mappo,0,1.0,gazebo\n"
        "static_clutter,mo_gat_mappo,1,1.0,gazebo\n",
        encoding="utf-8",
    )

    rows = build_statistical_tests(csv_path, metrics=("success_rate",))

    assert len(rows) == 1
    assert rows[0]["scenario"] == "static_clutter"
    assert rows[0]["metric"] == "success_rate"
    assert rows[0]["test_name"] == "paired_wilcoxon_signed_rank"
    assert rows[0]["adjusted_p_value"] != ""
