import json
import subprocess
from pathlib import Path

from PIL import Image


SCENARIOS = (
    ("static_clutter", 2),
    ("pedestrian_dynamic", 2),
    ("mixed_complex", 0),
)


def test_selects_five_sixths_and_preserves_final_image(tmp_path: Path) -> None:
    snapshot_root = tmp_path / "snapshots"
    for scenario, seed in SCENARIOS:
        directory_name = f"{scenario}_seed{seed}"
        directory = snapshot_root / directory_name
        directory.mkdir(parents=True)
        (directory / "evaluate_gui.log").write_text(
            "Final GUI hold ready at step 600. Create release.signal to close.\n",
            encoding="utf-8",
        )
        for step in range(25, 600, 25):
            image = Image.new("RGB", (4, 4), (step % 256, 10, 20))
            image.save(directory / f"{directory_name}_step{step}.png")
        final_path = directory / f"{directory_name}_final.png"
        Image.new("RGB", (4, 4), (255, 255, 255)).save(final_path)

    subprocess.run(
        [
            "python3",
            "scripts/select_even_gazebo_snapshots.py",
            "--snapshot-root",
            str(snapshot_root),
        ],
        check=True,
    )

    manifest = json.loads(
        (snapshot_root / "selection_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["scenarios"]) == 3
    for scenario_result in manifest["scenarios"]:
        assert scenario_result["final_step"] == 600
        assert [sample["source_step"] for sample in scenario_result["samples"]] == [
            100,
            200,
            300,
            400,
            500,
        ]
        directory = snapshot_root / (
            f"{scenario_result['scenario']}_seed{scenario_result['seed']}"
        )
        assert all(
            (directory / f"{directory.name}_sample{index}.png").is_file()
            for index in range(1, 6)
        )
        assert (directory / scenario_result["final_file"]).is_file()
