#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


FINAL_STEP_PATTERN = re.compile(r"Final GUI hold ready at step (\d+)\.")
STEP_FILE_PATTERN = re.compile(r"_step(\d+)\.png$")


@dataclass(frozen=True)
class ScenarioSelection:
    scenario: str
    seed: int

    @property
    def directory_name(self) -> str:
        return f"{self.scenario}_seed{self.seed}"


SCENARIOS = (
    ScenarioSelection("static_clutter", 2),
    ScenarioSelection("pedestrian_dynamic", 2),
    ScenarioSelection("mixed_complex", 0),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _final_step(log_path: Path) -> int:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = FINAL_STEP_PATTERN.findall(text)
    if not matches:
        raise ValueError(f"Final hold step is missing from {log_path}")
    return int(matches[-1])


def _step_images(directory: Path, final_step: int) -> list[tuple[int, Path]]:
    images: list[tuple[int, Path]] = []
    for path in directory.glob("*_step*.png"):
        match = STEP_FILE_PATTERN.search(path.name)
        if match:
            step = int(match.group(1))
            if 0 < step < final_step:
                images.append((step, path))
    images.sort()
    if len(images) < 5:
        raise ValueError(
            f"Need at least five pre-final images in {directory}, got {len(images)}"
        )
    return images


def select_scenario(
    snapshot_root: Path,
    selection: ScenarioSelection,
) -> dict[str, object]:
    directory = snapshot_root / selection.directory_name
    log_path = directory / "evaluate_gui.log"
    final_path = directory / f"{selection.directory_name}_final.png"
    if not final_path.is_file():
        raise FileNotFoundError(final_path)

    final_step = _final_step(log_path)
    candidates = _step_images(directory, final_step)
    unused = set(range(len(candidates)))
    samples: list[dict[str, object]] = []
    for index in range(1, 6):
        fraction = index / 6.0
        target_step = final_step * fraction
        candidate_index = min(
            unused,
            key=lambda item_index: (
                abs(candidates[item_index][0] - target_step),
                candidates[item_index][0],
            ),
        )
        unused.remove(candidate_index)
        source_step, source_path = candidates[candidate_index]
        output_path = directory / (
            f"{selection.directory_name}_sample{index}.png"
        )
        shutil.copy2(source_path, output_path)
        samples.append(
            {
                "sample_index": index,
                "target_fraction": fraction,
                "target_step": target_step,
                "source_step": source_step,
                "source_file": source_path.name,
                "selected_file": output_path.name,
                "sha256": _sha256(output_path),
            }
        )

    samples.sort(key=lambda item: int(item["sample_index"]))
    source_steps = [int(item["source_step"]) for item in samples]
    if source_steps != sorted(source_steps):
        raise ValueError(
            f"Selected steps are not monotonic for {selection.scenario}: {source_steps}"
        )
    return {
        "scenario": selection.scenario,
        "seed": selection.seed,
        "final_step": final_step,
        "samples": samples,
        "final_file": final_path.name,
        "final_sha256": _sha256(final_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select five evenly spaced pre-arrival Gazebo snapshots."
    )
    parser.add_argument("--snapshot-root", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            "Nearest captured steps to 1/6, 2/6, 3/6, 4/6, and 5/6 "
            "of the realized final-arrival step; final arrival is a separate frame."
        ),
        "scenarios": [
            select_scenario(args.snapshot_root, selection)
            for selection in SCENARIOS
        ],
    }
    output_path = args.snapshot_root / "selection_manifest.json"
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Evenly spaced snapshot selection written to {output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
