#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_STEPS = {
    "static_clutter": [50, 150, 250, 350, 400, 450],
    "pedestrian_dynamic": [25, 75, 125, 175, 225, 275],
    "mixed_complex": [100, 250, 400, 550, 700, 850],
}

DEFAULT_TITLES = {
    "static_clutter": "Static clutter",
    "pedestrian_dynamic": "Dynamic pedestrians",
    "mixed_complex": "Mixed complex",
}


@dataclass(frozen=True)
class TimeModel:
    intercept: float
    slope: float

    def video_time(self, step: int) -> float:
        return self.intercept + self.slope * step


def parse_anchor(raw: str) -> tuple[int, float]:
    try:
        step_text, time_text = raw.split(":", 1)
        return int(step_text), float(time_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Anchor must use step:seconds format, got {raw!r}"
        ) from exc


def build_time_model(
    *,
    anchors: list[tuple[int, float]],
    start_offset_sec: float,
    dt: float,
    time_scale: float,
) -> TimeModel:
    if len(anchors) >= 2:
        n = float(len(anchors))
        mean_step = sum(step for step, _ in anchors) / n
        mean_time = sum(t for _, t in anchors) / n
        denom = sum((step - mean_step) ** 2 for step, _ in anchors)
        if denom <= 0:
            raise ValueError("Anchor steps must not all be identical.")
        slope = sum((step - mean_step) * (t - mean_time) for step, t in anchors) / denom
        intercept = mean_time - slope * mean_step
        return TimeModel(intercept=intercept, slope=slope)

    if len(anchors) == 1:
        anchor_step, anchor_time = anchors[0]
        slope = dt * time_scale
        return TimeModel(intercept=anchor_time - slope * anchor_step, slope=slope)

    return TimeModel(intercept=start_offset_sec, slope=dt * time_scale)


def run_ffmpeg_extract(
    *,
    video: Path,
    output: Path,
    timestamp_sec: float,
    crop: str | None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(timestamp_sec, 0.0):.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
    ]
    if crop:
        cmd.extend(["-vf", f"crop={crop}"])
    cmd.extend(["-q:v", "2", str(output)])
    subprocess.run(cmd, check=True)


def video_duration_sec(video: Path) -> float | None:
    try:
        output = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video),
            ],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    try:
        return float(output.strip())
    except ValueError:
        return None


def make_panel(
    *,
    frames: list[tuple[int, Path]],
    output: Path,
    title: str,
    target_height: int,
) -> None:
    if len(frames) != 6:
        raise ValueError(f"Expected 6 frames for a 2x3 panel, got {len(frames)}")

    font = ImageFont.load_default()
    title_h = 48
    label_h = 28
    gap_x = 24
    gap_y = 28
    margin = 24

    loaded: list[tuple[int, Image.Image]] = []
    target_width = None
    for step, path in frames:
        image = Image.open(path).convert("RGB")
        scale = target_height / image.height
        width = int(round(image.width * scale))
        resized = image.resize((width, target_height), Image.Resampling.LANCZOS)
        loaded.append((step, resized))
        target_width = width if target_width is None else min(target_width, width)

    if target_width is None:
        raise ValueError("No frames loaded")

    normalized: list[tuple[int, Image.Image]] = []
    for step, image in loaded:
        if image.width > target_width:
            left = (image.width - target_width) // 2
            image = image.crop((left, 0, left + target_width, image.height))
        normalized.append((step, image))

    canvas_w = margin * 2 + target_width * 3 + gap_x * 2
    canvas_h = margin * 2 + title_h + (label_h + target_height) * 2 + gap_y
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), title, fill=(25, 25, 25), font=font)

    for idx, (step, image) in enumerate(normalized):
        col = idx % 3
        row = idx // 3
        x = margin + col * (target_width + gap_x)
        y = margin + title_h + row * (label_h + target_height + gap_y)
        draw.text((x, y), f"step {step}", fill=(30, 30, 30), font=font)
        canvas.paste(image, (x, y + label_h))

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Gazebo paper snapshots from a recorded video and build a 2x3 panel."
        )
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument(
        "--scenario",
        required=True,
        choices=sorted(DEFAULT_STEPS),
    )
    parser.add_argument("--steps", nargs="+", type=int, default=None)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--start-offset-sec", type=float, default=0.0)
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Multiplier for dt when no two-point anchor calibration is provided.",
    )
    parser.add_argument(
        "--anchor",
        action="append",
        type=parse_anchor,
        default=[],
        help=(
            "Calibration point in step:video_seconds format. With two or more "
            "anchors, the script fits video_time = a + b * step."
        ),
    )
    parser.add_argument(
        "--crop",
        default=None,
        help="Optional ffmpeg crop filter value, e.g. 1280:720:0:80.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--panel-output", type=Path, default=None)
    parser.add_argument("--panel-height", type=int, default=360)
    parser.add_argument("--title", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.video.exists():
        raise SystemExit(f"Video does not exist: {args.video}")
    if args.dt <= 0:
        raise SystemExit("--dt must be positive")
    if args.time_scale <= 0:
        raise SystemExit("--time-scale must be positive")

    steps = args.steps if args.steps is not None else DEFAULT_STEPS[args.scenario]
    if len(steps) != 6:
        raise SystemExit("Exactly six steps are required for the paper 2x3 panel.")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path("results/eval/final_selected_20260704/gazebo_video_frames")
            / args.scenario
        )
    panel_output = args.panel_output
    if panel_output is None:
        panel_output = output_dir / f"{args.scenario}_video_sequence_panel.png"

    time_model = build_time_model(
        anchors=args.anchor,
        start_offset_sec=args.start_offset_sec,
        dt=args.dt,
        time_scale=args.time_scale,
    )
    duration = video_duration_sec(args.video)
    frame_specs: list[tuple[int, float, Path]] = []
    for step in steps:
        t = time_model.video_time(step)
        if duration is not None and t > duration:
            raise SystemExit(
                f"Requested step {step} maps to {t:.3f}s, "
                f"but video duration is only {duration:.3f}s."
            )
        frame_specs.append((step, t, output_dir / f"{args.scenario}_step{step}.png"))

    print("Frame extraction plan:")
    print(f"  video: {args.video}")
    if duration is not None:
        print(f"  duration: {duration:.3f}s")
    print(f"  time model: video_time = {time_model.intercept:.6f} + {time_model.slope:.6f} * step")
    for step, t, path in frame_specs:
        print(f"  step {step}: {t:.3f}s -> {path}")

    if args.dry_run:
        return 0

    for step, t, path in frame_specs:
        run_ffmpeg_extract(video=args.video, output=path, timestamp_sec=t, crop=args.crop)

    make_panel(
        frames=[(step, path) for step, _, path in frame_specs],
        output=panel_output,
        title=args.title or DEFAULT_TITLES[args.scenario],
        target_height=args.panel_height,
    )
    print(f"Panel written to {panel_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
