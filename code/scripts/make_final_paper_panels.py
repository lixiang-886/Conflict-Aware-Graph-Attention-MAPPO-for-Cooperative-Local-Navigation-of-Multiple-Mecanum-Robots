#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


@dataclass(frozen=True)
class PanelItem:
    path: str
    label: str


GAZEBO_ITEMS = (
    PanelItem(
        "gazebo_snapshots/static_clutter_seed2/static_clutter_seed2_initial.png",
        "(a) Static clutter",
    ),
    PanelItem(
        "gazebo_snapshots/pedestrian_dynamic_seed2/pedestrian_dynamic_seed2_initial.png",
        "(b) Dynamic pedestrians",
    ),
    PanelItem(
        "gazebo_snapshots/mixed_complex_seed0/mixed_complex_seed0_initial.png",
        "(c) Mixed complex",
    ),
)

GAZEBO_SEQUENCE_ROWS = (
    (
        "Static clutter",
        (
            PanelItem(
                "gazebo_snapshots/static_clutter_seed2/static_clutter_seed2_sample1.png",
                "1/6 progress",
            ),
            PanelItem(
                "gazebo_snapshots/static_clutter_seed2/static_clutter_seed2_sample2.png",
                "2/6 progress",
            ),
            PanelItem(
                "gazebo_snapshots/static_clutter_seed2/static_clutter_seed2_sample3.png",
                "3/6 progress",
            ),
            PanelItem(
                "gazebo_snapshots/static_clutter_seed2/static_clutter_seed2_sample4.png",
                "4/6 progress",
            ),
            PanelItem(
                "gazebo_snapshots/static_clutter_seed2/static_clutter_seed2_sample5.png",
                "5/6 progress",
            ),
            PanelItem(
                "gazebo_snapshots/static_clutter_seed2/static_clutter_seed2_final.png",
                "final arrival",
            ),
        ),
    ),
    (
        "Dynamic pedestrians",
        (
            PanelItem(
                "gazebo_snapshots/pedestrian_dynamic_seed2/pedestrian_dynamic_seed2_sample1.png",
                "1/6 progress",
            ),
            PanelItem(
                "gazebo_snapshots/pedestrian_dynamic_seed2/pedestrian_dynamic_seed2_sample2.png",
                "2/6 progress",
            ),
            PanelItem(
                "gazebo_snapshots/pedestrian_dynamic_seed2/pedestrian_dynamic_seed2_sample3.png",
                "3/6 progress",
            ),
            PanelItem(
                "gazebo_snapshots/pedestrian_dynamic_seed2/pedestrian_dynamic_seed2_sample4.png",
                "4/6 progress",
            ),
            PanelItem(
                "gazebo_snapshots/pedestrian_dynamic_seed2/pedestrian_dynamic_seed2_sample5.png",
                "5/6 progress",
            ),
            PanelItem(
                "gazebo_snapshots/pedestrian_dynamic_seed2/pedestrian_dynamic_seed2_final.png",
                "final arrival",
            ),
        ),
    ),
    (
        "Mixed complex",
        (
            PanelItem(
                "gazebo_snapshots/mixed_complex_seed0/mixed_complex_seed0_sample1.png",
                "1/6 progress",
            ),
            PanelItem(
                "gazebo_snapshots/mixed_complex_seed0/mixed_complex_seed0_sample2.png",
                "2/6 progress",
            ),
            PanelItem(
                "gazebo_snapshots/mixed_complex_seed0/mixed_complex_seed0_sample3.png",
                "3/6 progress",
            ),
            PanelItem(
                "gazebo_snapshots/mixed_complex_seed0/mixed_complex_seed0_sample4.png",
                "4/6 progress",
            ),
            PanelItem(
                "gazebo_snapshots/mixed_complex_seed0/mixed_complex_seed0_sample5.png",
                "5/6 progress",
            ),
            PanelItem(
                "gazebo_snapshots/mixed_complex_seed0/mixed_complex_seed0_final.png",
                "final arrival",
            ),
        ),
    ),
)

TRAJECTORY_ITEMS = (
    PanelItem(
        "comparison/paper_assets/representative_figures_clean/static_clutter_mo_gat_mappo_seed2.png",
        "(a) Static clutter",
    ),
    PanelItem(
        "comparison/paper_assets/representative_figures_clean/pedestrian_dynamic_mo_gat_mappo_seed2.png",
        "(b) Dynamic pedestrians",
    ),
    PanelItem(
        "comparison/paper_assets/representative_figures_clean/mixed_complex_mo_gat_mappo_seed0.png",
        "(c) Mixed complex",
    ),
)

GAZEBO_TARGET_GRID_PITCH_PX = 59.0
GAZEBO_TARGET_ASPECT = 4.0 / 3.0


def make_panel(
    root: Path,
    items: tuple[PanelItem, ...],
    output: Path,
    target_height: int,
    gutter: int = 28,
    padding: int = 34,
    label_height: int = 54,
    normalize_gazebo_zoom: bool = True,
) -> None:
    images = []
    for item in items:
        image = _load_item_image(
            root,
            item,
            normalize_gazebo_zoom=normalize_gazebo_zoom,
        )
        scale = target_height / image.height
        width = max(1, round(image.width * scale))
        images.append(image.resize((width, target_height), Image.Resampling.LANCZOS))

    font = _load_font(28)
    label_font = _load_font(30)
    total_width = padding * 2 + sum(image.width for image in images) + gutter * (len(images) - 1)
    total_height = padding * 2 + label_height + target_height
    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)

    x = padding
    for image, item in zip(images, items):
        draw.text((x, padding), item.label, fill=(20, 20, 20), font=label_font)
        canvas.paste(image, (x, padding + label_height))
        x += image.width + gutter

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def make_sequence_panel_2x3(
    root: Path,
    row_label: str,
    items: tuple[PanelItem, ...],
    output: Path,
    target_height: int,
    columns: int = 3,
    gutter: int = 22,
    padding: int = 32,
    title_height: int = 58,
    label_height: int = 44,
    row_gutter: int = 22,
    normalize_gazebo_zoom: bool = True,
) -> None:
    if len(items) != 6:
        raise ValueError(f"Expected 6 snapshots for {row_label}, got {len(items)}")

    title_font = _load_font(30)
    label_font = _load_font(24)
    images = []
    max_cell_width = 0
    for item in items:
        image = _load_item_image(
            root,
            item,
            normalize_gazebo_zoom=normalize_gazebo_zoom,
        )
        scale = target_height / image.height
        width = max(1, round(image.width * scale))
        resized = image.resize((width, target_height), Image.Resampling.LANCZOS)
        images.append((resized, item.label))
        max_cell_width = max(max_cell_width, resized.width)

    rows = (len(items) + columns - 1) // columns
    total_width = padding * 2 + columns * max_cell_width + (columns - 1) * gutter
    total_height = (
        padding * 2
        + title_height
        + rows * (label_height + target_height)
        + (rows - 1) * row_gutter
    )
    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, padding), row_label, fill=(20, 20, 20), font=title_font)

    y = padding + title_height
    for row_index in range(rows):
        x = padding
        for col_index in range(columns):
            item_index = row_index * columns + col_index
            if item_index >= len(images):
                break
            image, label = images[item_index]
            draw.text((x, y), label, fill=(35, 35, 35), font=label_font)
            paste_x = x + (max_cell_width - image.width) // 2
            canvas.paste(image, (paste_x, y + label_height))
            x += max_cell_width + gutter
        y += label_height + target_height + row_gutter

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def make_grid_panel(
    root: Path,
    rows: tuple[tuple[str, tuple[PanelItem, ...]], ...],
    output: Path,
    target_height: int,
    gutter: int = 22,
    padding: int = 32,
    row_label_width: int = 330,
    col_label_height: int = 48,
    row_gutter: int = 30,
) -> None:
    font = _load_font(26)
    label_font = _load_font(28)

    resized_rows = []
    max_cell_width = 0
    for row_label, items in rows:
        images = []
        for item in items:
            image = _load_item_image(root, item)
            scale = target_height / image.height
            width = max(1, round(image.width * scale))
            resized = image.resize((width, target_height), Image.Resampling.LANCZOS)
            images.append((resized, item.label))
            max_cell_width = max(max_cell_width, resized.width)
        resized_rows.append((row_label, images))

    column_count = max(len(images) for _row_label, images in resized_rows)
    total_width = (
        padding * 2
        + row_label_width
        + column_count * max_cell_width
        + (column_count - 1) * gutter
    )
    total_height = (
        padding * 2
        + len(resized_rows) * (target_height + col_label_height)
        + (len(resized_rows) - 1) * row_gutter
    )
    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)

    y = padding
    for row_label, images in resized_rows:
        label_text = row_label.replace(" ", "\n")
        bbox = draw.multiline_textbbox((0, 0), label_text, font=label_font, spacing=6)
        label_height = bbox[3] - bbox[1]
        draw.multiline_text(
            (padding, y + col_label_height + target_height // 2 - label_height // 2),
            label_text,
            fill=(20, 20, 20),
            font=label_font,
            spacing=6,
        )
        x = padding + row_label_width
        for image, label in images:
            draw.text((x, y), label, fill=(35, 35, 35), font=font)
            paste_x = x + (max_cell_width - image.width) // 2
            canvas.paste(image, (paste_x, y + col_label_height))
            x += max_cell_width + gutter
        y += col_label_height + target_height + row_gutter

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _load_item_image(
    root: Path,
    item: PanelItem,
    normalize_gazebo_zoom: bool = True,
) -> Image.Image:
    image = Image.open(root / item.path).convert("RGB")
    if item.path.startswith("gazebo_snapshots/"):
        image = _strip_gazebo_window_chrome(image)
        if normalize_gazebo_zoom:
            image = _remove_bright_green_gui_overlay(image)
            image = _normalize_gazebo_zoom(image)
    return image


def _strip_gazebo_window_chrome(image: Image.Image) -> Image.Image:
    width, height = image.size
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    search_limit = max(1, min(height, round(height * 0.15)))
    row_luma = arr[:search_limit].mean(axis=(1, 2))
    darker_rows = np.flatnonzero(row_luma < 220.0)
    top = int(darker_rows[0] + 1) if darker_rows.size else min(56, height - 1)
    return image.crop((0, top, width, height))


def _remove_bright_green_gui_overlay(image: Image.Image) -> Image.Image:
    width, height = image.size
    center_left = width * 0.18
    center_right = width * 0.82
    center_top = height * 0.12
    center_bottom = height * 0.82
    mask_data = []
    for index, (red, green, blue) in enumerate(image.getdata()):
        x = index % width
        y = index // width
        central_overlay = (
            center_left <= x <= center_right
            and center_top <= y <= center_bottom
            and green > 180
            and red > 100
            and blue > 100
            and green - red > 8
            and green - blue > 8
        )
        mask_data.append(255 if central_overlay else 0)
    mask = Image.new("L", image.size, 0)
    mask.putdata(mask_data)
    mask = mask.filter(ImageFilter.MaxFilter(5))
    return _inpaint_masked_pixels(image, mask)


def _inpaint_masked_pixels(image: Image.Image, mask: Image.Image) -> Image.Image:
    arr = np.asarray(image, dtype=np.float32).copy()
    pending = np.asarray(mask, dtype=np.uint8) > 0
    known = ~pending
    if not pending.any():
        return image

    for _ in range(24):
        coords = np.argwhere(pending)
        if coords.size == 0:
            break
        updates: list[tuple[int, int, np.ndarray]] = []
        for y, x in coords:
            y0 = max(0, y - 1)
            y1 = min(arr.shape[0], y + 2)
            x0 = max(0, x - 1)
            x1 = min(arr.shape[1], x + 2)
            local_known = known[y0:y1, x0:x1]
            if local_known.any():
                updates.append((y, x, arr[y0:y1, x0:x1][local_known].mean(axis=0)))
        if not updates:
            break
        for y, x, value in updates:
            arr[y, x] = value
            pending[y, x] = False
            known[y, x] = True

    arr[pending] = 203.0
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def _normalize_gazebo_zoom(image: Image.Image) -> Image.Image:
    width, height = image.size
    grid_pitch = _estimate_grid_pitch_px(image)
    zoom_ratio = min(1.0, max(0.35, grid_pitch / GAZEBO_TARGET_GRID_PITCH_PX))
    crop_height = round(height * zoom_ratio)
    crop_width = round(crop_height * GAZEBO_TARGET_ASPECT)
    if crop_width > width:
        crop_width = width
        crop_height = round(crop_width / GAZEBO_TARGET_ASPECT)

    center_x = width // 2
    center_y = height // 2
    left = max(0, min(width - crop_width, center_x - crop_width // 2))
    top = max(0, min(height - crop_height, center_y - crop_height // 2))
    right = left + crop_width
    bottom = top + crop_height
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


def _estimate_grid_pitch_px(image: Image.Image) -> float:
    arr = np.asarray(image.convert("RGB"))
    height, width = arr.shape[:2]
    row_indices = np.linspace(
        round(height * 0.08),
        round(height * 0.92),
        num=13,
        dtype=int,
    )
    pitches: list[int] = []
    for y in row_indices:
        row = arr[min(max(y, 0), height - 1)]
        red = row[:, 0].astype(int)
        green = row[:, 1].astype(int)
        blue = row[:, 2].astype(int)
        line_mask = (
            (np.abs(red - green) < 4)
            & (np.abs(green - blue) < 4)
            & (red > 70)
            & (red < 175)
        )
        centers = _thin_run_centers(line_mask)
        pitches.extend(
            b - a
            for a, b in zip(centers, centers[1:])
            if 20 <= b - a <= 100
        )
    if not pitches:
        return 39.0
    return float(np.median(pitches))


def _thin_run_centers(mask: np.ndarray) -> list[int]:
    centers: list[int] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            end = index - 1
            if end - start <= 3:
                centers.append((start + end) // 2)
            start = None
    if start is not None:
        end = len(mask) - 1
        if end - start <= 3:
            centers.append((start + end) // 2)
    return centers


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create final three-panel paper figures from selected result PNGs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/eval/final_selected_20260704"),
        help="Final selected result root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/eval/final_selected_20260704/comparison/paper_assets/panel_figures"),
        help="Output directory for combined panel figures.",
    )
    parser.add_argument("--gazebo-height", type=int, default=560)
    parser.add_argument("--gazebo-sequence-height", type=int, default=430)
    parser.add_argument("--trajectory-height", type=int, default=980)
    parser.add_argument(
        "--preserve-gazebo-camera",
        action="store_true",
        help="Keep each captured Gazebo camera scale instead of applying a second crop.",
    )
    parser.add_argument(
        "--gazebo-only",
        action="store_true",
        help="Generate Gazebo panels without requiring trajectory figures under the root.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    make_panel(
        args.root,
        GAZEBO_ITEMS,
        args.output_dir / "gazebo_scenarios_panel.png",
        target_height=args.gazebo_height,
        normalize_gazebo_zoom=not args.preserve_gazebo_camera,
    )
    sequence_outputs = []
    for row_label, items in GAZEBO_SEQUENCE_ROWS:
        output = args.output_dir / f"gazebo_sequence_{row_label.lower().replace(' ', '_')}.png"
        sequence_outputs.append(output)
        make_sequence_panel_2x3(
            args.root,
            row_label,
            items,
            output,
            target_height=args.gazebo_sequence_height,
            normalize_gazebo_zoom=not args.preserve_gazebo_camera,
        )
    make_stacked_existing_panels(
        sequence_outputs,
        args.output_dir / "gazebo_sequence_panel.png",
        gutter=38,
    )
    if not args.gazebo_only:
        make_panel(
            args.root,
            TRAJECTORY_ITEMS,
            args.output_dir / "mo_gat_mappo_trajectories_panel.png",
            target_height=args.trajectory_height,
        )
    print(f"Panel figures written to {args.output_dir}")
    return 0


def make_stacked_existing_panels(
    inputs: list[Path],
    output: Path,
    gutter: int = 32,
    padding: int = 24,
) -> None:
    images = [Image.open(path).convert("RGB") for path in inputs]
    max_width = max(image.width for image in images)
    total_height = padding * 2 + sum(image.height for image in images) + gutter * (len(images) - 1)
    canvas = Image.new("RGB", (max_width + padding * 2, total_height), "white")
    y = padding
    for image in images:
        x = padding + (max_width - image.width) // 2
        canvas.paste(image, (x, y))
        y += image.height + gutter
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


if __name__ == "__main__":
    raise SystemExit(main())
