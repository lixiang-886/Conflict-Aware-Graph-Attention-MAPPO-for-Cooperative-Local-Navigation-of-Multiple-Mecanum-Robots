"""Shared drawing helpers for the manuscript's editable vector figures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import svgwrite


ROOT = Path(__file__).resolve().parents[2]
EDITABLE_DIR = ROOT / "paper" / "mdpi" / "figures_editable"
VECTOR_DIR = ROOT / "paper" / "mdpi" / "figures_vector"

COLORS = {
    "ink": "#24313A",
    "muted": "#64727D",
    "line": "#87949D",
    "shared": "#EEF1F3",
    "shared_stroke": "#7A8790",
    "proposed": "#ECE9F7",
    "proposed_stroke": "#7566A8",
    "safety": "#FCEAD8",
    "safety_stroke": "#C5742F",
    "sensor": "#E4F2EC",
    "sensor_stroke": "#4A8C72",
    "blue": "#4C78A8",
    "orange": "#F58518",
    "green": "#54A24B",
    "purple": "#B279A2",
    "red": "#C65050",
    "panel": "#FAFBFC",
    "white": "#FFFFFF",
}

ARROW_MARKERS = {
    "arrow": COLORS["ink"],
    "arrow-muted": COLORS["muted"],
    "arrow-shared": COLORS["shared_stroke"],
    "arrow-proposed": COLORS["proposed_stroke"],
    "arrow-safety": COLORS["safety_stroke"],
    "arrow-sensor": COLORS["sensor_stroke"],
    "arrow-blue": COLORS["blue"],
    "arrow-orange": COLORS["orange"],
    "arrow-green": COLORS["green"],
    "arrow-purple": COLORS["purple"],
    "arrow-red": COLORS["red"],
}


def make_drawing(name: str, width: int, height: int) -> svgwrite.Drawing:
    """Create a full-profile SVG with a shared arrow marker and white canvas."""
    EDITABLE_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    output = EDITABLE_DIR / f"{name}.svg"
    height_mm = 180.0 * height / width
    dwg = svgwrite.Drawing(
        str(output),
        size=("180mm", f"{height_mm:.2f}mm"),
        viewBox=f"0 0 {width} {height}",
        profile="full",
    )
    for marker_id, marker_color in ARROW_MARKERS.items():
        marker = dwg.marker(
            insert=(9, 5),
            size=(10, 10),
            orient="auto",
            markerUnits="strokeWidth",
            id=marker_id,
        )
        marker.add(dwg.path(d="M 0 0 L 10 5 L 0 10 z", fill=marker_color))
        dwg.defs.add(marker)
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill=COLORS["white"]))
    return dwg


def add_text(
    dwg: svgwrite.Drawing,
    parent: svgwrite.container.Group,
    lines: str | Iterable[str],
    x: float,
    y: float,
    *,
    size: int = 16,
    weight: str = "normal",
    anchor: str = "middle",
    fill: str | None = None,
    line_height: float = 1.25,
    italic: bool = False,
) -> svgwrite.text.Text:
    """Add editable SVG text, using tspans for explicit line breaks."""
    if isinstance(lines, str):
        split_lines = [lines]
    else:
        split_lines = list(lines)
    text = dwg.text(
        "",
        insert=(x, y),
        text_anchor=anchor,
        font_family="DejaVu Sans, Liberation Sans, Arial, sans-serif",
        font_size=size,
        font_weight=weight,
        font_style="italic" if italic else "normal",
        fill=fill or COLORS["ink"],
    )
    offset = -0.5 * (len(split_lines) - 1) * size * line_height
    for index, line in enumerate(split_lines):
        text.add(
            dwg.tspan(
                line,
                x=[x],
                dy=[offset if index == 0 else size * line_height],
            )
        )
        offset = 0
    parent.add(text)
    return text


def add_math_text(
    dwg: svgwrite.Drawing,
    parent: svgwrite.container.Group,
    parts: Iterable[tuple[str, str | None]],
    x: float,
    y: float,
    *,
    size: int = 18,
    anchor: str = "middle",
    fill: str | None = None,
    weight: str = "normal",
) -> svgwrite.text.Text:
    """Add editable math-like text with SVG sub/superscript tspans."""
    text = dwg.text(
        "",
        insert=(x, y),
        text_anchor=anchor,
        font_family="DejaVu Sans, Liberation Sans, Arial, sans-serif",
        font_size=size,
        font_style="italic",
        font_weight=weight,
        fill=fill or COLORS["ink"],
    )
    for value, script in parts:
        span = dwg.tspan(value)
        if script is not None:
            span["baseline-shift"] = script
            span["font-size"] = f"{0.72 * size:.1f}px"
        text.add(span)
    parent.add(text)
    return text


def add_box(
    dwg: svgwrite.Drawing,
    parent: svgwrite.container.Group,
    *,
    box_id: str,
    x: float,
    y: float,
    width: float,
    height: float,
    lines: str | Iterable[str],
    fill: str,
    stroke: str,
    font_size: int = 15,
    radius: int = 7,
    stroke_width: float = 1.6,
) -> svgwrite.container.Group:
    group = dwg.g(id=box_id)
    group.add(
        dwg.rect(
            insert=(x, y),
            size=(width, height),
            rx=radius,
            ry=radius,
            fill=fill,
            stroke=stroke,
            stroke_width=stroke_width,
        )
    )
    add_text(
        dwg,
        group,
        lines,
        x + width / 2,
        y + height / 2 + font_size * 0.34,
        size=font_size,
        weight="bold",
    )
    parent.add(group)
    return group


def add_arrow(
    dwg: svgwrite.Drawing,
    parent: svgwrite.container.Group,
    points: Iterable[tuple[float, float]],
    *,
    stroke: str | None = None,
    width: float = 1.8,
    dashed: bool = False,
    end: bool = True,
) -> svgwrite.shapes.Polyline:
    stroke_color = stroke or COLORS["ink"]
    line = dwg.polyline(
        points=list(points),
        fill="none",
        stroke=stroke_color,
        stroke_width=width,
        stroke_linecap="round",
        stroke_linejoin="round",
    )
    if dashed:
        line["stroke-dasharray"] = "7,5"
    if end:
        marker_id = next(
            (
                name
                for name, color in ARROW_MARKERS.items()
                if color.lower() == stroke_color.lower()
            ),
            "arrow",
        )
        line["marker-end"] = f"url(#{marker_id})"
    parent.add(line)
    return line


def add_panel(
    dwg: svgwrite.Drawing,
    parent: svgwrite.container.Group,
    panel_id: str,
    label: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> svgwrite.container.Group:
    panel = dwg.g(id=panel_id)
    panel.add(
        dwg.rect(
            insert=(x, y),
            size=(width, height),
            rx=4,
            ry=4,
            fill=COLORS["panel"],
            stroke="#CDD6DC",
            stroke_width=1.2,
        )
    )
    add_text(
        dwg,
        panel,
        label,
        x + 12,
        y + 25,
        size=17,
        weight="bold",
        anchor="start",
    )
    parent.add(panel)
    return panel


def save(dwg: svgwrite.Drawing) -> Path:
    dwg.save(pretty=True)
    return Path(dwg.filename)
