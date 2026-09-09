#!/usr/bin/env python3
"""Generate an editable top-view mecanum geometry diagram."""

from __future__ import annotations

import svgwrite

from common import COLORS, add_arrow, add_math_text, add_text, make_drawing, save


def add_wheel(
    dwg: svgwrite.Drawing,
    parent: svgwrite.container.Group,
    *,
    wheel_id: str,
    x: float,
    y: float,
    label: str,
    roller_slope: int,
    label_above: bool,
) -> None:
    group = dwg.g(id=wheel_id)
    group.add(
        dwg.rect(
            insert=(x, y),
            size=(100, 52),
            rx=5,
            ry=5,
            fill="#D7DEE3",
            stroke=COLORS["ink"],
            stroke_width=2,
        )
    )
    for offset in (8, 31, 54, 77):
        if roller_slope > 0:
            start = (x + offset, y + 44)
            end = (x + offset + 20, y + 8)
        else:
            start = (x + offset, y + 8)
            end = (x + offset + 20, y + 44)
        group.add(dwg.line(start=start, end=end, stroke=COLORS["line"], stroke_width=2.2))
    add_text(
        dwg,
        group,
        label,
        x + 50,
        y - 12 if label_above else y + 79,
        size=21,
        weight="bold",
    )
    parent.add(group)


def add_horizontal_dimension(
    dwg: svgwrite.Drawing,
    parent: svgwrite.container.Group,
    x1: float,
    x2: float,
    y: float,
) -> None:
    parent.add(dwg.line(start=(x1, y), end=(x2, y), stroke=COLORS["ink"], stroke_width=1.6))
    parent.add(dwg.polygon(points=[(x1, y), (x1 + 10, y - 5), (x1 + 10, y + 5)], fill=COLORS["ink"]))
    parent.add(dwg.polygon(points=[(x2, y), (x2 - 10, y - 5), (x2 - 10, y + 5)], fill=COLORS["ink"]))


def add_vertical_dimension(
    dwg: svgwrite.Drawing,
    parent: svgwrite.container.Group,
    x: float,
    y1: float,
    y2: float,
) -> None:
    parent.add(dwg.line(start=(x, y1), end=(x, y2), stroke=COLORS["ink"], stroke_width=1.6))
    parent.add(dwg.polygon(points=[(x, y1), (x - 5, y1 + 10), (x + 5, y1 + 10)], fill=COLORS["ink"]))
    parent.add(dwg.polygon(points=[(x, y2), (x - 5, y2 - 10), (x + 5, y2 - 10)], fill=COLORS["ink"]))


def main() -> None:
    dwg = make_drawing("mecanum_geometry", 900, 610)
    root = dwg.g(id="mecanum_geometry")

    world = dwg.g(id="world_frame")
    origin_w = (105, 485)
    add_arrow(dwg, world, [origin_w, (230, 485)], width=2.2)
    add_arrow(dwg, world, [origin_w, (105, 355)], width=2.2)
    add_math_text(dwg, world, [("x", None), ("W", "sub")], 240, 493, size=21, anchor="start")
    add_math_text(dwg, world, [("y", None), ("W", "sub")], 95, 343, size=21)
    add_text(dwg, world, "{W}", 72, 522, size=22, weight="bold")
    yaw = dwg.path(
        d="M 205 478 A 78 78 0 0 0 151 405",
        fill="none",
        stroke=COLORS["blue"],
        stroke_width=2.2,
    )
    yaw["marker-end"] = "url(#arrow)"
    world.add(yaw)
    add_math_text(dwg, world, [("yaw\u00a0", None), ("ψ", None)], 197, 405, size=20, fill=COLORS["blue"])
    root.add(world)

    robot = dwg.g(id="robot_top_view")
    robot.add(
        dwg.rect(
            insert=(320, 145),
            size=(410, 320),
            rx=18,
            ry=18,
            fill=COLORS["shared"],
            stroke=COLORS["ink"],
            stroke_width=2.4,
        )
    )
    add_wheel(dwg, robot, wheel_id="wheel_bl", x=280, y=100, label="BL", roller_slope=-1, label_above=True)
    add_wheel(dwg, robot, wheel_id="wheel_fl", x=670, y=100, label="FL", roller_slope=1, label_above=True)
    add_wheel(dwg, robot, wheel_id="wheel_br", x=280, y=440, label="BR", roller_slope=1, label_above=False)
    add_wheel(dwg, robot, wheel_id="wheel_fr", x=670, y=440, label="FR", roller_slope=-1, label_above=False)

    center = (525, 305)
    robot.add(dwg.circle(center=center, r=5, fill=COLORS["ink"]))
    add_arrow(dwg, robot, [center, (675, 305)], width=2.4)
    add_arrow(dwg, robot, [center, (525, 165)], width=2.4)
    add_text(dwg, robot, "{B}", 492, 337, size=21, weight="bold")
    add_math_text(
        dwg,
        robot,
        [("+x", None), ("B", "sub"), (", v", None), ("x", "sub")],
        684,
        312,
        size=21,
        anchor="start",
    )
    add_math_text(
        dwg,
        robot,
        [("+y", None), ("B", "sub"), (", v", None), ("y", "sub")],
        542,
        164,
        size=21,
        anchor="start",
    )
    add_text(dwg, robot, "front", 746, 347, size=19, anchor="start", fill=COLORS["muted"])
    omega = dwg.path(
        d="M 588 305 A 63 63 0 0 0 530 242",
        fill="none",
        stroke=COLORS["purple"],
        stroke_width=2.2,
    )
    omega["marker-end"] = "url(#arrow)"
    robot.add(omega)
    add_math_text(dwg, robot, [("ω", None)], 601, 250, size=23, fill=COLORS["purple"])

    dimensions = dwg.g(id="dimensions")
    add_horizontal_dimension(dwg, dimensions, 525, 720, 402)
    add_math_text(dwg, dimensions, [("L", None), ("x", "sub")], 622, 389, size=21)
    add_vertical_dimension(dwg, dimensions, 415, 126, 305)
    add_math_text(dwg, dimensions, [("L", None), ("y", "sub")], 398, 221, size=21)
    add_arrow(dwg, dimensions, [(860, 126), (780, 126)], width=1.6)
    add_math_text(dwg, dimensions, [("r", None), ("w", "sub")], 855, 112, size=21, anchor="end")
    robot.add(dimensions)
    add_text(dwg, robot, "X-arranged roller directions", 525, 570, size=19, fill=COLORS["muted"])
    root.add(robot)

    dwg.add(root)
    save(dwg)


if __name__ == "__main__":
    main()
