#!/usr/bin/env python3
"""Generate the editable conflict-aware coordination mechanism figure."""

from __future__ import annotations

import svgwrite

from common import COLORS, add_arrow, add_text, make_drawing, save


PANEL_FILL = "#FAFBFC"
PANEL_STROKE = "#C9D3DA"
CORE_FILL = "#F2EFF9"
CORE_STROKE = "#6F62A8"
CONFLICT_FILL = "#FBEAEA"
CONFLICT_STROKE = "#C65050"
GUIDE_FILL = "#EAF1F7"
GUIDE_STROKE = "#5E7F98"


def add_panel(
    dwg: svgwrite.Drawing,
    parent: svgwrite.container.Group,
    *,
    panel_id: str,
    x: float,
    y: float,
    width: float,
    height: float,
    letter: str,
    title: str,
) -> svgwrite.container.Group:
    panel = dwg.g(id=panel_id)
    panel.add(
        dwg.rect(
            insert=(x, y),
            size=(width, height),
            rx=6,
            ry=6,
            fill=PANEL_FILL,
            stroke=PANEL_STROKE,
            stroke_width=1.4,
        )
    )
    panel.add(dwg.circle(center=(x + 24, y + 26), r=14, fill=CORE_STROKE))
    add_text(dwg, panel, letter, x + 24, y + 32, size=18, weight="bold", fill=COLORS["white"])
    add_text(dwg, panel, title, x + 47, y + 31, size=18, weight="bold", anchor="start")
    panel.add(
        dwg.line(
            start=(x + 18, y + 52),
            end=(x + width - 18, y + 52),
            stroke=PANEL_STROKE,
            stroke_width=1.0,
        )
    )
    parent.add(panel)
    return panel


def add_card(
    dwg: svgwrite.Drawing,
    parent: svgwrite.container.Group,
    *,
    card_id: str,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    lines: list[str],
    fill: str,
    stroke: str,
    title_size: int = 19,
    text_size: int = 18,
) -> None:
    card = dwg.g(id=card_id)
    card.add(
        dwg.rect(
            insert=(x, y),
            size=(width, height),
            rx=5,
            ry=5,
            fill=fill,
            stroke=stroke,
            stroke_width=1.3,
        )
    )
    add_text(dwg, card, title, x + width / 2, y + 27, size=title_size, weight="bold")
    add_text(
        dwg,
        card,
        lines,
        x + width / 2,
        y + 64,
        size=text_size,
        fill=COLORS["muted"],
        line_height=1.22,
    )
    parent.add(card)


def add_robot(
    dwg: svgwrite.Drawing,
    parent: svgwrite.container.Group,
    *,
    x: float,
    y: float,
    label: str,
    color: str,
    radius: float = 17,
    self_ring: bool = False,
) -> None:
    robot = dwg.g(id=f"robot_{label}_{int(x)}_{int(y)}")
    if self_ring:
        robot.add(dwg.circle(center=(x, y), r=radius + 7, fill="none", stroke=color, stroke_width=1.3))
    robot.add(dwg.circle(center=(x, y), r=radius, fill=color, stroke=COLORS["ink"], stroke_width=1.2))
    add_text(dwg, robot, label, x, y + 6, size=18, weight="bold", fill=COLORS["white"])
    parent.add(robot)


def add_pill(
    dwg: svgwrite.Drawing,
    parent: svgwrite.container.Group,
    *,
    pill_id: str,
    x: float,
    y: float,
    width: float,
    text: str,
    fill: str,
    stroke: str,
) -> None:
    pill = dwg.g(id=pill_id)
    pill.add(
        dwg.rect(
            insert=(x, y),
            size=(width, 40),
            rx=5,
            ry=5,
            fill=fill,
            stroke=stroke,
            stroke_width=1.2,
        )
    )
    add_text(dwg, pill, text, x + width / 2, y + 26, size=18, weight="bold")
    parent.add(pill)


def main() -> None:
    dwg = make_drawing("conflict_aware_coordination", 1320, 700)
    root = dwg.g(id="conflict_aware_coordination")

    predict = add_panel(
        dwg,
        root,
        panel_id="panel_conflict_prediction",
        x=15,
        y=25,
        width=300,
        height=480,
        letter="a",
        title="Predict pairwise conflict",
    )
    add_robot(dwg, predict, x=72, y=315, label="i", color=COLORS["blue"], radius=18)
    add_robot(dwg, predict, x=258, y=315, label="j", color=COLORS["orange"], radius=18)
    add_arrow(dwg, predict, [(87, 300), (137, 175)], stroke=COLORS["blue"], width=2.4)
    add_arrow(dwg, predict, [(243, 300), (193, 175)], stroke=COLORS["orange"], width=2.4)
    predict.add(dwg.circle(center=(142, 162), r=6, fill=COLORS["white"], stroke=COLORS["blue"], stroke_width=2.0))
    predict.add(dwg.circle(center=(188, 162), r=6, fill=COLORS["white"], stroke=COLORS["orange"], stroke_width=2.0))
    predict.add(
        dwg.line(
            start=(148, 162),
            end=(182, 162),
            stroke=CONFLICT_STROKE,
            stroke_width=2.3,
            stroke_dasharray="6,4",
        )
    )
    add_text(dwg, predict, "dᵢⱼ (CPA)", 165, 149, size=19, weight="bold", fill=CONFLICT_STROKE)
    add_text(dwg, predict, "predicted positions at τᵢⱼ", 165, 224, size=18, fill=COLORS["muted"])
    add_text(dwg, predict, "relative state: rᵢⱼ, qᵢⱼ", 165, 354, size=18, fill=COLORS["muted"])
    add_card(
        dwg,
        predict,
        card_id="edge_test",
        x=40,
        y=380,
        width=250,
        height=98,
        title="Retain conflict edge",
        lines=["0 < τᵢⱼ ≤ 1.25H", "dᵢⱼ (CPA) ≤ 1.35R"],
        fill=CONFLICT_FILL,
        stroke=CONFLICT_STROKE,
        title_size=19,
        text_size=18,
    )

    graph = add_panel(
        dwg,
        root,
        panel_id="panel_sparse_graph_attention",
        x=340,
        y=25,
        width=300,
        height=480,
        letter="b",
        title="Attend only to conflicts",
    )
    node_positions = {
        "1": (410, 145, COLORS["blue"]),
        "2": (565, 145, COLORS["orange"]),
        "3": (410, 245, COLORS["green"]),
        "4": (565, 245, COLORS["purple"]),
    }
    graph.add(dwg.line(start=(410, 145), end=(565, 145), stroke=CONFLICT_STROKE, stroke_width=3.2))
    graph.add(dwg.line(start=(565, 145), end=(565, 245), stroke=CONFLICT_STROKE, stroke_width=3.2))
    for label, (x, y, color) in node_positions.items():
        add_robot(dwg, graph, x=x, y=y, label=label, color=color, self_ring=True)
    add_text(dwg, graph, "Gₜ: self-loops + conflict edges", 490, 294, size=18, fill=COLORS["muted"])
    add_arrow(dwg, graph, [(490, 307), (490, 328)], width=1.4)
    add_card(
        dwg,
        graph,
        card_id="residual_graph_encoder",
        x=365,
        y=334,
        width=250,
        height=112,
        title="Residual GAT encoder",
        lines=["hᵢ = LN(GAT(o)+Wᵣoᵢ)"],
        fill=CORE_FILL,
        stroke=CORE_STROKE,
        title_size=19,
        text_size=18,
    )
    add_text(dwg, graph, "hᵢ → shared actor → aᵢ", 490, 478, size=19, weight="bold", fill=CORE_STROKE)

    coordinate = add_panel(
        dwg,
        root,
        panel_id="panel_right_of_way",
        x=665,
        y=25,
        width=300,
        height=480,
        letter="c",
        title="Coordinate right-of-way",
    )
    add_text(dwg, coordinate, "risk κᵢⱼ", 815, 96, size=18, weight="bold", fill=CONFLICT_STROKE)
    add_robot(dwg, coordinate, x=735, y=155, label="L", color=COLORS["blue"], radius=18)
    add_robot(dwg, coordinate, x=895, y=155, label="F", color=COLORS["orange"], radius=18)
    add_arrow(dwg, coordinate, [(735, 135), (735, 100)], stroke=COLORS["blue"], width=2.2)
    add_arrow(dwg, coordinate, [(895, 135), (895, 113), (860, 96)], stroke=COLORS["orange"], width=2.2)
    add_text(dwg, coordinate, "leader: proceed", 735, 191, size=18, fill=COLORS["blue"])
    add_text(dwg, coordinate, "follower: yield", 895, 191, size=18, fill=COLORS["orange"])
    add_card(
        dwg,
        coordinate,
        card_id="clearance_time_priority",
        x=685,
        y=218,
        width=260,
        height=66,
        title="Clearance-time priority",
        lines=[],
        fill=GUIDE_FILL,
        stroke=GUIDE_STROKE,
        title_size=18,
        text_size=18,
    )
    add_arrow(dwg, coordinate, [(815, 284), (815, 314)], width=1.4)
    add_card(
        dwg,
        coordinate,
        card_id="bounded_coordination_outputs",
        x=685,
        y=320,
        width=260,
        height=122,
        title="Bounded follower control",
        lines=["s_f=1-(1-s_min)κᵢⱼ", "‖c_f‖ ≤ v⊥"],
        fill=CORE_FILL,
        stroke=CORE_STROKE,
        title_size=18,
        text_size=18,
    )
    add_text(dwg, coordinate, "outputs: speed sᵢ, lateral cᵢ", 815, 478, size=18, weight="bold", fill=CORE_STROKE)

    gate = add_panel(
        dwg,
        root,
        panel_id="panel_interaction_gate",
        x=990,
        y=25,
        width=315,
        height=480,
        letter="d",
        title="Gate residual authority",
    )
    axis_left = 1035
    axis_right = 1265
    axis_bottom = 310
    axis_top = 120
    gate.add(dwg.line(start=(axis_left, axis_bottom), end=(axis_right, axis_bottom), stroke=COLORS["line"], stroke_width=1.3))
    gate.add(dwg.line(start=(axis_left, axis_bottom), end=(axis_left, axis_top), stroke=COLORS["line"], stroke_width=1.3))
    gate.add(dwg.line(start=(axis_left + 5, axis_bottom), end=(1105, axis_bottom), stroke=COLORS["green"], stroke_width=4.0))
    gate.add(dwg.line(start=(1105, axis_bottom), end=(1190, axis_top + 28), stroke=COLORS["orange"], stroke_width=4.0))
    gate.add(dwg.line(start=(1190, axis_top + 28), end=(axis_right, axis_top + 28), stroke=CONFLICT_STROKE, stroke_width=4.0))
    add_text(dwg, gate, "gᵢ", axis_left - 12, axis_top + 7, size=18, weight="bold", anchor="end")
    add_text(dwg, gate, "0", axis_left - 10, axis_bottom + 6, size=18, anchor="end", fill=COLORS["muted"])
    add_text(dwg, gate, "1", axis_left - 10, axis_top + 34, size=18, anchor="end", fill=COLORS["muted"])
    add_text(dwg, gate, "clear", 1060, 337, size=18, fill=COLORS["green"])
    add_text(dwg, gate, "pedestrian", 1147, 337, size=18, fill=COLORS["orange"])
    add_text(dwg, gate, "conflict", 1233, 337, size=18, fill=CONFLICT_STROKE)
    add_text(dwg, gate, "interaction risk", 1150, 362, size=18, fill=COLORS["muted"])
    add_card(
        dwg,
        gate,
        card_id="gated_action_equations",
        x=1007,
        y=378,
        width=281,
        height=104,
        title="Gated residual action",
        lines=["ãₓ=max(aₓ,0)+gᵢ min(aₓ,0)", "ãᵧ=gᵢaᵧ;  ãω=gᵢaω"],
        fill=CORE_FILL,
        stroke=CORE_STROKE,
        title_size=19,
        text_size=18,
    )

    outputs = dwg.g(id="mechanism_outputs")
    outputs.add(
        dwg.rect(
            insert=(80, 545),
            size=(1225, 115),
            rx=7,
            ry=7,
            fill="#F8F6FC",
            stroke=CORE_STROKE,
            stroke_width=1.5,
        )
    )
    add_text(
        dwg,
        outputs,
        "POLICY-SIDE OUTPUTS TO RESIDUAL COMMAND COMPOSITION",
        100,
        576,
        size=20,
        weight="bold",
        anchor="start",
        fill=CORE_STROKE,
    )
    add_pill(
        dwg,
        outputs,
        pill_id="action_output",
        x=110,
        y=594,
        width=250,
        text="hᵢ → shared actor → aᵢ",
        fill=COLORS["proposed"],
        stroke=CORE_STROKE,
    )
    add_pill(
        dwg,
        outputs,
        pill_id="speed_output",
        x=385,
        y=594,
        width=180,
        text="speed scale  sᵢ",
        fill=COLORS["proposed"],
        stroke=CORE_STROKE,
    )
    add_pill(
        dwg,
        outputs,
        pill_id="lateral_output",
        x=590,
        y=594,
        width=220,
        text="lateral preference  cᵢ",
        fill=COLORS["proposed"],
        stroke=CORE_STROKE,
    )
    add_pill(
        dwg,
        outputs,
        pill_id="gate_output",
        x=820,
        y=594,
        width=190,
        text="residual gate  gᵢ",
        fill=COLORS["proposed"],
        stroke=CORE_STROKE,
    )
    add_pill(
        dwg,
        outputs,
        pill_id="bounded_command",
        x=1046,
        y=589,
        width=220,
        text="local command  uᵢᵖʳᵉ",
        fill="#E5E0F4",
        stroke=CORE_STROKE,
    )
    merge_centers = [235, 475, 700, 915]
    for center_x in merge_centers:
        add_arrow(
            dwg,
            outputs,
            [(center_x, 634), (center_x, 646)],
            stroke=COLORS["proposed_stroke"],
            width=1.2,
            end=False,
        )
    add_arrow(
        dwg,
        outputs,
        [(235, 646), (1156, 646)],
        stroke=COLORS["proposed_stroke"],
        width=1.2,
        end=False,
    )
    add_arrow(
        dwg,
        outputs,
        [(1156, 646), (1156, 635)],
        stroke=COLORS["proposed_stroke"],
        width=1.3,
    )
    root.add(outputs)

    links = dwg.g(id="mechanism_links")
    add_arrow(dwg, links, [(315, 165), (334, 165)], stroke=CONFLICT_STROKE, width=1.4)
    add_arrow(dwg, links, [(490, 505), (490, 539)], stroke=CORE_STROKE, width=1.4)
    add_arrow(dwg, links, [(815, 505), (815, 539)], stroke=CORE_STROKE, width=1.4)
    add_arrow(dwg, links, [(1147, 505), (1147, 539)], stroke=CORE_STROKE, width=1.4)
    root.add(links)

    dwg.add(root)
    save(dwg)


if __name__ == "__main__":
    main()
