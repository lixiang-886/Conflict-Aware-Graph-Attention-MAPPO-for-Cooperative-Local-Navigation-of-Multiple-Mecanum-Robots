#!/usr/bin/env python3
"""Generate the editable overview of the MO-GAT-MAPPO navigation stack."""

from __future__ import annotations

from collections.abc import Iterable

import svgwrite

from common import COLORS, add_arrow, add_text, make_drawing, save


GUIDE_FILL = "#EAF1F7"
GUIDE_STROKE = "#5E7F98"
CORE_FILL = "#F7F5FC"
CORE_STROKE = "#6F62A8"
TRAIN_FILL = "#FCFBFE"
SAFETY_FILL = "#FFF0DD"
ENV_FILL = "#E7F4EF"
CONFLICT_FILL = "#FBEAEA"
MIN_FONT_SIZE = 20
PANEL_TITLE_SIZE = 22


def add_card(
    dwg: svgwrite.Drawing,
    parent: svgwrite.container.Group,
    *,
    card_id: str,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str | Iterable[str],
    subtitle: str | Iterable[str] | None,
    fill: str,
    stroke: str,
    title_size: int = MIN_FONT_SIZE,
    subtitle_size: int = MIN_FONT_SIZE,
    dashed: bool = False,
) -> svgwrite.container.Group:
    """Add a two-level module card with an editable title and subtitle."""
    group = dwg.g(id=card_id)
    border = dwg.rect(
        insert=(x, y),
        size=(width, height),
        rx=6,
        ry=6,
        fill=fill,
        stroke=stroke,
        stroke_width=1.5,
    )
    if dashed:
        border["stroke-dasharray"] = "7,5"
    group.add(border)

    if subtitle is None:
        add_text(
            dwg,
            group,
            title,
            x + width / 2,
            y + height / 2 + title_size * 0.34,
            size=title_size,
            weight="bold",
        )
    else:
        add_text(
            dwg,
            group,
            title,
            x + width / 2,
            y + 27,
            size=title_size,
            weight="bold",
            line_height=1.05,
        )
        add_text(
            dwg,
            group,
            subtitle,
            x + width / 2,
            y + height - 13,
            size=subtitle_size,
            fill=COLORS["muted"],
            line_height=1.05,
        )
    parent.add(group)
    return group


def add_group_panel(
    dwg: svgwrite.Drawing,
    parent: svgwrite.container.Group,
    *,
    panel_id: str,
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    stroke: str,
    title: str,
    subtitle: str | None = None,
    title_size: int = PANEL_TITLE_SIZE,
    dashed: bool = False,
) -> svgwrite.container.Group:
    panel = dwg.g(id=panel_id)
    border = dwg.rect(
        insert=(x, y),
        size=(width, height),
        rx=7,
        ry=7,
        fill=fill,
        stroke=stroke,
        stroke_width=1.6,
    )
    if dashed:
        border["stroke-dasharray"] = "8,5"
    panel.add(border)
    add_text(
        dwg,
        panel,
        title,
        x + 18,
        y + 31,
        size=title_size,
        weight="bold",
        anchor="start",
        fill=stroke,
    )
    if subtitle:
        add_text(
            dwg,
            panel,
            subtitle,
            x + width - 18,
            y + 30,
            size=MIN_FONT_SIZE,
            anchor="end",
            fill=COLORS["muted"],
        )
    parent.add(panel)
    return panel


def add_scene_icon(
    dwg: svgwrite.Drawing,
    parent: svgwrite.container.Group,
    x: float,
    y: float,
) -> None:
    """Draw a small flat top-down multi-robot scene without raster assets."""
    scene = dwg.g(id="gazebo_state_icon")
    scene.add(
        dwg.rect(
            insert=(x, y),
            size=(174, 82),
            rx=4,
            ry=4,
            fill=COLORS["white"],
            stroke="#BDD0C8",
            stroke_width=1.1,
        )
    )
    for ox, oy, ow, oh in [(x + 67, y + 9, 40, 15), (x + 67, y + 58, 40, 15)]:
        scene.add(dwg.rect(insert=(ox, oy), size=(ow, oh), fill="#D9DEE2", stroke="#9CA8B0"))
    robots = [
        (x + 28, y + 24, COLORS["blue"]),
        (x + 146, y + 24, COLORS["orange"]),
        (x + 28, y + 61, COLORS["green"]),
        (x + 146, y + 61, COLORS["purple"]),
    ]
    for index, (rx, ry, color) in enumerate(robots, start=1):
        scene.add(dwg.circle(center=(rx, ry), r=12, fill=color, stroke=COLORS["ink"], stroke_width=1.0))
        add_text(
            dwg,
            scene,
            str(index),
            rx,
            ry + 7,
            size=MIN_FONT_SIZE,
            weight="bold",
            fill=COLORS["white"],
        )
    scene.add(dwg.circle(center=(x + 87, y + 41), r=7, fill="#E9B44C", stroke="#A46D16", stroke_width=1.0))
    parent.add(scene)


def main() -> None:
    dwg = make_drawing("system_architecture", 1400, 850)
    root = dwg.g(id="system_architecture")

    guidance = add_group_panel(
        dwg,
        root,
        panel_id="shared_route_guidance",
        x=20,
        y=20,
        width=1360,
        height=135,
        fill="#F7FAFC",
        stroke=GUIDE_STROKE,
        title="SHARED ROUTE GUIDANCE",
        subtitle="common to every compared controller",
    )
    route_cards = [
        ("scenario_map", 245, 72, 190, "Scenario / map", "starts, goals, map"),
        ("astar_path", 470, 72, 180, "Static-map A*", "reference path  Pᵢ"),
        ("waypoint_target", 685, 72, 200, "Waypoint target", "active lookahead  ĝᵢᵗ"),
        ("waypoint_velocity", 920, 72, 195, "Velocity command", "waypoint  uᵢʷᵖ"),
    ]
    for card_id, x, y, width, title, subtitle in route_cards:
        add_card(
            dwg,
            guidance,
            card_id=card_id,
            x=x,
            y=y,
            width=width,
            height=62,
            title=title,
            subtitle=subtitle,
            fill=GUIDE_FILL,
            stroke=GUIDE_STROKE,
        )
    for start_x, end_x in [(435, 470), (650, 685), (885, 920)]:
        add_arrow(dwg, guidance, [(start_x, 103), (end_x - 7, 103)], width=1.6)

    state = add_group_panel(
        dwg,
        root,
        panel_id="online_state",
        x=20,
        y=185,
        width=230,
        height=335,
        fill="#F5FBF8",
        stroke=COLORS["sensor_stroke"],
        title="ONLINE STATE",
    )
    add_scene_icon(dwg, state, 48, 232)
    add_card(
        dwg,
        state,
        card_id="robot_state",
        x=45,
        y=334,
        width=180,
        height=54,
        title="Robot state",
        subtitle="pose + velocity",
        fill=ENV_FILL,
        stroke=COLORS["sensor_stroke"],
    )
    add_card(
        dwg,
        state,
        card_id="local_lidar",
        x=45,
        y=399,
        width=180,
        height=54,
        title="Local sensing",
        subtitle="8-sector LiDAR",
        fill=ENV_FILL,
        stroke=COLORS["sensor_stroke"],
    )
    add_card(
        dwg,
        state,
        card_id="dynamic_agents",
        x=45,
        y=452,
        width=180,
        height=54,
        title=["Neighbors and", "pedestrians"],
        subtitle=None,
        fill=ENV_FILL,
        stroke=COLORS["sensor_stroke"],
    )

    proposed = add_group_panel(
        dwg,
        root,
        panel_id="proposed_online_policy",
        x=280,
        y=185,
        width=730,
        height=335,
        fill=CORE_FILL,
        stroke=CORE_STROKE,
        title="COMMUNICATION-ENABLED ACTOR INFERENCE",
        subtitle=None,
    )
    add_card(
        dwg,
        proposed,
        card_id="cpa_predictor",
        x=310,
        y=250,
        width=150,
        height=72,
        title=["CPA conflict", "test"],
        subtitle="τᵢⱼ,  d_CPA",
        fill=CONFLICT_FILL,
        stroke=COLORS["red"],
    )
    add_card(
        dwg,
        proposed,
        card_id="sparse_conflict_graph",
        x=490,
        y=225,
        width=160,
        height=72,
        title=["Sparse", "conflict graph"],
        subtitle="adjacency  Gₜ",
        fill=COLORS["proposed"],
        stroke=CORE_STROKE,
    )
    add_card(
        dwg,
        proposed,
        card_id="residual_gat",
        x=670,
        y=225,
        width=160,
        height=72,
        title=["Residual GAT", "encoder"],
        subtitle="self + messages",
        fill=COLORS["proposed"],
        stroke=CORE_STROKE,
    )
    add_card(
        dwg,
        proposed,
        card_id="actor_head",
        x=850,
        y=225,
        width=130,
        height=72,
        title=["Shared  πθ", "64→64→3"],
        subtitle="hᵢ → aᵢ",
        fill=COLORS["proposed"],
        stroke=CORE_STROKE,
    )
    add_card(
        dwg,
        proposed,
        card_id="right_of_way",
        x=490,
        y=360,
        width=150,
        height=72,
        title="Right-of-way",
        subtitle="bounded  sᵢ, cᵢ",
        fill=COLORS["proposed"],
        stroke=CORE_STROKE,
    )
    add_card(
        dwg,
        proposed,
        card_id="interaction_gate",
        x=670,
        y=360,
        width=150,
        height=72,
        title=["Interaction", "gate"],
        subtitle="authority  gᵢ",
        fill=COLORS["proposed"],
        stroke=CORE_STROKE,
    )
    add_card(
        dwg,
        proposed,
        card_id="gated_residuals",
        x=850,
        y=360,
        width=130,
        height=72,
        title=["Gated", "residuals"],
        subtitle="ãᵢ",
        fill="#E5E0F4",
        stroke=CORE_STROKE,
    )

    add_arrow(dwg, proposed, [(460, 275), (484, 261)], width=1.5)
    add_arrow(dwg, proposed, [(650, 261), (664, 261)], width=1.5)
    add_arrow(dwg, proposed, [(830, 261), (844, 261)], width=1.5)
    add_arrow(dwg, proposed, [(385, 322), (385, 396), (484, 396)], width=1.5)
    add_arrow(dwg, proposed, [(820, 396), (844, 396)], width=1.5)
    add_arrow(dwg, proposed, [(915, 297), (915, 354)], width=1.5)
    add_arrow(dwg, proposed, [(640, 396), (650, 396), (650, 454), (915, 454), (915, 438)], width=1.5)
    add_arrow(dwg, proposed, [(385, 322), (385, 350), (745, 350), (745, 354)], width=1.4)
    add_text(
        dwg,
        proposed,
        "conflict flag",
        565,
        342,
        size=MIN_FONT_SIZE,
        fill=COLORS["red"],
    )

    execution = add_group_panel(
        dwg,
        root,
        panel_id="shared_execution",
        x=1040,
        y=185,
        width=340,
        height=430,
        fill="#FFFCF7",
        stroke=COLORS["safety_stroke"],
        title="COMMON EXECUTION",
        title_size=MIN_FONT_SIZE,
        subtitle=None,
    )
    add_card(
        dwg,
        execution,
        card_id="orca_prior",
        x=1065,
        y=230,
        width=290,
        height=68,
        title="ORCA reciprocal prior",
        subtitle="robot + pedestrian states",
        fill=GUIDE_FILL,
        stroke=GUIDE_STROKE,
    )
    add_card(
        dwg,
        execution,
        card_id="residual_composition",
        x=1065,
        y=325,
        width=290,
        height=76,
        title=["Residual command", "composition"],
        subtitle="waypoint + prior + residual",
        fill=COLORS["proposed"],
        stroke=CORE_STROKE,
    )
    add_card(
        dwg,
        execution,
        card_id="command_limits",
        x=1065,
        y=422,
        width=125,
        height=78,
        title=["Limits &", "smoothing"],
        subtitle="L(uᵖʳᵉ)",
        fill=COLORS["shared"],
        stroke=COLORS["shared_stroke"],
    )
    add_card(
        dwg,
        execution,
        card_id="final_safety_filter",
        x=1200,
        y=422,
        width=155,
        height=78,
        title=["Shared safety", "envelope"],
        subtitle="LiDAR | RR | RP",
        fill=SAFETY_FILL,
        stroke=COLORS["safety_stroke"],
    )
    add_card(
        dwg,
        execution,
        card_id="gazebo_mecanum",
        x=1065,
        y=535,
        width=290,
        height=64,
        title="ROS 2 / Gazebo robots",
        subtitle="mecanum; cmd_vel; sensors",
        fill=ENV_FILL,
        stroke=COLORS["sensor_stroke"],
    )
    add_arrow(dwg, execution, [(1210, 298), (1210, 319)], width=1.6)
    add_arrow(dwg, execution, [(1210, 401), (1210, 412), (1132, 412), (1132, 416)], width=1.5)
    add_arrow(dwg, execution, [(1190, 461), (1194, 461)], width=1.5)
    add_arrow(dwg, execution, [(1287, 500), (1287, 529)], width=1.6)
    add_text(
        dwg,
        execution,
        "safe uᵢ",
        1335,
        526,
        size=MIN_FONT_SIZE,
        fill=COLORS["safety_stroke"],
    )

    training = add_group_panel(
        dwg,
        root,
        panel_id="training_only",
        x=280,
        y=540,
        width=730,
        height=235,
        fill=TRAIN_FILL,
        stroke=CORE_STROKE,
        title="CENTRALIZED CRITIC TRAINING",
        subtitle=None,
        dashed=True,
    )
    add_text(
        dwg,
        training,
        "no value-to-action path",
        990,
        570,
        size=MIN_FONT_SIZE,
        anchor="end",
        fill=COLORS["muted"],
    )
    training_cards = [
        (
            "adaptive_objective",
            300,
            590,
            185,
            ["Adaptive", "scalarization"],
            "weights  wᵢ → rᵢ",
        ),
        (
            "joint_state_pool",
            500,
            590,
            215,
            "Joint observations",
            "Oₜ: 16→64; pool  c̄ₜ",
        ),
        (
            "centralized_critic",
            730,
            590,
            255,
            ["Central critic  Vφ,ψ", "128→64→1"],
            "joint [hᵢ;c̄ₜ] → Vᵢ",
        ),
        (
            "per_robot_credit",
            395,
            687,
            260,
            "Per-robot GAE / credit",
            "Ãᵢ = .75 Aᵢ + .25 Ā",
        ),
        (
            "ppo_update",
            685,
            687,
            260,
            "Clipped PPO update",
            "actor + encoder + critic",
        ),
    ]
    for card_id, x, y, width, title, subtitle in training_cards:
        add_card(
            dwg,
            training,
            card_id=card_id,
            x=x,
            y=y,
            width=width,
            height=72 if y > 680 else 78,
            title=title,
            subtitle=subtitle,
            fill=COLORS["white"],
            stroke=CORE_STROKE,
            dashed=True,
        )
    add_arrow(
        dwg,
        training,
        [(392, 668), (392, 678), (450, 678), (450, 681)],
        stroke=COLORS["proposed_stroke"],
        width=1.4,
        dashed=True,
    )
    add_arrow(
        dwg,
        training,
        [(715, 629), (724, 629)],
        stroke=COLORS["proposed_stroke"],
        width=1.4,
        dashed=True,
    )
    add_arrow(
        dwg,
        training,
        [(857, 668), (857, 678), (620, 678), (620, 681)],
        stroke=COLORS["proposed_stroke"],
        width=1.4,
        dashed=True,
    )
    add_arrow(
        dwg,
        training,
        [(655, 723), (679, 723)],
        stroke=COLORS["proposed_stroke"],
        width=1.4,
        dashed=True,
    )

    links = dwg.g(id="cross_group_links")
    add_arrow(dwg, links, [(1017, 134), (1017, 166), (385, 166), (385, 244)], width=1.5)
    add_text(
        dwg,
        links,
        "nominal waypoint motion",
        690,
        179,
        size=MIN_FONT_SIZE,
        fill=GUIDE_STROKE,
    )
    add_arrow(dwg, links, [(1115, 103), (1370, 103), (1370, 264), (1361, 264)], width=1.5)
    add_arrow(dwg, links, [(250, 286), (304, 286)], stroke=COLORS["sensor_stroke"], width=1.7)
    add_text(dwg, links, "states", 273, 275, size=MIN_FONT_SIZE, fill=COLORS["sensor_stroke"])
    add_arrow(
        dwg,
        links,
        [(250, 485), (270, 485), (270, 480), (655, 480), (655, 396), (664, 396)],
        stroke=COLORS["sensor_stroke"],
        width=1.5,
    )
    add_text(
        dwg,
        links,
        "pedestrian range",
        460,
        478,
        size=MIN_FONT_SIZE,
        fill=COLORS["sensor_stroke"],
    )
    add_arrow(dwg, links, [(980, 396), (1059, 363)], width=1.8)
    add_text(dwg, links, "policy signals", 1015, 348, size=MIN_FONT_SIZE, fill=CORE_STROKE)
    add_arrow(
        dwg,
        links,
        [(250, 360), (265, 360), (265, 580), (605, 580), (605, 589)],
        stroke=COLORS["proposed_stroke"],
        width=1.4,
        dashed=True,
    )
    add_arrow(
        dwg,
        links,
        [(745, 297), (745, 325), (1018, 325), (1018, 627), (991, 627)],
        stroke=COLORS["proposed_stroke"],
        width=1.4,
        dashed=True,
    )
    add_arrow(
        dwg,
        training,
        [(900, 685), (975, 675), (975, 666)],
        stroke=COLORS["proposed_stroke"],
        width=1.4,
        dashed=True,
    )
    add_arrow(
        dwg,
        links,
        [(1065, 565), (1030, 565), (1030, 785), (290, 785), (290, 627), (299, 627)],
        stroke=COLORS["sensor_stroke"],
        width=1.4,
        dashed=True,
    )
    add_text(
        dwg,
        links,
        "transition and outcome",
        665,
        804,
        size=MIN_FONT_SIZE,
        fill=COLORS["sensor_stroke"],
    )
    add_arrow(
        dwg,
        links,
        [(1210, 593), (1210, 820), (135, 820), (135, 526)],
        stroke=COLORS["sensor_stroke"],
        width=1.7,
    )
    add_text(
        dwg,
        links,
        "closed-loop state feedback",
        695,
        842,
        size=MIN_FONT_SIZE,
        fill=COLORS["sensor_stroke"],
    )
    root.add(links)

    dwg.add(root)
    save(dwg)


if __name__ == "__main__":
    main()
