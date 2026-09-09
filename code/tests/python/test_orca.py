from mrpp_navigation.orca import AgentState, VelocityCommand2D, reciprocal_avoidance_velocity


def test_reciprocal_avoidance_pushes_away_from_close_neighbor() -> None:
    agent = AgentState(x=0.0, y=0.0, vx=0.0, vy=0.0)
    neighbor = AgentState(x=0.2, y=0.0, vx=0.0, vy=0.0)
    preferred = VelocityCommand2D(vx=0.5, vy=0.0)

    command = reciprocal_avoidance_velocity(agent, [neighbor], preferred, safety_radius=0.5)

    assert command.vx < preferred.vx


def test_reciprocal_avoidance_reacts_to_predicted_head_on_collision() -> None:
    agent = AgentState(x=0.0, y=0.0, vx=0.0, vy=0.0)
    neighbor = AgentState(x=1.4, y=0.0, vx=-0.5, vy=0.0)
    preferred = VelocityCommand2D(vx=0.5, vy=0.0)

    command = reciprocal_avoidance_velocity(
        agent,
        [neighbor],
        preferred,
        safety_radius=0.6,
        gain=1.2,
        time_horizon=2.0,
    )

    assert command.vx < preferred.vx
