from mrpp_rl.reward import RewardWeights, compute_reward


def test_collision_reward_is_strongly_negative() -> None:
    reward = compute_reward(
        progress=0.1,
        reached_goal=False,
        collision=True,
        unsafe_distance=False,
        control_effort=0.2,
        waiting=False,
        weights=RewardWeights(),
    )

    assert reward < -5.0


def test_goal_reward_is_positive() -> None:
    reward = compute_reward(
        progress=0.5,
        reached_goal=True,
        collision=False,
        unsafe_distance=False,
        control_effort=0.1,
        waiting=False,
        weights=RewardWeights(),
    )

    assert reward > 5.0


def test_smoothness_penalty_reduces_reward() -> None:
    smooth = compute_reward(
        progress=0.5,
        reached_goal=False,
        collision=False,
        unsafe_distance=False,
        control_effort=0.1,
        waiting=False,
        weights=RewardWeights(),
        smoothness=0.0,
    )
    abrupt = compute_reward(
        progress=0.5,
        reached_goal=False,
        collision=False,
        unsafe_distance=False,
        control_effort=0.1,
        waiting=False,
        weights=RewardWeights(),
        smoothness=2.0,
    )

    assert abrupt < smooth


def test_clearance_and_intervention_penalties_reduce_reward() -> None:
    weights = RewardWeights(
        clearance=1.0,
        robot_clearance=1.0,
        pedestrian_clearance=1.0,
        safety_intervention=1.0,
    )
    clear = compute_reward(
        progress=0.5,
        reached_goal=False,
        collision=False,
        unsafe_distance=False,
        control_effort=0.1,
        waiting=False,
        weights=weights,
    )
    risky = compute_reward(
        progress=0.5,
        reached_goal=False,
        collision=False,
        unsafe_distance=False,
        control_effort=0.1,
        waiting=False,
        weights=weights,
        clearance_penalty=0.5,
        robot_clearance_penalty=0.5,
        pedestrian_clearance_penalty=0.5,
        safety_intervention=0.5,
    )

    assert risky < clear
