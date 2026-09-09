from mrpp_rl.adaptive_weights import SceneContext, adaptive_reward_weights


def test_dense_scene_increases_collision_weight() -> None:
    sparse = adaptive_reward_weights(SceneContext(local_robot_density=0.1, nearest_obstacle_distance=2.0))
    dense = adaptive_reward_weights(SceneContext(local_robot_density=0.9, nearest_obstacle_distance=0.4))

    assert dense.collision > sparse.collision
    assert dense.unsafe_distance > sparse.unsafe_distance


def test_pedestrian_risk_increases_pedestrian_clearance_weight() -> None:
    far = adaptive_reward_weights(
        SceneContext(
            local_robot_density=0.1,
            nearest_obstacle_distance=2.0,
            nearest_pedestrian_distance=3.0,
        )
    )
    close = adaptive_reward_weights(
        SceneContext(
            local_robot_density=0.1,
            nearest_obstacle_distance=2.0,
            nearest_pedestrian_distance=0.8,
        )
    )

    assert close.pedestrian_clearance > far.pedestrian_clearance
    assert close.safety_intervention > far.safety_intervention
