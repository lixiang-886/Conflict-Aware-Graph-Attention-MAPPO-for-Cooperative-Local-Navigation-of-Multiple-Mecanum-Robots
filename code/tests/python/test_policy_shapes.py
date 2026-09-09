def test_policy_import_message_is_clear_without_torch() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        from mrpp_rl.policy import TORCH_IMPORT_ERROR

        assert "PyTorch" in TORCH_IMPORT_ERROR
    else:
        from mrpp_rl.policy import MOGATMAPPOPolicy

        policy = MOGATMAPPOPolicy(node_dim=16, hidden_dim=16)
        nodes = torch.zeros((1, 4, 16))
        adjacency = torch.ones((1, 4, 4))
        commands = policy(nodes, adjacency)
        assert commands.shape == (1, 4, 3)


def test_mappo_policy_deterministic_action_is_repeatable() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return

    from mrpp_rl.mappo import MOGATMAPPOPolicy

    policy = MOGATMAPPOPolicy(node_dim=16, hidden_dim=16)
    nodes = torch.zeros((1, 4, 16))
    adjacency = torch.ones((1, 4, 4))

    action_1, _, _, _ = policy.get_action_and_value(
        nodes, adjacency, deterministic=True
    )
    action_2, _, _, _ = policy.get_action_and_value(
        nodes, adjacency, deterministic=True
    )

    assert torch.allclose(action_1, action_2)


def test_centralized_critic_sees_joint_state_without_changing_local_actor() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return

    from mrpp_rl.mappo import MOGATMAPPOPolicy

    policy = MOGATMAPPOPolicy(
        node_dim=16,
        hidden_dim=16,
        centralized_critic=True,
    )
    adjacency = torch.eye(2).unsqueeze(0)
    first = torch.zeros((1, 2, 16))
    second = first.clone()
    second[0, 1, 0] = 1.0

    action_1, _, _, value_1 = policy.get_action_and_value(
        first,
        adjacency,
        deterministic=True,
    )
    action_2, _, _, value_2 = policy.get_action_and_value(
        second,
        adjacency,
        deterministic=True,
    )

    assert torch.allclose(action_1[:, 0], action_2[:, 0])
    assert not torch.allclose(value_1[:, 0], value_2[:, 0])


def test_legacy_gat_checkpoint_uses_original_encoder_forward() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return

    from mrpp_rl.mappo import MOGATMAPPOPolicy
    from mrpp_rl.policy import LEGACY_GAT_MISSING_KEYS
    from mrpp_rl.policy import load_mogat_policy_state_dict

    source = MOGATMAPPOPolicy(node_dim=16, hidden_dim=16)
    legacy_state = {
        key: value
        for key, value in source.state_dict().items()
        if key not in LEGACY_GAT_MISSING_KEYS
    }
    restored = MOGATMAPPOPolicy(node_dim=16, hidden_dim=16)

    assert load_mogat_policy_state_dict(restored, legacy_state) is True
    assert restored.encoder.residual_norm_enabled is False

    nodes = torch.randn((1, 4, 16))
    adjacency = torch.ones((1, 4, 4))
    with torch.no_grad():
        encoded = restored.encoder(nodes, adjacency)
        query = restored.encoder.query(nodes)
        key = restored.encoder.key(nodes)
        value = restored.encoder.value(nodes)
        scores = torch.matmul(query, key.transpose(-1, -2)) / (16 ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        expected = restored.encoder.output(torch.matmul(weights, value))

    assert torch.allclose(encoded, expected)


def test_mappo_policy_initial_residual_is_zero_centered() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return

    from mrpp_rl.mappo import MOGATMAPPOPolicy

    policy = MOGATMAPPOPolicy(node_dim=16, hidden_dim=16)
    nodes = torch.zeros((1, 4, 16))
    adjacency = torch.ones((1, 4, 4))

    action, _, _, _ = policy.get_action_and_value(
        nodes,
        adjacency,
        deterministic=True,
    )

    assert torch.allclose(action, torch.zeros_like(action))


def test_mappo_trainer_applies_action_std_config() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return

    from mrpp_rl.mappo import MAPPOConfig, MAPPOTrainer, MOGATMAPPOPolicy

    policy = MOGATMAPPOPolicy(node_dim=16, hidden_dim=16)
    trainer = MAPPOTrainer(
        policy=policy,
        config=MAPPOConfig(action_std=0.12),
    )

    assert trainer.policy.action_std == 0.12


def test_centralized_critic_parameters_are_optimized() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return

    from mrpp_rl.mappo import MAPPOConfig, MAPPOTrainer, MOGATMAPPOPolicy

    policy = MOGATMAPPOPolicy(
        node_dim=16,
        hidden_dim=16,
        centralized_critic=True,
    )
    trainer = MAPPOTrainer(
        policy=policy,
        config=MAPPOConfig(centralized_critic=True),
    )
    optimized = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }

    assert all(
        id(parameter) in optimized
        for parameter in policy.central_state_encoder.parameters()
    )
    assert all(
        id(parameter) in optimized
        for parameter in policy.central_critic.parameters()
    )


def test_rollout_buffer_can_keep_independent_agent_advantages() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return

    from mrpp_rl.mappo import RolloutBuffer

    buffer = RolloutBuffer()
    buffer.add(
        torch.zeros(2, 16),
        torch.zeros(2, 3),
        torch.tensor([1.0, 0.5]),
        torch.zeros(2),
        torch.zeros(2),
        torch.zeros(2),
    )

    joint_returns, joint_advantages = buffer.compute_returns_and_advantages(
        torch.zeros(2),
        gamma=0.99,
        gae_lambda=0.95,
        average_agents=True,
    )
    ippo_returns, ippo_advantages = buffer.compute_returns_and_advantages(
        torch.zeros(2),
        gamma=0.99,
        gae_lambda=0.95,
        average_agents=False,
    )

    assert joint_returns.shape == torch.Size([1])
    assert joint_advantages.shape == torch.Size([1])
    assert ippo_returns.shape == torch.Size([1, 2])
    assert ippo_advantages.shape == torch.Size([1, 2])


def test_mappo_trainer_describes_independent_agent_mode() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return

    from mrpp_rl.mappo import MAPPOConfig, MAPPOTrainer, MOGATMAPPOPolicy

    policy = MOGATMAPPOPolicy(node_dim=16, hidden_dim=16)
    trainer = MAPPOTrainer(
        policy=policy,
        config=MAPPOConfig(independent_agents=True),
    )

    assert trainer.describe()["independent_agents"] is True


def test_mappo_trainer_uses_configured_team_advantage_mix() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return

    from mrpp_rl.mappo import MAPPOConfig, MAPPOTrainer, MOGATMAPPOPolicy

    policy = MOGATMAPPOPolicy(node_dim=16, hidden_dim=16)
    trainer = MAPPOTrainer(
        policy=policy,
        config=MAPPOConfig(team_advantage_mix=0.25),
    )

    assert trainer.describe()["team_advantage_mix"] == 0.25


def test_joint_mappo_update_keeps_per_agent_credit_shapes() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return

    from mrpp_rl.mappo import MAPPOConfig, MAPPOTrainer, MOGATMAPPOPolicy

    policy = MOGATMAPPOPolicy(node_dim=16, hidden_dim=16)
    trainer = MAPPOTrainer(
        policy=policy,
        config=MAPPOConfig(
            rollout_steps=2,
            mini_batch_size=2,
            team_advantage_mix=0.25,
        ),
    )
    observations = torch.zeros(2, 16)
    adjacency = torch.ones(2, 2)
    for rewards in (torch.tensor([1.0, -0.5]), torch.tensor([0.2, 0.8])):
        action, log_prob, _, value = policy.get_action_and_value(
            observations.unsqueeze(0),
            adjacency.unsqueeze(0),
        )
        trainer.buffer.add(
            observations,
            action.squeeze(0),
            rewards,
            torch.zeros(2),
            log_prob.squeeze(0),
            value.squeeze(0),
            adjacency=adjacency,
        )

    losses = trainer.update(observations, adjacency)

    assert set(losses) == {"policy_loss", "value_loss", "entropy"}
    assert not trainer.buffer.observations


def test_maddpg_policy_and_trainer_update_shapes() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return

    from mrpp_rl.maddpg import MADDPGConfig, MADDPGPolicy, MADDPGTrainer

    policy = MADDPGPolicy(node_dim=16, hidden_dim=16)
    trainer = MADDPGTrainer(
        policy=policy,
        config=MADDPGConfig(rollout_steps=2, mini_batch_size=2),
    )
    obs = torch.zeros(2, 16)
    adjacency = torch.eye(2)
    action, log_prob, entropy, value = policy.get_action_and_value(
        obs.unsqueeze(0),
        adjacency.unsqueeze(0),
        deterministic=True,
    )

    assert action.shape == torch.Size([1, 2, 3])
    assert log_prob.shape == torch.Size([1, 2])
    assert entropy.shape == torch.Size([1, 2])
    assert value.shape == torch.Size([1, 2])

    for reward in (torch.tensor([1.0, 0.5]), torch.tensor([0.2, 0.1])):
        trainer.buffer.add(
            obs,
            torch.zeros(2, 3),
            reward,
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
        )

    loss_info = trainer.update(obs, adjacency)

    assert set(loss_info) == {"policy_loss", "value_loss", "entropy"}
    assert not trainer.buffer.observations


def test_mappo_default_exploration_is_small_for_residual_control() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return

    from mrpp_rl.mappo import MAPPOConfig, MOGATMAPPOPolicy

    assert MAPPOConfig().action_std <= 0.03
    assert MOGATMAPPOPolicy(node_dim=16, hidden_dim=16).action_std <= 0.03
