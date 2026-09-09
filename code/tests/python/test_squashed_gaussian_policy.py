import math

import pytest


torch = pytest.importorskip("torch")

from mrpp_rl.mappo import (  # noqa: E402
    MAPPOConfig,
    MAPPOTrainer,
    MOGATMAPPOPolicy,
    _inverse_tanh,
    _squashed_gaussian_log_prob,
)


def _policy_inputs(
    *,
    batch_size: int = 2,
    num_agents: int = 3,
    node_dim: int = 8,
):
    torch.manual_seed(7)
    nodes = torch.randn(batch_size, num_agents, node_dim)
    adjacency = torch.ones(batch_size, num_agents, num_agents)
    return nodes, adjacency


def test_sampled_and_deterministic_actions_use_tanh_bounds() -> None:
    policy = MOGATMAPPOPolicy(node_dim=8, hidden_dim=16)
    policy.action_std = 0.3
    nodes, adjacency = _policy_inputs()

    assert isinstance(policy.actor[-1], torch.nn.Linear)
    action, raw_action, log_prob, entropy, value = (
        policy.sample_action_and_value(nodes, adjacency)
    )
    with torch.no_grad():
        raw_mean = policy.actor(policy.encoder(nodes, adjacency))
        expected_deterministic = torch.tanh(raw_mean)
        deterministic, _, _, _ = policy.get_action_and_value(
            nodes,
            adjacency,
            deterministic=True,
        )

    assert torch.all(action > -1.0)
    assert torch.all(action < 1.0)
    assert torch.allclose(action, torch.tanh(raw_action))
    assert torch.allclose(deterministic, expected_deterministic)
    assert torch.allclose(policy(nodes, adjacency), expected_deterministic)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(value).all()


def test_squashed_log_prob_matches_transformed_distribution() -> None:
    dtype = torch.float64
    mean = torch.tensor(
        [[-0.8, 0.1, 1.2], [0.5, -1.1, 0.3]],
        dtype=dtype,
    )
    std = torch.tensor(
        [[0.4, 0.7, 0.2], [0.8, 0.3, 0.6]],
        dtype=dtype,
    )
    raw_action = torch.tensor(
        [[-1.3, 0.25, 0.9], [0.7, -0.6, 1.1]],
        dtype=dtype,
    )
    base_dist = torch.distributions.Normal(mean, std)
    reference_dist = torch.distributions.TransformedDistribution(
        base_dist,
        [torch.distributions.transforms.TanhTransform(cache_size=1)],
    )

    actual = _squashed_gaussian_log_prob(base_dist, raw_action)
    expected = reference_dist.log_prob(torch.tanh(raw_action)).sum(dim=-1)

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_inverse_and_log_prob_are_finite_at_action_boundaries() -> None:
    action = torch.tensor(
        [[-1.0, -1.0 + 1e-8, 0.0, 1.0 - 1e-8, 1.0]],
        dtype=torch.float32,
    )
    raw_action = _inverse_tanh(action)
    base_dist = torch.distributions.Normal(
        torch.zeros_like(raw_action),
        torch.full_like(raw_action, 0.5),
    )
    log_prob = _squashed_gaussian_log_prob(base_dist, raw_action)

    assert torch.isfinite(raw_action).all()
    assert torch.isfinite(log_prob).all()

    policy = MOGATMAPPOPolicy(node_dim=8, hidden_dim=16)
    nodes, adjacency = _policy_inputs(
        batch_size=1,
        num_agents=2,
        node_dim=8,
    )
    boundary_action = torch.tensor(
        [[[-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]]]
    )
    _, evaluated_log_prob, entropy, value = policy.get_action_and_value(
        nodes,
        adjacency,
        action=boundary_action,
    )

    assert torch.isfinite(evaluated_log_prob).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(value).all()


def test_squashed_policy_backward_gradients_are_finite() -> None:
    torch.manual_seed(11)
    policy = MOGATMAPPOPolicy(node_dim=8, hidden_dim=16)
    policy.action_std = 0.25
    nodes, adjacency = _policy_inputs()

    _, _, log_prob, entropy, _ = policy.sample_action_and_value(
        nodes,
        adjacency,
    )
    loss = -(log_prob + 0.01 * entropy).mean()
    loss.backward()

    actor_gradients = [
        parameter.grad
        for parameter in policy.actor.parameters()
        if parameter.grad is not None
    ]
    assert actor_gradients
    assert all(torch.isfinite(gradient).all() for gradient in actor_gradients)
    assert any(torch.count_nonzero(gradient) for gradient in actor_gradients)


def test_evaluating_sampled_action_recomputes_the_same_log_prob() -> None:
    torch.manual_seed(19)
    policy = MOGATMAPPOPolicy(node_dim=8, hidden_dim=16)
    policy.action_std = 0.3
    nodes, adjacency = _policy_inputs()

    action, raw_action, sampled_log_prob, _, _ = (
        policy.sample_action_and_value(nodes, adjacency)
    )
    returned_action, stored_raw_log_prob, _, _ = (
        policy.get_action_and_value(
            nodes,
            adjacency,
            action=action,
            raw_action=raw_action,
        )
    )
    _, reconstructed_log_prob, _, _ = policy.get_action_and_value(
        nodes,
        adjacency,
        action=action,
    )

    assert torch.equal(returned_action, action)
    assert torch.allclose(stored_raw_log_prob, sampled_log_prob)
    assert torch.allclose(
        reconstructed_log_prob,
        sampled_log_prob,
        atol=2e-5,
        rtol=2e-5,
    )


@pytest.mark.parametrize("independent_agents", [False, True])
def test_one_ppo_update_runs_with_squashed_actions(
    independent_agents: bool,
) -> None:
    torch.manual_seed(23)
    policy = MOGATMAPPOPolicy(node_dim=8, hidden_dim=16)
    trainer = MAPPOTrainer(
        policy,
        MAPPOConfig(
            action_std=0.2,
            rollout_steps=3,
            ppo_epochs=1,
            mini_batch_size=2,
            independent_agents=independent_agents,
        ),
    )
    assert (
        trainer.describe()["action_distribution"]
        == "tanh_squashed_gaussian_v1"
    )
    num_agents = 2
    adjacency = torch.ones(num_agents, num_agents)
    last_observation = None

    for step in range(3):
        observation = torch.randn(num_agents, 8)
        action, raw_action, log_prob, _, value = (
            policy.sample_action_and_value(
                observation.unsqueeze(0),
                adjacency.unsqueeze(0),
            )
        )
        trainer.buffer.add(
            observation,
            action.squeeze(0),
            torch.tensor([1.0 - 0.1 * step, 0.4 + 0.1 * step]),
            torch.zeros(num_agents),
            log_prob.squeeze(0),
            value.squeeze(0),
            adjacency=adjacency,
            raw_action=raw_action.squeeze(0),
        )
        last_observation = observation

    losses = trainer.update(last_observation, adjacency)

    assert set(losses) == {"policy_loss", "value_loss", "entropy"}
    assert all(math.isfinite(value) for value in losses.values())
    assert not trainer.buffer.observations
    assert not trainer.buffer.raw_actions
