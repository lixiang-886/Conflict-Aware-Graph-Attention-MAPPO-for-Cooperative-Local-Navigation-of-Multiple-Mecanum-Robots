# Copyright 2026 lixiang-886
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from mrpp_rl.graph_attention import GraphAttentionEncoder, TORCH_IMPORT_ERROR, nn

LEGACY_GAT_MISSING_KEYS = {
    "encoder.residual.weight",
    "encoder.residual.bias",
    "encoder.norm.weight",
    "encoder.norm.bias",
}
CENTRAL_CRITIC_PREFIXES = (
    "central_state_encoder.",
    "central_critic.",
)


if nn is not None:

    class MOGATMAPPOPolicy(nn.Module):
        def __init__(self, node_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.encoder = GraphAttentionEncoder(
                node_dim=node_dim, hidden_dim=hidden_dim
            )
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 3),
            )

        def forward(self, nodes, adjacency):
            encoded = self.encoder(nodes, adjacency)
            return self.actor(encoded).tanh()

    def load_mogat_policy_state_dict(policy, state_dict) -> bool:
        """Load current and explicitly supported legacy policy checkpoints."""
        expected = set(policy.state_dict())
        provided = set(state_dict)
        missing = expected - provided
        unexpected = provided - expected
        central_keys = {
            key
            for key in expected
            if key.startswith(CENTRAL_CRITIC_PREFIXES)
        }
        allowed_missing: set[str] = set()

        missing_gat = missing.intersection(LEGACY_GAT_MISSING_KEYS)
        if missing_gat:
            if missing_gat != LEGACY_GAT_MISSING_KEYS:
                raise RuntimeError(
                    "Partially missing residual GAT checkpoint state: "
                    f"{sorted(missing_gat)}"
                )
            allowed_missing.update(LEGACY_GAT_MISSING_KEYS)

        missing_central = missing.intersection(central_keys)
        if missing_central:
            if missing_central != central_keys:
                raise RuntimeError(
                    "Partially missing centralized critic checkpoint state: "
                    f"{sorted(missing_central)}"
                )
            if getattr(policy, "centralized_critic_enabled", False):
                raise RuntimeError(
                    "Checkpoint predates the centralized critic required by "
                    "this Sensors algorithm. Re-train the requested method."
                )
            allowed_missing.update(central_keys)

        unsupported_missing = missing - allowed_missing
        if unsupported_missing or unexpected:
            raise RuntimeError(
                "Policy checkpoint mismatch: "
                f"missing={sorted(unsupported_missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

        policy.load_state_dict(state_dict, strict=False)
        if missing_gat:
            policy.encoder.residual_norm_enabled = False
        return bool(missing)

else:

    class MOGATMAPPOPolicy:
        def __init__(self, *_args, **_kwargs) -> None:
            raise ModuleNotFoundError(TORCH_IMPORT_ERROR)

    def load_mogat_policy_state_dict(*_args, **_kwargs) -> bool:
        """Raise the stored import error when PyTorch is unavailable."""
        raise ModuleNotFoundError(TORCH_IMPORT_ERROR)
