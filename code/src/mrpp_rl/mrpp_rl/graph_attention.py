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

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = (
        "PyTorch is required for GraphAttentionEncoder."
        " Install torch on the Ubuntu training server."
    )
    _IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = ""
    _IMPORT_ERROR = None


if nn is not None:

    class GraphAttentionEncoder(nn.Module):
        def __init__(self, node_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.query = nn.Linear(node_dim, hidden_dim)
            self.key = nn.Linear(node_dim, hidden_dim)
            self.value = nn.Linear(node_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, hidden_dim)
            self.residual = nn.Linear(node_dim, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)
            self.residual_norm_enabled = True

        def forward(self, nodes: "torch.Tensor", adjacency: "torch.Tensor") -> "torch.Tensor":
            q = self.query(nodes)
            k = self.key(nodes)
            v = self.value(nodes)
            scores = torch.matmul(q, k.transpose(-1, -2)) / (k.shape[-1] ** 0.5)
            scores = scores.masked_fill(adjacency <= 0, -1e9)
            weights = torch.softmax(scores, dim=-1)
            weights = torch.nan_to_num(weights, nan=1.0 / nodes.shape[-2])
            encoded = torch.matmul(weights, v)
            attended = self.output(encoded)
            if not self.residual_norm_enabled:
                return attended
            node_residual = self.residual(nodes)
            return self.norm(attended + node_residual)

else:

    class GraphAttentionEncoder:
        def __init__(self, *_args, **_kwargs) -> None:
            raise ModuleNotFoundError(TORCH_IMPORT_ERROR) from _IMPORT_ERROR
