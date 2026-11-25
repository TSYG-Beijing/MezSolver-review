# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

r"""Local graph message passing, regional attention and core-node pooling."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch.nn import functional as F


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim)
    )


def _segment_softmax(scores: Tensor, index: Tensor, count: int) -> Tensor:
    r"""Column-wise softmax within patches, with float32 reduction under AMP."""
    scores = scores.float()
    expanded = index[:, None].expand_as(scores)
    maxima = scores.new_full((count, scores.shape[1]), -torch.inf)
    maxima.scatter_reduce_(
        0, expanded, scores.detach(), reduce="amax", include_self=True
    )
    numerator = torch.exp(scores - maxima[index])
    denominator = scores.new_zeros(maxima.shape).index_add_(0, index, numerator)
    return numerator / denominator[index].clamp_min(1e-12)


class RegionAttention(nn.Module):
    """Multihead attention over the incoming physical neighbors of each region."""

    def __init__(
        self, width: int, edge_dim: int, heads: int, attention_width: int
    ) -> None:
        super().__init__()
        if heads < 1 or attention_width < 1 or attention_width % heads:
            raise ValueError(
                "Region attention width must be positive and divisible by its head count"
            )
        self.heads = heads
        self.head_dim = attention_width // heads
        self.query = nn.Linear(width, attention_width, bias=False)
        self.key = nn.Linear(width, attention_width, bias=False)
        self.value = nn.Linear(width, attention_width, bias=False)
        self.edge_bias = nn.Linear(edge_dim, heads, bias=False)
        self.output = nn.Linear(attention_width, width, bias=False)

    def forward(
        self, nodes: Tensor, source: Tensor, target: Tensor, edge_attr: Tensor
    ) -> Tensor:
        count = nodes.shape[0]
        shape = (count, self.heads, self.head_dim)
        query = self.query(nodes).reshape(shape)
        key = self.key(nodes).reshape(shape)
        value = self.value(nodes).reshape(shape)
        # Only connected source-target pairs contribute. All reductions are FP32 under AMP.
        scores = (query[target].float() * key[source].float()).sum(
            -1
        ) * self.head_dim**-0.5
        scores = scores + self.edge_bias(edge_attr).float()
        weights = _segment_softmax(scores, target, count)
        pooled = torch.zeros(shape, dtype=torch.float32, device=nodes.device)
        pooled.index_add_(0, target, weights[..., None] * value[source].float())
        return self.output(pooled.flatten(1).to(nodes.dtype))


class MeshBlock(nn.Module):
    r"""One graph hop with static edge attributes and per-node normalization.

    Edges are directed source -> target. The data builder includes both
    directions of physical mesh edges. Factorized projections avoid materializing
    an E x (3H) concatenation. No batch/global normalization mixes graph patches.

    Parameters
    ----------
    width : int
        Number of node hidden features.
    edge_dim : int, optional, default=6
        Number of static edge features.

    Forward
    -------
    nodes : torch.Tensor
        Node states of shape :math:`(V,H)`.
    edge_index : torch.Tensor
        Directed source and target indices of shape :math:`(2,E)`.
    edge_attr : torch.Tensor
        Static edge attributes of shape :math:`(E,C_e)`.
    in_degree : torch.Tensor | None, optional, default=None
        Optional precomputed incoming edge counts of shape :math:`(V,)`.

    Outputs
    -------
    torch.Tensor
        Updated node states of shape :math:`(V,H)`.
    """

    def __init__(
        self,
        width: int,
        edge_dim: int = 6,
        *,
        region_attention_heads: int = 0,
        region_attention_width: int = 0,
    ) -> None:
        super().__init__()
        # PhysicsNeMo MeshGraphEdgeMLPSum uses a different projection layout,
        # initialization and graph container. These factorized projections keep
        # the original checkpoint contract and avoid an E x 3H concatenation.
        self.source_projection = nn.Linear(width, width, bias=False)
        self.target_projection = nn.Linear(width, width, bias=False)
        self.edge_projection = nn.Linear(edge_dim, width)
        self.message_projection = nn.Linear(width, width)
        self.node_update = _mlp(2 * width, width, width)
        self.norm = nn.LayerNorm(width)
        if region_attention_heads < 0 or region_attention_width < 0:
            raise ValueError(
                "Region attention head count and width must be nonnegative"
            )
        self.region_attention = (
            RegionAttention(
                width, edge_dim, region_attention_heads, region_attention_width or width
            )
            if region_attention_heads
            else None
        )

    def forward(
        self,
        nodes: Float[Tensor, "node hidden"],
        edge_index: Int[Tensor, "2 edge"],
        edge_attr: Float[Tensor, "edge feature"],
        in_degree: Tensor | None = None,
    ) -> Float[Tensor, "node hidden"]:
        source, target = edge_index
        message = self.source_projection(nodes)[source]
        message = message + self.target_projection(nodes)[target]
        message = self.message_projection(
            F.silu(message + self.edge_projection(edge_attr))
        )
        # Keep large sums in float32, including when linear layers use bf16/fp16.
        aggregate = torch.zeros(nodes.shape, dtype=torch.float32, device=nodes.device)
        aggregate.index_add_(0, target, message.float())
        # Static graph batches can compute this once on the CPU. Repeating
        # bincount on CUDA at every layer adds both work and a host sync.
        degree = (
            torch.bincount(target, minlength=nodes.shape[0]).float()
            if in_degree is None
            else in_degree.float()
        )
        aggregate = (aggregate / degree.clamp_min(1)[:, None]).to(nodes.dtype)
        if self.region_attention is not None:
            # Both branches read this hop's inputs, preserving the original context radius.
            aggregate = aggregate + self.region_attention(
                nodes, source, target, edge_attr
            )
        return self.norm(nodes + self.node_update(torch.cat((nodes, aggregate), -1)))


class MultiscaleCorePooling(nn.Module):
    r"""One whole-patch token and two fixed-coordinate local tokens per patch.

    Parameters
    ----------
    width : int
        Number of input node hidden features.
    latent_dims : Sequence[int]
        Three output widths: whole-patch, lower-coordinate and upper-coordinate.

    Forward
    -------
    nodes : torch.Tensor
        Node states of shape :math:`(V,H)`.
    node_patch : torch.Tensor
        Owning patch indices of shape :math:`(V,)`.
    core_mask : torch.Tensor
        Boolean core ownership mask of shape :math:`(V,)`; halo nodes are excluded.
    pool_coordinate : torch.Tensor
        Fixed reference pooling coordinates of shape :math:`(V,)`.
    patch_count : int
        Number of output patch tokens.
    core_index : torch.Tensor | None, optional, default=None
        Optional precomputed core indices of shape :math:`(V_c,)`.

    Outputs
    -------
    torch.Tensor
        Concatenated patch tokens of shape :math:`(P,L)`.
    """

    def __init__(self, width: int, latent_dims: Sequence[int]) -> None:
        super().__init__()
        self.score = nn.Linear(width, 3)
        self.projections = nn.ModuleList(nn.Linear(width, int(d)) for d in latent_dims)
        self.register_buffer("fine_centers", torch.tensor((-0.5, 0.5)))
        self.fine_sigma = 0.4

    def forward(
        self,
        nodes: Float[Tensor, "node hidden"],
        node_patch: Int[Tensor, "node"],  # noqa: F821 - jaxtyping dimension name
        core_mask: Bool[Tensor, "node"],  # noqa: F821 - jaxtyping dimension name
        pool_coordinate: Float[Tensor, "node"],  # noqa: F821 - jaxtyping dimension name
        patch_count: int,
        core_index: Int[Tensor, "core"] | None = None,  # noqa: F821 - jaxtyping dimension name
    ) -> Float[Tensor, "patch latent"]:
        # A provided index avoids dynamic CUDA nonzero and its host sync.
        core = (
            torch.nonzero(core_mask, as_tuple=False).flatten()
            if core_index is None
            else core_index
        )
        patch = node_patch[core]
        h = nodes[core]
        scores = self.score(h).float()
        # The prior is based on immutable reference coordinates, never q(t).
        coordinate = pool_coordinate[core].float().reshape(-1, 1)
        fine_prior = (
            -0.5
            * ((coordinate - self.fine_centers[None, :]) / self.fine_sigma).square()
        )
        scores = torch.cat((scores[:, :1], scores[:, 1:] + fine_prior), -1)
        attention = _segment_softmax(scores, patch, patch_count)
        tokens = []
        for slot, projection in enumerate(self.projections):
            pooled = torch.zeros(
                (patch_count, h.shape[1]), dtype=torch.float32, device=h.device
            )
            pooled.index_add_(0, patch, h.float() * attention[:, slot, None])
            tokens.append(projection(pooled.to(h.dtype)))
        return torch.cat(tokens, -1)
