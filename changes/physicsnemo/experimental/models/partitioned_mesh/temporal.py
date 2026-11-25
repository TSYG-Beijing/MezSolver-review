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

r"""One-shot latent trajectories with internal input-selective state rollout.

Only the initial tokens, undeformed geometry/design, part categories and query
times are inputs. The predictor never observes future latent tokens or responses.
Spatial message passing is local. Each patch evolves a diagonal state using a
causal scan. An optional same-time update exchanges hidden features between
neighboring patches after the first scan. The model reads out all requested
latent states in one call.
"""

from __future__ import annotations

import math

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from physicsnemo.core import Module

from ._metadata import PartitionedMeshMetaData
from .layers import MeshBlock
from .selective_scan import SelectiveStateSpaceBlock


class TemporalPredictor(Module):
    r"""Predict a complete latent trajectory from known initial conditions.

    Parameters
    ----------
    latent_dim : int
        Number of latent features per patch.
    part_count : int
        Size of the categorical part vocabulary.
    patch_feature_dim : int, optional, default=17
        Number of known patch features. The first three columns contain centroid XYZ.
    hidden_dim : int, optional, default=128
        Width of the condition embedding and selective state-space channels.
    spatial_layers : int, optional, default=2
        Number of initial patch-graph message hops before the SSM.
    region_attention_heads, region_attention_width : int, optional, default=0
        Additional multihead attention on the edges of each patch-graph hop.
        Zero heads preserves the original mean aggregation. Zero width uses hidden_dim.
    temporal_layers : int, optional, default=2
        Number of residual selective state-space layers. Must be positive.
    state_dim : int, optional, default=16
        Internal recurrent state width per channel, independent of latent_dim.
    scan_backend : str, optional, default="parallel"
        Portable tensor prefix scan, or explicit "mamba" fused CUDA scan.
    heads : None, optional, default=None
        Reserved legacy argument. Its only supported value is None.
    dropout : float, optional, default=0.0
        Residual-output dropout probability.
    fourier_frequencies : int, optional, default=6
        Number of fixed physical-time Fourier bands.
    dynamic_region_exchange : bool, optional, default=False
        Insert one same-time neighbor update between the first and second SSM blocks.
    exchange_frame_chunk : int, optional, default=4
        Number of independent temporal graphs per message-passing kernel call.

    Forward
    -------
    initial_latent : torch.Tensor
        Known initial latent states of shape :math:`(P,L)`.
    patch_features : torch.Tensor
        Known patch geometry and design features of shape :math:`(P,C)`.
    patch_part_index : torch.Tensor
        Integer categorical part indices of shape :math:`(P,)`.
    region_edge_index : torch.Tensor
        Directed patch graph of shape :math:`(2,E_r)`.
    times : torch.Tensor
        Physical query times of shape :math:`(T,)`, starting at zero and
        strictly increasing thereafter.
    time_scale : float
        Positive reference duration in the same units as the query times.
        Use a consistent scale across partial or complete queries of a trajectory.

    Outputs
    -------
    torch.Tensor
        Latent trajectory of shape :math:`(T,P,L)`. The zero-time slice equals
        ``initial_latent`` exactly.

    Notes
    -----
    The first three conditioning features are normalized patch centroids.
    Time and time_scale must have the same physical units. For a restricted
    region graph, provide one hop of surrounding patches per initial spatial
    layer, plus one more hop when the inter-SSM exchange is enabled.
    The requested times are evaluated together on a nonuniform physical grid.
    Appending later queries does not change a preceding prediction in evaluation.
    Inserting an intermediate time changes the integration path and may change
    later outputs. No recurrent state persists between forward calls.

    Examples
    --------
    >>> model = TemporalPredictor(latent_dim=8, part_count=3,
    ...                                hidden_dim=12, state_dim=4)
    >>> model.latent_dim
    8
    """

    config_version = "partitioned_mesh_selective_scan_v4"

    def __init__(
        self,
        latent_dim: int,
        part_count: int,
        patch_feature_dim: int = 17,
        hidden_dim: int = 128,
        spatial_layers: int = 2,
        temporal_layers: int = 2,
        heads: int | None = None,
        dropout: float = 0.0,
        fourier_frequencies: int = 6,
        state_dim: int = 16,
        scan_backend: str = "parallel",
        region_attention_heads: int = 0,
        region_attention_width: int = 0,
        dynamic_region_exchange: bool = False,
        exchange_frame_chunk: int = 4,
    ) -> None:
        super().__init__(meta=PartitionedMeshMetaData())
        if heads is not None:
            raise ValueError(
                "The selective-scan model has no attention heads. Remove 'heads'; "
                "legacy Transformer checkpoints require their archived implementation."
            )
        dimensions = (
            latent_dim,
            part_count,
            patch_feature_dim,
            hidden_dim,
            state_dim,
            temporal_layers,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in dimensions
        ):
            raise ValueError(
                "Latent, PART, feature, hidden, state and temporal-layer dimensions must be positive integers"
            )
        counts = (spatial_layers, fourier_frequencies)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counts
        ):
            raise ValueError(
                "Spatial layers and Fourier count must be nonnegative integers"
            )
        if patch_feature_dim < 3:
            raise ValueError(
                "patch_features must include centroid XYZ in its first three columns"
            )
        if not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")

        self.latent_dim = latent_dim
        self.part_count = part_count
        self.patch_feature_dim = patch_feature_dim
        if not isinstance(dynamic_region_exchange, bool):
            raise TypeError("dynamic_region_exchange must be boolean")
        if dynamic_region_exchange and temporal_layers < 2:
            raise ValueError(
                "The inter-SSM exchange requires at least two temporal blocks"
            )
        if type(exchange_frame_chunk) is not int or exchange_frame_chunk < 1:
            raise ValueError("exchange_frame_chunk must be a positive integer")
        self.dynamic_region_exchange = dynamic_region_exchange
        self.exchange_frame_chunk = exchange_frame_chunk
        self.initial_spatial_layers = spatial_layers
        # Native training, validation and inference use this attribute for the
        # induced graph radius. The constructor retains the static layer count.
        self.spatial_layers = spatial_layers + int(dynamic_region_exchange)
        self.model_kwargs = dict(
            latent_dim=latent_dim,
            part_count=part_count,
            patch_feature_dim=patch_feature_dim,
            hidden_dim=hidden_dim,
            spatial_layers=spatial_layers,
            temporal_layers=temporal_layers,
            dropout=float(dropout),
            fourier_frequencies=fourier_frequencies,
            state_dim=state_dim,
            scan_backend=scan_backend,
        )
        if region_attention_heads or region_attention_width:
            self.model_kwargs.update(
                region_attention_heads=region_attention_heads,
                region_attention_width=region_attention_width,
            )
        part_embedding_dim = 8
        self.part_embedding = nn.Embedding(part_count, part_embedding_dim)
        self.condition = nn.Sequential(
            nn.Linear(latent_dim + patch_feature_dim + part_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.spatial_blocks = nn.ModuleList(
            MeshBlock(
                hidden_dim,
                edge_dim=4,
                region_attention_heads=region_attention_heads,
                region_attention_width=region_attention_width,
            )
            for _ in range(spatial_layers)
        )
        self.register_buffer(
            "time_bands",
            torch.pi * 2.0 ** torch.arange(fourier_frequencies, dtype=torch.float32),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(2 + 2 * fourier_frequencies, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.temporal_blocks = nn.ModuleList(
            SelectiveStateSpaceBlock(
                hidden_dim, state_dim, float(dropout), scan_backend
            )
            for _ in range(temporal_layers)
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.delta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        if dynamic_region_exchange:
            self.model_kwargs.update(
                dynamic_region_exchange=True, exchange_frame_chunk=exchange_frame_chunk
            )
            self.dynamic_exchange_block = MeshBlock(
                hidden_dim,
                edge_dim=4,
                region_attention_heads=region_attention_heads,
                region_attention_width=region_attention_width,
            )
            self.dynamic_exchange_gate = nn.Parameter(torch.zeros(()))

    def _exchange_at_each_time(
        self, sequence, region_edge_index, edge_attr, region_in_degree=None
    ):
        patch_count, _, width = sequence.shape
        outputs = []
        for chunk in sequence.split(self.exchange_frame_chunk, dim=1):
            frame_count = chunk.shape[1]
            # Disjoint copies of the physical graph. A source at time t can
            # only send to a target at time t, including at chunk boundaries.
            offsets = torch.arange(frame_count, device=sequence.device) * patch_count
            batched_edges = (
                region_edge_index[:, None, :] + offsets[None, :, None]
            ).flatten(1)
            nodes = chunk.transpose(0, 1).reshape(frame_count * patch_count, width)
            degree = (
                None
                if region_in_degree is None
                else region_in_degree.repeat(frame_count)
            )
            update = self.dynamic_exchange_block(
                nodes, batched_edges, edge_attr.repeat(frame_count, 1), degree
            )
            # The block has its normal nonzero initialization, so the gate
            # receives a gradient even when upgrading an archived checkpoint.
            mixed = nodes + self.dynamic_exchange_gate.to(nodes.dtype) * (
                update - nodes
            )
            outputs.append(
                mixed.reshape(frame_count, patch_count, width).transpose(0, 1)
            )
        return torch.cat(outputs, dim=1)

    def _validate(
        self,
        initial_latent: Tensor,
        patch_features: Tensor,
        patch_part_index: Tensor,
        region_edge_index: Tensor,
        times: Tensor,
        time_scale: float,
    ) -> float:
        inputs = (
            initial_latent,
            patch_features,
            patch_part_index,
            region_edge_index,
            times,
        )
        if not all(isinstance(value, Tensor) for value in inputs):
            raise TypeError(
                "All predictor inputs except time_scale must be torch tensors"
            )
        if initial_latent.ndim != 2 or initial_latent.shape[1] != self.latent_dim:
            raise ValueError("initial_latent must be [patch_count, latent_dim]")
        patch_count = initial_latent.shape[0]
        if patch_count == 0:
            raise ValueError("At least one patch is required")
        if patch_features.shape != (patch_count, self.patch_feature_dim):
            raise ValueError("patch_features has the wrong patch or feature dimension")
        if (
            patch_part_index.shape != (patch_count,)
            or patch_part_index.dtype != torch.long
        ):
            raise ValueError("patch_part_index must be int64 [patch_count]")
        if ((patch_part_index < 0) | (patch_part_index >= self.part_count)).any():
            raise ValueError("patch_part_index is outside the learned PART vocabulary")
        if (
            region_edge_index.ndim != 2
            or region_edge_index.shape[0] != 2
            or region_edge_index.dtype != torch.long
        ):
            raise ValueError("region_edge_index must be int64 [2, edge_count]")
        if ((region_edge_index < 0) | (region_edge_index >= patch_count)).any():
            raise ValueError("region_edge_index refers to a patch outside this context")
        if any(value.device != initial_latent.device for value in inputs[1:]):
            raise ValueError("Predictor input tensors must share one device")
        if (
            not initial_latent.is_floating_point()
            or not patch_features.is_floating_point()
        ):
            raise ValueError(
                "Initial latent and geometry/design features must be floating point"
            )
        if initial_latent.dtype != patch_features.dtype:
            raise ValueError(
                "Initial latent and geometry/design features must have the same dtype"
            )
        if (
            not torch.isfinite(initial_latent).all()
            or not torch.isfinite(patch_features).all()
        ):
            raise ValueError(
                "Initial latent and geometry/design features must be finite"
            )
        if times.ndim != 1 or times.numel() < 1 or not times.is_floating_point():
            raise ValueError(
                "times must be a nonempty floating-point [query_time] axis"
            )
        if not torch.isfinite(times).all() or (times < 0).any():
            raise ValueError("times must be finite nonnegative seconds")
        if times[0] != 0 or (times[1:] <= times[:-1]).any():
            raise ValueError("times must start at zero and then strictly increase")
        try:
            scale = float(time_scale)
        except (TypeError, ValueError) as error:
            raise ValueError("time_scale must be a positive finite scalar") from error
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("time_scale must be a positive finite scalar")
        return scale

    def forward(
        self,
        initial_latent: Float[Tensor, "patch latent"],
        patch_features: Float[Tensor, "patch feature"],
        patch_part_index: Int[Tensor, "patch"],  # noqa: F821 - jaxtyping dimension name
        region_edge_index: Int[Tensor, "2 edge"],
        times: Float[Tensor, "time"],  # noqa: F821 - jaxtyping dimension name
        time_scale: float,
    ) -> Float[Tensor, "time patch latent"]:
        r"""Return all query-time tokens; no future-target argument is accepted."""
        scale = float(time_scale)
        if not torch.compiler.is_compiling():
            scale = self._validate(
                initial_latent,
                patch_features,
                patch_part_index,
                region_edge_index,
                times,
                time_scale,
            )
        return self._forward_known(
            initial_latent,
            patch_features,
            patch_part_index,
            region_edge_index,
            times,
            scale,
        )

    def forward_known(
        self,
        initial_latent: Float[Tensor, "patch latent"],
        patch_features: Float[Tensor, "patch feature"],
        patch_part_index: Int[Tensor, "patch"],  # noqa: F821 - jaxtyping dimension name
        region_edge_index: Int[Tensor, "2 edge"],
        times: Float[Tensor, "time"],  # noqa: F821 - jaxtyping dimension name
        time_scale: float,
        region_in_degree: Float[Tensor, "patch"] | None = None,  # noqa: F821 - jaxtyping dimension name
    ) -> Float[Tensor, "time patch latent"]:
        r"""Training path for graph indices, features and times validated by the caller.

        Public ``forward`` retains full input validation. The training pipeline
        validates geometry, time axes and graph indices on the CPU and checks
        finite losses/gradients, avoiding GPU-host reductions each forward.

        Parameters
        ----------
        initial_latent : torch.Tensor
            Initial regional tokens of shape :math:`(P,L)`.
        patch_features : torch.Tensor
            Known geometry and conditions of shape :math:`(P,C)`.
        patch_part_index : torch.Tensor
            Categorical indices of shape :math:`(P,)`.
        region_edge_index : torch.Tensor
            Directed region adjacency of shape :math:`(2,E_r)`.
        times : torch.Tensor
            Increasing physical query grid of shape :math:`(T,)`, starting at zero.
        time_scale : float
            Fixed physical time reference, shared by all prefixes.
        region_in_degree : torch.Tensor, optional
            Precomputed incoming degree of shape :math:`(P,)`.

        Returns
        -------
        torch.Tensor
            Causal latent trajectory of shape :math:`(T,P,L)`.
        """
        return self._forward_known(
            initial_latent,
            patch_features,
            patch_part_index,
            region_edge_index,
            times,
            float(time_scale),
            region_in_degree,
        )

    def _forward_known(
        self,
        initial_latent,
        patch_features,
        patch_part_index,
        region_edge_index,
        times,
        scale,
        region_in_degree=None,
    ):
        known = torch.cat(
            (initial_latent, patch_features, self.part_embedding(patch_part_index)),
            dim=-1,
        )
        spatial = self.condition(known)
        source, target = region_edge_index
        centroid = patch_features[:, :3]
        relative = centroid[source] - centroid[target]
        edge_attr = torch.cat(
            (relative, relative.float().norm(dim=-1, keepdim=True).to(relative.dtype)),
            dim=-1,
        )
        for block in self.spatial_blocks:
            if region_in_degree is None:
                spatial = block(spatial, region_edge_index, edge_attr)
            else:
                spatial = block(spatial, region_edge_index, edge_attr, region_in_degree)

        tau = (times / scale).to(initial_latent.dtype)
        phase = tau[:, None] * self.time_bands[None, :]
        time_features = torch.cat(
            (tau[:, None], tau.square()[:, None], phase.sin(), phase.cos()), dim=-1
        )
        # The hidden recurrence is independent for each patch. Physical time
        # intervals scale the input-selective transition, including irregular grids.
        sequence = spatial[:, None, :] + self.time_embedding(time_features)[None, :, :]
        # Difference physical timestamps before converting to a model/AMP dtype:
        # adjacent valid times can collapse to the same BF16 value otherwise.
        physical_times = times.to(torch.float64)
        time_deltas = torch.cat(
            (
                physical_times.new_zeros(1),
                (physical_times[1:] - physical_times[:-1]) / scale,
            )
        ).float()
        for index, block in enumerate(self.temporal_blocks):
            sequence = block(sequence, time_deltas)
            if self.dynamic_region_exchange and index == 0:
                sequence = self._exchange_at_each_time(
                    sequence, region_edge_index, edge_attr, region_in_degree
                )
        hidden = self.temporal_norm(sequence)
        delta = self.delta(hidden).transpose(0, 1)
        return initial_latent[None, :, :] + tau[:, None, None] * delta
