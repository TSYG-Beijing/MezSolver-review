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

r"""Partitioned spatial autoencoder for displacement on an unstructured mesh.

The response ``q`` is normalized (Ux, Uy, Uz). Only the encoder observes it;
the decoder uses latent tokens and known, undeformed geometry/design features.
Core ownership and pooling coordinates come from a fixed reference topology.
Halo nodes transmit messages but never contribute directly to token pooling.

An L-hop halo reproduces L local message-passing layers at core nodes. Regional
message passing additionally needs context patches covering its own hops; halo
alone does not imply equivalence to an arbitrary full-mesh autoencoder.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from physicsnemo.core import Module

from ._metadata import PartitionedMeshMetaData
from .layers import MeshBlock, MultiscaleCorePooling, _mlp


class DecoderResidual(nn.Module):
    """Region-conditioned FiLM residual block with neutral initialization."""

    def __init__(self, width, condition_dim):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.film = _mlp(condition_dim, width, 2 * width)
        self.network = _mlp(width, width, width)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, hidden, scale, bias):
        return hidden + self.network(
            F.silu(self.norm(hidden) * (1 + scale.tanh()) + bias)
        )


class PartitionAutoencoder(Module):
    r"""Encode graph patches and reconstruct normalized displacement.

    Parameters
    ----------
    node_static_dim : int, optional, default=12
        Number of known node features; columns 3:6 contain local XYZ.
    patch_feature_dim : int, optional, default=17
        Number of patch features; the first three columns contain centroid XYZ.
    part_count : int, optional, default=905
        Size of the categorical part vocabulary.
    hidden_dim : int, optional, default=128
        Width of node message-passing layers.
    latent_dims : Sequence[int], optional, default=(64, 32, 32)
        Widths of the whole-patch token and two coordinate-local tokens.
    message_layers : int, optional, default=4
        Number of local graph hops; the caller must provide a sufficient halo.
    region_layers : int, optional, default=1
        Number of patch-graph hops; context must cover these hops separately.
    region_attention_heads, region_attention_width : int, optional, default=0
        Additional edge attention in each regional hop. Zero heads preserves legacy
        mean aggregation. Zero projection width uses the regional latent width.
    part_embedding_dim : int, optional, default=8
        Width of the categorical part embedding.
    fourier_frequencies : int, optional, default=4
        Number of fixed Fourier bands for decoder local coordinates.
    decoder_hidden_dim : int, optional, default=256
        Width of the geometry-conditioned decoder.
    checkpoint_encoder : bool, optional, default=False
        Recompute local block activations with non-reentrant checkpointing.
    response_dim : int, optional, default=3
        Number of Cartesian displacement components, fixed at three.
    edge_dim : int, optional, default=6
        Number of input features on each directed mesh edge.
    property_region_count, topology_region_count : int, optional, default=0
        Vocabulary sizes for node memberships. Zero disables the respective embedding.
    membership_embedding_dim : int, optional, default=8
        Width of each membership embedding.
    observed_mask : bool, optional, default=False
        Concatenate a component-wise observation mask to the encoder input
        and zero missing displacements.
    token_attention_heads : int, optional, default=4
        Decoder query heads attending to each region's three latent slots.
    decoder_residual_layers : int, optional, default=4
        Number of residual blocks conditioned on the decoded region.
    decode_chunk_size : int, optional, default=8192
        Maximum number of output nodes processed together in a decoder chunk.

    Forward
    -------
    batch : Mapping[str, torch.Tensor]
        Encoder arrays: ``node_static`` :math:`(V,S)`, ``q`` :math:`(V,3)`,
        ``edge_index`` :math:`(2,E)`, ``edge_attr`` :math:`(E,6)`,
        ``node_patch``, ``core_mask`` and ``pool_coordinate`` :math:`(V,)`,
        ``patch_features`` :math:`(P,C)`, ``patch_part_index`` :math:`(P,)`,
        and ``region_edge_index`` :math:`(2,E_r)`. Decoder selectors
        ``decode_index`` and ``decode_patch`` have shape :math:`(D,)`.
        Optional ``core_index``, ``node_in_degree`` and ``region_in_degree``
        reuse caller-computed indices or degrees. With ``observed_mask=True``,
        supply an ``observed`` tensor of shape (V,3). Enabled property and topology
        embeddings use CSR arrays ``property_ptr``, ``property_indices``,
        ``topology_ptr``, and ``topology_indices`` for the corresponding vocabulary.
    decoder_spatial : torch.Tensor | None, optional, default=None
        Shared decoder geometry embedding of shape :math:`(D,H_d)`, valid only
        while the model weights and geometry remain unchanged.

    Outputs
    -------
    dict[str, torch.Tensor]
        ``prediction`` :math:`(D,3)`, ``latent`` :math:`(P,L)` and
        ``local_latent`` :math:`(P,L)`, where :math:`L` is the sum of latent
        slot widths. Halo nodes do not contribute directly to pooled tokens.

    Notes
    -----
    PART_ID is represented only as a categorical embedding; its numeric value
    carries no material or structural-role meaning. Input arrays must be converted
    to tensors by the caller. Membership embeddings use sparse node-to-region lists.

    Checkpoints use :meth:`physicsnemo.core.Module.save` and
    :meth:`physicsnemo.core.Module.from_checkpoint`. The original recipe's
    ``model_kwargs`` and ``state_dict`` remain compatible as well.

    Examples
    --------
    >>> model = PartitionAutoencoder(part_count=3, latent_dims=(4, 2, 2))
    >>> model.latent_dim
    8
    """

    config_version = "partitioned_mesh_token_query_residual_v5"

    def __init__(
        self,
        node_static_dim: int = 12,
        patch_feature_dim: int = 17,
        part_count: int = 905,
        hidden_dim: int = 128,
        latent_dims: Sequence[int] = (64, 32, 32),
        message_layers: int = 4,
        region_layers: int = 1,
        part_embedding_dim: int = 8,
        fourier_frequencies: int = 4,
        decoder_hidden_dim: int = 256,
        checkpoint_encoder: bool = False,
        response_dim: int = 3,
        edge_dim: int = 6,
        property_region_count: int = 0,
        topology_region_count: int = 0,
        membership_embedding_dim: int = 8,
        observed_mask: bool = False,
        token_attention_heads: int = 4,
        decoder_residual_layers: int = 4,
        decode_chunk_size: int = 8192,
        region_attention_heads: int = 0,
        region_attention_width: int = 0,
    ) -> None:
        super().__init__(meta=PartitionedMeshMetaData())
        latent_dims = tuple(int(d) for d in latent_dims)
        positive = (
            node_static_dim,
            patch_feature_dim,
            part_count,
            hidden_dim,
            part_embedding_dim,
            decoder_hidden_dim,
            *latent_dims,
        )
        if any(int(v) <= 0 for v in positive) or len(latent_dims) != 3:
            raise ValueError(
                "Model dimensions must be positive; latent_dims needs 3 slots"
            )
        if min(message_layers, region_layers, fourier_frequencies) < 0:
            raise ValueError(
                "Layer counts and Fourier frequency count must be nonnegative"
            )
        if node_static_dim < 6 or patch_feature_dim < 3:
            raise ValueError("Features must include local XYZ and patch centroid XYZ")
        if response_dim != 3:
            raise ValueError("response_dim must be three for Cartesian displacement")
        self.model_kwargs = dict(
            node_static_dim=int(node_static_dim),
            patch_feature_dim=int(patch_feature_dim),
            part_count=int(part_count),
            hidden_dim=int(hidden_dim),
            latent_dims=latent_dims,
            message_layers=int(message_layers),
            region_layers=int(region_layers),
            part_embedding_dim=int(part_embedding_dim),
            fourier_frequencies=int(fourier_frequencies),
            decoder_hidden_dim=int(decoder_hidden_dim),
            checkpoint_encoder=bool(checkpoint_encoder),
            response_dim=int(response_dim),
        )
        self.node_static_dim = int(node_static_dim)
        self.patch_feature_dim = int(patch_feature_dim)
        self.latent_dim = sum(latent_dims)
        self.response_dim = int(response_dim)
        self.message_layers = int(message_layers)
        self.region_layers = int(region_layers)
        self.checkpoint_encoder = bool(checkpoint_encoder)
        if (
            edge_dim < 1
            or membership_embedding_dim < 1
            or token_attention_heads < 1
            or decoder_hidden_dim % token_attention_heads
            or decoder_residual_layers < 0
            or decode_chunk_size < 1
            or min(property_region_count, topology_region_count) < 0
        ):
            raise ValueError("Invalid graph, membership or decoder dimensions")
        self.model_kwargs.update(
            edge_dim=edge_dim,
            property_region_count=property_region_count,
            topology_region_count=topology_region_count,
            membership_embedding_dim=membership_embedding_dim,
            observed_mask=observed_mask,
            token_attention_heads=token_attention_heads,
            decoder_residual_layers=decoder_residual_layers,
            decode_chunk_size=decode_chunk_size,
        )
        if region_attention_heads or region_attention_width:
            self.model_kwargs.update(
                region_attention_heads=region_attention_heads,
                region_attention_width=region_attention_width,
            )
        self.observed_mask = observed_mask
        self.latent_dims = latent_dims
        self.decode_chunk_size = decode_chunk_size
        self.memberships = nn.ModuleDict(
            {
                name: nn.EmbeddingBag(
                    count + 1,
                    membership_embedding_dim,
                    mode="mean",
                    include_last_offset=True,
                )
                for name, count in (
                    ("property", property_region_count),
                    ("topology", topology_region_count),
                )
                if count
            }
        )
        known_dim = node_static_dim + len(self.memberships) * membership_embedding_dim
        self.part_embedding = nn.Embedding(part_count, part_embedding_dim)
        self.node_input = _mlp(
            known_dim + self.response_dim * (2 if observed_mask else 1),
            hidden_dim,
            hidden_dim,
        )
        self.patch_condition = _mlp(
            patch_feature_dim + part_embedding_dim, hidden_dim, hidden_dim
        )
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.local_blocks = nn.ModuleList(
            MeshBlock(hidden_dim, edge_dim=edge_dim) for _ in range(message_layers)
        )
        self.pool = MultiscaleCorePooling(hidden_dim, latent_dims)
        self.latent_condition = nn.Linear(hidden_dim, self.latent_dim)
        self.latent_norm = nn.LayerNorm(self.latent_dim)
        self.region_blocks = nn.ModuleList(
            MeshBlock(
                self.latent_dim,
                edge_dim=4,
                region_attention_heads=region_attention_heads,
                region_attention_width=region_attention_width,
            )
            for _ in range(region_layers)
        )
        frequencies = torch.pi * 2.0 ** torch.arange(
            fourier_frequencies, dtype=torch.float32
        )
        self.register_buffer("fourier_bands", frequencies)
        spatial_dim = known_dim + part_embedding_dim + 6 * fourier_frequencies
        self.decoder_spatial = _mlp(spatial_dim, decoder_hidden_dim, decoder_hidden_dim)
        self.decoder_film = _mlp(
            self.latent_dim + patch_feature_dim,
            decoder_hidden_dim,
            2 * decoder_hidden_dim,
        )
        self.decoder_output = _mlp(
            decoder_hidden_dim, decoder_hidden_dim, self.response_dim
        )
        self.attention_heads = token_attention_heads
        self.decoder_query = nn.Linear(decoder_hidden_dim, decoder_hidden_dim)
        self.decoder_keys = nn.ModuleList(
            nn.Linear(d, decoder_hidden_dim) for d in latent_dims
        )
        self.decoder_values = nn.ModuleList(
            nn.Linear(d, decoder_hidden_dim) for d in latent_dims
        )
        self.decoder_query_output = nn.Linear(decoder_hidden_dim, decoder_hidden_dim)
        self.decoder_residuals = nn.ModuleList(
            DecoderResidual(decoder_hidden_dim, self.latent_dim + patch_feature_dim)
            for _ in range(decoder_residual_layers)
        )

    def known_features(
        self, batch: Mapping[str, Tensor]
    ) -> Float[Tensor, "node known"]:
        r"""Embed the fixed reference data for each node.

        Parameters
        ----------
        batch : Mapping[str, torch.Tensor]
            Static features and optional membership CSR arrays from the graph.

        Returns
        -------
        torch.Tensor
            Geometry and membership embedding of shape :math:`(V,K)`.
        """
        values = [batch["node_static"]]
        for name, embedding in self.memberships.items():
            values.append(embedding(batch[name + "_indices"], batch[name + "_ptr"]))
        return torch.cat(values, -1)

    @property
    def checkpoint_encoder(self) -> bool:
        r"""Whether local encoder activations are recomputed during training."""
        return self._checkpoint_encoder

    @checkpoint_encoder.setter
    def checkpoint_encoder(self, enabled: bool) -> None:
        # The recipe may change this runtime option after loading a checkpoint.
        # Keep both legacy kwargs and PhysicsNeMo's captured constructor aligned
        # so native .mdlus export reconstructs the effective execution setting.
        self._checkpoint_encoder = bool(enabled)
        self.model_kwargs["checkpoint_encoder"] = self._checkpoint_encoder
        self._args["__args__"]["checkpoint_encoder"] = self._checkpoint_encoder

    def _patch_condition(self, batch: Mapping[str, Tensor]) -> Tensor:
        return self.patch_condition(
            torch.cat(
                (
                    batch["patch_features"],
                    self.part_embedding(batch["patch_part_index"]),
                ),
                -1,
            )
        )

    def encode_nodes(self, batch: Mapping[str, Tensor]) -> Float[Tensor, "node hidden"]:
        r"""Apply the local graph encoder before core pooling.

        Parameters
        ----------
        batch : Mapping[str, torch.Tensor]
            Node responses, optional masks, static features and local mesh edges.

        Returns
        -------
        torch.Tensor
            Node embeddings of shape :math:`(V,H)`, including halo nodes.
        """
        node_static, q = batch["node_static"], batch["q"]
        if node_static.ndim != 2 or node_static.shape[1] != self.node_static_dim:
            raise ValueError("node_static has the wrong feature dimension")
        if q.shape != (node_static.shape[0], self.response_dim):
            raise ValueError(
                f"q must be [node_count, {self.response_dim}] normalized displacement components"
            )
        # A repeated halo node must have the same initial embedding in every
        # local graph. Patch conditioning belongs AFTER pooling; adding the
        # current graph's condition to halo nodes would break L-hop equivalence.
        features = [self.known_features(batch), q]
        if self.observed_mask:
            mask = batch["observed"].to(q.dtype)
            if mask.shape != q.shape:
                raise ValueError("observed must match displacement shape")
            features[1] = torch.where(mask.bool(), q, 0.0)
            features.append(mask)
        nodes = self.input_norm(self.node_input(torch.cat(features, -1)))
        for block in self.local_blocks:
            if self.checkpoint_encoder and self.training and torch.is_grad_enabled():
                nodes = checkpoint(
                    block,
                    nodes,
                    batch["edge_index"],
                    batch["edge_attr"],
                    batch.get("node_in_degree"),
                    use_reentrant=False,
                )
            else:
                nodes = block(
                    nodes,
                    batch["edge_index"],
                    batch["edge_attr"],
                    batch.get("node_in_degree"),
                )
        return nodes

    def _encode(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        condition = self._patch_condition(batch)
        nodes = self.encode_nodes(batch)
        local = self.pool(
            nodes,
            batch["node_patch"],
            batch["core_mask"],
            batch["pool_coordinate"],
            condition.shape[0],
            batch.get("core_index"),
        )
        local = self.latent_norm(local + self.latent_condition(condition))
        latent = local
        if self.region_blocks:
            source, target = batch["region_edge_index"]
            centroid = batch["patch_features"][:, :3]
            delta = centroid[source] - centroid[target]
            edge_attr = torch.cat(
                (delta, delta.float().norm(dim=-1, keepdim=True).to(delta.dtype)), -1
            )
            for block in self.region_blocks:
                latent = block(
                    latent,
                    batch["region_edge_index"],
                    edge_attr,
                    batch.get("region_in_degree"),
                )
        return latent, local

    def encode(self, batch: Mapping[str, Tensor]) -> Float[Tensor, "patch latent"]:
        r"""Compute regional latent tokens for a graph and its context.

        Parameters
        ----------
        batch : Mapping[str, torch.Tensor]
            Node, pooling and regional inputs specified in the class contract.

        Returns
        -------
        torch.Tensor
            Context patch embeddings of shape :math:`(P,L)`.
        """
        return self._encode(batch)[0]

    def _decoder_spatial_features(self, batch: Mapping[str, Tensor]) -> Tensor:
        r"""Compute known geometry features once, shared by all decoded frames."""
        index, patch = batch["decode_index"], batch["decode_patch"]
        static = self.known_features(batch)[index]
        local_xyz = static[:, 3:6]
        phase = local_xyz[..., None] * self.fourier_bands[None, None, :]
        fourier = torch.cat((phase.sin(), phase.cos()), -1).flatten(1)
        part = self.part_embedding(batch["patch_part_index"])[patch]
        return self.decoder_spatial(torch.cat((static, part, fourier), -1))

    def prepare_decoder(
        self, batch: Mapping[str, Tensor]
    ) -> Float[Tensor, "decoded hidden"]:
        r"""Prepare spatial features for reuse within one unchanged weight state.

        Training callers may share this differentiable tensor within one step.
        Inference callers may retain it for a fixed graph while weights stay fixed.
        Recompute it after each optimizer update.

        Parameters
        ----------
        batch : Mapping[str, torch.Tensor]
            Known node geometry, memberships, patch identifiers and decode rows.

        Returns
        -------
        torch.Tensor
            Decoder geometry features of shape :math:`(D,H_d)`.
        """
        return self._decoder_spatial_features(batch)

    def decode(
        self,
        batch: Mapping[str, Tensor],
        latent: Float[Tensor, "patch latent"],
        *,
        spatial: Float[Tensor, "decoded hidden"] | None = None,
    ) -> Float[Tensor, "decoded 3"]:
        r"""Decode fields on selected nodes from known geometry and regional tokens.

        Parameters
        ----------
        batch : Mapping[str, torch.Tensor]
            Static mesh and selectors for the requested core nodes.
        latent : torch.Tensor
            Context patch tokens of shape :math:`(P,L)`.
        spatial : torch.Tensor, optional
            Cached features of shape :math:`(D,H_d)` from :meth:`prepare_decoder`.

        Returns
        -------
        torch.Tensor
            Physical-field predictions in normalized coordinates for the decoded rows.
        """
        if not torch.compiler.is_compiling():
            if latent.shape != (len(batch["patch_features"]), self.latent_dim):
                raise ValueError("latent must have shape [patch_count, latent_dim]")
            if spatial is not None and spatial.shape != (
                len(batch["decode_patch"]),
                self.model_kwargs["decoder_hidden_dim"],
            ):
                raise ValueError(
                    "spatial must have shape [decoded_count, decoder_hidden_dim]"
                )
        spatial = self.prepare_decoder(batch) if spatial is None else spatial
        patch = batch["decode_patch"]
        # FiLM is constant inside a patch. Project P patch tokens before the
        # gather instead of projecting D repeated node copies (typically D >> P).
        condition = torch.cat((latent, batch["patch_features"]), -1)
        scale, bias = self.decoder_film(condition).chunk(2, -1)
        scale, bias = scale[patch], bias[patch]
        tokens = latent.split(self.latent_dims, dim=-1)
        width = spatial.shape[-1]
        heads, head_dim = self.attention_heads, width // self.attention_heads
        keys = torch.stack([layer(z) for layer, z in zip(self.decoder_keys, tokens)], 1)
        values = torch.stack(
            [layer(z) for layer, z in zip(self.decoder_values, tokens)], 1
        )
        keys = keys.reshape(-1, len(tokens), heads, head_dim)
        values = values.reshape_as(keys)
        films = [layer.film(condition).chunk(2, -1) for layer in self.decoder_residuals]
        outputs = []
        for start in range(0, len(patch), self.decode_chunk_size):
            stop = start + self.decode_chunk_size
            p = patch[start:stop]
            space = spatial[start:stop]
            query = self.decoder_query(space).reshape(-1, heads, head_dim)
            score = (query[:, None] * keys[p]).sum(-1) / head_dim**0.5
            if "pool_coordinate" in batch:
                coordinate = batch["pool_coordinate"][batch["decode_index"][start:stop]]
                prior = (
                    -0.5
                    * (
                        (coordinate[:, None] - coordinate.new_tensor([-0.5, 0.5])) / 0.4
                    ).square()
                )
                score = (
                    score
                    + torch.cat((torch.zeros_like(prior[:, :1]), prior), -1)[:, :, None]
                )
            attention = score.float().softmax(1).to(values.dtype)
            context = (attention[..., None] * values[p]).sum(1).flatten(1)
            hidden = F.silu(
                space * (1 + torch.tanh(scale[start:stop])) + bias[start:stop]
            )
            hidden = hidden + self.decoder_query_output(context)
            for layer, (s, b) in zip(self.decoder_residuals, films):
                hidden = layer(hidden, s[p], b[p])
            outputs.append(self.decoder_output(hidden))
        return torch.cat(outputs)

    def decode_frames(
        self,
        batch: Mapping[str, Tensor],
        latents: Float[Tensor, "time patch latent"],
        frame_batch_size: int = 4,
        *,
        spatial: Float[Tensor, "decoded hidden"] | None = None,
    ) -> Float[Tensor, "time decoded 3"]:
        r"""Decode [T,P,L] tokens as [T,D,3], sharing static spatial computation.

        Frames are vectorized in chunks to bound temporary hidden tensors.
        ``frame_batch_size=0`` processes all frames together. Autograd keeps the
        shared spatial graph, including PART embeddings, connected to every frame.
        Training still retains the activations needed for backward across chunks.

        Parameters
        ----------
        batch : Mapping[str, torch.Tensor]
            Known graph features and output selectors, shared across time.
        latents : torch.Tensor
            A trajectory of shape :math:`(T,P,L)`.
        frame_batch_size : int, optional
            Maximum number of query frames processed per chunk.
        spatial : torch.Tensor, optional
            Cached features of shape :math:`(D,H_d)` from :meth:`prepare_decoder`.

        Returns
        -------
        torch.Tensor
            Normalized fields for every requested time and node.
        """
        if latents.ndim != 3 or latents.shape[1:] != (
            batch["patch_features"].shape[0],
            self.latent_dim,
        ):
            raise ValueError("latents must be [frame_count, patch_count, latent_dim]")
        if frame_batch_size < 0:
            raise ValueError("frame_batch_size must be nonnegative; 0 means all frames")
        if latents.shape[0] == 0:
            raise ValueError("decode_frames requires at least one frame")
        spatial = self.prepare_decoder(batch) if spatial is None else spatial
        step = frame_batch_size or latents.shape[0]
        outputs = []
        for start in range(0, latents.shape[0], step):
            chunk = latents[start : start + step]
            outputs.append(
                torch.stack([self.decode(batch, z, spatial=spatial) for z in chunk])
            )
        return torch.cat(outputs, dim=0)

    def _validate_graph(self, batch: Mapping[str, Tensor]) -> None:
        """Check graph metadata without reducing CUDA tensors on the host."""
        required = {
            "node_static",
            "q",
            "edge_index",
            "edge_attr",
            "node_patch",
            "core_mask",
            "pool_coordinate",
            "patch_features",
            "patch_part_index",
            "decode_index",
            "decode_patch",
        }
        if self.region_blocks:
            required.add("region_edge_index")
        if self.observed_mask:
            required.add("observed")
        for name in self.memberships:
            required.update((name + "_ptr", name + "_indices"))
        missing = required.difference(batch)
        if missing:
            raise ValueError(f"Missing graph fields: {sorted(missing)}")
        operands = required.union(
            name
            for name in ("core_index", "node_in_degree", "region_in_degree")
            if name in batch
        )
        if not all(isinstance(batch[key], Tensor) for key in operands):
            raise ValueError("Graph fields must be torch tensors")
        if any(batch[key].device != batch["q"].device for key in operands):
            raise ValueError("Graph tensors must share the response device")
        static, features = batch["node_static"], batch["patch_features"]
        if static.ndim != 2 or features.ndim != 2:
            raise ValueError("node_static and patch_features must be matrices")
        n, p = len(static), len(features)
        index = batch["decode_index"]
        if index.ndim != 1 or not index.numel():
            raise ValueError("decode_index must be a nonempty vector")
        shapes = {
            "node_static": (n, self.node_static_dim),
            "q": (n, self.response_dim),
            "patch_features": (p, self.patch_feature_dim),
            "patch_part_index": (p,),
            "node_patch": (n,),
            "core_mask": (n,),
            "pool_coordinate": (n,),
            "decode_patch": index.shape,
        }
        integer = {
            "node_patch",
            "patch_part_index",
            "decode_index",
            "decode_patch",
            "edge_index",
        }
        masks = {"core_mask"}
        for key in ("edge_index", "region_edge_index"):
            if key in required:
                edge = batch[key]
                if edge.ndim != 2 or edge.shape[0] != 2:
                    raise ValueError(f"{key} must have shape [2, edge_count]")
                integer.add(key)
        shapes["edge_attr"] = (
            batch["edge_index"].shape[1],
            self.model_kwargs["edge_dim"],
        )
        if self.observed_mask:
            shapes["observed"] = (n, self.response_dim)
            masks.add("observed")
        for key, length in (("node_in_degree", n), ("region_in_degree", p)):
            if key in operands:
                shapes[key] = (length,)
        if "core_index" in operands:
            integer.add("core_index")
            if batch["core_index"].ndim != 1:
                raise ValueError("core_index must be a vector")
        for name in self.memberships:
            values, ptr = name + "_indices", name + "_ptr"
            integer.update((values, ptr))
            shapes[ptr] = (n + 1,)
            if batch[values].ndim != 1:
                raise ValueError(f"{values} must be a vector")
        for key, shape in shapes.items():
            if batch[key].shape != shape:
                raise ValueError(f"{key} must have shape {tuple(shape)}")
        if any(batch[key].dtype != torch.long for key in integer):
            raise ValueError("Graph indices and membership pointers must be int64")
        if batch["core_mask"].dtype != torch.bool:
            raise ValueError("core_mask must be Boolean")
        if any(
            not batch[key].is_floating_point() for key in operands - integer - masks
        ):
            raise ValueError(
                "Geometry, edge features, responses and degrees must be floating point"
            )

    def forward(
        self,
        batch: Mapping[str, Tensor],
        *,
        decoder_spatial: Float[Tensor, "decoded hidden"] | None = None,
    ) -> dict[str, Tensor]:
        r"""Return normalized fields, pooled tokens and spatially exchanged tokens."""
        if not torch.compiler.is_compiling():
            self._validate_graph(batch)
        latent, local = self._encode(batch)
        prediction = (
            self.decode(batch, latent)
            if decoder_spatial is None
            else self.decode(batch, latent, spatial=decoder_spatial)
        )
        return {"prediction": prediction, "latent": latent, "local_latent": local}
