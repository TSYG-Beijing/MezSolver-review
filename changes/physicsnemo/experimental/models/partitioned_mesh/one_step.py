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

r"""Small, independently checkpointed dynamics regularizer for the spatial AE.

The two observed latent frames stay in autograd. A per-patch residual network
predicts the next latent state; its decoded field is supervised by Stage 1.
There is no node-field shortcut or mixing between unrelated patches/cases.
This auxiliary module does not change the spatial AE checkpoint contract.
"""

from __future__ import annotations

from numbers import Integral

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from physicsnemo.core import Module

from ._metadata import PartitionedMeshMetaData


def _positive_dimension(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class OneStepPredictor(Module):
    r"""Predict ``z_next = z_current + delta`` from two latent history frames.

    Parameters
    ----------
    latent_dim : int
        Number of latent features per patch.
    part_count : int
        Size of the categorical part vocabulary.
    hidden_dim : int, optional, default=128
        Width of the residual multilayer perceptron.
    patch_feature_dim : int, optional, default=17
        Number of known geometry and design features per patch.
    part_embedding_dim : int, optional, default=16
        Width of the categorical part embedding.

    Forward
    -------
    z_previous : torch.Tensor
        Observed previous latent states of shape :math:`(P,L)`.
    z_current : torch.Tensor
        Observed current latent states of shape :math:`(P,L)`.
    patch_features : torch.Tensor
        Known patch geometry and design features of shape :math:`(P,C)`.
    patch_part_index : torch.Tensor
        Integer categorical part indices of shape :math:`(P,)`.
    dt_previous : float | torch.Tensor
        Positive previous physical time interval; scalar or shape :math:`(P,)`.
    dt_next : float | torch.Tensor
        Positive next physical time interval; scalar or shape :math:`(P,)`.
    time_reference : float | torch.Tensor
        Positive scalar physical time scale fitted on training data by the caller.
    current_time : float | torch.Tensor | None, optional, default=None
        Nonnegative current physical time; scalar or shape :math:`(P,)`.
        ``None`` uses zero.

    Outputs
    -------
    torch.Tensor
        Predicted next latent states of shape :math:`(P,L)`.

    Notes
    -----
    ``patch_features`` contains known geometry and conditioning features for each patch.
    Part IDs use a separate categorical embedding. The network receives both
    time intervals divided by the reference interval, their logarithms, and
    the current and next scaled times. Its output is an additive latent increment.

    Examples
    --------
    >>> model = OneStepPredictor(latent_dim=8, part_count=3)
    >>> model(torch.zeros(2, 8), torch.zeros(2, 8), torch.zeros(2, 17),
    ...       torch.zeros(2, dtype=torch.long), 0.01, 0.02, 0.01).shape
    torch.Size([2, 8])
    """

    config_version = "partitioned_mesh_two_frame_delta_v2"

    def __init__(
        self,
        latent_dim: int,
        part_count: int,
        hidden_dim: int = 128,
        patch_feature_dim: int = 17,
        part_embedding_dim: int = 16,
    ) -> None:
        super().__init__(meta=PartitionedMeshMetaData())
        self.model_kwargs = {
            name: _positive_dimension(value, name)
            for name, value in (
                ("latent_dim", latent_dim),
                ("part_count", part_count),
                ("hidden_dim", hidden_dim),
                ("patch_feature_dim", patch_feature_dim),
                ("part_embedding_dim", part_embedding_dim),
            )
        }
        self.latent_dim = self.model_kwargs["latent_dim"]
        self.part_count = self.model_kwargs["part_count"]
        self.patch_feature_dim = self.model_kwargs["patch_feature_dim"]
        self.part_embedding = nn.Embedding(self.part_count, int(part_embedding_dim))
        input_dim = (
            3 * self.latent_dim + self.patch_feature_dim + int(part_embedding_dim) + 6
        )
        self.network = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), self.latent_dim),
        )
        # Exact zero initialization would prevent the first dynamics loss from
        # reaching the previous-frame encoder. Keep a small nonzero residual.
        nn.init.normal_(self.network[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.network[-1].bias)

    @staticmethod
    def _time_tensor(
        value, name, count, device, *, scalar=False, positive=True, check_values=True
    ):
        try:
            # Inspect scalar/list dtypes on CPU; a tensor already on CUDA is
            # returned as-is. Only the final float64 physical value is moved.
            tensor = torch.as_tensor(value)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"{name} must contain real physical times") from exc
        if tensor.is_complex() or tensor.dtype == torch.bool:
            raise ValueError(f"{name} must contain real physical times")
        # Construct Python scalars directly in float64: casting an already
        # rounded float32 tensor afterwards would lose small physical values.
        tensor = torch.as_tensor(value, dtype=torch.float64, device=device)
        if scalar:
            valid_shape = tensor.ndim == 0
        else:
            valid_shape = tensor.ndim == 0 or tensor.shape == (count,)
        if not valid_shape or (check_values and not bool(torch.isfinite(tensor).all())):
            shape = "a scalar" if scalar else "a scalar or [patch_count]"
            raise ValueError(f"{name} must be {shape} with finite values")
        if check_values and bool((tensor <= 0 if positive else tensor < 0).any()):
            raise ValueError(
                f"{name} must be {'positive' if positive else 'nonnegative'}"
            )
        return tensor.expand(count) if not scalar else tensor

    def forward(
        self,
        z_previous: Float[Tensor, "patch latent"],
        z_current: Float[Tensor, "patch latent"],
        patch_features: Float[Tensor, "patch feature"],
        patch_part_index: Int[Tensor, "patch"],  # noqa: F821 - jaxtyping dimension name
        dt_previous: float | Tensor,
        dt_next: float | Tensor,
        time_reference: float | Tensor,
        current_time: float | Tensor | None = None,
    ) -> Float[Tensor, "patch latent"]:
        r"""Validate arbitrary inputs before predicting the next latent frame."""
        return self._forward(
            z_previous,
            z_current,
            patch_features,
            patch_part_index,
            dt_previous,
            dt_next,
            time_reference,
            current_time,
            check_values=not torch.compiler.is_compiling(),
        )

    def forward_known(
        self,
        z_previous: Float[Tensor, "patch latent"],
        z_current: Float[Tensor, "patch latent"],
        patch_features: Float[Tensor, "patch feature"],
        patch_part_index: Int[Tensor, "patch"],  # noqa: F821 - jaxtyping dimension name
        dt_previous: float | Tensor,
        dt_next: float | Tensor,
        time_reference: float | Tensor,
        current_time: float | Tensor | None = None,
    ) -> Float[Tensor, "patch latent"]:
        r"""Fast path for inputs already validated by the CPU dataset loader.

        Training validates physical time grids and categorical topology indices
        before transferring batches. This path keeps metadata shape/dtype/device
        checks while avoiding any tensor-value boolean or scalar extraction,
        so predictor evaluation introduces no CUDA-to-host synchronization.
        The training loop must retain its final finite-loss guard.

        Parameters
        ----------
        z_previous, z_current : torch.Tensor
            Two observed latent fields of shape :math:`(P,L)`.
        patch_features : torch.Tensor
            Known geometry and conditions of shape :math:`(P,C)`.
        patch_part_index : torch.Tensor
            Categorical labels of shape :math:`(P,)`.
        dt_previous, dt_next : float or torch.Tensor
            Positive physical intervals before and after the current observation.
        time_reference : float or torch.Tensor
            Positive reference step in the same physical units.
        current_time : float or torch.Tensor, optional
            Physical observation time relative to the case origin.

        Returns
        -------
        torch.Tensor
            Next latent state of shape :math:`(P,L)`.
        """
        return self._forward(
            z_previous,
            z_current,
            patch_features,
            patch_part_index,
            dt_previous,
            dt_next,
            time_reference,
            current_time,
            check_values=False,
        )

    def _forward(
        self,
        z_previous,
        z_current,
        patch_features,
        patch_part_index,
        dt_previous,
        dt_next,
        time_reference,
        current_time,
        *,
        check_values,
    ):
        for name, value in (
            ("z_previous", z_previous),
            ("z_current", z_current),
            ("patch_features", patch_features),
        ):
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise ValueError(f"{name} must be a floating-point tensor")
            if check_values and not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain finite values")
        if (
            z_current.ndim != 2
            or z_current.shape[0] < 1
            or z_current.shape[1] != self.latent_dim
            or z_previous.shape != z_current.shape
        ):
            raise ValueError(
                "History latents must have matching nonempty [patch_count, latent_dim] shapes"
            )
        count = z_current.shape[0]
        if z_previous.dtype != z_current.dtype:
            raise ValueError("History latents must have the same dtype")
        if patch_features.shape != (count, self.patch_feature_dim):
            raise ValueError(
                "patch_features must have shape [patch_count, patch_feature_dim]"
            )
        device = z_current.device
        if any(value.device != device for value in (z_previous, patch_features)):
            raise ValueError("All latent and patch tensors must be on the same device")
        if (
            not isinstance(patch_part_index, Tensor)
            or patch_part_index.shape != (count,)
            or patch_part_index.dtype not in (torch.int32, torch.int64)
            or patch_part_index.device != device
        ):
            raise ValueError(
                "patch_part_index must be an integer [patch_count] tensor on the latent device"
            )
        if check_values and bool(
            ((patch_part_index < 0) | (patch_part_index >= self.part_count)).any()
        ):
            raise ValueError(
                "patch_part_index contains an unknown categorical part index"
            )
        reference = self._time_tensor(
            time_reference,
            "time_reference",
            count,
            device,
            scalar=True,
            check_values=check_values,
        )
        previous = self._time_tensor(
            dt_previous, "dt_previous", count, device, check_values=check_values
        )
        following = self._time_tensor(
            dt_next, "dt_next", count, device, check_values=check_values
        )
        time = self._time_tensor(
            0.0 if current_time is None else current_time,
            "current_time",
            count,
            device,
            positive=False,
            check_values=check_values,
        )
        # Physical ratios/logs are formed in float64 before conversion for AMP.
        previous, following = previous / reference, following / reference
        current = time / reference
        time_features = torch.stack(
            (
                previous,
                following,
                previous.log(),
                following.log(),
                current,
                current + following,
            ),
            dim=-1,
        )
        time_features = time_features.to(dtype=self.network[0].weight.dtype)
        if check_values and not bool(torch.isfinite(time_features).all()):
            raise ValueError(
                "Scaled physical time features exceed the predictor dtype range"
            )
        embedding = self.part_embedding(patch_part_index)
        features = torch.cat(
            (
                z_previous,
                z_current,
                z_current - z_previous,
                patch_features,
                embedding,
                time_features,
            ),
            dim=-1,
        )
        delta = self.network(features.to(dtype=self.network[0].weight.dtype))
        return z_current + delta
