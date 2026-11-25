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

"""Training-only region and material weighting for normalized displacements."""

import torch


class NoValidTarget(ValueError):
    """Signal a sampled graph or frame with no supervised displacement components."""

    pass


def displacement_mses(
    prediction, batch, region_power=0.0, material_power=0.0, channel_weights=None
):
    """Return weighted and ordinary MSE over present channels.

    Each supervised core node in region r has weight n_r**(-region_power)
    times the physical node material factor raised to material_power. Older
    meshes use a single factor per core. Counts are computed separately
    for each frame and channel after the label mask. At region_power=1 every
    sampled, nonempty region has the same total weight before the material
    factor is applied. Halo and unsupervised nodes make no contribution.
    """
    valid = batch["valid"]
    residual = torch.where(valid, prediction.float() - batch["target"].float(), 0.0)
    squared = residual.square()
    count = valid.sum(0)
    present = count > 0
    if not bool(present.any()):
        raise NoValidTarget("Selected training target has no valid displacement labels")
    ordinary = (squared.sum(0) / count.clamp_min(1))[present].mean()
    if region_power == 0 and material_power == 0 and channel_weights is None:
        return ordinary, ordinary

    region = batch["decode_patch"]
    weights = valid.float()
    if region_power:
        region_count = weights.new_zeros((len(batch["patch_features"]), valid.shape[1]))
        region_count.index_add_(0, region, weights)
        weights = weights * region_count[region].clamp_min(1).pow(-region_power)
    if material_power:
        if "decode_material_weight" in batch:
            material_factor = batch["decode_material_weight"].float()
        elif "patch_material_weight" in batch:
            material_factor = batch["patch_material_weight"].float()[region]
        else:
            raise ValueError(
                "Material weighting requires per-region material factors in the training batch"
            )
        weights = weights * material_factor.pow(material_power).unsqueeze(-1)

    channel_mse = (weights * squared).sum(0)[present] / weights.sum(0)[present]
    if channel_weights is None:
        weighted = channel_mse.mean()
    else:
        factors = channel_mse.new_tensor(channel_weights)[present]
        weighted = (channel_mse * factors).sum() / factors.sum()
    return weighted, ordinary


def wheel_edge_mses(prediction, batch):
    """Normalized displacement-difference MSE on active interior and interface edges.

    Each returned scalar is an equal mean over supervised components in that
    edge class. A class with no active, supervised edges contributes zero.
    """
    edges = batch["wheel_edge_index"]
    interface = batch["wheel_edge_interface"]
    active = batch["wheel_edge_valid"]
    if (
        edges.ndim != 2
        or edges.shape[0] != 2
        or interface.shape != (edges.shape[1],)
        or active.shape != interface.shape
    ):
        raise ValueError("Wheel edges require [2,E] endpoint indices and two [E] masks")
    a, b = edges
    valid = active.bool()[:, None] & batch["valid"][a] & batch["valid"][b]
    residual = prediction.float() - batch["target"].float()
    delta = torch.where(valid, residual[b] - residual[a], 0.0)
    squared = delta.square()
    result = []
    for group in (~interface.bool(), interface.bool()):
        keep = valid & group[:, None]
        count = keep.sum(0)
        by_component = torch.where(keep, squared, 0.0).sum(0) / count.clamp_min(1)
        result.append(by_component.sum() / (count > 0).sum().clamp_min(1))
    return tuple(result)
