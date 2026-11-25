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

"""Test the physical-time scan and its same-time region exchange."""

import pytest
import torch

from physicsnemo import Module
from physicsnemo.experimental.models.partitioned_mesh import TemporalPredictor
from physicsnemo.experimental.models.partitioned_mesh.selective_scan import affine_scan


def test_parallel_affine_scan_matches_recurrence():
    generator = torch.Generator().manual_seed(247)
    decay = torch.rand(2, 7, 3, 4, generator=generator, requires_grad=True)
    drive = torch.randn(2, 7, 3, 4, generator=generator, requires_grad=True)
    state = torch.zeros_like(drive[:, 0])
    serial = []
    for a, b in zip(decay.unbind(1), drive.unbind(1)):
        state = a * state + b
        serial.append(state)
    reference = torch.stack(serial, dim=1)
    observed = affine_scan(decay, drive)
    torch.testing.assert_close(observed, reference)
    gradient = torch.autograd.grad(observed.sum(), (decay, drive))
    expected = torch.autograd.grad(reference.sum(), (decay, drive))
    for actual, target in zip(gradient, expected):
        torch.testing.assert_close(actual, target)


@pytest.mark.parametrize("exchange", [False, True])
def test_nonuniform_trajectory_prefix_halo_and_checkpoint(exchange, tmp_path):
    generator = torch.Generator().manual_seed(704)
    model = TemporalPredictor(
        latent_dim=6,
        part_count=3,
        patch_feature_dim=5,
        hidden_dim=12,
        state_dim=3,
        spatial_layers=2,
        temporal_layers=2,
        dynamic_region_exchange=exchange,
        exchange_frame_chunk=2,
    ).eval()
    count = 11
    latent = torch.randn(count, 6, generator=generator)
    features = torch.randn(count, 5, generator=generator)
    part = torch.arange(count) % 3
    indices = torch.arange(count - 1)
    directed = torch.stack((indices, indices + 1))
    edges = torch.cat((directed, directed.flip(0)), dim=1)
    times = torch.tensor([0.0, 0.016, 0.023, 0.044, 0.051], dtype=torch.float64)
    if exchange:
        # Exercise the trained branch as well as its neutral initialization.
        torch.testing.assert_close(model.dynamic_exchange_gate, torch.tensor(0.0))
        with torch.no_grad():
            model.dynamic_exchange_gate.fill_(0.3)
    expected = model(latent, features, part, edges, times, 0.05)
    torch.testing.assert_close(expected[0], latent, atol=0, rtol=0)
    stretched = model(latent, features, part, edges, 2 * times, 0.05)
    assert not torch.allclose(stretched[1:], expected[1:])
    truncated = model(latent, features, part, edges, times[:3], 0.05)
    torch.testing.assert_close(expected[:3], truncated)
    radius = 2 + int(exchange)
    assert model.spatial_layers == radius
    center = count // 2
    start, stop = center - radius, center + radius + 1
    inside = ((edges >= start) & (edges < stop)).all(0)
    local = model(
        latent[start:stop],
        features[start:stop],
        part[start:stop],
        edges[:, inside] - start,
        times,
        0.05,
    )
    torch.testing.assert_close(local[:, radius], expected[:, center])
    path = tmp_path / "temporal.mdlus"
    model.save(path)
    restored = Module.from_checkpoint(path).eval()
    assert isinstance(restored, TemporalPredictor)
    torch.testing.assert_close(
        restored(latent, features, part, edges, times, 0.05), expected
    )
    expected.square().mean().backward()
    if exchange:
        assert model.dynamic_exchange_gate.grad.abs() > 0
        assert (
            model.dynamic_exchange_block.source_projection.weight.grad.abs().sum() > 0
        )
