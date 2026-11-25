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

"""CPU contracts for masked encoding, decoding selected nodes and two-history prediction."""

import pytest
import torch

from physicsnemo import Module
from physicsnemo.experimental.models.partitioned_mesh import (
    OneStepPredictor,
    PartitionAutoencoder,
)


def mesh_batch():
    random = torch.Generator().manual_seed(452)
    source = torch.tensor([0, 1, 1, 2, 3, 4, 4, 5])
    target = torch.tensor([1, 0, 2, 1, 4, 3, 5, 4])
    owned = torch.tensor([0, 1, 3, 4])
    return dict(
        node_static=torch.randn(6, 12, generator=random),
        q=torch.randn(6, 3, generator=random),
        observed=torch.ones(6, 3, dtype=torch.bool),
        edge_index=torch.stack((source, target)),
        edge_attr=torch.randn(8, 6, generator=random),
        node_patch=torch.tensor([0, 0, 0, 1, 1, 1]),
        core_mask=torch.tensor([1, 1, 0, 1, 1, 0], dtype=torch.bool),
        core_index=owned,
        pool_coordinate=torch.tensor([-0.5, 0.5, 0.0, -0.5, 0.5, 0.0]),
        patch_features=torch.randn(2, 17, generator=random),
        patch_part_index=torch.tensor([0, 1]),
        region_edge_index=torch.tensor([[0, 1], [1, 0]]),
        decode_index=owned,
        decode_patch=torch.tensor([0, 0, 1, 1]),
    )


def test_masked_values_do_not_leak_and_both_modules_roundtrip(tmp_path):
    batch = mesh_batch()
    model = PartitionAutoencoder(
        part_count=2,
        latent_dims=(4, 2, 2),
        hidden_dim=8,
        message_layers=1,
        region_layers=1,
        region_attention_heads=2,
        region_attention_width=8,
        decoder_hidden_dim=16,
        decoder_residual_layers=1,
        observed_mask=True,
    )
    one = OneStepPredictor(latent_dim=8, part_count=2, hidden_dim=8)
    batch["observed"][0, 1] = False
    altered = batch["q"].clone()
    altered[0, 1] = 1.0e6
    output = model(batch)
    torch.testing.assert_close(
        output["prediction"], model(dict(batch, q=altered))["prediction"]
    )
    assert output["prediction"].shape == (4, 3)
    current = output["latent"]
    previous = model.encode(dict(batch, q=batch["q"] * 0.5))
    temporal_inputs = (
        previous,
        current,
        batch["patch_features"],
        batch["patch_part_index"],
        0.01,
        0.015,
        0.01,
    )
    next_latent = one(*temporal_inputs, current_time=0.1)
    next_field = model.decode(batch, next_latent)
    (output["prediction"].square().mean() + next_field.square().mean()).backward()
    for module in (model, one):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert (
            grads
            and all(torch.isfinite(g).all() for g in grads)
            and any(torch.count_nonzero(g) for g in grads)
        )
    for filename, module in (("spatial.mdlus", model), ("predictor.mdlus", one)):
        module.save(tmp_path / filename)
        loaded = Module.from_checkpoint(tmp_path / filename)
        assert type(loaded) is type(module)
        if isinstance(loaded, PartitionAutoencoder):
            torch.testing.assert_close(
                loaded(batch)["prediction"], output["prediction"]
            )
        else:
            torch.testing.assert_close(
                loaded(*temporal_inputs, current_time=0.1), next_latent
            )


def test_one_step_rejects_zero_and_negative_intervals():
    batch = mesh_batch()
    one = OneStepPredictor(latent_dim=4, part_count=2, hidden_dim=8)
    latent = torch.zeros(2, 4)
    args = (latent, latent, batch["patch_features"], batch["patch_part_index"])
    for dt_previous, dt_next in ((0.0, 0.01), (0.01, -0.01)):
        with pytest.raises(ValueError):
            one(*args, dt_previous, dt_next, 0.01)


def test_public_graph_validates_shapes_and_tensor_roles():
    """Give callers the offending field before it reaches a message-passing kernel."""
    model = PartitionAutoencoder(part_count=3, latent_dims=(4, 2, 2))
    batch = mesh_batch()
    for key, malformed in (
        ("decode_patch", torch.tensor([0])),
        ("region_edge_index", torch.zeros(2, 2, 2, dtype=torch.long)),
        ("edge_attr", torch.ones(8, 9)),
    ):
        with pytest.raises(ValueError, match=key):
            model(dict(batch, **{key: malformed}))
    with pytest.raises(ValueError, match="Missing graph fields"):
        model({k: v for k, v in batch.items() if k != "q"})
