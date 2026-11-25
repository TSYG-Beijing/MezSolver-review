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

"""Test displacement masks, material weighting and wheel edge losses."""

import numpy as np
import pytest
import torch
from displacement.data import PartitionedDisplacementDataset
from displacement.losses import displacement_mses, wheel_edge_mses
from make_synthetic_data import generate


def test_masked_materials_and_annular_edges(tmp_path):
    """Check split isolation, mixed-material weights and invariance to translation."""
    manifest = generate(tmp_path / "annuli")
    with PartitionedDisplacementDataset(manifest, "val") as dataset:
        assemblies = dataset.enable_wheels(manifest.parent / "wheels.npz")
        assert len(assemblies) == 4
        assert all(len(wheel.patches) == 6 for wheel in assemblies.values())
        graph = dataset.graph_batch("val_mid", assemblies["rear_right"].patches)
        frame = dataset.attach("val_mid", 4, graph)
        assert np.isfinite(frame["target"]).all()
        assert (~frame["valid"]).sum() == 3
        rows = graph["output_nodes"]
        np.testing.assert_array_equal(
            frame["decode_material_weight"], dataset.material_weight_by_node[rows]
        )
        # All nodes of the interface ring decode the same token but retain two material factors.
        assert (
            len(
                np.unique(
                    frame["decode_material_weight"][
                        dataset.mesh["node_patch"][rows] == 20
                    ]
                )
            )
            == 2
        )
        with pytest.raises(ValueError, match="split"):
            dataset.frame("test_mid", 0, rows)
        batch = {
            k: torch.from_numpy(v)
            for k, v in frame.items()
            if isinstance(v, np.ndarray)
        }
        # Uniform translation cancels in relative edge displacement, including at material interfaces.
        target = torch.zeros_like(batch["target"])
        batch["target"] = target
        shifted = target + torch.tensor([4.0, -1.0, 2.0])
        for edge_type in (False, True):
            assert (
                batch["wheel_edge_valid"] & (batch["wheel_edge_interface"] == edge_type)
            ).any()
        interior, interface = wheel_edge_mses(shifted, batch)
        assert interior.item() == 0.0 and interface.item() == 0.0
        weighted, ordinary = displacement_mses(
            shifted, batch, region_power=1.0, material_power=1.0
        )
        for metric in (weighted, ordinary):
            torch.testing.assert_close(metric, torch.tensor(7.0))


def test_per_node_material_weight_and_per_component_masks():
    """Compare weighted and unweighted losses with a hand-calculated masked example."""
    # In one region node weights 1 and 3 give U1 MSE (1 + 3*9)/4 = 7.
    # U2 has one valid node (MSE 4), and U3 has no labels and is omitted.
    prediction = torch.tensor([[1.0, 2.0, 1.0], [3.0, 100.0, 1.0]])
    batch = dict(
        target=torch.zeros_like(prediction),
        valid=torch.tensor([[True, True, False], [True, False, False]]),
        decode_patch=torch.tensor([0, 0]),
        patch_features=torch.zeros(1, 1),
        decode_material_weight=torch.tensor([1.0, 3.0]),
        patch_material_weight=torch.tensor([1.0]),
    )
    weighted, ordinary = displacement_mses(
        prediction, batch, region_power=1.0, material_power=1.0
    )
    torch.testing.assert_close(weighted, torch.tensor(5.5))
    torch.testing.assert_close(ordinary, torch.tensor(4.5))
