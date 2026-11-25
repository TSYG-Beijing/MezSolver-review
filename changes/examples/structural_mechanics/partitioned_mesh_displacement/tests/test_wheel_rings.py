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

"""Check radius-only partitioning on a translated and tilted wheel."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def test_ring_labels_depend_only_on_distance_to_axle():
    """Verify rigid-motion invariance of radial labels and reject coincident bead centers."""
    spec = importlib.util.spec_from_file_location(
        "wheel_rings", Path(__file__).resolve().parents[1] / "wheel_rings.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    theta = np.arange(16) * 2 * np.pi / 16
    direction = np.column_stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))
    rotation = np.array([[0.0, 0.0, 1.0], [0.6, 0.8, 0.0], [-0.8, 0.6, 0.0]])
    shift = np.array([21.0, -32.0, 12.0])
    points = np.concatenate(
        [
            radius * direction + [0.0, 0.0, z]
            for z in (-1.0, 1.0)
            for radius in (2.0, 4.0)
        ]
    )
    faces = np.array([[0, 1, 33, 33], [16, 17, 49, 48]])
    beads = [(3.0 * direction + [0.0, 0.0, z]) @ rotation + shift for z in (-1.0, 1.0)]
    result = module.radial_partition(points @ rotation + shift, faces, *beads, [3.0])
    np.testing.assert_allclose(result["center"], shift, atol=1e-12)
    np.testing.assert_array_equal(result["node_ring"], np.repeat([1, 2, 1, 2], 16))
    np.testing.assert_array_equal(result["cell_ring"], [1, 2])
    with pytest.raises(ValueError, match="distinct"):
        module.radial_partition(points, faces, beads[0], beads[0], [3.0])
