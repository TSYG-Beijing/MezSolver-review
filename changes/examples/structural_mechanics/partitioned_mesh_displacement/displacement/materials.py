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

"""Map source-topology material weights to target nodes or regions."""

import numpy as np


def patch_material_weights(data, core_owners, topology_region_count, mesh_sha256):
    """Map each core owner to its mesh-verified material loss weight."""
    if (
        data.get("format") != "mesh_material_weights_v1"
        or data["mesh_sha256"] != mesh_sha256
    ):
        raise ValueError("Material table and mesh identities disagree")
    cards = data["material_types"]
    types = np.asarray(data["region_type_index"])
    if (
        not cards
        or len(cards) != len(set(cards))
        or types.shape != (topology_region_count,)
        or not np.issubdtype(types.dtype, np.integer)
        or types.min() < 0
        or types.max() >= len(cards)
    ):
        raise ValueError("Material types do not cover all topology regions")
    factors = np.asarray(
        [data["type_weights"][card] for card in cards], dtype=np.float32
    )
    if (
        not np.isfinite(factors).all()
        or np.any(factors <= 0)
        or data["unassigned_weight"] != 1.0
    ):
        raise ValueError("Invalid material loss weights")
    owners = np.asarray(core_owners)
    if owners.min() < -1 or owners.max() >= topology_region_count:
        raise ValueError("Invalid core owner in material mapping")
    weights = np.ones(len(owners), np.float32)
    valid = owners >= 0
    weights[valid] = factors[types[owners[valid]]]
    return weights
