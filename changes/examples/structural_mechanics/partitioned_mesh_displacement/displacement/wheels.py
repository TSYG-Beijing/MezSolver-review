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

"""Optional tyre and rim shell groups, physical edges, and complete-wheel sampling."""

import hashlib
from dataclasses import dataclass, fields

import numpy as np

DEFAULT_WHEELS = ("front_left", "front_right", "rear_left", "rear_right")
FORMAT = "wheel_shell_groups_v1"


def array_hash(values):
    """Hash the contiguous byte representation of an array."""
    return hashlib.sha256(
        memoryview(np.ascontiguousarray(values)).cast("B")
    ).hexdigest()


@dataclass
class WheelGroup:
    """Store node rows, shell incidence and edge categories for one wheel."""

    rows: np.ndarray
    tire_rows: np.ndarray
    rim_rows: np.ndarray
    patches: np.ndarray
    edges: np.ndarray
    edge_shells: np.ndarray
    interface: np.ndarray
    attachment: np.ndarray
    tire_edge: np.ndarray
    rim_edge: np.ndarray
    tire_triangles: np.ndarray
    tire_elements: np.ndarray

    def active_edges(self, deleted, indices=None):
        """An edge survives when any incident wheel shell survives, including non-manifold joins."""
        support = self.edge_shells if indices is None else self.edge_shells[indices]
        if not len(deleted):
            return np.ones(len(support), dtype=bool)
        return ((support >= 0) & ~np.isin(support, deleted)).any(axis=1)


class WheelGroups(dict):
    """Manage verified wheel groups used for sampling and edge supervision."""

    @classmethod
    def load(cls, path, mesh, mesh_sha):
        """Read wheel groups and verify their correspondence with the reference mesh."""
        result = cls()
        ids, patch = mesh["node_ids"], mesh["node_patch"]
        with np.load(path, allow_pickle=False) as asset:
            if (
                str(asset["format"]) != FORMAT
                or str(asset["mesh_sha256"]) != mesh_sha
                or str(asset["node_ids_sha256"]) != array_hash(ids)
            ):
                raise ValueError(
                    "Wheel asset does not match the supplied mesh and node order"
                )
            names = (
                asset["wheel_names"].tolist()
                if "wheel_names" in asset
                else DEFAULT_WHEELS
            )
            if not names or len(set(names)) != len(names):
                raise ValueError("Expected a nonempty list of distinct wheel names")
            for name in names:
                parts = asset[name + "_parts"]
                if (
                    parts.ndim != 1
                    or not len(parts)
                    or parts.dtype.kind not in "iu"
                    or len(np.unique(parts)) != len(parts)
                ):
                    raise ValueError("Wheel part inventory mismatch for " + name)
                g = WheelGroup(
                    **{f.name: asset[name + "_" + f.name] for f in fields(WheelGroup)}
                )
                g_rows = np.union1d(g.tire_rows, g.rim_rows)
                if (
                    not np.array_equal(g.rows, g_rows)
                    or not len(g.rows)
                    or g.rows.min() < 0
                    or g.rows.max() >= len(ids)
                    or not np.array_equal(g.patches, np.unique(patch[g.rows]))
                    or not np.array_equal(asset[name + "_node_ids"], ids[g.rows])
                ):
                    raise ValueError("Invalid wheel rows or core ownership for " + name)
                for rows in (g.rows, g.tire_rows, g.rim_rows):
                    if not np.issubdtype(rows.dtype, np.integer) or np.any(
                        np.diff(rows) <= 0
                    ):
                        raise ValueError("Expected strictly ordered wheel rows")
                e = g.edges
                if (
                    e.ndim != 2
                    or e.shape[1] != 2
                    or not len(e)
                    or not np.issubdtype(e.dtype, np.integer)
                    or np.any(e[:, 0] >= e[:, 1])
                    or not np.isin(e, g.rows).all()
                    or len(np.unique(e, axis=0)) != len(e)
                    or g.edge_shells.ndim != 2
                    or len(g.edge_shells) != len(e)
                    or np.any((g.edge_shells < 0).all(axis=1))
                    or not np.issubdtype(g.edge_shells.dtype, np.integer)
                    or np.any(g.edge_shells < -1)
                ):
                    raise ValueError(
                        "Invalid physical shell edge or shell incidence for " + name
                    )
                for mask in (g.interface, g.attachment, g.tire_edge, g.rim_edge):
                    if mask.shape != (len(e),) or mask.dtype != bool:
                        raise ValueError("Invalid wheel edge category")
                is_tire = np.isin(e, g.tire_rows)
                if (
                    not np.array_equal(g.attachment, is_tire[:, 0] ^ is_tire[:, 1])
                    or not np.array_equal(
                        g.interface, g.attachment | (patch[e[:, 0]] != patch[e[:, 1]])
                    )
                    or not np.all(g.tire_edge | g.rim_edge)
                ):
                    raise ValueError("Incorrect wheel interface categories")
                if (
                    g.tire_triangles.ndim != 2
                    or g.tire_triangles.shape[1] != 3
                    or g.tire_elements.shape != (len(g.tire_triangles),)
                    or np.min(g.tire_triangles) < 0
                    or np.max(g.tire_triangles) >= len(ids)
                    or np.min(g.tire_elements) < 0
                ):
                    raise ValueError("Invalid reference tyre shell triangles")
                result[name] = g
        joined = np.concatenate([g.rows for g in result.values()])
        if len(joined) != len(np.unique(joined)):
            raise ValueError("Wheel assemblies must be disjoint")
        return result

    def sample_patches(self, rng, available, batch_size):
        """Select an eligible complete assembly, then fill uniformly. A large wheel expands the batch."""
        available = np.asarray(available, dtype=np.int64)
        if batch_size < 1 or not len(available):
            raise ValueError("Need a positive batch size and eligible mesh patches")
        choices = [
            g.patches for g in self.values() if np.isin(g.patches, available).all()
        ]
        if not choices:
            return rng.choice(available, min(batch_size, len(available)), replace=False)
        wheel = choices[int(rng.integers(len(choices)))]
        remaining = np.setdiff1d(available, wheel)
        extra = min(len(remaining), max(0, batch_size - len(wheel)))
        return (
            np.sort(np.r_[wheel, rng.choice(remaining, extra, replace=False)])
            if extra
            else wheel.copy()
        )

    def add_edges(self, graph, selected):
        """Map only fully decoded edges to decoder-output row indices; context/halo is excluded."""
        available = [
            (name, g) for name, g in self.items() if np.isin(g.patches, selected).any()
        ]
        local, interface, retained = [], [], []
        if available:
            output = graph["output_nodes"]
            order = np.argsort(output)
            sorted_output = output[order]
            for name, g in available:
                positions = np.searchsorted(sorted_output, g.edges)
                in_bounds = (positions < len(output)).all(1)
                keep = np.flatnonzero(in_bounds)
                keep = keep[(sorted_output[positions[keep]] == g.edges[keep]).all(1)]
                if len(keep):
                    local.append(order[positions[keep]])
                    interface.append(g.interface[keep])
                    retained.append((name, keep))
        graph["wheel_edge_index"] = (
            np.concatenate(local).T.copy() if local else np.empty((2, 0), np.int64)
        )
        graph["wheel_edge_interface"] = (
            np.concatenate(interface) if interface else np.empty(0, bool)
        )
        graph["_wheel_edge_selection"] = tuple(retained)

    def edge_valid(self, graph, endpoint_valid, deleted):
        """Require valid endpoint labels and at least one surviving supporting shell."""
        retained = graph["_wheel_edge_selection"]
        if not retained:
            return np.empty(0, bool)
        alive = np.concatenate(
            [self[name].active_edges(deleted, indices) for name, indices in retained]
        )
        return alive & endpoint_valid[graph["wheel_edge_index"]].all(axis=0)
