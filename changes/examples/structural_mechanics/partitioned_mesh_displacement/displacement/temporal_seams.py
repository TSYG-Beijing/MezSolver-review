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

"""Build and use undeformed shell perimeter edges across same-owner partitions."""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from .data import file_hash, resolve_path

FORMAT = "shell_partition_seams_v1"


def build_arrays(mesh, mapping):
    """Keep true perimeter edges on shell elements of the two endpoints' topology owner."""
    node_ids, xyz = mesh["node_ids"], mesh["xyz"]
    owner, patch = mesh["owner_region"], mesh["node_patch"]
    n, core_count = len(owner), len(mesh["core_ptr"]) - 1
    with h5py.File(mapping, "r") as m:
        np.testing.assert_array_equal(m["nodes/ids"][:], node_ids)
        offsets = m["elements/type_offsets"][:]
        shell_owner = m["topology/element_region_index"][offsets[1] : offsets[2]]
        kinds = m["topology/kind"].asstr()[:]
    ptr, neighbors, flags = mesh["edge_ptr"], mesh["neighbors"], mesh["edge_flags"]
    shell_nodes = (owner >= 0) & (kinds[owner] == "shell") & ~mesh["wall"].astype(bool)
    keys = []
    for start in range(0, n, 100000):
        stop = min(n, start + 100000)
        a = np.repeat(np.arange(start, stop), np.diff(ptr[start : stop + 1]))
        lo, hi = ptr[start], ptr[stop]
        b = neighbors[lo:hi]
        keep = (a < b) & ((flags[lo:hi] & 2) != 0) & shell_nodes[a] & shell_nodes[b]
        keep &= (owner[a] == owner[b]) & (patch[a] != patch[b])
        a, b = a[keep], b[keep]
        keep = np.linalg.norm(xyz[a] - xyz[b], axis=1) > 1e-5
        keys.append(a[keep] * n + b[keep])
    candidates = np.unique(np.concatenate(keys))
    if not len(candidates):
        raise ValueError("Mesh has no eligible cross-core shell edges")
    # Reconstruct original shell perimeter incidences. A quadrilateral's internal
    # display diagonal occurs twice in that same shell row and is not a support.
    triangles, elements = mesh["triangles"], mesh["tri_element"]
    if elements.min() < 0 or elements.max() >= len(shell_owner):
        raise ValueError("Triangle shell-row numbering differs from topology mapping")
    joined = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = triangles[:, i], triangles[:, j]
        keep = (
            (patch[a] != patch[b])
            & (owner[a] == owner[b])
            & (shell_owner[elements] == owner[a])
        )
        a, b, rows = a[keep], b[keep], elements[keep]
        keys = np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b)
        loc = np.searchsorted(candidates, keys)
        hit = (loc < len(candidates)) & (
            candidates[np.minimum(loc, len(candidates) - 1)] == keys
        )
        joined.append(loc[hit] * len(shell_owner) + rows[hit])
    unique, repetitions = np.unique(np.concatenate(joined), return_counts=True)
    ids, shells = np.divmod(unique[repetitions == 1], len(shell_owner))
    edge_ids, counts = np.unique(ids, return_counts=True)
    edges = np.column_stack(np.divmod(candidates[edge_ids], n))
    if not len(edges):
        raise ValueError("No cross-core physical edges have same-part shell support")
    pairs = np.sort(patch[edges], axis=1)
    pair_key = pairs[:, 0].astype(np.int64) * core_count + pairs[:, 1]
    pair_keys, edge_pair = np.unique(pair_key, return_inverse=True)
    return dict(
        edges=edges.astype(np.int32),
        patch_pairs=np.column_stack(np.divmod(pair_keys, core_count)).astype(np.int32),
        edge_pair=edge_pair.astype(np.int32),
        support_ptr=np.r_[0, counts.cumsum()].astype(np.int64),
        support_shells=shells.astype(np.int32),
    )


class SeamEdges:
    """Sample and index physically supported shell edges crossing a partition."""

    def __init__(self, data, cache):
        """Verify the cached shell seam topology against the loaded reference mesh."""
        self.node_patch = data.mesh["node_patch"]
        self.patch_count = data.patch_count
        signature = data.manifest["mesh_sha256"]
        with np.load(cache, allow_pickle=False) as asset:
            if str(asset["format"]) != FORMAT or str(asset["mesh_sha256"]) != signature:
                raise ValueError("Seam topology cache identity mismatch")
            for name in (
                "edges",
                "patch_pairs",
                "edge_pair",
                "support_ptr",
                "support_shells",
            ):
                setattr(self, name, asset[name])
        if (
            self.edges.shape != (len(self.edge_pair), 2)
            or not len(self.edge_pair)
            or self.edges.min() < 0
            or self.edges.max() >= len(self.node_patch)
            or self.edge_pair.min() < 0
            or self.edge_pair.max() >= len(self.patch_pairs)
            or self.support_ptr.shape != (len(self.edges) + 1,)
            or self.support_ptr[0] != 0
            or self.support_ptr[-1] != len(self.support_shells)
            or np.any(np.diff(self.support_ptr) <= 0)
            or np.any(self.support_shells < 0)
        ):
            raise ValueError("Invalid seam topology arrays")
        lengths = np.linalg.norm(
            data.mesh["xyz"][self.edges[:, 0]].astype(np.float64)
            - data.mesh["xyz"][self.edges[:, 1]],
            axis=1,
        )
        if not np.isfinite(lengths).all() or np.any(lengths <= 1e-5):
            raise ValueError("Expected finite, nondegenerate reference seam edges")
        self.inverse_length = np.reciprocal(lengths).astype(np.float32)
        self.std = np.asarray(data.std, np.float32)
        np.testing.assert_array_equal(
            np.sort(self.node_patch[self.edges], axis=1),
            self.patch_pairs[self.edge_pair],
        )
        self._pair_keys = (
            self.patch_pairs[:, 0].astype(np.int64) * self.patch_count
            + self.patch_pairs[:, 1]
        )
        if np.any(np.diff(self._pair_keys) <= 0) or np.any(
            self.patch_pairs[:, 0] >= self.patch_pairs[:, 1]
        ):
            raise ValueError("Invalid or duplicate physical core pair")
        self._edge_order = np.argsort(self.edge_pair, kind="stable")
        self._pair_ptr = np.r_[
            0, np.bincount(self.edge_pair, minlength=len(self.patch_pairs)).cumsum()
        ]

    def sample_patches(self, rng, available, budget, pairs_per_batch=1):
        """Draw real neighboring core pairs uniformly, then fill the patch budget uniformly."""
        available = np.unique(np.asarray(available, dtype=np.int64))
        if budget < 1 or not len(available) or pairs_per_batch < 1:
            raise ValueError(
                "Need a positive core budget, pair count, and eligible cores"
            )
        count = min(budget, len(available))
        present = np.zeros(self.patch_count, dtype=bool)
        present[available] = True
        eligible = np.flatnonzero(present[self.patch_pairs].all(1))
        selected = set()
        if count >= 2 and len(eligible):
            for i in rng.choice(
                eligible,
                size=min(count // 2, pairs_per_batch, len(eligible)),
                replace=False,
            ):
                selected.update(map(int, self.patch_pairs[i]))
        chosen = np.array(sorted(selected), dtype=np.int64)
        others = np.setdiff1d(available, chosen, assume_unique=True)
        extra = rng.choice(others, count - len(chosen), replace=False)
        return np.sort(np.r_[chosen, extra].astype(np.int64))

    def bind_graph(self, graph):
        """Return static edge IDs and [E,2] local output row indices for fully decoded pairs."""
        nodes = np.asarray(graph["output_nodes"])
        cores = np.unique(self.node_patch[nodes]).astype(np.int64)
        a, b = np.triu_indices(len(cores), k=1)
        keys = cores[a] * self.patch_count + cores[b]
        match = np.searchsorted(self._pair_keys, keys)
        match = match[
            (match < len(self._pair_keys))
            & (self._pair_keys[np.minimum(match, len(self._pair_keys) - 1)] == keys)
        ]
        if not len(match):
            return np.empty(0, np.int64), np.empty((0, 2), np.int64)
        ids = np.concatenate(
            [self._edge_order[self._pair_ptr[i] : self._pair_ptr[i + 1]] for i in match]
        )
        order = np.argsort(nodes)
        sorted_nodes = nodes[order]
        pos = np.searchsorted(sorted_nodes, self.edges[ids])
        keep = (pos < len(nodes)).all(1) & (
            sorted_nodes[np.minimum(pos, len(nodes) - 1)] == self.edges[ids]
        ).all(1)
        return ids[keep], order[pos[keep]]

    def active_mask(self, edge_ids, deleted):
        """True if at least one supporting same-part shell remains alive at this frame."""
        if not len(deleted) or not len(edge_ids):
            return np.ones(len(edge_ids), bool)
        ids = np.asarray(edge_ids)
        start, stop = self.support_ptr[ids], self.support_ptr[ids + 1]
        count = stop - start
        begin = np.r_[0, np.cumsum(count)[:-1]]
        pos = np.repeat(start - begin, count) + np.arange(count.sum())
        return np.logical_or.reduceat(
            ~np.isin(self.support_shells[pos], np.asarray(deleted)), begin
        )

    def residual_sums(self, bound, deleted, prediction, batch):
        """Return U1 jump, vector jump and vector gradient SSE plus the live edge count."""
        edge_ids, local = bound
        active = self.active_mask(edge_ids, deleted)
        a, b = torch.as_tensor(local[active], device=prediction.device).unbind(1)
        valid = batch["valid"].all(-1)
        good = valid[a] & valid[b]
        residual = torch.where(
            valid[:, None], prediction.float() - batch["target"].float(), 0.0
        )
        scale = torch.as_tensor(self.std, device=prediction.device)
        jump = torch.where(good[:, None], residual[b] - residual[a], 0.0) * scale
        inverse_length = torch.as_tensor(
            self.inverse_length[edge_ids[active]], device=prediction.device
        )
        square = jump.square()
        return torch.stack(
            (
                square[:, 0].sum(),
                square.sum(),
                (jump * inverse_length[:, None]).square().sum(),
            )
        ), good.sum()


def enable_seams(data, asset_path=None):
    """Attach a mesh-matched seam asset and open the split's shell deletion table."""
    path = asset_path or data.manifest.get("seam_edges")
    if not path:
        raise ValueError(
            "Seam supervision requires --seam-edges or manifest.seam_edges"
        )
    resolved = resolve_path(path, data.path.parent)
    if data.manifest.get("seam_edges") and resolved == resolve_path(
        data.manifest["seam_edges"], data.path.parent
    ):
        expected = data.manifest.get("seam_edges_sha256")
        if expected and file_hash(resolved) != expected:
            raise ValueError("Seam asset checksum differs from the dataset manifest")
    data.seams = SeamEdges(data, resolved)
    data.enable_shell_activity()
    return data.seams


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.mesh, allow_pickle=False) as mesh:
        arrays = build_arrays(mesh, args.mapping)
    with args.output.open("xb") as stream:
        np.savez_compressed(
            stream, format=FORMAT, mesh_sha256=file_hash(args.mesh), **arrays
        )
