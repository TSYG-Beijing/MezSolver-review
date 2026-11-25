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

"""Partition core/halo assembly preserving physical edges and memberships."""

import numpy as np


def csr_take(ptr, indices, rows):
    """Gather CSR rows and rebase the output offsets to zero."""
    rows = np.asarray(rows, np.int64)
    sizes = ptr[rows + 1] - ptr[rows]
    out_ptr = np.r_[0, np.cumsum(sizes)].astype(np.int64)
    positions = np.repeat(ptr[rows] - out_ptr[:-1], sizes) + np.arange(out_ptr[-1])
    return out_ptr, indices[positions]


class PartitionGraph:
    """Build local physical graphs plus a coarse region-neighborhood graph."""

    def _local_graph(self, patch):
        """Gather a patch's core and halo nodes with their induced physical edges."""
        cache_key = (patch, self.cfg.halo_hops)
        if cache_key in self.graphs:
            self.graphs.move_to_end(cache_key)
            return self.graphs[cache_key]
        m = self.mesh
        core = m["core_indices"][m["core_ptr"][patch] : m["core_ptr"][patch + 1]]
        nodes = np.sort(core)
        for _ in range(self.cfg.halo_hops):
            _, near = csr_take(m["edge_ptr"], m["neighbors"], nodes)
            nodes = np.union1d(nodes, near)
        ptr, near = csr_take(m["edge_ptr"], m["neighbors"], nodes)
        _, flags = csr_take(m["edge_ptr"], m["edge_flags"], nodes)
        target = np.searchsorted(nodes, near)
        keep = target < len(nodes)
        keep[keep] &= nodes[target[keep]] == near[keep]
        source = np.repeat(np.arange(len(nodes)), np.diff(ptr))[keep]
        target = target[keep]
        delta = (m["xyz"][nodes[source]] - m["xyz"][nodes[target]]) / float(m["scale"])
        attr = np.column_stack(
            (
                delta,
                np.linalg.norm(delta, axis=1),
                *((flags[keep] & (1 << i)) != 0 for i in range(3)),
            )
        ).astype(np.float32)
        value = dict(
            nodes=nodes.astype(np.int64),
            edge=np.stack((source, target)).astype(np.int64),
            attr=attr,
            core=np.searchsorted(nodes, core).astype(np.int64),
        )
        size = sum(v.nbytes for v in value.values())
        if size <= self.cfg.graph_cache_mb * 2**20:
            while (
                self.graphs
                and self.graph_bytes + size > self.cfg.graph_cache_mb * 2**20
            ):
                old = self.graphs.popitem(last=False)[1]
                self.graph_bytes -= sum(v.nbytes for v in old.values())
            self.graphs[cache_key] = value
            self.graph_bytes += size
        return value

    def batch(self, cid, selected):
        """Assemble local node graphs and region context for a batch of selected patches."""
        m = self.mesh
        selected = np.sort(np.asarray(selected, np.int64))
        context = selected
        for _ in range(self.cfg.region_layers):
            _, near = csr_take(m["region_ptr"], m["region_neighbors"], context)
            context = np.union1d(context, near)
        nodes = []
        edges = []
        attrs = []
        core = []
        pool_patch = []
        output = []
        decode_patch = []
        offset = 0
        for local, p in enumerate(context):
            g = self._local_graph(int(p))
            size = len(g["nodes"])
            nodes.append(g["nodes"])
            edges.append(g["edge"] + offset)
            attrs.append(g["attr"])
            core.append(g["core"] + offset)
            pool_patch.append(np.full(len(g["core"]), local, np.int64))
            if p in selected:
                output.append(g["core"] + offset)
                decode_patch.append(np.full(len(g["core"]), local, np.int64))
            offset += size
        node = np.concatenate(nodes)
        edge = np.concatenate(edges, axis=1)
        if (
            len(node) > self.cfg.max_graph_nodes
            or edge.shape[1] > self.cfg.max_graph_edges
        ):
            raise MemoryError(
                f"Batch has {len(node)} nodes/{edge.shape[1]} edges; reduce --patches-per-batch to fit the complete halo."
            )
        out = np.concatenate(output)
        out_nodes = node[out]
        pptr, pnear = csr_take(m["region_ptr"], m["region_neighbors"], context)
        rt = np.searchsorted(context, pnear)
        keep = rt < len(context)
        keep[keep] &= context[rt[keep]] == pnear[keep]
        re = np.stack(
            (np.repeat(np.arange(len(context)), np.diff(pptr))[keep], rt[keep])
        ).astype(np.int64)
        feat = np.column_stack(
            (
                m["patch_features"][context],
                np.full(
                    len(context),
                    (self.speeds[self.case_index[cid]] - self.stats["speed_mean"])
                    / self.stats["speed_std"],
                ),
            )
        ).astype(np.float32)
        batch = dict(
            global_core_ids=context.astype(np.int64),
            node_ids=node,
            node_static=m["static"][node],
            edge_index=edge,
            edge_attr=np.concatenate(attrs),
            core_index=np.concatenate(core),
            core_patch=np.concatenate(pool_patch),
            pool_coordinate=m["pool_coordinate"][node],
            patch_features=feat,
            region_edge_index=re,
            selected=np.searchsorted(context, selected),
            decode_index=out,
            decode_patch=np.concatenate(decode_patch),
            output_nodes=out_nodes,
            node_in_degree=np.bincount(edge[1], minlength=len(node)).astype(np.float32),
            region_in_degree=np.bincount(re[1], minlength=len(context)).astype(
                np.float32
            ),
        )
        for name in ["property", "topology"]:
            ptr, index = csr_take(m[name + "_ptr"], m[name + "_indices"], node)
            # Unknown category 0: retain isolated/unassigned nodes in the encoder.
            sizes = np.diff(ptr)
            total = sizes + (sizes == 0)
            newptr = np.r_[0, np.cumsum(total)]
            newindex = np.zeros(newptr[-1], np.int64)
            if len(index):
                positions = np.repeat(newptr[:-1] - ptr[:-1], sizes) + np.arange(
                    len(index)
                )
                newindex[positions] = index
            batch[name + "_ptr"], batch[name + "_indices"] = (
                newptr.astype(np.int64),
                newindex,
            )
        return batch
