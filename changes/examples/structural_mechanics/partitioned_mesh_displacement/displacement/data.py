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

"""Case-wise HDF5 reader for three-component nodal displacements."""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from .graph import PartitionGraph
from .materials import patch_material_weights
from .wheels import WheelGroups

FORMAT = "partitioned_mesh_displacement_v1"


def file_hash(path):
    """Compute a file's SHA-256 digest without loading it entirely into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    """Hash a canonical JSON serialization of the manifest."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def resolve_path(value, base):
    """Resolve a manifest asset relative to its declared base directory."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


class PartitionedDisplacementDataset(PartitionGraph):
    """Assemble supervised graphs for the train, validation or test split."""

    def __init__(
        self,
        manifest,
        split="train",
        *,
        halo_hops=3,
        region_layers=1,
        graph_cache_mb=256,
        max_graph_nodes=500000,
        max_graph_edges=5000000,
        verify_hashes=True,
        track_reads=False,
        response_cache_mb=0,
        open_cases=3,
    ):
        """Open a case split and validate its mesh, features and normalization data."""
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val or test")
        self.path = Path(manifest).resolve()
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        m = self.manifest
        if m.get("format") != FORMAT:
            raise ValueError("Unsupported displacement manifest")
        self.signature = identity(m)
        self.root = resolve_path(m["dataset_dir"], self.path.parent)
        self.mesh_path = resolve_path(m["mesh_cache"], self.path.parent)
        sdf_path = resolve_path(m["sdf_cache"], self.path.parent)
        if verify_hashes:
            for key, path in (
                ("mesh_sha256", self.mesh_path),
                ("sdf_sha256", sdf_path),
            ):
                if file_hash(path) != m[key]:
                    raise ValueError("Geometry identity mismatch: " + key)
            for name, expected in m.get("dataset_files", {}).items():
                if file_hash(self.root / name) != expected:
                    raise ValueError("Dataset identity mismatch: " + name)
        with np.load(self.mesh_path, allow_pickle=False) as z:
            self.mesh = {key: z[key] for key in z.files}
        sdf = torch.load(sdf_path, map_location="cpu", weights_only=True)
        self.node_sdf = np.asarray(sdf["node_features"], dtype=np.float32)
        self.core_sdf = np.asarray(sdf["core_features"], dtype=np.float32)
        self.node_count = len(self.mesh["node_ids"])
        self.patch_count = len(self.mesh["core_ptr"]) - 1
        if (
            self.mesh["static"].shape != (self.node_count, 47)
            or self.mesh["patch_features"].shape != (self.patch_count, 45)
            or self.node_sdf.shape != (self.node_count, 12)
            or self.core_sdf.shape != (self.patch_count, 24)
        ):
            raise ValueError("Require 47/45 reference features and 12/24 SDF features")
        for array in (
            self.mesh["static"],
            self.mesh["patch_features"],
            self.node_sdf,
            self.core_sdf,
        ):
            if not np.isfinite(array).all():
                raise ValueError("Nonfinite reference geometry features")
        self.stats = m["normalization"]
        self.splits = m["splits"]
        lists = [self.splits[k] for k in ("train", "val", "test")]
        if any(not v or len(v) != len(set(v)) for v in lists):
            raise ValueError("Splits must be nonempty and unique")
        if any(set(lists[i]) & set(lists[j]) for i in range(3) for j in range(i)):
            raise ValueError("Overlapping train/validation/test cases")
        if set(self.stats["fit_cases"]) != set(self.splits["train"]):
            raise ValueError("Normalization must use the training split only")
        self.mean = np.asarray(self.stats["mean"][:3], np.float32)
        self.std = np.asarray(self.stats["std"][:3], np.float32)
        if (
            self.mean.shape != (3,)
            or self.std.shape != (3,)
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.std).all()
            or np.any(self.std <= 0)
        ):
            raise ValueError("Invalid displacement moments")
        self.ids = m["case_ids"]
        if set(self.ids) != set().union(*map(set, lists)) or len(set(self.ids)) != len(
            self.ids
        ):
            raise ValueError("Manifest case inventory disagrees with splits")
        self.case_ids = list(self.splits[split])
        self.split = split
        self.case_index = {cid: i for i, cid in enumerate(self.ids)}
        self.speeds = np.asarray(m["speed_kmh"], np.float32)
        self.times = np.asarray(m["times"], np.float64)
        if len(self.speeds) != len(self.ids) or self.times.shape[0] != len(self.ids):
            raise ValueError("Invalid case condition/time inventory")
        if (
            self.times.ndim != 2
            or self.times.shape[1] < 3
            or not np.isfinite(self.times).all()
            or np.any(self.times[:, 0] != 0)
            or np.any(np.diff(self.times, axis=1) <= 0)
        ):
            raise ValueError("Invalid physical time grids")
        # Derive a single shared clock even for a pre-existing manifest.
        # Only the training split may contribute statistics used by the model.
        train_rows = [self.case_index[cid] for cid in self.splits["train"]]
        self.temporal_reference_time = m.get(
            "temporal_reference_time", float(np.median(self.times[train_rows, -1]))
        )
        if (
            type(self.temporal_reference_time) not in (int, float)
            or not np.isfinite(self.temporal_reference_time)
            or self.temporal_reference_time <= 0
        ):
            raise ValueError(
                "temporal_reference_time must be a finite positive physical time"
            )
        self.cfg = SimpleNamespace(
            halo_hops=halo_hops,
            region_layers=region_layers,
            max_graph_nodes=max_graph_nodes,
            max_graph_edges=max_graph_edges,
            graph_cache_mb=graph_cache_mb,
        )
        self.graphs, self.graph_bytes = OrderedDict(), 0
        self.handles = OrderedDict()
        self.open_cases = max(1, int(open_cases))
        self.response_cache = OrderedDict()
        self.response_cache_limit = max(0, int(response_cache_mb)) * 2**20
        self.response_cache_bytes = 0
        self.cache_hits = self.cache_misses = 0
        # Opt-in test instrumentation; full validation can read millions of frames.
        self.response_reads = [] if track_reads else None
        self._training_patch_cache = {}
        self.node_static_dim, self.patch_feature_dim = 59, 70
        self.property_region_count = int(self.mesh["property_region_count"])
        self.topology_region_count = int(self.mesh["topology_region_count"])
        self.part_count = self.topology_region_count + 1
        self.material_weight_by_node = None
        self.material_weight_by_patch = None
        if "material_weights" in m:
            path = resolve_path(m["material_weights"], self.path.parent)
            if file_hash(path) != m["material_weights_sha256"]:
                raise ValueError("Material loss weight identity mismatch")
            table = json.loads(path.read_text(encoding="utf-8"))
            calibration = table.get("calibration")
            if calibration is not None and not (
                set(calibration["fit_cases"])
                and set(calibration["fit_cases"]).issubset(self.splits["train"])
            ):
                raise ValueError(
                    "Material deformation statistics must fit only train cases"
                )
            self.material_weight_by_patch = patch_material_weights(
                table,
                self.mesh["core_owner_region"],
                self.topology_region_count,
                m["mesh_sha256"],
            )
            if table.get("material_weight_scope") == "per_node_topology_owner":
                self.material_weight_by_node = patch_material_weights(
                    table,
                    self.mesh["owner_region"],
                    self.topology_region_count,
                    m["mesh_sha256"],
                )
        speed_std = self.stats["speed_std"]
        if (
            not np.isfinite(self.speeds).all()
            or not np.isfinite(self.stats["speed_mean"])
            or not np.isfinite(speed_std)
            or speed_std <= 0
        ):
            raise ValueError(
                "Expected finite case speeds and a positive training speed scale"
            )
        self.activity = h5py.File(self.root / "training_masks.h5", "r")
        self.wheels = None
        self.seams = None
        self._shell_activity = None
        self._deleted_cache = OrderedDict()
        self._initial_wheel_masks = OrderedDict()

    def trajectory_time_scale(self, case):
        """Return the shared physical reference time, or the legacy case duration."""
        if self.temporal_reference_time is None:
            return float(self.times[self.case_index[case], -1])
        return float(self.temporal_reference_time)

    def restore_trajectory_clock(self, checkpoint):
        """Keep archived per-case clocks and verify the clock on newer checkpoints."""
        if checkpoint.get("temporal_reference_time") is None:
            self.temporal_reference_time = None
        elif checkpoint["temporal_reference_time"] != self.temporal_reference_time:
            raise ValueError(
                "Checkpoint and manifest disagree on temporal_reference_time"
            )

    def enable_wheels(self, asset_path):
        """Enable shell supervision and complete-wheel batches using a fixed reference asset."""
        groups = WheelGroups.load(
            resolve_path(asset_path, self.path.parent),
            self.mesh,
            self.manifest["mesh_sha256"],
        )
        self.enable_shell_activity()
        self.wheels = groups
        return groups

    def enable_shell_activity(self):
        """Open the case-specific deletion table shared by wheel and seam supervision."""
        if self._shell_activity is None:
            self._shell_activity = h5py.File(self.root / "element_activity.h5", "r")

    def deleted_wheel_shells(self, case, frame):
        """Read the deleted shell element rows for a case and frame."""
        if self._shell_activity is None:
            raise RuntimeError(
                "Enable shell activity before requesting shell deletions"
            )
        if (
            case not in self.case_ids
            or frame < 0
            or frame >= len(self.times[self.case_index[case]])
        ):
            raise ValueError("Wheel activity outside the opened split or time grid")
        key = (case, int(frame))
        if key not in self._deleted_cache:
            group = self._shell_activity[f"cases/{case}/shell"]
            start, stop = group["deleted_indptr"][frame : frame + 2]
            self._deleted_cache[key] = group["deleted_element_rows"][start:stop]
            if len(self._deleted_cache) > 256:
                self._deleted_cache.popitem(last=False)
        self._deleted_cache.move_to_end(key)
        return self._deleted_cache[key]

    def handle(self, case):
        """Return a read-only case file from the bounded cache of HDF5 handles."""
        if case not in self.case_ids:
            raise ValueError(f"{case} is outside the opened {self.split} split")
        if case not in self.handles:
            self.handles[case] = h5py.File(self.root / "cases" / (case + ".h5"), "r")
            if len(self.handles) > self.open_cases:
                self.handles.popitem(last=False)[1].close()
        self.handles.move_to_end(case)
        return self.handles[case]

    def training_patches(self, case):
        """Exclude permanently unsupervised wall/isolated cores using train masks."""
        if self.split != "train" or case not in self.case_ids:
            raise ValueError("Sampling statistics are training-only")
        if case not in self._training_patch_cache:
            mask = self.activity[f"cases/{case}/valid_displacement_training"]
            present = np.zeros(self.patch_count, bool)
            for frame in range(1, mask.shape[0]):
                valid = mask[frame]
                present |= (
                    np.bincount(
                        self.mesh["node_patch"],
                        weights=valid,
                        minlength=self.patch_count,
                    )
                    > 0
                )
            self._training_patch_cache[case] = np.flatnonzero(present)
        return self._training_patch_cache[case]

    def frame(self, case, frame, nodes):
        """Only three displacement columns are read; never stress or PEEQ."""
        nodes = np.asarray(nodes, np.int64)
        unique, inverse = np.unique(nodes, return_inverse=True)
        h = self.handle(case)
        raw = np.empty((len(unique), 3), np.float32)
        valid = np.empty(len(unique), bool)
        # Bounded contiguous HDF5 slices avoid huge fancy-index reads.
        blocks = unique // 32768
        cuts = np.r_[0, np.flatnonzero(np.diff(blocks)) + 1, len(unique)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            lo = int(blocks[a]) * 32768
            hi = min(lo + 32768, self.node_count)
            key = (case, int(frame), lo)
            if key in self.response_cache:
                block, activity = self.response_cache[key]
                self.response_cache.move_to_end(key)
                self.cache_hits += 1
            else:
                block = h["fields"][frame, lo:hi, :3]
                activity = self.activity[f"cases/{case}/valid_displacement_training"][
                    frame, lo:hi
                ]
                self.cache_misses += 1
                size = block.nbytes + activity.nbytes
                if size <= self.response_cache_limit:
                    while self.response_cache_bytes + size > self.response_cache_limit:
                        _, old = self.response_cache.popitem(last=False)
                        self.response_cache_bytes -= sum(v.nbytes for v in old)
                    self.response_cache[key] = (block, activity)
                    self.response_cache_bytes += size
            raw[a:b] = block[unique[a:b] - lo]
            valid[a:b] = activity[unique[a:b] - lo]
        finite = np.isfinite(raw).all(-1)
        if np.any(valid & ~finite):
            raise ValueError("Nonfinite valid displacement")
        if frame == 0:
            valid = finite  # all known finite initial displacements are inputs
        mask = np.repeat(valid[:, None], 3, axis=1)
        q = np.where(mask, (raw - self.mean) / self.std, 0.0).astype(np.float32)
        if self.response_reads is not None:
            self.response_reads.append((case, int(frame)))
        return q[inverse], mask[inverse]

    def graph_batch(self, case, selected, *, region_layers=None):
        """Build core, halo and context tensors for the requested case and patches."""
        if case not in self.case_ids:
            raise ValueError("Case not in selected split")
        selected = np.unique(np.asarray(selected, np.int64))
        if not len(selected) or selected[0] < 0 or selected[-1] >= self.patch_count:
            raise ValueError("Invalid selected patches")
        old = self.cfg.region_layers
        try:
            if region_layers is not None:
                self.cfg.region_layers = region_layers
            b = super().batch(case, selected)
        finally:
            self.cfg.region_layers = old
        b["node_static"] = np.concatenate(
            (b["node_static"], self.node_sdf[b["node_ids"]]), -1
        )
        b["patch_features"] = np.concatenate(
            (b["patch_features"], self.core_sdf[b["global_core_ids"]]), -1
        )
        b["patch_part_index"] = (
            self.mesh["core_owner_region"][b["global_core_ids"]].astype(np.int64) + 1
        )
        if self.material_weight_by_patch is not None:
            b["patch_material_weight"] = self.material_weight_by_patch[
                b["global_core_ids"]
            ]
        if self.material_weight_by_node is not None:
            b["decode_material_weight"] = self.material_weight_by_node[
                b["output_nodes"]
            ]
        b["node_patch"] = np.zeros(len(b["node_ids"]), np.int64)
        b["node_patch"][b["core_index"]] = b["core_patch"]
        b["core_mask"] = np.zeros(len(b["node_ids"]), bool)
        b["core_mask"][b["core_index"]] = True
        if self.wheels is not None:
            self.wheels.add_edges(b, selected)
        return b

    def attach(self, case, frame, graph):
        """Attach normalized displacements, component masks and wheel activity for one frame."""
        q, mask = self.frame(case, frame, graph["node_ids"])
        batch = dict(
            graph,
            q=q,
            observed=mask.astype(np.float32),
            target=q[graph["decode_index"]],
            valid=mask[graph["decode_index"]],
        )
        if self.wheels is not None:
            batch["wheel_edge_valid"] = np.empty(0, bool)
            if graph["_wheel_edge_selection"]:
                supervised = batch["valid"].all(1)
                if frame == 0:
                    # Finite initial positions can be model inputs even when not supervised.
                    if case not in self._initial_wheel_masks:
                        self._initial_wheel_masks[case] = self.activity[
                            f"cases/{case}/valid_displacement_training"
                        ][0]
                        if len(self._initial_wheel_masks) > self.open_cases:
                            self._initial_wheel_masks.popitem(last=False)
                    self._initial_wheel_masks.move_to_end(case)
                    supervised &= self._initial_wheel_masks[case][graph["output_nodes"]]
                batch["wheel_edge_valid"] = self.wheels.edge_valid(
                    graph, supervised, self.deleted_wheel_shells(case, frame)
                )
        return batch

    def model_kwargs(self, **overrides):
        """Combine dataset feature sizes and vocabulary sizes with model settings."""
        values = dict(
            node_static_dim=59,
            patch_feature_dim=70,
            part_count=self.part_count,
            property_region_count=self.property_region_count,
            topology_region_count=self.topology_region_count,
            membership_embedding_dim=8,
            observed_mask=True,
            edge_dim=7,
            hidden_dim=128,
            latent_dims=(64, 32, 32),
            message_layers=3,
            region_layers=1,
            decoder_hidden_dim=256,
            decoder_residual_layers=4,
            token_attention_heads=4,
        )
        values.update(overrides)
        return values

    def close(self):
        """Release the open case files and shared HDF5 resources."""
        for handle in self.handles.values():
            handle.close()
        self.activity.close()
        if self._shell_activity is not None:
            self._shell_activity.close()
        self.response_cache.clear()
        self.response_cache_bytes = 0

    def __enter__(self):
        """Return the opened case split for use in a context manager."""
        return self

    def __exit__(self, *args):
        """Close the split's HDF5 resources when leaving the context."""
        self.close()
