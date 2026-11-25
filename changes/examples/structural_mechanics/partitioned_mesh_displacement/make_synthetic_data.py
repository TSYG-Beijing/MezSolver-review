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

"""Generate four independent, triangulated annuli with analytic displacement histories.

The example exercises mesh membership, ring sampling, masks, mixed-material weights
and interface edges. Its trajectories are prescribed functions, not solver output.
All paths in the generated manifest are relative to its parent directory.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from displacement.temporal_seams import FORMAT as SEAM_FORMAT
from displacement.temporal_seams import build_arrays

WHEELS = ("front_left", "front_right", "rear_left", "rear_right")
RINGS, SECTORS = 6, 8


def file_hash(path):
    """Return the SHA-256 digest recorded for an asset in the manifest."""
    with path.open("rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def write_json(path, data):
    """Write indented UTF-8 JSON and reject nonfinite numbers."""
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def adjacency(pairs, count):
    """Encode unique undirected mesh edges as a symmetric CSR neighbor table."""
    directed = np.unique(np.concatenate((pairs, pairs[:, ::-1])), axis=0)
    degrees = np.bincount(directed[:, 0], minlength=count)
    return np.r_[0, np.cumsum(degrees)], directed[:, 1]


def make_mesh(directory):
    """Write mesh, wheel, material and distance features and return coordinates and patch IDs."""
    nodes_per_wheel = RINGS * SECTORS
    count = len(WHEELS) * nodes_per_wheel
    node_ids = np.arange(1, count + 1)
    rows = np.arange(count)
    patch = rows // SECTORS
    ring = patch % RINGS
    sector = rows % SECTORS
    angle = 2 * np.pi * sector / SECTORS
    radius = 1.0 + ring
    wheel = rows // nodes_per_wheel
    xyz = np.column_stack(
        (
            20.0 * (wheel // 2) + radius * np.cos(angle),
            15.0 * (wheel % 2),
            radius * np.sin(angle),
        )
    ).astype(np.float32)
    group_count = len(WHEELS) * RINGS
    wheels = {}
    all_edges = []
    all_triangles = []
    for idx, name in enumerate(WHEELS):
        local_rows = rows[wheel == idx].reshape(RINGS, SECTORS)
        # Each quad between two rings produces two triangles in the undeformed mesh.
        inner, outer = local_rows[:-1], local_rows[1:]
        next_inner = np.roll(inner, -1, axis=1)
        next_outer = np.roll(outer, -1, axis=1)
        triangles = np.stack(
            (
                np.stack((inner, outer, next_outer), axis=-1),
                np.stack((inner, next_outer, next_inner), axis=-1),
            ),
            axis=2,
        ).reshape(-1, 3)
        elements = idx * len(triangles) + np.arange(len(triangles))
        incidence = {}
        for tri, element in zip(triangles, elements, strict=True):
            for a, b in zip(tri, np.roll(tri, -1), strict=True):
                edge = (min(a, b), max(a, b))
                incidence.setdefault(edge, []).append(element)
        edges = np.array(sorted(incidence))
        support = np.full((len(edges), 2), -1)
        for j, edge in enumerate(edges):
            incident = incidence[tuple(edge)]
            support[j, : len(incident)] = incident
        wheel_rows = local_rows.ravel()
        tire_rows, rim_rows = local_rows[2:].ravel(), local_rows[:3].ravel()
        in_tire = np.isin(edges, tire_rows)
        attachment = in_tire[:, 0] ^ in_tire[:, 1]
        tire_faces = np.isin(triangles, tire_rows).all(axis=1)
        fields = dict(
            rows=wheel_rows,
            node_ids=node_ids[wheel_rows],
            parts=np.arange(100 + 2 * idx, 102 + 2 * idx),
            tire_rows=tire_rows,
            rim_rows=rim_rows,
            patches=np.unique(patch[wheel_rows]),
            edges=edges,
            edge_shells=support,
            attachment=attachment,
            interface=attachment | (patch[edges[:, 0]] != patch[edges[:, 1]]),
            tire_edge=in_tire.all(axis=1),
            rim_edge=np.isin(edges, rim_rows).all(axis=1),
            tire_triangles=triangles[tire_faces],
            tire_elements=elements[tire_faces],
        )
        wheels.update({name + "_" + k: v for k, v in fields.items()})
        all_edges.append(edges)
        all_triangles.append(triangles)
    edges = np.concatenate(all_edges)
    edge_ptr, neighbors = adjacency(edges, count)
    pairs = patch[edges]
    region_ptr, region_neighbors = adjacency(
        pairs[pairs[:, 0] != pairs[:, 1]], group_count
    )
    static = np.zeros((count, 47), np.float32)
    static[:, :3] = xyz / 30
    static[:, 3:6] = np.column_stack((np.cos(angle), radius / RINGS, np.sin(angle)))
    centers = xyz.reshape(group_count, SECTORS, 3).mean(axis=1)
    patch_features = np.zeros((group_count, 45), np.float32)
    patch_features[:, :3] = centers / 30
    # Introduce one mixed-material seam so point-wise weights differ inside a region.
    owner = ((ring > 2) | ((ring == 2) & (sector < 2))).astype(np.int64)
    core_owner = (owner.reshape(group_count, SECTORS).mean(axis=1) > 0.5).astype(
        np.int64
    )
    np.savez_compressed(
        directory / "mesh.npz",
        xyz=xyz,
        node_ids=node_ids,
        node_patch=patch,
        core_ptr=np.arange(0, count + 1, SECTORS),
        core_indices=rows,
        owner_region=owner,
        core_owner_region=core_owner,
        static=static,
        patch_features=patch_features,
        edge_ptr=edge_ptr,
        neighbors=neighbors,
        edge_flags=np.full(len(neighbors), 2, np.uint8),
        region_ptr=region_ptr,
        region_neighbors=region_neighbors,
        scale=np.array(30.0),
        property_ptr=np.arange(count + 1),
        property_indices=owner + 1,
        property_region_count=np.array(2),
        topology_ptr=np.arange(count + 1),
        topology_indices=owner + 1,
        topology_region_count=np.array(2),
        wall=np.zeros(count, bool),
        triangles=np.concatenate(all_triangles),
        tri_element=np.arange(sum(map(len, all_triangles))),
        pool_coordinate=np.cos(angle).astype(np.float32),
    )
    sha = file_hash(directory / "mesh.npz")
    mapping = directory / "shell_topology.h5"
    triangles = np.concatenate(all_triangles)
    with h5py.File(mapping, "w") as handle:
        handle["nodes/ids"] = node_ids
        handle["elements/type_offsets"] = [0, 0, len(triangles)]
        handle["topology/kind"] = np.array(["shell", "shell"], dtype="S5")
        handle["topology/element_region_index"] = (owner[triangles].sum(1) >= 2).astype(
            np.int64
        )
    with np.load(directory / "mesh.npz", allow_pickle=False) as mesh:
        seams = build_arrays(mesh, mapping)
    np.savez_compressed(
        directory / "seams.npz", format=SEAM_FORMAT, mesh_sha256=sha, **seams
    )
    np.savez_compressed(
        directory / "wheels.npz",
        format="wheel_shell_groups_v1",
        mesh_sha256=sha,
        node_ids_sha256=hashlib.sha256(node_ids.tobytes()).hexdigest(),
        **wheels,
    )
    features = np.zeros((count, 12), np.float32)
    features[:, :2] = np.column_stack((radius - 1.0, RINGS - radius)) / RINGS
    per_patch = features.reshape(group_count, SECTORS, -1)
    torch.save(
        dict(
            node_features=torch.from_numpy(features),
            core_features=torch.from_numpy(np.c_[per_patch.mean(1), per_patch.max(1)]),
        ),
        directory / "sdf.pt",
    )
    write_json(
        directory / "materials.json",
        dict(
            format="mesh_material_weights_v1",
            mesh_sha256=sha,
            material_weight_scope="per_node_topology_owner",
            material_types=["rim", "tire"],
            region_type_index=[0, 1],
            type_weights={"rim": 1.0, "tire": 1.25},
            unassigned_weight=1.0,
        ),
    )
    return xyz, patch


def generate(directory):
    """Write the analytic dataset to an empty directory and return the manifest path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {directory}")
    (directory / "cases").mkdir()
    xyz, patch = make_mesh(directory)
    cases = ["train_slow", "train_fast", "val_mid", "test_mid"]
    speeds = np.array([24.0, 40.0, 32.0, 36.0])
    times = np.array([0.0, 0.01, 0.023, 0.037, 0.05])
    training = []
    with (
        h5py.File(directory / "training_masks.h5", "w") as masks,
        h5py.File(directory / "element_activity.h5", "w") as activity,
    ):
        for i, (case, speed) in enumerate(zip(cases, speeds, strict=True)):
            t = times[:, None]
            x, y, z = xyz.T
            displacement = np.stack(
                (
                    speed * t + 0.012 * x * t**2,
                    0.04 * speed * t * np.cos(x / 10) + 0.1 * y * t**2,
                    0.2 * z * np.sin(speed * t / 10),
                ),
                axis=-1,
            ).astype(np.float32)
            valid = np.ones(displacement.shape[:2], bool)
            valid[-1, -1] = False
            displacement[~valid] = np.nan
            if i < 2:
                training.append(displacement[valid])
            with h5py.File(directory / "cases" / f"{case}.h5", "w") as handle:
                handle["fields"] = displacement
            masks[f"cases/{case}/valid_displacement_training"] = valid
            group = activity.create_group(f"cases/{case}/shell")
            group["deleted_indptr"] = np.zeros(len(times) + 1, np.int64)
            group["deleted_element_rows"] = np.array([], np.int64)
    train_samples = np.concatenate(training).astype(np.float64)
    panel = np.flatnonzero(
        np.isin(patch, [1, 5, 10, 14, 21]) & (np.arange(len(patch)) % SECTORS < 4)
    )
    np.save(directory / "validation_rows.npy", panel)
    patch_ids = np.unique(patch[panel]).tolist()
    write_json(directory / "patches.json", patch_ids)
    write_json(
        directory / "manifest.json",
        dict(
            format="partitioned_mesh_displacement_v1",
            dataset_dir=".",
            mesh_cache="mesh.npz",
            sdf_cache="sdf.pt",
            mesh_sha256=file_hash(directory / "mesh.npz"),
            sdf_sha256=file_hash(directory / "sdf.pt"),
            material_weights="materials.json",
            material_weights_sha256=file_hash(directory / "materials.json"),
            seam_edges="seams.npz",
            seam_edges_sha256=file_hash(directory / "seams.npz"),
            splits=dict(train=cases[:2], val=cases[2:3], test=cases[3:]),
            case_ids=cases,
            nodes=len(xyz),
            patches=int(patch.max()) + 1,
            output_channels=["U1", "U2", "U3"],
            times=[times.tolist() for _ in cases],
            temporal_reference_time=0.05,
            speed_kmh=speeds.tolist(),
            normalization=dict(
                mean=train_samples.mean(0).tolist(),
                std=train_samples.std(0).tolist(),
                speed_mean=float(speeds[:2].mean()),
                speed_std=float(speeds[:2].std()),
                fit_cases=cases[:2],
            ),
            stage1_validation_panel=dict(
                node_rows="validation_rows.npy",
                node_rows_sha256=file_hash(directory / "validation_rows.npy"),
                patches=patch_ids,
                node_count=len(panel),
            ),
        ),
    )
    return directory / "manifest.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Empty directory for the generated data",
    )
    print(generate(parser.parse_args().output))
