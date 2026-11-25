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

"""Wheel validation on active tyre shell nodes and physical tyre-to-rim edges."""

import numpy as np
import torch


def _rotation_from_reference(x, deformed):
    """Fit the proper rotation between corresponding reference and deformed points."""
    a = x - x.mean(0)
    b = deformed - deformed.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    if s[1] <= np.finfo(float).eps * s[0]:
        return None
    u[:, -1] *= np.sign(np.linalg.det(u @ vt))
    return u @ vt


def wheel_rotation_error(x, predicted, target):
    """Geodesic angle between fitted proper rotations from the same corresponding reference nodes."""
    if len(x) < 3:
        return None
    rp = _rotation_from_reference(x, x + predicted)
    rt = _rotation_from_reference(x, x + target)
    if rp is None or rt is None:
        return None
    r = rp.T @ rt
    sine = np.linalg.norm([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / 2
    return float(np.degrees(np.arctan2(sine, (np.trace(r) - 1) / 2)))


class WheelFields:
    """Gather predictions from existing general batches and the missing wheel patches only."""

    def __init__(self, n_frames, n_nodes):
        """Allocate frame-by-node buffers for three-component wheel displacements."""
        self.predicted = np.empty((n_frames, n_nodes, 3), np.float32)
        self.target = np.empty_like(self.predicted)
        self.valid = np.zeros((n_frames, n_nodes), bool)
        self.seen = np.zeros(n_nodes, bool)

    def select(self, positions, device):
        """Map unseen wheel outputs to unique buffer slots and decoder rows."""
        model_rows = np.flatnonzero(positions >= 0)
        slots = positions[model_rows]
        if self.seen[slots].any() or len(np.unique(slots)) != len(slots):
            raise ValueError("Wheel validation encountered duplicate output nodes")
        self.seen[slots] = True
        return slots, torch.as_tensor(model_rows, device=device)

    def add(self, j, slots, model_rows, pred, target, valid):
        """Store a frame's selected wheel predictions, targets and validity flags."""
        if len(slots):
            self.predicted[j, slots] = (
                pred.index_select(0, model_rows).float().cpu().numpy()
            )
            self.target[j, slots] = (
                target.index_select(0, model_rows).float().cpu().numpy()
            )
            self.valid[j, slots] = (
                valid.index_select(0, model_rows).all(-1).cpu().numpy()
            )


class WheelValidation:
    """Accumulate displacement, attachment and rotation errors for each wheel."""

    def __init__(self, data):
        """Build lookups from mesh rows to wheel buffer slots and reference coordinates."""
        self.data = data
        self.std = np.asarray(data.std, np.float64)
        self.mean = np.asarray(data.mean, np.float64)
        required = []
        for group in data.wheels.values():
            required.extend(
                (
                    group.tire_rows,
                    group.tire_triangles.ravel(),
                    group.edges[group.attachment].ravel(),
                )
            )
        self.nodes = np.unique(np.concatenate(required))
        self.patches = np.unique(data.mesh["node_patch"][self.nodes])
        self.lookup = np.full(data.node_count, -1, np.int32)
        self.lookup[self.nodes] = np.arange(len(self.nodes), dtype=np.int32)
        self.wall = data.mesh["wall"][self.nodes].astype(bool)
        self.wheels = {}
        for name, g in data.wheels.items():
            self.wheels[name] = (
                g,
                self.lookup[g.tire_rows],
                self.lookup[g.tire_triangles],
                self.lookup[g.edges[g.attachment]],
                data.mesh["xyz"][g.tire_rows].astype(np.float64),
            )
        self.records = []

    def finish_case(self, case, frames, fields):
        """Record per-wheel metrics for a case using valid nodes and active shells."""
        if not fields.seen.all():
            raise ValueError("Incomplete wheel validation inference")
        wheels = {name: [] for name in self.wheels}
        for j, frame in enumerate(frames):
            valid = fields.valid[j] & ~self.wall
            predicted = fields.predicted[j].astype(np.float64) * self.std + self.mean
            target = fields.target[j].astype(np.float64) * self.std + self.mean
            error = predicted - target
            if not np.isfinite(error[valid]).all():
                raise ValueError(
                    f"Invalid supervised wheel validation prediction: {case}, frame {frame}"
                )
            deleted = self.data.deleted_wheel_shells(case, int(frame))
            for name, (g, nodes, tri, edges, xyz) in self.wheels.items():
                active_faces = valid[tri].all(1) & ~np.isin(g.tire_elements, deleted)
                active_nodes = valid[nodes] & np.isin(nodes, tri[active_faces])
                active_edges = g.active_edges(deleted, g.attachment) & valid[edges].all(
                    1
                )
                a, b = edges[active_edges].T
                keep = nodes[active_nodes]
                wheels[name].append(
                    dict(
                        frame=int(frame),
                        nodes=len(keep),
                        interface_edges=len(a),
                        node_squared_error_mm2=np.sum(
                            error[keep] ** 2, axis=0
                        ).tolist(),
                        interface_squared_error_mm2=np.sum(
                            (error[a] - error[b]) ** 2, axis=0
                        ).tolist(),
                        rotation_error_deg=wheel_rotation_error(
                            xyz[active_nodes], predicted[keep], target[keep]
                        ),
                    )
                )
        self.records.append(dict(case=case, frames=frames.tolist(), wheels=wheels))

    def report(self):
        """Summarize per-wheel observations and compute equally weighted wheel scores."""
        per_wheel = {}
        for name in self.wheels:
            records = [case["wheels"][name] for case in self.records]
            summary = {}
            for caption, count_key, error_key in (
                ("node", "nodes", "node_squared_error_mm2"),
                ("interface", "interface_edges", "interface_squared_error_mm2"),
            ):
                case_mses = []
                count = 0
                for rows in records:
                    n = sum(row[count_key] for row in rows)
                    if n:
                        case_mses.append(
                            np.sum([row[error_key] for row in rows], axis=0) / n
                        )
                        count += n
                mse = np.mean(case_mses, axis=0) if case_mses else None
                summary[caption + "_valid_samples"] = count
                summary[caption + "_valid_cases"] = len(case_mses)
                summary[caption + "_vector_rmse_mm"] = (
                    float(np.sqrt(mse.sum())) if mse is not None else None
                )
                summary[caption + "_normalized_mse"] = (
                    float(np.mean(mse / self.std**2)) if mse is not None else None
                )
            angles = [
                [
                    r["rotation_error_deg"]
                    for r in rows
                    if r["rotation_error_deg"] is not None
                ]
                for rows in records
            ]
            case_angles = [np.mean(a) for a in angles if a]
            summary["mean_rotation_error_deg"] = (
                float(np.mean(case_angles)) if case_angles else None
            )
            summary["valid_rotation_frames"] = sum(map(len, angles))
            per_wheel[name] = summary
        result = dict(
            wheels=per_wheel,
            records=self.records,
            scope=dict(
                split=self.data.split,
                cases=[c["case"] for c in self.records],
                frames_by_case={c["case"]: c["frames"] for c in self.records},
                wheel_support_patches=self.patches.tolist(),
                total_support_rows=len(self.nodes),
            ),
            rotation_definition="Unsigned 3D angle separating proper reference-to-model and reference-to-truth Kabsch rotations of the same active tyre nodes, in degrees.",
            reduction="Pool valid node/edge samples over frames within each case, then weight cases and wheels equally. Average the three channel MSEs after dividing by training channel variances.",
        )
        for label in ("node", "interface"):
            values = [w[label + "_normalized_mse"] for w in per_wheel.values()]
            result["equal_wheel_" + label + "_normalized_mse"] = (
                float(np.mean(values)) if all(v is not None for v in values) else None
            )
        for key, compact_key in (
            ("node_vector_rmse_mm", "mean_wheel_node_vector_rmse_mm"),
            ("interface_vector_rmse_mm", "mean_wheel_interface_vector_rmse_mm"),
            ("mean_rotation_error_deg", "mean_wheel_rotation_error_deg"),
        ):
            values = [w[key] for w in per_wheel.values()]
            result[compact_key] = (
                float(np.mean(values)) if all(v is not None for v in values) else None
            )
        return result
