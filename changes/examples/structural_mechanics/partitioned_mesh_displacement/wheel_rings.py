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

"""Assign a wheel's mesh nodes and shell faces to cylindrical radial bands."""

import argparse
from pathlib import Path

import numpy as np


def _circle_center(points):
    """Fit a circle in the least-squares plane through a reference bead."""
    points = np.asarray(points, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 3
        or not np.isfinite(points).all()
    ):
        raise ValueError("Each bead requires at least three finite 3-D points.")
    mean = points.mean(axis=0)
    centered = points - mean
    basis = np.linalg.svd(centered, full_matrices=False)[2][:2]
    uv = centered @ basis.T
    design = np.column_stack((2 * uv, np.ones(len(uv))))
    solution, _, rank, _ = np.linalg.lstsq(design, (uv**2).sum(axis=1), rcond=None)
    if rank < 3:
        raise ValueError("Bead points must span a circle.")
    return mean + solution[:2] @ basis


def radial_partition(points, cells, bead_a, bead_b, boundaries):
    """Fit the axle using two bead point clouds and label nodes and shell centroids.

    All coordinates and the strictly increasing positive boundaries share one
    length unit. Connectivity uses zero-based node rows. Triangle or quadrilateral
    connectivity is supported; a repeated triangle corner contributes only once.
    Returned bands are one-based, in order of increasing radius.
    """
    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells)
    boundaries = np.asarray(boundaries, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not len(points)
        or not np.isfinite(points).all()
    ):
        raise ValueError("Expected finite wheel coordinates of shape (nodes, 3).")
    if (
        boundaries.ndim != 1
        or not np.isfinite(boundaries).all()
        or np.any(boundaries <= 0)
        or np.any(np.diff(boundaries) <= 0)
    ):
        raise ValueError("Radial boundaries must be positive and increasing.")
    if (
        cells.dtype.kind not in "iu"
        or cells.ndim != 2
        or cells.shape[1] not in (3, 4)
        or np.any(cells < 0)
        or np.any(cells >= len(points))
    ):
        raise ValueError(
            "Shell connectivity must reference three or four wheel vertices."
        )
    a, b = _circle_center(bead_a), _circle_center(bead_b)
    axis = b - a
    length = np.linalg.norm(axis)
    if length <= np.finfo(float).eps * max(np.linalg.norm(a), np.linalg.norm(b), 1):
        raise ValueError("The two fitted bead centers must be distinct.")
    axis /= length
    center = (a + b) * 0.5
    ordered = np.sort(cells, axis=1)
    distinct = np.ones(cells.shape, dtype=bool)
    distinct[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    counts = distinct.sum(axis=1)
    if np.any(counts < 3):
        raise ValueError("A shell face needs at least three distinct vertices.")
    centroids = (points[ordered] * distinct[..., None]).sum(axis=1) / counts[:, None]

    def labels(xyz):
        """Assign one-based bands by perpendicular distance from the axle."""
        radius = np.linalg.norm(np.cross(xyz - center, axis), axis=1)
        return np.searchsorted(boundaries, radius, side="right").astype(np.int32) + 1

    return dict(
        center=center,
        axis=axis,
        boundaries=boundaries,
        node_ring=labels(points),
        cell_ring=labels(centroids),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="NPZ containing points, cells, bead_a and bead_b.",
    )
    parser.add_argument(
        "--boundaries",
        type=float,
        nargs="+",
        required=True,
        help="Internal radial cuts.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        result = radial_partition(
            data["points"],
            data["cells"],
            data["bead_a"],
            data["bead_b"],
            args.boundaries,
        )
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **result)
