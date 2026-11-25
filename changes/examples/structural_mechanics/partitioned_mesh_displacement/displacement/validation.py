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

"""Score reconstruction on fixed nodes and optionally score full wheel assemblies."""

import numpy as np
import torch

from .data import file_hash
from .objective import precision_context, tensor_batch
from .wheel_validation import WheelFields, WheelValidation


@torch.no_grad()
def evaluate(
    ae,
    data,
    device,
    *,
    max_cases=0,
    max_frames=0,
    max_patches=0,
    patch_ids=None,
    patches_per_batch=1,
    precision="fp32",
    wheel_selection_weight=0.0,
    dynamics=None,
):
    """Score reconstruction or an initial-condition trajectory for checkpoint selection."""
    if not np.isfinite(wheel_selection_weight) or wheel_selection_weight < 0:
        raise ValueError("Wheel selection weight must be finite and nonnegative")
    if wheel_selection_weight and (
        data.split != "val" or not getattr(data, "wheels", None)
    ):
        raise ValueError(
            "Wheel-weighted checkpoint selection requires validation data with wheel groups"
        )
    ae.eval()
    if dynamics is not None:
        from .temporal_objective import predict_trajectory

        dynamics.eval()
    data.cfg.halo_hops = ae.message_layers
    data.cfg.region_layers = ae.region_layers + (
        dynamics.spatial_layers if dynamics is not None else 0
    )
    cases = data.case_ids[: max_cases or None]
    patches = (
        np.asarray(patch_ids, np.int64)
        if patch_ids is not None
        else np.arange(data.patch_count)[: max_patches or None]
    )
    if len(np.unique(patches)) != len(patches) or not len(patches):
        raise ValueError("Evaluation patches must be nonempty and unique")
    scoring_rows = None
    panel = data.manifest.get("stage1_validation_panel", {})
    if (
        data.split == "val"
        and dynamics is None
        and patch_ids is not None
        and np.array_equal(np.sort(patches), panel.get("patches", []))
    ):
        rows_file = data.path.parent / panel["node_rows"]
        if file_hash(rows_file) != panel["node_rows_sha256"]:
            raise ValueError("Validation scoring nodes differ from the fixed panel")
        row_ids = np.load(rows_file, allow_pickle=False)
        if (
            row_ids.ndim != 1
            or not len(row_ids)
            or len(row_ids) != panel["node_count"]
            or row_ids.dtype.kind not in "iu"
            or row_ids.min() < 0
            or row_ids.max() >= data.node_count
            or np.any(row_ids[1:] <= row_ids[:-1])
        ):
            raise ValueError("Invalid fixed validation node rows")
        np.testing.assert_array_equal(
            np.unique(data.mesh["node_patch"][row_ids]), np.sort(patches)
        )
        scoring_rows = np.zeros(data.node_count, dtype=bool)
        scoring_rows[row_ids] = True
    wheel = WheelValidation(data) if getattr(data, "wheels", None) else None
    groups = [
        (True, patches[start : start + patches_per_batch])
        for start in range(0, len(patches), patches_per_batch)
    ]
    if wheel is not None:
        missing = np.setdiff1d(wheel.patches, patches)
        groups.extend(
            (False, missing[start : start + patches_per_batch])
            for start in range(0, len(missing), patches_per_batch)
        )
    rows = []
    for case in cases:
        frames = np.arange(1, len(data.times[data.case_index[case]]))
        if max_frames and len(frames) > max_frames:
            frames = frames[
                np.unique(np.linspace(0, len(frames) - 1, max_frames).astype(int))
            ]
        error, energy, counts = np.zeros(3), np.zeros(3), np.zeros(3, np.int64)
        fields = (
            WheelFields(len(frames), len(wheel.nodes)) if wheel is not None else None
        )
        for general, selected in groups:
            graph = data.graph_batch(case, selected)
            score_mask = (
                torch.as_tensor(scoring_rows[graph["output_nodes"]], device=device)[
                    :, None
                ]
                if general and scoring_rows is not None
                else None
            )
            if wheel is not None:
                slots, model_rows = fields.select(
                    wheel.lookup[graph["output_nodes"]], device
                )
            if dynamics is not None:
                with precision_context(device, precision):
                    predictions, _, _ = predict_trajectory(
                        ae, dynamics, data, case, graph, device, frames
                    )
            for j, frame in enumerate(frames):
                target = tensor_batch(data.attach(case, int(frame), graph), device)
                with precision_context(device, precision):
                    pred = (
                        ae(target)["prediction"] if dynamics is None else predictions[j]
                    )
                valid = target["valid"]
                if wheel is not None:
                    fields.add(j, slots, model_rows, pred, target["target"], valid)
                if not general:
                    continue
                if score_mask is not None:
                    valid = valid & score_mask
                scale = torch.tensor(data.std, device=pred.device, dtype=torch.float64)
                physical_error = (pred - target["target"]).double() * scale.double()
                truth = target["target"].double() * scale + torch.tensor(
                    data.mean, device=pred.device, dtype=torch.float64
                )
                error += (
                    torch.where(valid, physical_error.square(), 0.0)
                    .sum(0)
                    .cpu()
                    .numpy()
                )
                energy += torch.where(valid, truth.square(), 0.0).sum(0).cpu().numpy()
                counts += valid.sum(0).cpu().numpy()
        if np.any(counts == 0):
            raise ValueError(
                f"{case}: evaluation scope contains no valid displacement labels"
            )
        if wheel is not None:
            wheel.finish_case(case, frames, fields)
        relative = [
            float(np.sqrt(e / d)) if d > 0 else None for e, d in zip(error, energy)
        ]
        rows.append(
            dict(
                case=case,
                frames=frames.tolist(),
                valid_node_frames=counts.tolist(),
                squared_error_mm2=error.tolist(),
                truth_energy_mm2=energy.tolist(),
                rmse_mm=np.sqrt(error / np.maximum(counts, 1)).tolist(),
                relative_l2=relative,
                normalized_mse=(error / np.maximum(counts, 1) / data.std**2).tolist(),
            )
        )
    selected_nodes = int(np.diff(data.mesh["core_ptr"])[patches].sum())
    if scoring_rows is not None:
        selected_nodes = int(scoring_rows.sum())
    all_times = all(
        len(row["frames"]) == len(data.times[data.case_index[row["case"]]]) - 1
        for row in rows
    )
    mse = np.mean([r["normalized_mse"] for r in rows], axis=0)
    report = dict(
        task="reconstruction" if dynamics is None else "initial_condition_trajectory",
        split=data.split,
        channels=["U1", "U2", "U3"],
        units="mm",
        records=rows,
        selection_score=float(mse.mean()),
        equal_case_normalized_mse=mse.tolist(),
        scope=dict(
            cases=cases,
            split_cases=len(data.case_ids),
            patches=patches.tolist(),
            total_patches=data.patch_count,
            nodes=selected_nodes,
            total_nodes=data.node_count,
            all_cases=len(cases) == len(data.case_ids),
            all_nodes=selected_nodes == data.node_count,
            all_positive_frames=all_times,
            full_split=len(cases) == len(data.case_ids)
            and selected_nodes == data.node_count
            and all_times,
        ),
        test_set_used=data.split == "test",
        manifest_signature=data.signature,
    )
    if wheel is not None:
        metrics = wheel.report()
        report.update(
            wheel_validation=metrics,
            global_selection_score=report["selection_score"],
            wheel_selection_weight=float(wheel_selection_weight),
        )
        if wheel_selection_weight:
            node = metrics["equal_wheel_node_normalized_mse"]
            edge = metrics["equal_wheel_interface_normalized_mse"]
            if node is None or edge is None:
                raise ValueError(
                    "Each wheel must have valid node and attachment-edge labels for weighted selection"
                )
            report["selection_score"] += wheel_selection_weight * (node + edge)
    return report
