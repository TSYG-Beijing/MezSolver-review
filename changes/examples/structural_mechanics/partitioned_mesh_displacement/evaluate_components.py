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

"""Score reconstruction and causal displacement forecasts on identical targets."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from displacement.data import PartitionedDisplacementDataset
from displacement.objective import tensor_batch

from physicsnemo.experimental.models.partitioned_mesh import (
    OneStepPredictor,
    PartitionAutoencoder,
    TemporalPredictor,
)


def summarize(sums, forecast_name="one_step"):
    """Use valid node-frame pairs as the denominator for each displacement channel."""
    recon, forecast, energy, count = sums
    if np.any(count <= 0):
        raise ValueError(
            "The selected nodes and frames need valid labels in each channel."
        )
    result = {"valid_node_frames": count.astype(int).tolist()}
    for name, error in (("reconstruction", recon), (forecast_name, forecast)):
        result[name] = {
            "rmse_mm": np.sqrt(error / count).tolist(),
            "relative_l2": [
                float(np.sqrt(e / t)) if t > 0 else None for e, t in zip(error, energy)
            ],
        }
    return result


@torch.inference_mode()
def evaluate(args):
    """Encode only earlier observations for a forecast; score on physical targets."""
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    stage = checkpoint.get("stage")
    if stage not in (1, 2, 3):
        raise ValueError("Expected a Stage 1, 2 or 3 checkpoint.")
    device = torch.device(args.device)
    ae = PartitionAutoencoder(**checkpoint["ae_kwargs"]).to(device).eval()
    if stage == 1:
        predictor = OneStepPredictor(**checkpoint["one_step_kwargs"])
        predictor.load_state_dict(checkpoint["one_step_state"])
        forecast_name = "one_step"
        region_layers = ae.region_layers
    else:
        from displacement.scan_backend import configure_scan_backend
        from displacement.temporal_objective import predict_trajectory

        predictor = TemporalPredictor(**checkpoint["dynamics_kwargs"])
        predictor.load_state_dict(checkpoint["dynamics_state"])
        configure_scan_backend(predictor)
        forecast_name = "trajectory"
        region_layers = ae.region_layers + predictor.spatial_layers
    predictor.to(device).eval()
    ae.load_state_dict(checkpoint["ae_state"])
    with PartitionedDisplacementDataset(
        args.manifest,
        args.split,
        halo_hops=ae.message_layers,
        region_layers=region_layers,
        max_graph_nodes=args.max_graph_nodes,
        max_graph_edges=args.max_graph_edges,
        response_cache_mb=args.response_cache_mb,
    ) as data:
        if data.signature != checkpoint["manifest_signature"]:
            raise ValueError("Checkpoint and manifest signatures differ.")
        if stage in (2, 3):
            data.restore_trajectory_clock(checkpoint)
        panel = (
            data.manifest.get("stage1_validation_panel")
            if stage == 1 and args.split == "val" and not args.full_mesh
            else None
        )
        keep_rows = np.ones(data.node_count, dtype=bool)
        if panel is not None:
            from displacement.data import file_hash

            path = args.manifest.resolve().parent / panel["node_rows"]
            if file_hash(path) != panel["node_rows_sha256"]:
                raise ValueError("Validation node identity mismatch.")
            rows = np.load(path, allow_pickle=False)
            keep_rows[:] = False
            keep_rows[rows] = True
            if len(rows) != panel["node_count"] or keep_rows.sum() != len(rows):
                raise ValueError(
                    "Fixed validation panel contains duplicate or missing rows."
                )
        patches = np.unique(data.mesh["node_patch"][keep_rows])
        mean = torch.tensor(data.mean, device=device, dtype=torch.float64)
        std = torch.tensor(data.std, device=device, dtype=torch.float64)
        cases = args.cases or data.case_ids
        if len(set(cases)) != len(cases) or not set(cases) <= set(data.case_ids):
            raise ValueError(
                "Each requested case must occur once in the selected split."
            )
        total = np.zeros((4, 3), dtype=np.float64)
        per_case = {}
        first_target = 2 if stage == 1 else 1
        for case in cases:
            times = data.times[data.case_index[case]]
            frames = (
                args.frames
                if args.frames is not None
                else list(range(first_target, len(times)))
            )
            if (
                not frames
                or min(frames) < first_target
                or max(frames) >= len(times)
                or len(set(frames)) != len(frames)
            ):
                raise ValueError(
                    f"Case {case} needs unique target frames in [{first_target}, {len(times) - 1}]."
                )
            dt_ref = float(np.median(np.diff(times)))
            sums = torch.zeros((4, 3), dtype=torch.float64, device=device)
            covered = 0
            for start in range(0, len(patches), args.patches_per_batch):
                graph = data.graph_batch(
                    case, patches[start : start + args.patches_per_batch]
                )
                mask = torch.as_tensor(keep_rows[graph["output_nodes"]], device=device)[
                    :, None
                ]
                covered += int(keep_rows[graph["output_nodes"]].sum())
                if stage != 1:
                    trajectory, _, _ = predict_trajectory(
                        ae, predictor, data, case, graph, device, frames=frames
                    )
                for index, t in enumerate(frames):
                    if stage == 1:
                        previous = tensor_batch(data.attach(case, t - 2, graph), device)
                        current = tensor_batch(data.attach(case, t - 1, graph), device)
                        latent = predictor.forward_known(
                            ae.encode(previous),
                            ae.encode(current),
                            current["patch_features"],
                            current["patch_part_index"],
                            float(times[t - 1] - times[t - 2]),
                            float(times[t] - times[t - 1]),
                            dt_ref,
                            current_time=float(times[t - 1]),
                        )
                        prediction = ae.decode(current, latent)
                    else:
                        prediction = trajectory[index]
                    target = tensor_batch(data.attach(case, t, graph), device)
                    reconstruction = ae(target)["prediction"]
                    valid = mask & target["valid"]
                    truth = target["target"].double()
                    for k, value in enumerate(
                        (
                            (reconstruction.double() - truth) * std,
                            (prediction.double() - truth) * std,
                            truth * std + mean,
                        )
                    ):
                        sums[k] += torch.where(valid, value.square(), 0.0).sum(0)
                    sums[3] += valid.sum(0)
            if covered != int(keep_rows.sum()):
                raise RuntimeError(
                    "Graph batches failed to cover the requested scoring nodes."
                )
            stats = sums.cpu().numpy()
            per_case[case] = {"frames": frames, **summarize(stats, forecast_name)}
            total += stats
        return dict(
            stage=stage,
            epoch=checkpoint["epoch"],
            split=args.split,
            channels=["U1", "U2", "U3"],
            scored_nodes=int(keep_rows.sum()),
            scope="fixed_nodes" if panel else "all_nodes",
            pooled=summarize(total, forecast_name),
            cases=per_case,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--frames", type=int, nargs="+")
    parser.add_argument("--full-mesh", action="store_true")
    parser.add_argument("--patches-per-batch", type=int, default=16)
    parser.add_argument("--response-cache-mb", type=int, default=512)
    parser.add_argument("--max-graph-nodes", type=int, default=1000000)
    parser.add_argument("--max-graph-edges", type=int, default=8000000)
    args = parser.parse_args()
    if args.patches_per_batch < 1:
        parser.error("--patches-per-batch must be positive")
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
