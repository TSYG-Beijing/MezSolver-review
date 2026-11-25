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

"""Exercise Stage 1 training, checkpointing and causal evaluation on synthetic data."""

import json
from types import SimpleNamespace

import numpy as np
import torch
from displacement.config import parser
from displacement.data import PartitionedDisplacementDataset
from evaluate_components import evaluate
from make_synthetic_data import generate
from train_stage_1 import train

from physicsnemo.experimental.models.partitioned_mesh import OneStepPredictor


def test_stage1_optimizer_checkpoint_and_held_out_forecast(tmp_path, monkeypatch):
    """Run two optimizer steps and check the held-out forecast reads its target last."""
    manifest = generate(tmp_path / "data")
    out = tmp_path / "stage1"
    args = parser().parse_args(
        [
            "--manifest",
            str(manifest),
            "--output",
            str(out),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--steps-per-epoch",
            "2",
            "--patches-per-batch",
            "6",
            "--hidden-dim",
            "16",
            "--coarse-dim",
            "8",
            "--fine-dim",
            "4",
            "--decoder-hidden-dim",
            "16",
            "--decoder-residual-layers",
            "1",
            "--message-layers",
            "1",
            "--region-attention-heads",
            "2",
            "--region-attention-width",
            "8",
            "--wheel-groups",
            str(manifest.parent / "wheels.npz"),
            "--wheel-batch-fraction",
            "1",
            "--wheel-edge-weight",
            "0.2",
            "--wheel-interface-weight",
            "0.1",
            "--wheel-selection-weight",
            "0.1",
            "--region-balance-power",
            "1",
            "--material-weight-power",
            "1",
            "--independent-reconstruction-frames",
            "--cpu-threads",
            "1",
            "--val-patch-ids-file",
            str(manifest.parent / "patches.json"),
        ]
    )
    original_threads = torch.get_num_threads()
    try:
        train(args)
        assert json.loads((out / "completion.json").read_text())["epochs"] == 1
        checkpoint = torch.load(out / "last.pt", weights_only=True, map_location="cpu")
        assert checkpoint["ddp_history"][-1]["global_optimizer_steps"] == 2
        assert checkpoint["ae_state"] and checkpoint["one_step_state"]
        assert checkpoint["test_set_used"] is False
        assert checkpoint["validation"]["scope"]["nodes"] == 20
        events = []
        original_frame = PartitionedDisplacementDataset.frame
        original_forecast = OneStepPredictor.forward_known

        def record_frame(self, case, frame, nodes):
            """Record the frame index each time the evaluator reads displacements."""
            events.append(frame)
            return original_frame(self, case, frame, nodes)

        def record_forecast(self, *args, **kwargs):
            """Mark when the predictor runs relative to recorded frame reads."""
            events.append("forecast")
            return original_forecast(self, *args, **kwargs)

        monkeypatch.setattr(PartitionedDisplacementDataset, "frame", record_frame)
        monkeypatch.setattr(OneStepPredictor, "forward_known", record_forecast)
        result = evaluate(
            SimpleNamespace(
                device="cpu",
                checkpoint=out / "best.pt",
                manifest=manifest,
                split="val",
                full_mesh=False,
                frames=[2, 3, 4],
                cases=None,
                patches_per_batch=6,
                max_graph_nodes=10000,
                max_graph_edges=100000,
                response_cache_mb=0,
            )
        )
        assert result["pooled"]["valid_node_frames"] == [60, 60, 60]
        assert events == [0, 1, "forecast", 2, 1, 2, "forecast", 3, 2, 3, "forecast", 4]
        for metric in ("reconstruction", "one_step"):
            assert np.isfinite(result["pooled"][metric]["rmse_mm"]).all()
    finally:
        torch.set_num_threads(original_threads)
