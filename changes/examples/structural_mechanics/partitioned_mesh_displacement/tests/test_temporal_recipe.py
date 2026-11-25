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

"""Exercise temporal handoffs, trainable parameter scopes and physical seam supervision."""

import json

import numpy as np
import pytest
import torch
from displacement import temporal_training
from displacement.config import parser as spatial_parser
from displacement.data import PartitionedDisplacementDataset
from displacement.temporal_objective import predict_trajectory
from displacement.temporal_seams import build_arrays, enable_seams
from make_synthetic_data import generate
from train_stage_1 import train as train_spatial


def test_mixed_shell_flag_keeps_the_same_supported_perimeter(tmp_path):
    """Mark both directions of a true shell perimeter with a second edge type."""
    manifest = generate(tmp_path / "annuli")
    with PartitionedDisplacementDataset(manifest) as data:
        mapping = manifest.parent / "shell_topology.h5"
        mesh = data.mesh
        expected = build_arrays(mesh, mapping)
        a, b = expected["edges"][0]
        rows = np.repeat(np.arange(data.node_count), np.diff(mesh["edge_ptr"]))
        ab = ((rows == a) & (mesh["neighbors"] == b)) | (
            (rows == b) & (mesh["neighbors"] == a)
        )
        assert ab.sum() == 2
        flags = mesh["edge_flags"].copy()
        flags[ab] |= 1
        actual = build_arrays(dict(mesh, edge_flags=flags), mapping)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])


def test_physical_clock_and_archived_case_duration(tmp_path):
    """Keep the reference time independent of the query grid and enforce checkpoint identity."""
    manifest = generate(tmp_path / "annuli")
    settings = json.loads(manifest.read_text())
    case = settings["splits"]["val"][0]
    row = settings["case_ids"].index(case)
    settings["times"][row] = [2 * t for t in settings["times"][row]]
    manifest.write_text(json.dumps(settings))
    with PartitionedDisplacementDataset(manifest, "val") as data:
        assert data.trajectory_time_scale(case) == pytest.approx(0.05)
        from physicsnemo.experimental.models.partitioned_mesh import (
            PartitionAutoencoder,
            TemporalPredictor,
        )

        graph = data.graph_batch(case, [0])
        ae = PartitionAutoencoder(
            node_static_dim=data.node_static_dim,
            patch_feature_dim=data.patch_feature_dim,
            part_count=data.part_count,
            edge_dim=graph["edge_attr"].shape[-1],
            latent_dims=(4, 2, 2),
            hidden_dim=8,
            message_layers=1,
            decoder_hidden_dim=8,
            decoder_residual_layers=1,
        ).eval()
        dynamics = TemporalPredictor(
            8,
            data.part_count,
            patch_feature_dim=data.patch_feature_dim,
            hidden_dim=8,
            spatial_layers=1,
            temporal_layers=1,
            state_dim=2,
        ).eval()
        with torch.no_grad():
            full = predict_trajectory(ae, dynamics, data, case, graph, "cpu")[0]
            original = data.times
            data.times = original[:, :3]
            prefix = predict_trajectory(ae, dynamics, data, case, graph, "cpu")[0]
            torch.testing.assert_close(prefix, full[:3])
            data.times = original * 2.0
            stretched = predict_trajectory(ae, dynamics, data, case, graph, "cpu")[0]
            assert not torch.allclose(stretched[1:], full[1:])
            data.times = original
        with pytest.raises(ValueError, match="temporal_reference_time"):
            data.restore_trajectory_clock({"temporal_reference_time": 1.5})
        data.restore_trajectory_clock({})
        assert data.trajectory_time_scale(case) == pytest.approx(0.10)
    del settings["temporal_reference_time"]
    manifest.write_text(json.dumps(settings))
    with PartitionedDisplacementDataset(manifest, "val") as data:
        # Only the held-out duration changed; training-end statistics stay fixed.
        assert data.trajectory_time_scale(case) == pytest.approx(0.05)
        data.restore_trajectory_clock({"temporal_reference_time": None})
        assert data.trajectory_time_scale(case) == pytest.approx(0.10)
    settings["temporal_reference_time"] = 0
    manifest.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="temporal_reference_time"):
        PartitionedDisplacementDataset(manifest)


def test_shell_seam_units_and_deleted_support(tmp_path):
    """Verify physical edge residuals, rigid translation, deleted shells and missing endpoints."""
    manifest = generate(tmp_path / "annuli")
    with PartitionedDisplacementDataset(manifest) as data:
        seams = enable_seams(data)
        # Isolate a seam whose two incident triangles can be deleted independently.
        eid = int(np.flatnonzero(np.diff(seams.support_ptr) == 2)[0])
        bound = np.array([eid]), np.array([[0, 1]])
        supported_by = seams.support_shells[
            seams.support_ptr[eid] : seams.support_ptr[eid + 1]
        ]
        prediction = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 5.0]])
        batch = dict(
            target=torch.zeros_like(prediction),
            valid=torch.ones_like(prediction, dtype=torch.bool),
        )
        delta = (prediction[1] - prediction[0]) * torch.from_numpy(data.std)
        a, b = seams.edges[eid]
        length = np.linalg.norm(data.mesh["xyz"][a] - data.mesh["xyz"][b])
        expected = torch.stack(
            (delta[0].square(), delta.square().sum(), (delta / length).square().sum())
        )
        for deleted in ([], supported_by[:1]):
            values, count = seams.residual_sums(bound, deleted, prediction + 2.0, batch)
            torch.testing.assert_close(values, expected)
            assert int(count) == 1
        values, count = seams.residual_sums(bound, supported_by, prediction, batch)
        assert values.count_nonzero() == 0 and int(count) == 0
        batch["valid"][1, 2] = False
        batch["target"][1, 2] = float("nan")
        values, count = seams.residual_sums(bound, [], prediction, batch)
        assert values.count_nonzero() == 0 and int(count) == 0


def test_three_stage_checkpoint_handoff_and_gradients(tmp_path, monkeypatch):
    """Check live gradients, parameter updates, seam labels and handoff rejection across stages."""
    manifest = generate(tmp_path / "annuli")
    common = [
        "--manifest",
        str(manifest),
        "--device",
        "cpu",
        "--epochs",
        "1",
        "--steps-per-epoch",
        "1",
        "--patches-per-batch",
        "6",
        "--val-max-frames",
        "1",
        "--val-max-patches",
        "2",
        "--cpu-threads",
        "1",
    ]
    spatial_dir = tmp_path / "stage1"
    stage1 = spatial_parser().parse_args(
        [
            *common,
            "--output",
            str(spatial_dir),
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
        ]
    )
    temporal = [
        "--frames-per-step",
        "2",
        "--wheel-groups",
        str(manifest.parent / "wheels.npz"),
        "--wheel-edge-weight",
        "0.2",
        "--wheel-interface-weight",
        "0.1",
        "--channel-weights",
        "3",
        "1",
        "1",
        "--region-balance-power",
        "1",
        "--material-weight-power",
        "1",
        "--seam-weight",
        "0.003",
        "--seam-gradient-weight",
        "0.001",
        "--seam-batch-fraction",
        "1",
    ]
    records = []
    original_step = temporal_training.epoch_step

    def record_gradients(module, *args, **kwargs):
        """Inspect gradients after the real optimizer update and before validation."""
        summary = original_step(module, *args, **kwargs)
        trainable = {
            name for name, p in module.ae.named_parameters() if p.requires_grad
        }
        gradients = {
            name
            for name, p in module.ae.named_parameters()
            if p.grad is not None and p.grad.count_nonzero()
        }
        dynamics_grad = any(
            p.grad is not None and p.grad.count_nonzero()
            for p in module.dynamics.parameters()
        )
        records.append((trainable, gradients, dynamics_grad))
        return summary

    def load(path):
        """Read a training checkpoint on the CPU using the public safe-loading contract."""
        return torch.load(path, map_location="cpu", weights_only=True)

    monkeypatch.setattr(temporal_training, "epoch_step", record_gradients)
    threads = torch.get_num_threads()
    precision = torch.get_float32_matmul_precision()
    try:
        train_spatial(stage1)
        s1 = spatial_dir / "best.pt"
        stage2_dir = tmp_path / "stage2"
        stage2 = temporal_training.parser(2).parse_args(
            [
                *common,
                *temporal,
                "--output",
                str(stage2_dir),
                "--initial-checkpoint",
                str(s1),
                "--hidden-dim",
                "16",
                "--spatial-layers",
                "2",
                "--temporal-layers",
                "2",
                "--state-dim",
                "4",
                "--dynamic-region-exchange",
            ]
        )
        temporal_training.train(stage2)
        s2 = stage2_dir / "best.pt"
        stage3_dir = tmp_path / "stage3"
        stage3 = temporal_training.parser(3).parse_args(
            [
                *common,
                *temporal,
                "--output",
                str(stage3_dir),
                "--initial-checkpoint",
                str(s2),
            ]
        )
        temporal_training.train(stage3)
        first, second, third = load(s1), load(s2), load(stage3_dir / "best.pt")
        assert records[0] == (set(), set(), True)
        trainable, gradients, dynamics_grad = records[1]
        assert dynamics_grad and gradients and gradients <= trainable
        assert all(name.startswith("decoder_") for name in trainable)
        for key in first["ae_state"]:
            assert torch.equal(first["ae_state"][key], second["ae_state"][key])
            if not key.startswith("decoder_"):
                assert torch.equal(second["ae_state"][key], third["ae_state"][key])
        assert any(
            not torch.equal(second["ae_state"][k], third["ae_state"][k])
            for k in gradients
        )
        assert any(
            not torch.equal(second["dynamics_state"][k], third["dynamics_state"][k])
            for k in second["dynamics_state"]
        )
        for stage, result in ((2, second), (3, third)):
            assert (
                result["stage"] == stage
                and result["parent_checkpoint"]["stage"] == stage - 1
            )
            assert result["temporal_reference_time"] == pytest.approx(0.05)
            assert result["validation"]["task"] == "initial_condition_trajectory"
            assert (
                result["test_set_used"] is False
                and result["validation"]["split"] == "val"
            )
            history = result["ddp_history"][-1]
            assert (
                history["prediction_seam_active_edges"] > 0
                and history["prediction_seam_gradient_mse"] > 0
            )
            assert (
                history["prediction_wheel_edge_mse"] > 0
                and history["prediction_wheel_interface_mse"] > 0
            )
            assert (history["reconstruction"] > 0) == (stage == 3)
        stage3.output = tmp_path / "bad_parent"
        stage3.initial_checkpoint = s1
        with pytest.raises(ValueError, match="Stage 2"):
            temporal_training.train(stage3)
        stage2.channel_weights = [0.0, 1.0, 1.0]
        with pytest.raises(ValueError, match="weights"):
            temporal_training.train(stage2)
        assert not stage3.output.exists()
    finally:
        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision(precision)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="The fused scan requires CUDA"
)
def test_cuda_selective_scan_matches_outputs_and_gradients():
    """Compare the optional installed CUDA kernel with the reference recurrence."""
    pytest.importorskip("mamba_ssm.ops.selective_scan_interface")
    from displacement.scan_backend import configure_scan_backend

    from physicsnemo.experimental.models.partitioned_mesh.selective_scan import (
        SelectiveStateSpaceBlock,
    )

    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(491)
        portable = SelectiveStateSpaceBlock(8, state_dim=4).cuda()
        fused = SelectiveStateSpaceBlock(8, state_dim=4, scan_backend="mamba").cuda()
        configure_scan_backend(fused)
        fused.load_state_dict(portable.state_dict())
        sequence = torch.randn(3, 9, 8, device="cuda", requires_grad=True)
        deltas = torch.tensor(
            [0, 0.01, 0.03, 0.02, 0.04, 0.07, 0.015, 0.01, 0.05], device="cuda"
        )
        expected = portable(sequence, deltas)
        actual = fused(sequence, deltas)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        weights = torch.randn_like(actual)
        left = torch.autograd.grad(actual, (sequence, *fused.parameters()), weights)
        right = torch.autograd.grad(
            expected, (sequence, *portable.parameters()), weights
        )
        for observed, reference in zip(left, right, strict=True):
            torch.testing.assert_close(observed, reference, rtol=5e-4, atol=1e-5)
