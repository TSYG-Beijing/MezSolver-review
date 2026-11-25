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

"""Command-line controls for initial-condition trajectory training."""

import argparse
import math
from pathlib import Path

import torch


def parser(stage):
    """Build the argument parser for the frozen or jointly trained trajectory model."""
    if stage not in (2, 3):
        raise ValueError("Choose Stage 2 or Stage 3")
    cli = argparse.ArgumentParser(description=__doc__)
    cli.set_defaults(stage=stage)
    cli.add_argument("--manifest", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    source = cli.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="Validation-best checkpoint from the preceding stage",
    )
    source.add_argument("--resume", type=Path, help="Resume a last.pt from this stage")
    if stage == 2:
        cli.add_argument("--hidden-dim", type=int, default=128)
        cli.add_argument("--spatial-layers", type=int, default=2)
        cli.add_argument("--temporal-layers", type=int, default=2)
        cli.add_argument("--state-dim", type=int, default=16)
        cli.add_argument(
            "--scan-backend", choices=("parallel", "mamba"), default="parallel"
        )
        cli.add_argument("--region-attention-heads", type=int, default=0)
        cli.add_argument("--region-attention-width", type=int, default=0)
        cli.add_argument(
            "--dynamic-region-exchange",
            action="store_true",
            help="Exchange neighboring region states once, between the first two SSM layers",
        )
    else:
        cli.add_argument("--unfreeze", choices=("decoder", "all"), default="decoder")
        cli.add_argument("--reconstruction-weight", type=float, default=0.1)
    cli.set_defaults(
        **({"unfreeze": "none", "reconstruction_weight": 0.0} if stage == 2 else {})
    )
    cli.add_argument(
        "--device", default="cpu", help="cpu or cuda; torchrun selects the local GPU"
    )
    cli.add_argument("--epochs", type=int, default=900 if stage == 2 else 200)
    cli.add_argument("--steps-per-epoch", type=int, default=256)
    cli.add_argument("--patches-per-batch", type=int, default=16)
    cli.add_argument("--patch-ids", nargs="+", type=int)
    cli.add_argument(
        "--frames-per-step",
        type=int,
        default=4,
        help="Number of future frames supervised per draw, or zero for all",
    )
    cli.add_argument(
        "--learning-rate", type=float, default=1e-4 if stage == 2 else 5e-5
    )
    cli.add_argument("--min-learning-rate", type=float, default=1e-6)
    cli.add_argument("--warmup-epochs", type=int, default=0)
    cli.add_argument("--weight-decay", type=float, default=1e-5)
    cli.add_argument("--grad-clip", type=float, default=1.0)
    cli.add_argument("--latent-weight", type=float, default=0.1)
    cli.add_argument(
        "--channel-weights",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        metavar=("U1", "U2", "U3"),
    )
    cli.add_argument("--region-balance-power", type=float, default=0.0)
    cli.add_argument("--material-weight-power", type=float, default=0.0)
    cli.add_argument("--wheel-groups", type=Path)
    cli.add_argument("--wheel-batch-fraction", type=float, default=0.0)
    cli.add_argument("--wheel-edge-weight", type=float, default=0.0)
    cli.add_argument("--wheel-interface-weight", type=float, default=0.0)
    cli.add_argument("--wheel-selection-weight", type=float, default=0.0)
    cli.add_argument(
        "--seam-edges",
        type=Path,
        help="Shell-supported seam catalog; defaults to the manifest asset",
    )
    cli.add_argument("--seam-weight", type=float, default=0.0)
    cli.add_argument("--seam-gradient-weight", type=float, default=0.0)
    cli.add_argument("--seam-batch-fraction", type=float, default=0.0)
    cli.add_argument("--seam-scale-mm", type=float, default=1.0)
    cli.add_argument("--val-every", type=int, default=1)
    cli.add_argument("--val-patch-ids-file", type=Path)
    cli.add_argument("--val-max-cases", type=int, default=0)
    cli.add_argument("--val-max-frames", type=int, default=0)
    cli.add_argument("--val-max-patches", type=int, default=0)
    cli.add_argument("--eval-patches-per-batch", type=int, default=0)
    cli.add_argument("--max-graph-nodes", type=int, default=500000)
    cli.add_argument("--max-graph-edges", type=int, default=5000000)
    cli.add_argument("--response-cache-mb", type=int, default=0)
    cli.add_argument("--validation-cache-mb", type=int, default=0)
    cli.add_argument("--open-cases", type=int, default=3)
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--cpu-threads", type=int, default=4)
    cli.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    cli.add_argument(
        "--matmul-precision", choices=("highest", "high"), default="highest"
    )
    cli.add_argument("--fused-adamw", action="store_true")
    return cli


def validate(args, device):
    """Validate objective weights, independent sampling fractions and resource limits."""
    if args.wheel_groups is not None:
        args.wheel_groups = args.wheel_groups.expanduser().resolve()
    if args.seam_edges is not None:
        args.seam_edges = args.seam_edges.expanduser().resolve()
    if (
        min(
            args.epochs,
            args.steps_per_epoch,
            args.patches_per_batch,
            args.val_every,
            args.max_graph_nodes,
            args.max_graph_edges,
            args.open_cases,
            args.cpu_threads,
        )
        < 1
    ):
        raise ValueError("Epoch, batch and resource counts must be positive")
    if (
        min(
            args.frames_per_step,
            args.val_max_cases,
            args.val_max_frames,
            args.val_max_patches,
            args.eval_patches_per_batch,
            args.validation_cache_mb,
            args.response_cache_mb,
            args.warmup_epochs,
        )
        < 0
    ):
        raise ValueError(
            "Frame counts, validation limits and cache sizes cannot be negative"
        )
    weights = (
        args.latent_weight,
        args.reconstruction_weight,
        args.wheel_selection_weight,
        args.wheel_edge_weight,
        args.wheel_interface_weight,
        args.seam_weight,
        args.seam_gradient_weight,
        args.min_learning_rate,
        args.weight_decay,
    )
    if any(not math.isfinite(x) or x < 0 for x in weights):
        raise ValueError(
            "Loss weights and optimizer coefficients must be finite and nonnegative"
        )
    if any(
        not math.isfinite(x) or x <= 0
        for x in (
            *args.channel_weights,
            args.grad_clip,
            args.learning_rate,
            args.seam_scale_mm,
        )
    ):
        raise ValueError(
            "Component weights, gradient clipping, learning rate and seam scale must be positive"
        )
    if (
        not 0 <= args.region_balance_power <= 1
        or not 0 <= args.material_weight_power <= 2
    ):
        raise ValueError(
            "Region balance must be in [0,1] and material weighting in [0,2]"
        )
    if (
        not 0 <= args.wheel_batch_fraction <= 1
        or not 0 <= args.seam_batch_fraction <= 1
        or args.wheel_batch_fraction + args.seam_batch_fraction > 1
    ):
        raise ValueError(
            "Wheel and seam sampling fractions must be nonnegative and sum to at most one"
        )
    if not args.wheel_groups and any(
        (
            args.wheel_batch_fraction,
            args.wheel_edge_weight,
            args.wheel_interface_weight,
            args.wheel_selection_weight,
        )
    ):
        raise ValueError("Wheel supervision or reporting requires --wheel-groups")
    if args.precision == "bf16" and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("BF16 training requires a supported CUDA device")
