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

"""Command line configuration for Stage 1 displacement training."""

import argparse
import json
import math
from pathlib import Path

import torch


def read_validation_patches(path, patch_count):
    """Read a JSON list of distinct patch indices and check the mesh bounds."""
    patches = json.loads(path.read_text(encoding="utf-8-sig"))
    if (
        not isinstance(patches, list)
        or not patches
        or any(
            type(patch) is not int or not 0 <= patch < patch_count for patch in patches
        )
        or len(set(patches)) != len(patches)
    ):
        raise ValueError(
            f"Validation patch file must contain a nonempty JSON list of unique integers in [0, {patch_count})"
        )
    return patches


def validate_training_args(args, device):
    """Validate the optimizer schedule, loss weights, wheel inputs and device precision."""
    if args.epochs < 1 or args.steps_per_epoch < 1 or args.patches_per_batch < 1:
        raise ValueError("Invalid training schedule")
    if (
        not 0 <= args.region_balance_power <= 1
        or not 0 <= args.material_weight_power <= 2
    ):
        raise ValueError(
            "Region balance power must be in [0,1] and material weight power in [0,2]"
        )
    if args.region_attention_heads < 0 or args.region_attention_width < 0:
        raise ValueError("Region attention width and head count must be nonnegative")
    wheel_weights = (
        args.wheel_edge_weight,
        args.wheel_interface_weight,
        args.wheel_selection_weight,
    )
    if (
        any(not math.isfinite(w) or w < 0 for w in wheel_weights)
        or not 0 <= args.wheel_batch_fraction <= 1
    ):
        raise ValueError(
            "Wheel weights must be finite and nonnegative and the batch fraction in [0,1]"
        )
    if not args.wheel_groups and (any(wheel_weights) or args.wheel_batch_fraction):
        raise ValueError(
            "Wheel objectives, validation scoring and sampling require --wheel-groups"
        )
    if args.wheel_groups:
        args.wheel_groups = args.wheel_groups.expanduser().resolve()
    if (
        min(
            args.learning_rate,
            args.min_learning_rate,
            args.weight_decay,
            args.latent_weight,
            args.dynamics_field_weight,
        )
        < 0
        or args.grad_clip <= 0
    ):
        raise ValueError("Invalid optimizer or loss settings")
    if args.precision == "bf16" and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("BF16 training requires a supported CUDA device")
    if args.val_every < 1 or args.eval_patches_per_batch < 0:
        raise ValueError("Invalid validation schedule")


def epoch_learning_rate(args, epoch):
    """Compute linear warmup followed by cosine decay for a zero-based epoch."""
    if epoch < args.warmup_epochs:
        return args.learning_rate * (epoch + 1) / args.warmup_epochs
    fraction = max(0, epoch - args.warmup_epochs) / max(
        1, args.epochs - args.warmup_epochs - 1
    )
    return args.min_learning_rate + 0.5 * (
        args.learning_rate - args.min_learning_rate
    ) * (1 + math.cos(math.pi * fraction))


def parser():
    """Build the Stage 1 command-line parser with the selected preset defaults."""
    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument("--preset", choices=["wheel-supervised"])
    selected, _ = preset_parser.parse_known_args()
    p = argparse.ArgumentParser(description=__doc__, parents=[preset_parser])
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--stage", type=int, choices=(1,), default=1)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--steps-per-epoch", type=int, default=256)
    p.add_argument("--patches-per-batch", type=int, default=1)
    p.add_argument("--max-graph-nodes", type=int, default=500000)
    p.add_argument("--max-graph-edges", type=int, default=5000000)
    p.add_argument(
        "--patch-ids",
        nargs="+",
        type=int,
        help="Explicit limited patch scope for diagnostics only",
    )
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--coarse-dim", type=int, default=64)
    p.add_argument("--fine-dim", type=int, default=32)
    p.add_argument("--decoder-hidden-dim", type=int, default=256)
    p.add_argument("--decoder-residual-layers", type=int, default=4)
    p.add_argument("--message-layers", type=int, default=3)
    p.add_argument("--region-layers", type=int, default=1)
    p.add_argument(
        "--region-attention-heads",
        type=int,
        default=0,
        help="Number of attention heads for region-to-region messages in new models; 0 uses legacy messages",
    )
    p.add_argument(
        "--region-attention-width",
        type=int,
        default=0,
        help="Width of region attention projections; 0 uses the region block width",
    )
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--min-learning-rate", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--latent-weight", type=float, default=0.1)
    p.add_argument("--dynamics-field-weight", type=float, default=0.02)
    p.add_argument(
        "--region-balance-power",
        type=float,
        default=0.0,
        help="Inverse valid region node-count exponent; 0 = node mean, 1 = equal total per nonempty region",
    )
    p.add_argument(
        "--material-weight-power",
        type=float,
        default=0.0,
        help="Exponent of the material deformation factor mapped to each region; 0 disables material weighting",
    )
    p.add_argument(
        "--wheel-groups", type=Path, help="Mesh-matched tyre and rim topology NPZ"
    )
    p.add_argument(
        "--wheel-batch-fraction",
        type=float,
        default=0.0,
        help="Probability of sampling one complete wheel assembly and filling remaining batch slots",
    )
    p.add_argument(
        "--wheel-edge-weight",
        type=float,
        default=0.0,
        help="Training loss weight for wheel interior relative-displacement error",
    )
    p.add_argument(
        "--wheel-interface-weight",
        type=float,
        default=0.0,
        help="Training loss weight for wheel interface relative-displacement error",
    )
    p.add_argument(
        "--wheel-selection-weight",
        type=float,
        default=0.0,
        help="Validation checkpoint score weight for wheel node and attachment errors",
    )
    p.add_argument(
        "--independent-reconstruction-frames",
        action="store_true",
        help="Sample Stage 1 reconstruction frames including the last, independently of the one-step triplet",
    )
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-max-cases", type=int, default=0)
    p.add_argument("--val-max-frames", type=int, default=0)
    p.add_argument("--val-max-patches", type=int, default=0)
    p.add_argument(
        "--val-patch-ids-file",
        type=Path,
        help="JSON array of validation-only patch IDs. Overrides --patch-ids and --val-max-patches during validation.",
    )
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--matmul-precision", choices=("highest", "high"), default="highest")
    p.add_argument("--fused-adamw", action="store_true")
    p.add_argument("--response-cache-mb", type=int, default=0)
    p.add_argument("--validation-cache-mb", type=int, default=0)
    p.add_argument("--open-cases", type=int, default=3)
    p.add_argument("--eval-patches-per-batch", type=int, default=0)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    if selected.preset == "wheel-supervised":
        p.set_defaults(
            device="cuda",
            epochs=500,
            steps_per_epoch=256,
            patches_per_batch=16,
            eval_patches_per_batch=16,
            learning_rate=3e-4,
            region_balance_power=1.0,
            material_weight_power=1.0,
            wheel_batch_fraction=0.4,
            wheel_edge_weight=0.2,
            wheel_interface_weight=0.1,
            wheel_selection_weight=0.1,
            independent_reconstruction_frames=True,
            region_attention_heads=4,
            region_attention_width=64,
            latent_weight=0.1,
            dynamics_field_weight=0.02,
            precision="bf16",
            fused_adamw=True,
            max_graph_nodes=1000000,
            max_graph_edges=8000000,
            response_cache_mb=65536,
            validation_cache_mb=8192,
            open_cases=128,
            val_max_cases=12,
            val_max_frames=5,
        )
    return p
