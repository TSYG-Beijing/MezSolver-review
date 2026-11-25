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

"""Train spatial reconstruction and a causal two-observation latent forecast."""

from contextlib import nullcontext

import numpy as np
import torch

from physicsnemo.experimental.models.partitioned_mesh import (
    OneStepPredictor,
    PartitionAutoencoder,
)

from .losses import NoValidTarget, displacement_mses, wheel_edge_mses

VERSION = "partitioned_mesh_displacement_training_v1"


def tensor_batch(batch, device):
    """Move NumPy arrays in a graph batch to the requested PyTorch device."""
    return {
        k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v
        for k, v in batch.items()
    }


def masked_latent_mse(prediction, target, batch):
    """Average latent prediction error over selected regions with observed core nodes."""
    counts = prediction.new_zeros(len(prediction))
    counts.index_add_(
        0,
        batch["core_patch"],
        batch["observed"][batch["core_index"]].any(-1).to(counts.dtype),
    )
    selected = batch["selected"]
    keep = counts[selected] > 0
    error = (prediction[selected] - target[selected].detach()).square().mean(-1)
    return torch.where(keep, error, 0.0).sum() / keep.sum().clamp_min(1)


def precision_context(device, precision):
    """Enable CUDA BF16 autocasting when explicitly configured."""
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if precision == "bf16" and torch.device(device).type == "cuda"
        else nullcontext()
    )


def build_models(args, data, device):
    """Build fresh networks or restore both networks from a training checkpoint."""
    previous = None
    if args.resume is not None:
        previous = torch.load(args.resume, map_location="cpu", weights_only=True)
        if (
            previous.get("format") != VERSION
            or previous.get("manifest_signature") != data.signature
        ):
            raise ValueError(
                "Resume checkpoint has a different training or dataset format"
            )
        ae = PartitionAutoencoder(**previous["ae_kwargs"])
        ae.load_state_dict(previous["ae_state"])
        one = OneStepPredictor(**previous["one_step_kwargs"])
        one.load_state_dict(previous["one_step_state"])
    else:
        ae = PartitionAutoencoder(
            **data.model_kwargs(
                hidden_dim=args.hidden_dim,
                latent_dims=(args.coarse_dim, args.fine_dim, args.fine_dim),
                message_layers=args.message_layers,
                region_layers=args.region_layers,
                decoder_hidden_dim=args.decoder_hidden_dim,
                decoder_residual_layers=args.decoder_residual_layers,
                region_attention_heads=args.region_attention_heads,
                region_attention_width=args.region_attention_width,
            )
        )
        one = OneStepPredictor(
            latent_dim=ae.latent_dim,
            part_count=data.part_count,
            patch_feature_dim=data.patch_feature_dim,
            hidden_dim=args.hidden_dim,
        )
    data.cfg.halo_hops = ae.message_layers
    data.cfg.region_layers = ae.region_layers
    return ae.to(device), one.to(device), previous


def training_loss(args, ae, one, data, case, graph, device, rng):
    """Compute reconstruction and optional one-step objectives on a single case graph."""
    count = len(data.times[data.case_index[case]])
    powers = (
        getattr(args, "region_balance_power", 0.0),
        getattr(args, "material_weight_power", 0.0),
    )
    edge_weight = getattr(args, "wheel_edge_weight", 0.0)
    interface_weight = getattr(args, "wheel_interface_weight", 0.0)
    edge_stats = {}

    def field_objective(prediction, target, name):
        """Combine supervised nodal errors with optional wheel displacement-difference errors."""
        field, ordinary = displacement_mses(prediction, target, *powers)
        if edge_weight or interface_weight:
            inner, interface = wheel_edge_mses(prediction, target)
            edge_stats.setdefault(name, []).append(
                torch.stack((inner.detach(), interface.detach()))
            )
            field = field + edge_weight * inner + interface_weight * interface
        return field, ordinary

    def add_edge_stats(details):
        """Append detached wheel edge and interface errors to the training statistics."""
        for name, values in edge_stats.items():
            inner, interface = torch.stack(values).mean(0).tolist()
            details[f"{name}_wheel_edge_mse"] = inner
            details[f"{name}_wheel_interface_mse"] = interface
        return details

    independent = getattr(args, "independent_reconstruction_frames", False)
    frame = int(rng.integers(1, count if independent else count - 1))
    attached = data.attach(case, frame, graph)
    batch = tensor_batch(attached, device)
    output = ae(batch)
    reconstruction, ordinary = field_objective(
        output["prediction"], batch, "reconstruction"
    )
    total = reconstruction
    details = dict(
        reconstruction=float(reconstruction.detach()),
        train_field_mse_unweighted=float(ordinary.detach()),
    )
    if args.latent_weight or args.dynamics_field_weight:
        # Only frame-dependent fields need a new device copy. Every frame
        # shares the same topology, node ordering and decoder selection.
        shared = {
            k: batch[k] for k, v in graph.items() if attached.get(k) is v and k in batch
        }

        def at(index):
            """Attach a new time slice while reusing the static graph tensors on the device."""
            if index == frame:
                return batch
            sample = data.attach(case, index, graph)
            changed = {
                k: v for k, v in sample.items() if k not in shared or v is not graph[k]
            }
            return {**shared, **tensor_batch(changed, device)}

        center = int(rng.integers(1, count - 1)) if independent else frame
        current, previous, following = at(center), at(center - 1), at(center + 1)
        zc = output["latent"] if center == frame else ae.encode(current)
        zp = output["latent"] if center - 1 == frame else ae.encode(previous)
        with torch.no_grad():
            zn = (
                output["latent"].detach()
                if center + 1 == frame
                else ae.encode(following)
            )
        times = data.times[data.case_index[case]]
        forecast = one.forward_known(
            zp,
            zc,
            current["patch_features"],
            current["patch_part_index"],
            float(times[center] - times[center - 1]),
            float(times[center + 1] - times[center]),
            float(np.median(np.diff(times))),
            current_time=float(times[center]),
        )
        latent_loss = masked_latent_mse(forecast, zn, following)
        field_loss = (
            field_objective(ae.decode(current, forecast), following, "one_step_field")[
                0
            ]
            if bool(following["valid"].any())
            else forecast.sum() * 0.0
        )
        total = (
            total
            + args.latent_weight * latent_loss
            + args.dynamics_field_weight * field_loss
        )
        details.update(
            latent=float(latent_loss.detach()),
            one_step_field=float(field_loss.detach()),
        )
    return total, add_edge_stats(details)


def sampled_training_loss(args, ae, one, data, candidates, device, rng):
    """Draw a case and patches, retrying samples that have no valid targets."""
    for _ in range(64):
        case = str(rng.choice(data.case_ids))
        available = np.intersect1d(candidates, data.training_patches(case))
        if not len(available):
            continue
        if args.wheel_batch_fraction and rng.random() < args.wheel_batch_fraction:
            patches = data.wheels.sample_patches(rng, available, args.patches_per_batch)
        else:
            patches = rng.choice(
                available, min(args.patches_per_batch, len(available)), replace=False
            )
        graph = data.graph_batch(case, patches)
        try:
            with precision_context(device, args.precision):
                return training_loss(args, ae, one, data, case, graph, device, rng)
        except NoValidTarget:
            continue
    raise NoValidTarget(
        "No valid target after 64 draws; inspect the requested patch/frame scope"
    )
