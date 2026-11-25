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

"""Train initial-condition trajectories with masked field, latent and physical edge losses."""

import numpy as np
import torch

from .losses import NoValidTarget, displacement_mses, wheel_edge_mses
from .objective import masked_latent_mse, precision_context, tensor_batch


def predict_trajectory(ae, dynamics, data, case, graph, device, frames=None):
    """Read only frame zero, evolve the entire time grid and decode the requested frames in order."""
    initial = tensor_batch(data.attach(case, 0, graph), device)
    times = torch.as_tensor(
        data.times[data.case_index[case]], dtype=torch.float64, device=device
    )
    latents = dynamics.forward_known(
        ae.encode(initial).float(),
        initial["patch_features"],
        initial["patch_part_index"],
        initial["region_edge_index"],
        times,
        data.trajectory_time_scale(case),
        initial["region_in_degree"],
    )
    spatial = ae.prepare_decoder(initial)
    indexes = range(len(times)) if frames is None else frames
    fields = [
        initial["q"][initial["decode_index"]]
        if frame == 0
        else ae.decode(initial, latents[frame], spatial=spatial)
        for frame in indexes
    ]
    return torch.stack(fields), latents, initial


def training_loss(
    args, ae, dynamics, data, case, graph, device, rng, require_seam=False
):
    """Compare an initial-condition forecast to sampled positive-time labels in one case."""
    available = np.arange(1, len(data.times[data.case_index[case]]))
    frames = (
        np.sort(
            rng.choice(
                available, min(len(available), args.frames_per_step), replace=False
            )
        )
        if args.frames_per_step
        else available
    )
    bound = data.seams.bind_graph(graph) if data.seams is not None else None
    if require_seam and (bound is None or not len(bound[0])):
        raise NoValidTarget("Seam draw contains no fully decoded shell perimeter edge")
    predictions, latents, initial = predict_trajectory(
        ae, dynamics, data, case, graph, device, frames
    )
    reconstruct = args.stage == 3 and args.reconstruction_weight > 0
    spatial = ae.prepare_decoder(initial) if reconstruct else None
    records = {}
    pred_losses, ordinary_losses, latent_losses, rec_losses = [], [], [], []
    seam_sums, seam_counts = [], []

    def field_loss(prediction, target, name):
        """Apply channel, region and material weights and optional wheel edge penalties."""
        weighted, ordinary = displacement_mses(
            prediction,
            target,
            args.region_balance_power,
            args.material_weight_power,
            args.channel_weights,
        )
        if args.wheel_edge_weight or args.wheel_interface_weight:
            edge, interface = wheel_edge_mses(prediction, target)
            weighted = (
                weighted
                + args.wheel_edge_weight * edge
                + args.wheel_interface_weight * interface
            )
            records.setdefault(f"{name}_wheel_edge_mse", []).append(edge.detach())
            records.setdefault(f"{name}_wheel_interface_mse", []).append(
                interface.detach()
            )
        return weighted, ordinary

    for frame, prediction in zip(frames, predictions, strict=True):
        target = tensor_batch(data.attach(case, int(frame), graph), device)
        if not bool(target["valid"].any()):
            continue
        field, ordinary = field_loss(prediction, target, "prediction")
        pred_losses.append(field)
        ordinary_losses.append(ordinary.detach())
        if args.latent_weight or reconstruct:
            # Target encodings supply detached labels. Only the Stage 3 reconstruction
            # branch needs a backward graph through an optionally unfrozen encoder.
            with torch.set_grad_enabled(torch.is_grad_enabled() and reconstruct):
                encoded = ae.encode(target)
            if args.latent_weight:
                latent_losses.append(masked_latent_mse(latents[frame], encoded, target))
            if reconstruct:
                rec = ae.decode(initial, encoded, spatial=spatial)
                rec_losses.append(field_loss(rec, target, "reconstruction")[0])
        if bound is not None:
            summed, count = data.seams.residual_sums(
                bound, data.deleted_wheel_shells(case, int(frame)), prediction, target
            )
            seam_sums.append(summed)
            seam_counts.append(count)
    if not pred_losses:
        raise NoValidTarget(
            "Selected future frames contain no valid displacement labels"
        )
    field = torch.stack(pred_losses).mean()
    latent = torch.stack(latent_losses).mean() if latent_losses else field.new_zeros(())
    rec = torch.stack(rec_losses).mean() if rec_losses else field.new_zeros(())
    total = field + args.latent_weight * latent + args.reconstruction_weight * rec
    details = {
        key: float(torch.stack(values).mean()) for key, values in records.items()
    }
    details.update(
        prediction=float(field.detach()),
        latent=float(latent.detach()),
        reconstruction=float(rec.detach()),
        train_field_mse_unweighted=float(torch.stack(ordinary_losses).mean()),
    )
    if bound is not None:
        active = int(torch.stack(seam_counts).sum())
        if require_seam and active == 0:
            raise NoValidTarget("No live seam has two valid endpoints in these frames")
        u1, vector, gradient = torch.stack(seam_sums).sum(0) / max(active, 1)
        total = (
            total
            + args.seam_gradient_weight * gradient
            + args.seam_weight * (u1 + vector) / (2 * args.seam_scale_mm**2)
        )
        details.update(
            prediction_seam_active_edges=active,
            seam_batch=int(require_seam),
            seam_supervised_batch=int(active > 0),
            prediction_seam_u1_jump_mse_mm2=float(u1.detach()),
            prediction_seam_vector_jump_mse_mm2=float(vector.detach()),
            prediction_seam_gradient_mse=float(gradient.detach()),
        )
    return total, details


def sampled_training_loss(args, ae, dynamics, data, candidates, device, rng):
    """Sample a uniformly chosen case and random patches, a full wheel or physical seam pairs."""
    draw = rng.random()
    wheel = draw < args.wheel_batch_fraction
    seam = not wheel and draw < args.wheel_batch_fraction + args.seam_batch_fraction
    for _ in range(64):
        case = str(rng.choice(data.case_ids))
        available = np.intersect1d(candidates, data.training_patches(case))
        if not len(available):
            continue
        if wheel:
            patches = data.wheels.sample_patches(rng, available, args.patches_per_batch)
        elif seam:
            patches = data.seams.sample_patches(
                rng,
                available,
                args.patches_per_batch,
                pairs_per_batch=max(1, args.patches_per_batch // 2),
            )
        else:
            patches = rng.choice(
                available, min(args.patches_per_batch, len(available)), replace=False
            )
        graph = data.graph_batch(case, patches)
        try:
            with precision_context(device, args.precision):
                return training_loss(
                    args,
                    ae,
                    dynamics,
                    data,
                    case,
                    graph,
                    device,
                    rng,
                    require_seam=seam,
                )
        except NoValidTarget:
            continue
    raise NoValidTarget(
        "No valid future supervision after 64 draws within the requested patch scope"
    )
