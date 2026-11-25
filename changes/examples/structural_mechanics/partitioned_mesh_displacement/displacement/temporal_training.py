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

"""Train an initial-condition temporal model or fine-tune it with the spatial decoder."""

import json
import math
import os
from collections import defaultdict
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from physicsnemo.experimental.models import partitioned_mesh
from physicsnemo.experimental.models.partitioned_mesh import (
    PartitionAutoencoder,
    TemporalPredictor,
)

from .config import epoch_learning_rate, read_validation_patches
from .data import PartitionedDisplacementDataset, file_hash, identity
from .objective import VERSION
from .scan_backend import configure_scan_backend
from .temporal_config import parser, validate
from .temporal_objective import predict_trajectory, sampled_training_loss
from .training_runtime import (
    atomic_save,
    gather_objects,
    restore_outputs,
    rng_state,
    save_history,
    write_json,
)
from .validation import evaluate

__all__ = ["main", "train", "parser", "predict_trajectory"]


class TemporalObjective(torch.nn.Module):
    """Apply the trajectory loss in one distributed forward call per graph draw."""

    def __init__(self, ae, dynamics, args, data, candidates, device, rng):
        """Bind the frozen or trainable spatial/temporal pair and rank-local sampler."""
        super().__init__()
        self.ae = ae
        self.dynamics = dynamics
        self.inputs = args, data, candidates, device, rng

    def forward(self):
        """Return the sampled training loss and detached diagnostics."""
        args, data, candidates, device, rng = self.inputs
        return sampled_training_loss(
            args, self.ae, self.dynamics, data, candidates, device, rng
        )


def load_models(args, data, device):
    """Restore spatial weights and initialize or restore the trajectory dynamics."""
    path = args.resume or args.initial_checkpoint
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (
        state.get("format") != VERSION
        or state.get("manifest_signature") != data.signature
        or state.get("test_set_used") is not False
    ):
        raise ValueError(
            "Checkpoint format, dataset identity or held-out split declaration differs"
        )
    expected_stage = args.stage if args.resume else args.stage - 1
    if state.get("stage") != expected_stage:
        raise ValueError(f"Expected a Stage {expected_stage} checkpoint")
    ae = PartitionAutoencoder(**state["ae_kwargs"]).to(device)
    ae.load_state_dict(state["ae_state"], strict=True)
    if expected_stage == 1:
        dynamics = TemporalPredictor(
            latent_dim=ae.latent_dim,
            part_count=data.part_count,
            patch_feature_dim=data.patch_feature_dim,
            hidden_dim=args.hidden_dim,
            spatial_layers=args.spatial_layers,
            temporal_layers=args.temporal_layers,
            state_dim=args.state_dim,
            scan_backend=args.scan_backend,
            region_attention_heads=args.region_attention_heads,
            region_attention_width=args.region_attention_width,
            dynamic_region_exchange=args.dynamic_region_exchange,
        ).to(device)
    else:
        data.restore_trajectory_clock(state)
        dynamics = TemporalPredictor(**state["dynamics_kwargs"]).to(device)
        dynamics.load_state_dict(state["dynamics_state"], strict=True)
    for name, parameter in ae.named_parameters():
        parameter.requires_grad_(
            args.stage == 3 and (args.unfreeze == "all" or name.startswith("decoder_"))
        )
    configure_scan_backend(dynamics)
    dynamics.requires_grad_(True)
    data.cfg.halo_hops = ae.message_layers
    data.cfg.region_layers = ae.region_layers + dynamics.spatial_layers
    return ae, dynamics, state


def source_manifest():
    """Bind resumable training state to the example and the imported model implementations."""
    package = Path(__file__).resolve().parent
    files = {f"displacement/{p.name}": file_hash(p) for p in package.glob("*.py")}
    for p in package.parent.glob("train_stage_*.py"):
        files[p.name] = file_hash(p)
    for p in Path(partitioned_mesh.__file__).resolve().parent.glob("*.py"):
        files[f"partitioned_mesh/{p.name}"] = file_hash(p)
    return files


def prepare_split(args, split, stack, verify):
    """Open the requested split, enabling wheel data and train-only seam sampling."""
    data = stack.enter_context(
        PartitionedDisplacementDataset(
            args.manifest,
            split,
            verify_hashes=verify,
            max_graph_nodes=args.max_graph_nodes,
            max_graph_edges=args.max_graph_edges,
            open_cases=args.open_cases,
            response_cache_mb=args.response_cache_mb
            if split == "train"
            else args.validation_cache_mb,
        )
    )
    if args.wheel_groups:
        data.enable_wheels(args.wheel_groups)
    if (
        split == "train"
        and args.material_weight_power
        and data.material_weight_by_patch is None
    ):
        raise ValueError("Material weighting requires the manifest material table")
    if split == "train" and (
        args.seam_weight or args.seam_gradient_weight or args.seam_batch_fraction
    ):
        from .temporal_seams import enable_seams

        if args.seam_edges is None and data.manifest.get("seam_edges"):
            args.seam_edges = (data.path.parent / data.manifest["seam_edges"]).resolve()
        enable_seams(data, args.seam_edges)
    return data


def validate_resume(state, config, source, world):
    """Require the original temporal source, optimizer schedule and rank-local RNG layout."""
    if (
        state.get("source_signature") != source
        or state.get("ddp_world_size") != world
        or len(state.get("ddp_rank_rng", [])) != world
    ):
        raise ValueError(
            "Resume requires this implementation and the original rank count"
        )
    paths = {
        "manifest",
        "output",
        "resume",
        "initial_checkpoint",
        "wheel_groups",
        "val_patch_ids_file",
        "seam_edges",
    }
    changed = [
        k for k, v in config.items() if k not in paths and state["config"].get(k) != v
    ]
    if changed:
        raise ValueError(f"Training schedule, assets or objective changed: {changed}")


def epoch_step(module, parameters, optimizer, args, rank, world, epoch, out):
    """Update trainable parameters and average diagnostics across participating workers."""
    local = defaultdict(list)
    module.train()
    (module.module if world > 1 else module).ae.train(args.stage == 3)
    for step in range(1, args.steps_per_epoch + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, details = module()
        value = float(loss.detach())
        if not math.isfinite(value):
            raise FloatingPointError(
                f"Nonfinite loss on rank {rank} at epoch {epoch}, step {step}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            parameters, args.grad_clip, error_if_nonfinite=True
        )
        optimizer.step()
        for key, val in {**details, "loss": value}.items():
            local[key].append(float(val))
        if rank == 0 and (step == 1 or step % 10 == 0 or step == args.steps_per_epoch):
            write_json(
                out / "progress.json",
                dict(
                    stage=args.stage,
                    epoch=epoch,
                    step=step,
                    rank=rank,
                    world_size=world,
                    loss=value,
                ),
            )
    workers = gather_objects({k: (sum(v), len(v)) for k, v in local.items()}, world)
    keys = set().union(*(worker.keys() for worker in workers))
    return {
        k: sum(w[k][0] for w in workers if k in w)
        / sum(w[k][1] for w in workers if k in w)
        for k in sorted(keys)
    }


def run(args, rank, world, device, stack):
    """Train one stage from a parent checkpoint and commit complete epochs on rank zero."""
    validate(args, device)
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, rank]))
    out = args.output.resolve()
    data = prepare_split(args, "train", stack, rank == 0)
    ae, dynamics, state = load_models(args, data, device)
    candidates = np.asarray(
        args.patch_ids if args.patch_ids else np.arange(data.patch_count),
        dtype=np.int64,
    )
    if (
        len(candidates) != len(np.unique(candidates))
        or candidates.min() < 0
        or candidates.max() >= data.patch_count
    ):
        raise ValueError("Training patch IDs must be unique and within the mesh")
    val_ids = (
        read_validation_patches(args.val_patch_ids_file, data.patch_count)
        if args.val_patch_ids_file
        else None
    )
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(
        val_patch_ids=val_ids,
        wheel_groups_sha256=file_hash(args.wheel_groups) if args.wheel_groups else None,
        seam_edges_sha256=file_hash(args.seam_edges) if args.seam_edges else None,
    )
    sources = source_manifest()
    signature = identity(sources)
    if args.resume:
        validate_resume(state, config, signature, world)
        parent = state["parent_checkpoint"]
    else:
        parent = dict(
            stage=state["stage"],
            epoch=state["epoch"],
            sha256=file_hash(args.initial_checkpoint),
            source_signature=state.get("source_signature"),
        )
    if rank == 0:
        if args.resume:
            restore_outputs(out, args.resume, state)
        else:
            out.mkdir(parents=True, exist_ok=False)
        write_json(out / "config.json", config)
        write_json(out / "source_manifest.json", sources)
        write_json(out / "parent_checkpoint.json", parent)
        validation = prepare_split(args, "val", stack, False)
        validation.temporal_reference_time = data.temporal_reference_time
    obj = TemporalObjective(ae, dynamics, args, data, candidates, device, rng)
    module = (
        DistributedDataParallel(obj, device_ids=None, find_unused_parameters=True)
        if world > 1
        else obj
    )
    parameters = [p for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=args.fused_adamw and device.type == "cuda",
    )
    start, best_epoch, best, history, latest = 0, 0, float("inf"), [], None
    if args.resume:
        optimizer.load_state_dict(state["optimizer"])
        local = state["ddp_rank_rng"][rank]
        rng.bit_generator.state = local["numpy_rng"]
        torch.set_rng_state(local["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(local["cuda_rng"], device)
        start, best_epoch, best = (
            state["epoch"],
            state["best_epoch"],
            state["best_score"],
        )
        history, latest = state["ddp_history"], state["latest_validation"]
    else:
        torch.manual_seed(args.seed + rank)
    del state
    for epoch in range(start + 1, args.epochs + 1):
        lr = epoch_learning_rate(args, epoch - 1)
        for group in optimizer.param_groups:
            group["lr"] = lr
        summary = epoch_step(
            module, parameters, optimizer, args, rank, world, epoch, out
        )
        if rank == 0:
            report = None
            if epoch == 1 or epoch % args.val_every == 0 or epoch == args.epochs:
                report = evaluate(
                    ae,
                    validation,
                    device,
                    dynamics=dynamics,
                    max_cases=args.val_max_cases,
                    max_frames=args.val_max_frames,
                    max_patches=args.val_max_patches,
                    patch_ids=val_ids or args.patch_ids,
                    patches_per_batch=args.eval_patches_per_batch
                    or args.patches_per_batch,
                    wheel_selection_weight=args.wheel_selection_weight,
                )
        states = gather_objects(rng_state(rng, device), world)
        if rank == 0:
            improved = report is not None and report["selection_score"] < best
            if improved:
                best, best_epoch = report["selection_score"], epoch
            if report is not None:
                latest = report
            history.append(
                dict(
                    epoch=epoch,
                    learning_rate=lr,
                    world_size=world,
                    global_optimizer_steps=epoch * args.steps_per_epoch,
                    validation=report["selection_score"] if report else None,
                    **summary,
                )
            )
            saved = dict(
                format=VERSION,
                stage=args.stage,
                epoch=epoch,
                manifest_signature=data.signature,
                temporal_reference_time=data.temporal_reference_time,
                source_signature=signature,
                config=config,
                parent_checkpoint=parent,
                ae_kwargs=ae.model_kwargs,
                ae_state=ae.state_dict(),
                dynamics_kwargs=dynamics.model_kwargs,
                dynamics_state=dynamics.state_dict(),
                one_step_kwargs=None,
                one_step_state=None,
                optimizer=optimizer.state_dict(),
                test_set_used=False,
                best_score=best,
                best_epoch=best_epoch,
                validation=report,
                latest_validation=latest,
                ddp_world_size=world,
                ddp_history=history,
                ddp_rank_rng=states,
            )
            atomic_save(out / "last.pt", saved)
            if improved:
                atomic_save(out / "best.pt", saved)
            if report is not None:
                write_json(out / "validation.json", report)
            save_history(out, history)
            print(json.dumps(history[-1]), flush=True)
        if world > 1:
            dist.barrier()
    if rank == 0:
        write_json(
            out / "completion.json",
            dict(
                status="complete",
                stage=args.stage,
                epochs=args.epochs,
                best_epoch=best_epoch,
                best_score=best,
                ddp_world_size=world,
                test_set_used=False,
            ),
        )
    return out


def train(args):
    """Select the local device and synchronize workers launched with torchrun."""
    rank, world, local_rank = (
        int(os.environ.get(name, default))
        for name, default in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0))
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        if world > 1 and device.index is not None and device.index != local_rank:
            raise ValueError("Use --device cuda with torchrun")
        device = torch.device(
            "cuda", local_rank if device.index is None else device.index
        )
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError("Choose --device cpu or cuda")
    if world > 1:
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo",
            rank=rank,
            world_size=world,
            timeout=timedelta(hours=4),
        )
    try:
        with ExitStack() as stack:
            return run(args, rank, world, device, stack)
    finally:
        if world > 1:
            dist.destroy_process_group()


def main(stage):
    """Parse the stage-specific arguments and run trajectory optimization."""
    return train(parser(stage).parse_args())
