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

"""Synchronous Stage 1 training. Launch one torchrun process per GPU.

One optimizer update averages independently sampled graph objectives across ranks.
The checkpoint includes rank-local random states for same-world-size restart.
"""

import json
import math
import os
import socket
import time
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from displacement.config import (
    epoch_learning_rate,
    parser,
    read_validation_patches,
    validate_training_args,
)
from displacement.data import PartitionedDisplacementDataset, file_hash, identity
from displacement.objective import VERSION, build_models, sampled_training_loss
from displacement.training_runtime import (
    atomic_save,
    gather_objects,
    restore_outputs,
    rng_state,
    save_history,
    write_json,
)
from displacement.validation import evaluate
from torch.nn.parallel import DistributedDataParallel

from physicsnemo.experimental.models import partitioned_mesh


class Stage1Objective(torch.nn.Module):
    """Expose both Stage 1 networks to distributed gradient reduction."""

    def __init__(self, ae, one, args, data, candidates, device, rng):
        """Bind the two networks and this rank's case sampler to a single module."""
        super().__init__()
        self.ae, self.one = ae, one
        self.args, self.data, self.candidates = args, data, candidates
        self.device, self.rng = device, rng

    def forward(self):
        # Retried unsupervised draws stay within this rank's one DDP forward.
        """Sample a supervised graph and return the differentiable Stage 1 loss."""
        return sampled_training_loss(
            self.args,
            self.ae,
            self.one,
            self.data,
            self.candidates,
            self.device,
            self.rng,
        )


def check_resume(previous, config, source, world):
    """Verify the checkpoint matches the sampler topology, source and training settings."""
    if (
        previous.get("ddp_world_size") != world
        or len(previous.get("ddp_rank_rng", [])) != world
        or previous.get("source_signature") != source
        or previous.get("test_set_used") is not False
    ):
        raise ValueError(
            "Resume requires the same distributed world size, source and Stage 1 checkpoint"
        )
    # Locations may change. The manifest signature, wheel asset digest and
    # parsed validation IDs bind the content of those inputs independently.
    paths = {"output", "resume", "manifest", "wheel_groups", "val_patch_ids_file"}
    differences = [
        k for k in config if k not in paths and config[k] != previous["config"].get(k)
    ]
    if differences:
        raise ValueError(
            f"Distributed training schedule, device or objective changed: {differences}"
        )


def metric_names(args):
    """List training metrics enabled by the chosen losses."""
    keys = ["loss", "train_field_mse_unweighted", "reconstruction"]
    fields = ["reconstruction"]
    if args.latent_weight or args.dynamics_field_weight:
        keys += ["latent", "one_step_field"]
        fields.append("one_step_field")
    if args.wheel_edge_weight or args.wheel_interface_weight:
        keys += [
            f"{field}_wheel_{kind}_mse"
            for field in fields
            for kind in ("edge", "interface")
        ]
    return keys


def mean_metrics(totals, names):
    """Divide each reduced metric sum by its number of observations."""
    return {key: s / n for key, (s, n) in zip(names, totals.cpu().tolist()) if n}


def run_distributed(args, rank, world, device, stack):
    """Run the rank-local optimizer loop and publish validation results on rank zero."""
    validate_training_args(args, device)
    out = args.output.resolve()
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, rank]))

    data_options = dict(
        max_graph_nodes=args.max_graph_nodes,
        max_graph_edges=args.max_graph_edges,
        open_cases=args.open_cases,
    )
    data = stack.enter_context(
        PartitionedDisplacementDataset(
            args.manifest,
            "train",
            verify_hashes=(rank == 0),
            response_cache_mb=args.response_cache_mb,
            **data_options,
        )
    )
    if args.material_weight_power and data.material_weight_by_patch is None:
        raise ValueError("Material weighting requires a verified material table")
    if args.wheel_groups:
        data.enable_wheels(args.wheel_groups)
    explicit_ids = (
        read_validation_patches(args.val_patch_ids_file, data.patch_count)
        if args.val_patch_ids_file
        else None
    )
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(
        val_patch_ids=explicit_ids,
        wheel_groups_sha256=file_hash(args.wheel_groups) if args.wheel_groups else None,
        ddp_world_size=world,
        ddp_backend=(dist.get_backend() if world > 1 else "single"),
    )
    package = Path(__file__).resolve().parent
    sources = {Path(__file__).name: file_hash(Path(__file__))}
    for path in (package / "displacement").glob("*.py"):
        sources[path.relative_to(package).as_posix()] = file_hash(path)
    model_dir = Path(partitioned_mesh.__file__).resolve().parent
    for path in model_dir.glob("*.py"):
        sources["partitioned_mesh/" + path.name] = file_hash(path)
    source = identity(sources)
    candidates = np.asarray(
        args.patch_ids if args.patch_ids else np.arange(data.patch_count), np.int64
    )
    if (
        len(np.unique(candidates)) != len(candidates)
        or min(candidates) < 0
        or max(candidates) >= data.patch_count
    ):
        raise ValueError("Invalid or duplicate training patches")
    ae, one, previous = build_models(args, data, device)
    if args.resume:
        check_resume(previous, config, source, world)
    else:
        if rank == 0:
            out.mkdir(parents=True, exist_ok=False)

    if rank == 0:
        if args.resume:
            restore_outputs(out, args.resume, previous)
        write_json(out / "source_manifest.json", sources)
        write_json(out / "config.json", config)
        validation = stack.enter_context(
            PartitionedDisplacementDataset(
                args.manifest,
                "val",
                verify_hashes=False,
                response_cache_mb=args.validation_cache_mb,
                **data_options,
            )
        )
        if args.wheel_groups:
            validation.enable_wheels(args.wheel_groups)
    objective = Stage1Objective(ae, one, args, data, candidates, device, rng)
    # The objective constructs its graph tensors directly on this rank's
    # device. Input scattering expects a positional input, while forward()
    # samples the graph itself. Parameter-gradient reduction stays on NCCL.
    ddp = (
        DistributedDataParallel(objective, device_ids=None, find_unused_parameters=True)
        if world > 1
        else objective
    )
    info = dict(
        rank=rank,
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        pid=os.getpid(),
        host=socket.gethostname(),
        device=str(device),
        backend=config["ddp_backend"],
    )
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info.update(
            gpu=properties.name,
            gpu_uuid=str(getattr(properties, "uuid", "")),
            total_memory_bytes=properties.total_memory,
        )
    workers = gather_objects(info, world)
    if rank == 0:
        write_json(
            out / "distributed_workers.json",
            dict(
                world_size=world,
                workers=workers,
                torch_version=str(torch.__version__),
                cuda_version=torch.version.cuda,
            ),
        )
    parameters = list(ddp.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=(args.fused_adamw and device.type == "cuda"),
    )
    start, best_epoch, best, latest, history = 0, 0, float("inf"), None, []
    if previous is not None:
        start, best, best_epoch = (
            previous["epoch"],
            previous["best_score"],
            previous["best_epoch"],
        )
        optimizer.load_state_dict(previous["optimizer"])
        local = previous["ddp_rank_rng"][rank]
        rng.bit_generator.state = local["numpy_rng"]
        torch.set_rng_state(local["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(local["cuda_rng"], device)
        if rank == 0:
            latest, history = previous["latest_validation"], previous["ddp_history"]
        del previous
    else:
        # Model parameters were initialized with a shared seed and broadcast by
        # DDP. Subsequent model-side stochastic operations use a rank-local stream.
        torch.manual_seed(args.seed + rank)
    keys = metric_names(args)
    for epoch in range(start, args.epochs):
        begin = time.monotonic()
        ddp.train()
        lr = epoch_learning_rate(args, epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        totals = torch.zeros((len(keys), 2), device=device, dtype=torch.float64)
        for step in range(args.steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            loss, details = ddp()
            loss_value = float(loss.detach())
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"Rank {rank}: nonfinite training loss")
            # DDP bucket-reduces and averages gradients from all ranks during
            # backward, including parameters used on only a subset of ranks.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters, args.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            details["loss"] = loss_value
            metrics = torch.tensor(
                [(details.get(k, 0.0), int(k in details)) for k in keys],
                device=device,
                dtype=torch.float64,
            )
            totals += metrics
            if step == 0 or (step + 1) % 10 == 0:
                if world > 1:
                    dist.reduce(metrics, dst=0)
                rank_losses = None
                if step == 0:
                    rank_losses = gather_objects(loss_value, world)
                if rank == 0:
                    progress = dict(
                        stage=1,
                        epoch=epoch + 1,
                        step=step + 1,
                        world_size=world,
                        **mean_metrics(metrics, keys),
                    )
                    if rank_losses is not None:
                        progress["rank_losses"] = rank_losses
                    write_json(out / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
        if world > 1:
            dist.reduce(totals, dst=0)
        training_seconds = time.monotonic() - begin
        report = None
        if rank == 0 and (
            epoch == 0 or (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs
        ):
            report = evaluate(
                ae,
                validation,
                device,
                patch_ids=explicit_ids or args.patch_ids,
                max_cases=args.val_max_cases,
                max_frames=args.val_max_frames,
                max_patches=args.val_max_patches,
                patches_per_batch=args.eval_patches_per_batch or args.patches_per_batch,
                wheel_selection_weight=args.wheel_selection_weight,
            )
        states = gather_objects(rng_state(rng, device), world)
        if rank == 0:
            improved = report is not None and report["selection_score"] < best
            if report is not None:
                latest = report
            if improved:
                best, best_epoch = report["selection_score"], epoch + 1
            row = dict(
                epoch=epoch + 1,
                learning_rate=lr,
                world_size=world,
                global_optimizer_steps=(epoch + 1) * args.steps_per_epoch,
                training_graph_batches=world * args.steps_per_epoch,
                training_seconds=training_seconds,
                validation_seconds=time.monotonic() - begin - training_seconds,
                seconds=time.monotonic() - begin,
                validation=report["selection_score"] if report else None,
                **mean_metrics(totals, keys),
            )
            if report and args.wheel_groups:
                row["global_validation"] = report["global_selection_score"]
                row.update(
                    {
                        k: report["wheel_validation"][k]
                        for k in (
                            "mean_wheel_node_vector_rmse_mm",
                            "mean_wheel_interface_vector_rmse_mm",
                            "mean_wheel_rotation_error_deg",
                            "equal_wheel_node_normalized_mse",
                            "equal_wheel_interface_normalized_mse",
                        )
                    }
                )
            history.append(row)
            payload = dict(
                format=VERSION,
                stage=1,
                epoch=epoch + 1,
                source_signature=source,
                manifest_signature=data.signature,
                parent_checkpoint=None,
                config=config,
                ae_kwargs=ae.model_kwargs,
                ae_state=ae.state_dict(),
                one_step_kwargs=one.model_kwargs,
                one_step_state=one.state_dict(),
                dynamics_kwargs=None,
                dynamics_state=None,
                optimizer=optimizer.state_dict(),
                best_score=best,
                best_epoch=best_epoch,
                validation=report,
                test_set_used=False,
                numpy_rng=states[0]["numpy_rng"],
                torch_rng=states[0]["torch_rng"],
                cuda_rng=[states[0]["cuda_rng"]] if device.type == "cuda" else None,
                ddp_world_size=world,
                ddp_rank_rng=states,
                ddp_workers=workers,
                ddp_history=history,
                latest_validation=latest,
            )
            # On interruption after committing last.pt, restore_outputs can
            # republish this epoch's best, validation and full history.
            atomic_save(out / "last.pt", payload)
            if improved:
                atomic_save(out / "best.pt", payload)
            if report is not None:
                write_json(out / "validation.json", report)
            save_history(out, history)
        if world > 1:
            dist.barrier()
    if rank == 0:
        write_json(
            out / "completion.json",
            dict(
                status="complete",
                stage=1,
                epochs=args.epochs,
                best_score=best,
                best_epoch=best_epoch,
                ddp_world_size=world,
                test_set_used=False,
            ),
        )
    if world > 1:
        dist.barrier()
    return out


def train(args):
    """Choose a device and manage the distributed group across the training run."""
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    requested = torch.device(args.device)
    if requested.type == "cuda":
        index = local_rank if requested.index is None else requested.index
        if world > 1 and index != local_rank:
            raise ValueError(
                "Use --device cuda under torchrun so each rank uses its local GPU"
            )
        torch.cuda.set_device(index)
        device, backend = torch.device("cuda", index), "nccl"
    elif requested.type == "cpu":
        device, backend = requested, "gloo"
    else:
        raise ValueError("Choose cpu or cuda as the training device")
    if world > 1:
        dist.init_process_group(
            backend,
            init_method="env://",
            rank=rank,
            world_size=world,
            timeout=timedelta(hours=4),
        )
    try:
        with ExitStack() as stack:
            return run_distributed(args, rank, world, device, stack)
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    train(parser().parse_args())
