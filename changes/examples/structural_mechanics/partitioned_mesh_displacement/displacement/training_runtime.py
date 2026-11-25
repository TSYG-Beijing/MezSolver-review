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

"""Shared checkpoint, run-report and distributed random-state utilities."""

import json
import shutil

import torch
import torch.distributed as dist


def atomic_save(path, value):
    """Write a PyTorch checkpoint beside its destination, then replace it atomically."""
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def write_text(path, value):
    """Atomically replace a UTF-8 run artifact."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)


def write_json(path, value):
    """Serialize a run report to JSON and publish it atomically."""
    write_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_history(out, history):
    """Write one JSON record per completed epoch in chronological order."""
    write_text(
        out / "history.jsonl",
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in history),
    )


def rng_state(rng, device):
    """Snapshot the rank's sampler and PyTorch random generators."""
    return dict(
        numpy_rng=rng.bit_generator.state,
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    )


def restore_outputs(out, resume, previous):
    """Rebuild the best checkpoint and run reports from a saved epoch."""
    out.mkdir(parents=True, exist_ok=True)
    last = out / "last.pt"
    if last.exists() and not last.samefile(resume):
        raise FileExistsError(
            f"Refusing to replace a different last checkpoint in {out}"
        )
    best_path = out / "best.pt"
    if previous["epoch"] == previous["best_epoch"]:
        atomic_save(best_path, previous)
    else:
        prior_best = resume.parent / "best.pt"
        if not prior_best.is_file():
            raise FileNotFoundError(
                f"Best checkpoint from epoch {previous['best_epoch']} is missing: {prior_best}"
            )
        best = torch.load(prior_best, map_location="cpu", weights_only=True)
        if (
            best["epoch"] != previous["best_epoch"]
            or best["best_score"] != previous["best_score"]
            or best["source_signature"] != previous["source_signature"]
            or best["manifest_signature"] != previous["manifest_signature"]
        ):
            raise ValueError("Best checkpoint does not correspond to this resume state")
        if best_path.resolve() != prior_best.resolve():
            atomic_save(best_path, best)
    if not last.exists():
        tmp = last.with_suffix(".tmp")
        shutil.copyfile(resume, tmp)
        tmp.replace(last)
    save_history(out, previous["ddp_history"])
    if previous["latest_validation"] is not None:
        write_json(out / "validation.json", previous["latest_validation"])


def gather_objects(value, world_size):
    """Collect one serializable value per rank in rank order."""
    if world_size == 1:
        return [value]
    gathered = [None] * world_size
    dist.all_gather_object(gathered, value)
    return gathered
