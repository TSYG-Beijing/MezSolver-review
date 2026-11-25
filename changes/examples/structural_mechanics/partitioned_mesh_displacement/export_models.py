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

"""Export a stage's trained spatial and prediction models in native PhysicsNeMo format."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from physicsnemo.experimental.models.partitioned_mesh import (
    OneStepPredictor,
    PartitionAutoencoder,
    TemporalPredictor,
)


def export_models(checkpoint: Path, output: Path) -> None:
    """Strictly load both state dictionaries and write an inference model pair."""
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    stage = payload.get("stage")
    if stage == 1:
        forecast_component = (
            "one_step",
            OneStepPredictor,
            ("one_step_kwargs", "one_step_state"),
        )
    elif stage in (2, 3):
        forecast_component = (
            "temporal",
            TemporalPredictor,
            ("dynamics_kwargs", "dynamics_state"),
        )
    else:
        raise ValueError("Expected a Stage 1, 2 or 3 training checkpoint.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        bundle = Path(temporary) / "models"
        bundle.mkdir()
        for name, component, keys in (
            ("autoencoder", PartitionAutoencoder, ("ae_kwargs", "ae_state")),
            forecast_component,
        ):
            model = component(**payload[keys[0]])
            model.load_state_dict(payload[keys[1]], strict=True)
            model.eval().save(str(bundle / f"{name}.mdlus"))
        if stage in (2, 3):
            clock = {"temporal_reference_time": payload.get("temporal_reference_time")}
            (bundle / "trajectory_clock.json").write_text(
                json.dumps(clock, indent=2) + "\n", encoding="utf-8"
            )
        bundle.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export_models(args.checkpoint, args.output)
