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

"""Connect the optional Mamba CUDA kernel to the portable model for this example."""

from importlib import import_module

from physicsnemo.core.version_check import check_version_spec
from physicsnemo.experimental.models.partitioned_mesh.selective_scan import (
    SelectiveStateSpaceBlock,
)


def configure_scan_backend(model):
    """Attach a compatible installed extension only to blocks requesting fused CUDA."""
    fused = [
        block
        for block in model.modules()
        if isinstance(block, SelectiveStateSpaceBlock) and block.scan_backend == "mamba"
    ]
    if not fused:
        return model
    message = (
        "Fused selective scan requires a CUDA-compatible mamba-ssm>=2.2.6.post3 "
        "on Linux. Install the example's requirements-mamba.txt after installing "
        "the matching CUDA/PyTorch toolchain, or use --scan-backend parallel."
    )
    if not check_version_spec("mamba-ssm", "2.2.6.post3", hard_fail=False):
        raise ImportError(message)
    try:
        fused_fn = import_module(
            "mamba_ssm.ops.selective_scan_interface"
        ).selective_scan_fn
    except (ImportError, OSError) as exc:
        raise ImportError(message) from exc

    def kernel(u, delta, a, b, c, skip):
        return fused_fn(u, delta, a, b, c, D=skip, delta_softplus=False)

    for block in fused:
        block.bind_scan_kernel(kernel)
    return model
