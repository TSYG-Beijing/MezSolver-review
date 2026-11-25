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

"""Capabilities shared by the partitioned-mesh displacement models."""

from dataclasses import dataclass

from physicsnemo.core import ModelMetaData


@dataclass
class PartitionedMeshMetaData(ModelMetaData):
    """Autograd and mixed-precision execution for the two public modules."""

    auto_grad: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True
    bf16: bool = True
