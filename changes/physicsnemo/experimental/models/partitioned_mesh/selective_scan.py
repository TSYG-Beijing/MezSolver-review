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

"""Input-selective diagonal dynamics with a parallel affine scan.

The portable backend composes affine transitions in logarithmic-depth tensor
passes. An optional caller-provided CUDA scan uses the same discretization. Neither backend feeds predicted output latents back in.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def affine_scan(decay: Tensor, drive: Tensor) -> Tensor:
    """Return all zero-initial states ``h[t] = decay[t]*h[t-1] + drive[t]``.

    Arrays have shape [patch, time, channel, state]. Affine composition is
    associative: (a2,b2) after (a1,b1) is (a2*a1, b2+a2*b1). Doubling offsets
    gives O(log T) parallel passes and O(T log T) work; this portable fallback
    is not the work-efficient fused CUDA kernel. Autograd retains every step.
    """
    if decay.shape != drive.shape or decay.ndim != 4 or decay.shape[1] < 1:
        raise ValueError(
            "Scan arrays must share a nonempty [patch,time,channel,state] shape"
        )
    transition, state = decay, drive
    offset = 1
    while offset < state.shape[1]:
        right = transition[:, offset:]
        next_state = torch.cat(
            (state[:, :offset], state[:, offset:] + right * state[:, :-offset]),
            dim=1,
        )
        transition = torch.cat(
            (transition[:, :offset], right * transition[:, :-offset]), dim=1
        )
        state = next_state
        offset *= 2
    return state


class SelectiveStateSpaceBlock(nn.Module):
    """Causal residual SSM block; sequence [P,T,H], physical deltas [T].

    Effective positive steps, B and C depend on the current input. A is a
    learned negative diagonal generator. Hidden states have shape [P,H,S]
    and start at zero for every call. Recurrence accumulation is float32.
    ``mamba`` requires CUDA and an explicitly installed compatible mamba_ssm;
    failures are surfaced instead of silently changing the selected backend.
    """

    def __init__(
        self,
        width: int,
        state_dim: int = 16,
        dropout: float = 0.0,
        scan_backend: str = "parallel",
    ) -> None:
        super().__init__()
        if scan_backend not in ("parallel", "mamba"):
            raise ValueError("scan_backend must be 'parallel' or 'mamba'")
        if width < 1 or state_dim < 1:
            raise ValueError("SSM width and state_dim must be positive")
        if scan_backend == "mamba" and state_dim > 256:
            raise ValueError(
                "The mamba selective-scan kernel requires state_dim <= 256"
            )
        self.width = width
        self.state_dim = state_dim
        self.scan_backend = scan_backend
        self._scan_kernel = None
        self.norm = nn.LayerNorm(width)
        self.in_projection = nn.Linear(width, 2 * width)
        self.dt_projection = nn.Linear(width, width)
        self.B_projection = nn.Linear(width, state_dim, bias=False)
        self.C_projection = nn.Linear(width, state_dim, bias=False)
        self.A_log = nn.Parameter(
            torch.arange(1, state_dim + 1, dtype=torch.float32).log().repeat(width, 1)
        )
        self.D = nn.Parameter(torch.ones(width, dtype=torch.float32))
        self.out_projection = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def bind_scan_kernel(
        self, kernel: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]
    ) -> None:
        r"""Bind a CUDA implementation for this instance's fused scan backend.

        Parameters
        ----------
        kernel : Callable
            Operates on contiguous float32 tensors using the recurrence implemented
            in :func:`affine_scan`. The caller supplies positive physical step sizes.

            .. code-block:: python

                def kernel(u, delta, A, B, C, D):
                    # u, delta: [P,H,T], A: [H,S], B,C: [P,S,T], D: [H]
                    # Return every state readout as [P,H,T].
                    ...
        """
        if not callable(kernel):
            raise TypeError("The scan kernel must be callable")
        self._scan_kernel = kernel

    def forward(self, sequence: Tensor, time_deltas: Tensor) -> Tensor:
        """Read the complete query sequence and return all causal outputs."""
        if sequence.ndim != 3 or sequence.shape[-1] != self.width:
            raise ValueError("sequence must have shape [patch,time,width]")
        if time_deltas.shape != (sequence.shape[1],):
            raise ValueError("time_deltas must match the query sequence length")
        projected, gate = self.in_projection(self.norm(sequence)).chunk(2, dim=-1)
        u = F.silu(projected)
        # Physical spacing multiplies the input-selective learned step; t=0
        # has zero elapsed time and does not advance the hidden state.
        delta = (
            F.softplus(self.dt_projection(u).float())
            * time_deltas.float()[None, :, None]
        )
        B = self.B_projection(u).float()
        C = self.C_projection(u).float()
        A = -self.A_log.float().exp()
        u32 = u.float()
        if self.scan_backend == "mamba":
            if not sequence.is_cuda:
                raise RuntimeError(
                    "scan_backend='mamba' requires CUDA; use 'parallel' on CPU"
                )
            if self._scan_kernel is None:
                raise RuntimeError(
                    "The fused scan requires a kernel bound to this model. Use the "
                    "partitioned mesh example's configure_scan_backend(model), or "
                    "load the same state dictionaries with scan_backend='parallel'."
                )
            y = self._scan_kernel(
                u32.transpose(1, 2).contiguous(),
                delta.transpose(1, 2).contiguous(),
                A,
                B.transpose(1, 2).contiguous(),
                C.transpose(1, 2).contiguous(),
                self.D.float(),
            ).transpose(1, 2)
        else:
            decay = torch.exp(delta[..., None] * A[None, None])
            drive = delta[..., None] * B[:, :, None, :] * u32[..., None]
            hidden = affine_scan(decay, drive)
            y = (hidden * C[:, :, None, :]).sum(dim=-1) + self.D.float() * u32
        y = y * F.silu(gate.float())
        update = self.out_projection(y.to(sequence.dtype))
        return sequence + self.dropout(update)
