# SPDX-License-Identifier: LGPL-3.0-or-later
"""Load OP library and provide edge-index helpers."""

from __future__ import annotations

import platform
from pathlib import Path

import torch

import deepmd_gnn.lib

SHARED_LIB_DIR = Path(deepmd_gnn.lib.__path__[0])


def load_library(module_name: str) -> None:
    """Load OP library.

    Parameters
    ----------
    module_name : str
        Name of the module

    Returns
    -------
    bool
        Whether the library is loaded successfully
    """
    if platform.system() == "Windows":
        ext = ".dll"
        prefix = ""
    else:
        ext = ".so"
        prefix = "lib"

    module_file = (SHARED_LIB_DIR / (prefix + module_name)).with_suffix(ext).resolve()

    torch.ops.load_library(module_file)


load_library("deepmd_gnn")


@torch.jit.script
def _edge_index_pytorch(
    nlist: torch.Tensor,
    atype: torch.Tensor,
    mm: torch.Tensor,
) -> torch.Tensor:
    """Build an edge index from a DeePMD neighbor list with PyTorch ops."""
    if nlist.dim() == 2:
        if atype.dim() != 1:
            msg = "atype must be 1D when nlist is 2D"
            raise ValueError(msg)
        nlist_3d = nlist.unsqueeze(0)
        atype_2d = atype.unsqueeze(0)
    elif nlist.dim() == 3:
        if atype.dim() != 2:
            msg = "atype must be 2D when nlist is 3D"
            raise ValueError(msg)
        if atype.shape[0] != nlist.shape[0]:
            msg = "atype must have the same frame count as nlist"
            raise ValueError(msg)
        nlist_3d = nlist
        atype_2d = atype
    else:
        msg = "nlist must be 2D or 3D"
        raise ValueError(msg)

    device = nlist.device
    nlist_3d = nlist_3d.to(device=device, dtype=torch.int64)
    atype_2d = atype_2d.to(device=device, dtype=torch.int64)
    mm = mm.to(device=device, dtype=torch.int64)

    nf = nlist_3d.shape[0]
    nloc = nlist_3d.shape[1]
    nall = atype_2d.shape[1]

    frame_offsets = torch.arange(
        nf,
        dtype=torch.int64,
        device=device,
    ).view(nf, 1, 1) * nall
    center_index = torch.arange(
        nloc,
        dtype=torch.int64,
        device=device,
    ).view(1, nloc, 1)

    neighbor_index = nlist_3d + frame_offsets
    center_index = center_index + frame_offsets
    center_index = center_index.expand_as(neighbor_index)

    valid_neighbor = nlist_3d >= 0
    atype_flat = atype_2d.reshape(-1)

    # Avoid negative indexing for padded neighbors. These entries are removed
    # by ``valid_neighbor`` before the edge index is materialized.
    safe_neighbor_index = torch.clamp(neighbor_index, min=0)
    neighbor_in_mm = torch.isin(atype_flat[safe_neighbor_index], mm)
    center_in_mm = torch.isin(atype_flat[center_index], mm)
    keep_edge = valid_neighbor & ~(neighbor_in_mm & center_in_mm)

    return torch.stack(
        (
            neighbor_index[keep_edge],
            center_index[keep_edge],
        ),
        dim=1,
    )


@torch.jit.script
def edge_index(
    nlist: torch.Tensor,
    atype: torch.Tensor,
    mm: torch.Tensor,
) -> torch.Tensor:
    """Build an edge index from a DeePMD neighbor list.

    CPU tensors use the existing C++ op, which is faster on CPU. Non-CPU
    tensors use the PyTorch implementation to avoid copying large neighbor-list
    tensors through host memory.
    """
    if nlist.device.type == "cpu":
        return torch.ops.deepmd_gnn.edge_index(nlist, atype, mm)
    return _edge_index_pytorch(nlist, atype, mm)
