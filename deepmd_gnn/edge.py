"""Neighbor-list C++ op wrappers used by graph model export paths."""

import torch

import deepmd_gnn.op  # noqa: F401


def _mm_tensor(mm_types: list[int]) -> torch.Tensor:
    return torch.tensor(mm_types, dtype=torch.int64, device="cpu")


@torch.library.register_fake("deepmd_gnn::dense_edge_index")
def _fake_dense_edge_index(
    nlist: torch.Tensor,
    _atype: torch.Tensor,
    _mm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if nlist.dim() == 2:
        dense_edge_size = nlist.shape[0] * nlist.shape[1]
    elif nlist.dim() == 3:
        dense_edge_size = nlist.shape[0] * nlist.shape[1] * nlist.shape[2]
    else:
        msg = "nlist must be 2D or 3D"
        raise ValueError(msg)
    edge_index = nlist.new_empty((dense_edge_size, 2), dtype=torch.int64)
    edge_mask = nlist.new_empty((dense_edge_size,), dtype=torch.bool)
    return edge_index, edge_mask


def dense_edge_index(
    nlist: torch.Tensor,
    extended_atype: torch.Tensor,
    mm_types: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build dense edge indices and validity mask with the C++ op."""
    return torch.ops.deepmd_gnn.dense_edge_index(
        nlist,
        extended_atype,
        _mm_tensor(mm_types),
    )
