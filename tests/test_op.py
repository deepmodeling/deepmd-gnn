"""Test custom operations."""

import pytest
import torch

from deepmd_gnn.op import _edge_index_pytorch
from deepmd_gnn.op import edge_index as build_edge_index


def test_one_frame() -> None:
    """Test one frame."""
    nlist_ff = torch.tensor(
        [
            [1, 2, -1, -1],
            [2, 0, -1, -1],
            [0, 1, -1, -1],
        ],
        dtype=torch.int64,
        device="cpu",
    )
    extended_atype_ff = torch.tensor(
        [0, 1, 2],
        dtype=torch.int64,
        device="cpu",
    )
    mm_types = [1, 2]
    expected_edge_index = torch.tensor(
        [
            [1, 0],
            [2, 0],
            [0, 1],
            [0, 2],
        ],
        dtype=torch.int64,
        device="cpu",
    )

    mm_tensor = torch.tensor(mm_types, dtype=torch.int64, device="cpu")
    edge_index = torch.ops.deepmd_gnn.edge_index(
        nlist_ff,
        extended_atype_ff,
        mm_tensor,
    )
    wrapped_edge_index = build_edge_index(
        nlist_ff,
        extended_atype_ff,
        mm_tensor,
    )
    python_edge_index = _edge_index_pytorch(
        nlist_ff,
        extended_atype_ff,
        mm_tensor,
    )

    assert torch.equal(edge_index, expected_edge_index)
    assert torch.equal(wrapped_edge_index, expected_edge_index)
    assert torch.equal(python_edge_index, expected_edge_index)


def test_two_frame() -> None:
    """Test one frame."""
    nlist = torch.tensor(
        [
            [
                [1, 2, -1, -1],
                [2, 0, -1, -1],
                [0, 1, -1, -1],
            ],
            [
                [1, 2, -1, -1],
                [2, 0, -1, -1],
                [0, 1, -1, -1],
            ],
        ],
        dtype=torch.int64,
        device="cpu",
    )
    extended_atype = torch.tensor(
        [
            [0, 1, 2],
            [0, 1, 2],
        ],
        dtype=torch.int64,
        device="cpu",
    )
    mm_types = [1, 2]
    expected_edge_index = torch.tensor(
        [
            [1, 0],
            [2, 0],
            [0, 1],
            [0, 2],
            [4, 3],
            [5, 3],
            [3, 4],
            [3, 5],
        ],
        dtype=torch.int64,
        device="cpu",
    )

    mm_tensor = torch.tensor(mm_types, dtype=torch.int64, device="cpu")
    edge_index = torch.ops.deepmd_gnn.edge_index(
        nlist,
        extended_atype,
        mm_tensor,
    )
    wrapped_edge_index = build_edge_index(
        nlist,
        extended_atype,
        mm_tensor,
    )
    python_edge_index = _edge_index_pytorch(
        nlist,
        extended_atype,
        mm_tensor,
    )

    assert torch.equal(edge_index, expected_edge_index)
    assert torch.equal(wrapped_edge_index, expected_edge_index)
    assert torch.equal(python_edge_index, expected_edge_index)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_python_edge_index_cuda() -> None:
    """Test the PyTorch implementation keeps CUDA tensors on CUDA."""
    nlist = torch.tensor(
        [
            [1, 2, -1, -1],
            [2, 0, -1, -1],
            [0, 1, -1, -1],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    extended_atype = torch.tensor(
        [0, 1, 2],
        dtype=torch.int64,
        device="cuda",
    )
    mm_tensor = torch.tensor([1, 2], dtype=torch.int64, device="cuda")
    expected_edge_index = torch.tensor(
        [
            [1, 0],
            [2, 0],
            [0, 1],
            [0, 2],
        ],
        dtype=torch.int64,
        device="cuda",
    )

    python_edge_index = build_edge_index(
        nlist,
        extended_atype,
        mm_tensor,
    )

    assert python_edge_index.is_cuda
    assert torch.equal(python_edge_index, expected_edge_index)
