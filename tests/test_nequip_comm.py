"""Tests for NequIP layer-wise DeePMD communication."""

import deepmd.pt.model  # noqa: F401
import pytest
import torch

from deepmd_gnn.nequip import NequipModel


def _make_nequip_model() -> NequipModel:
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(20240824)
        return NequipModel(
            type_map=["O", "H"],
            sel=4,
            r_max=5.0,
            num_layers=2,
            precision="float64",
        )
    finally:
        torch.set_default_dtype(old_default_dtype)


def _empty_comm_dict() -> dict[str, torch.Tensor]:
    empty = torch.empty(0, dtype=torch.int64)
    return {
        "send_list": empty,
        "send_proc": empty,
        "recv_proc": empty,
        "send_num": empty,
        "recv_num": empty,
        "communicator": empty,
    }


def test_nequip_comm_branch_matches_regular_lower_without_ghosts(monkeypatch) -> None:
    """Identity communication should reproduce the regular lower path."""

    def fake_border_op(
        send_list: torch.Tensor,
        send_proc: torch.Tensor,
        recv_proc: torch.Tensor,
        send_num: torch.Tensor,
        recv_num: torch.Tensor,
        node_feats: torch.Tensor,
        communicator: torch.Tensor,
        nloc: torch.Tensor,
        nghost: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        _ = (
            send_list,
            send_proc,
            recv_proc,
            send_num,
            recv_num,
            communicator,
            nloc,
            nghost,
        )
        return (node_feats,)

    monkeypatch.setattr(torch.ops.deepmd, "border_op", fake_border_op, raising=False)

    model = _make_nequip_model()
    device = next(model.parameters()).device
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        dtype=torch.float64,
        device=device,
    )
    atype = torch.tensor([[0, 1, 1]], dtype=torch.int64, device=device)
    nlist = torch.tensor(
        [[[1, 2, -1, -1], [0, 2, -1, -1], [0, 1, -1, -1]]],
        dtype=torch.int64,
        device=device,
    )

    regular = model.forward_lower(coord, atype, nlist)
    communicated = model.forward_lower(
        coord,
        atype,
        nlist,
        comm_dict=_empty_comm_dict(),
    )

    for key in ["atom_energy", "energy", "extended_force", "virial"]:
        torch.testing.assert_close(
            communicated[key],
            regular[key],
            atol=1e-6,
            rtol=0.0,
        )


def test_nequip_lower_with_ghosts_requires_comm_or_mapping() -> None:
    """Ghost lower inference should not rebuild a multi-hop NequIP nlist."""
    model = _make_nequip_model()
    device = next(model.parameters()).device
    coord = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
        ],
        dtype=torch.float64,
        device=device,
    )
    atype = torch.tensor([[0, 1, 1, 1]], dtype=torch.int64, device=device)
    nlist = torch.tensor(
        [[[1, 2, -1, -1], [0, 2, -1, -1], [0, 1, -1, -1]]],
        dtype=torch.int64,
        device=device,
    )

    with pytest.raises(ValueError, match="requires either comm_dict"):
        model.forward_lower(coord, atype, nlist)
