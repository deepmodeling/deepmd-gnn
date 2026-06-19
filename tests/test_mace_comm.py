"""Tests for MACE layer-wise DeePMD communication."""

import deepmd.pt.model  # noqa: F401
import torch

from deepmd_gnn.mace import MaceModel


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


def test_mace_comm_branch_matches_regular_lower_without_ghosts(monkeypatch) -> None:
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

    torch.manual_seed(20240823)
    model = MaceModel(type_map=["O", "H"], sel=4, r_max=5.0, precision="float64")
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        dtype=torch.float64,
    )
    atype = torch.tensor([[0, 1, 1]], dtype=torch.int64)
    nlist = torch.tensor(
        [[[1, 2, -1, -1], [0, 2, -1, -1], [0, 1, -1, -1]]],
        dtype=torch.int64,
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
