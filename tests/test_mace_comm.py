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


def _empty_export_comm_dict() -> dict[str, torch.Tensor]:
    comm_dict = _empty_comm_dict()
    comm_dict["nlocal"] = torch.tensor(2, dtype=torch.int32)
    comm_dict["nghost"] = torch.tensor(2, dtype=torch.int32)
    return comm_dict


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


def test_mace_exportable_comm_uses_deepmd_export_border_op(monkeypatch) -> None:
    """The pt2 with-comm trace must use the exportable border op namespace."""
    captured: dict[str, torch.Tensor] = {}

    def fake_border_op(
        send_list: torch.Tensor,
        send_proc: torch.Tensor,
        recv_proc: torch.Tensor,
        send_num: torch.Tensor,
        recv_num: torch.Tensor,
        node_feats: torch.Tensor,
        communicator: torch.Tensor,
        nlocal: torch.Tensor,
        nghost: torch.Tensor,
    ) -> torch.Tensor:
        captured.update(
            {
                "send_list": send_list,
                "send_proc": send_proc,
                "recv_proc": recv_proc,
                "send_num": send_num,
                "recv_num": recv_num,
                "node_feats": node_feats,
                "communicator": communicator,
                "nlocal": nlocal,
                "nghost": nghost,
            },
        )
        return node_feats + 1.0

    monkeypatch.setattr(
        torch.ops.deepmd_export,
        "border_op",
        fake_border_op,
        raising=False,
    )

    model = MaceModel(type_map=["O", "H"], sel=4, r_max=5.0, precision="float64")
    monkeypatch.setattr(model, "_use_exportable_border_op", True)
    node_feats = torch.arange(6, dtype=torch.float64).view(2, 3)
    comm_dict = _empty_export_comm_dict()

    exchanged = model.communicate_node_features(node_feats, 2, 4, comm_dict)

    assert exchanged.shape == (4, 3)
    torch.testing.assert_close(exchanged[:2], node_feats + 1.0)
    torch.testing.assert_close(captured["node_feats"][:2], node_feats)
    torch.testing.assert_close(captured["node_feats"][2:], torch.zeros(2, 3))
    assert captured["nlocal"] is comm_dict["nlocal"]
    assert captured["nghost"] is comm_dict["nghost"]
