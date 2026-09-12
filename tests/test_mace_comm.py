"""Tests for MACE layer-wise DeePMD communication."""

import deepmd.pt.model  # noqa: F401
import pytest
import torch

import deepmd_gnn.mace as mace_module
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
    torch.testing.assert_close(captured["node_feats"][2:], torch.zeros_like(node_feats))
    assert captured["nlocal"] is comm_dict["nlocal"]
    assert captured["nghost"] is comm_dict["nghost"]


def test_mace_runtime_comm_pads_ghost_slots(monkeypatch) -> None:
    """Runtime communication pads local node features before border_op."""
    captured: dict[str, torch.Tensor] = {}

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
        _ = send_list, send_proc, recv_proc, send_num, recv_num, communicator
        captured["node_feats"] = node_feats
        captured["nloc"] = nloc
        captured["nghost"] = nghost
        return (node_feats,)

    monkeypatch.setattr(torch.ops.deepmd, "border_op", fake_border_op, raising=False)

    model = MaceModel(type_map=["O", "H"], sel=4, r_max=5.0, precision="float64")
    node_feats = torch.arange(6, dtype=torch.float64).view(2, 3)
    exchanged = model.communicate_node_features(node_feats, 2, 4, _empty_comm_dict())

    assert exchanged.shape == (4, 3)
    torch.testing.assert_close(captured["node_feats"][:2], node_feats)
    torch.testing.assert_close(captured["node_feats"][2:], torch.zeros_like(node_feats))
    assert captured["nloc"].item() == 2
    assert captured["nghost"].item() == 2


def test_mace_forward_common_lower_exportable_with_comm_traces(
    monkeypatch,
) -> None:
    """The with-comm export hook traces through comm tensors and restores flags."""

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
        _ = send_list, send_proc, recv_proc, send_num, recv_num, communicator
        _ = nlocal, nghost
        return node_feats

    captured: dict[str, object] = {}

    def fake_make_fx(fn, **kwargs):
        captured["make_fx_kwargs"] = kwargs

        def runner(*args):
            return fn(*args)

        return runner

    def fake_clear_export_guards_once(traced: object) -> None:
        captured["traced"] = traced

    monkeypatch.setattr(torch.ops.deepmd_export, "border_op", fake_border_op)
    monkeypatch.setattr(mace_module, "make_fx", fake_make_fx)
    monkeypatch.setattr(
        mace_module,
        "_clear_export_guards_once",
        fake_clear_export_guards_once,
    )

    model = MaceModel(type_map=["O", "H"], sel=4, r_max=5.0, precision="float64")
    extended_coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    extended_atype = torch.tensor([[0, 1]], dtype=torch.int64)
    nlist = torch.tensor([[[1, -1, -1, -1], [0, -1, -1, -1]]], dtype=torch.int64)
    comm_dict = _empty_export_comm_dict()

    traced = model.forward_common_lower_exportable_with_comm(
        extended_coord,
        extended_atype,
        nlist,
        None,
        None,
        None,
        None,
        comm_dict["send_list"],
        comm_dict["send_proc"],
        comm_dict["recv_proc"],
        comm_dict["send_num"],
        comm_dict["recv_num"],
        comm_dict["communicator"],
        torch.tensor(2, dtype=torch.int32),
        torch.tensor(0, dtype=torch.int32),
        tracing_mode="symbolic",
    )

    assert "energy" in traced
    assert captured["traced"] is traced
    assert captured["make_fx_kwargs"] == {"tracing_mode": "symbolic"}
    assert model._use_exportable_edge_index is False  # noqa: SLF001
    assert model._use_exportable_border_op is False  # noqa: SLF001


def test_mace_runtime_comm_rejects_multi_frame_inputs() -> None:
    """Runtime comm lower inference accepts only one flattened frame."""
    model = MaceModel(type_map=["O", "H"], sel=4, r_max=5.0, precision="float64")
    edge_index = torch.empty((2, 0), dtype=torch.int64)
    node_attrs = torch.zeros((2, 2), dtype=torch.float64)
    positions = torch.zeros((2, 3), dtype=torch.float64)
    shifts = torch.empty((0, 3), dtype=torch.float64)

    with pytest.raises(
        ValueError,
        match="MACE comm_dict lower inference only supports one frame",
    ):
        model.forward_mace_with_comm(
            2,
            4,
            edge_index,
            node_attrs,
            positions,
            shifts,
            _empty_comm_dict(),
        )
