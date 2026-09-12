# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for DeePMD-kit's PyTorch exportable backend integration."""

import json
import sys
import types

import pytest
import torch
from deepmd.pt.model.model.model import (
    BaseModel as PtBaseModel,
)
from deepmd.pt_expt.model.model import (
    BaseModel as PtExptBaseModel,
)
from deepmd.pt_expt.utils.serialization import (
    _build_dynamic_shapes,
    _make_sample_inputs,
    _needs_with_comm_artifact,
)

import deepmd_gnn.pt
import deepmd_gnn.pt_expt
from deepmd_gnn.mace import MaceModel
from deepmd_gnn.nequip import NequipModel


def test_pt_expt_registration_defers_during_partial_mace_import(monkeypatch) -> None:
    """Registration is skipped while the MACE module is partially initialized."""
    monkeypatch.setitem(sys.modules, "deepmd_gnn.mace", types.ModuleType("mace"))
    deepmd_gnn.pt_expt._register()  # noqa: SLF001


def test_pt_registration_defers_during_partial_nequip_import(monkeypatch) -> None:
    """PyTorch registration is skipped while the NeQuIP module initializes."""
    monkeypatch.setitem(sys.modules, "deepmd_gnn.nequip", types.ModuleType("nequip"))
    deepmd_gnn.pt._register()  # noqa: SLF001


def test_pt_and_pt_expt_entry_points_register_models() -> None:
    """Model classes are available through their supported backend registries."""
    assert PtBaseModel.get_class_by_type("mace") is MaceModel
    assert PtBaseModel.get_class_by_type("nequip") is NequipModel
    assert PtExptBaseModel.get_class_by_type("mace") is MaceModel
    with pytest.raises(RuntimeError, match="Unknown model type: nequip"):
        PtExptBaseModel.get_class_by_type("nequip")


def test_mace_export_metadata_defaults() -> None:
    """MACE advertises the default optional pt_expt input metadata."""
    model = MaceModel(type_map=["O", "H"], r_max=6.0, sel=4, hidden_irreps="16x0e")

    assert model.atomic_output_def() is not None
    assert model.has_default_fparam() is False
    assert model.get_default_fparam() is None
    assert model.has_chg_spin_ebd() is False
    assert model.has_default_chg_spin() is False
    assert model.get_default_chg_spin() is None


def test_mace_freeze_disables_cueq_metadata(monkeypatch) -> None:
    """Freeze-time MACE construction exports e3nn metadata for cueq checkpoints."""
    monkeypatch.setattr(sys, "argv", ["dp", "--pt-expt", "freeze"])
    model_params = {
        "type": "mace",
        "type_map": ["O", "H"],
        "r_max": 6.0,
        "sel": 4,
        "hidden_irreps": "16x0e",
        "enable_cueq": True,
    }

    model = MaceModel.get_model(model_params)

    assert model_params["enable_cueq"] is False
    assert json.loads(model.get_model_def_script())["enable_cueq"] is False
    assert model.serialize()["enable_cueq"] is False


def test_mace_pt_expt_reports_with_comm_artifact_requirement() -> None:
    """Multi-layer MACE exposes the descriptor hook used by pt2 export."""
    model = MaceModel(
        type_map=["O", "H"],
        r_max=6.0,
        sel=16,
        hidden_irreps="16x0e",
        num_interactions=2,
    )

    assert model.atomic_model is model
    assert model.descriptor is model
    assert model.has_message_passing_across_ranks()
    assert _needs_with_comm_artifact(model)


def test_mace_pt_expt_export_handles_dynamic_nloc(tmp_path) -> None:
    """Exported MACE graph keeps atom-count dimensions dynamic after save/load."""
    model = MaceModel(type_map=["O", "H"], r_max=6.0, sel=116, hidden_irreps="16x0e")
    sample_inputs = _make_sample_inputs(model, nframes=2, nloc=2)
    extended_coord, extended_atype, nlist, mapping, fparam, aparam, charge_spin = (
        sample_inputs
    )
    traced = model.forward_common_lower_exportable(
        extended_coord,
        extended_atype,
        nlist,
        mapping,
        fparam,
        aparam,
        charge_spin,
        do_atomic_virial=True,
        tracing_mode="symbolic",
        _allow_non_fake_inputs=True,
    )
    exported = torch.export.export(
        traced,
        sample_inputs,
        dynamic_shapes=_build_dynamic_shapes(
            *sample_inputs,
            model_nnei=sum(model.get_sel()),
        ),
        strict=False,
    )
    assert exported._guards_code == []  # noqa: SLF001
    assert not any(
        node.op == "call_function"
        and node.target is torch.ops.aten._assert_scalar.default  # noqa: SLF001
        for node in exported.graph_module.graph.nodes
    )
    assert all(
        value_range.lower <= 1 for value_range in exported.range_constraints.values()
    )

    model_file = tmp_path / "model.pt2"
    torch.export.save(exported, model_file)
    module = torch.export.load(model_file).module()

    for nloc, nnei in ((2, model.get_nnei()), (3, 48)):
        inputs = list(_make_sample_inputs(model, nframes=2, nloc=nloc))
        inputs[2] = inputs[2][..., :nnei]
        runtime_inputs = tuple(inputs)
        outputs = module(*runtime_inputs)
        assert outputs["energy"].shape == (2, nloc, 1)
