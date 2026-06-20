# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for DeePMD-kit's PyTorch exportable backend integration."""

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
)

import deepmd_gnn.pt
import deepmd_gnn.pt_expt  # noqa: F401
from deepmd_gnn.mace import MaceModel
from deepmd_gnn.nequip import NequipModel


def test_pt_and_pt_expt_entry_points_register_models() -> None:
    """Model classes are available through both PyTorch backend registries."""
    assert PtBaseModel.get_class_by_type("mace") is MaceModel
    assert PtBaseModel.get_class_by_type("nequip") is NequipModel
    assert PtExptBaseModel.get_class_by_type("mace") is MaceModel
    assert PtExptBaseModel.get_class_by_type("nequip") is NequipModel


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
