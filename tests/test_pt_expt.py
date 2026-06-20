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
    extended_coord, extended_atype, nlist, mapping, fparam, aparam = sample_inputs
    traced = model.forward_common_lower_exportable(
        extended_coord,
        extended_atype,
        nlist,
        mapping,
        fparam,
        aparam,
        do_atomic_virial=True,
        tracing_mode="symbolic",
        _allow_non_fake_inputs=True,
    )
    exported = torch.export.export(
        traced,
        sample_inputs,
        dynamic_shapes=_build_dynamic_shapes(*sample_inputs),
        strict=False,
    )
    assert exported._guards_code == []  # noqa: SLF001

    model_file = tmp_path / "model.pt2"
    torch.export.save(exported, model_file)
    module = torch.export.load(model_file).module()

    for nloc in (2, 3):
        inputs = _make_sample_inputs(model, nframes=2, nloc=nloc)
        outputs = module(*inputs)
        assert outputs["energy"].shape == (2, nloc, 1)
