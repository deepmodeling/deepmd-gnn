# SPDX-License-Identifier: LGPL-3.0-or-later
"""Correctness tests for the original MACE energy head as a DeePMD fitting."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
from deepmd.pt.model.atomic_model.energy_atomic_model import DPEnergyAtomicModel
from deepmd.pt.model.model import get_model
from deepmd.pt.train.training import get_model_for_wrapper
from deepmd.pt.train.wrapper import ModelWrapper
from deepmd.pt.utils import env
from deepmd.pt.utils.multi_task import preprocess_shared_params

import deepmd_gnn.pt  # noqa: F401
import deepmd_gnn.pt as pt_mod
from deepmd_gnn.mace import MaceModel
from deepmd_gnn.mace_checkpoint import (
    MaceEnergyHead,
    ZeroPairRepulsion,
    load_native_mace_checkpoint,
    product_layer_feature_dims,
)
from deepmd_gnn.mace_descriptor import MaceDescriptor
from deepmd_gnn.mace_ener import MaceEnergyFitting, split_packed_layer_features
from tests.test_mace_descriptor import _inputs, _write_mace_checkpoint

if TYPE_CHECKING:
    from pathlib import Path


def _energy_model(checkpoint: Path, *, sel: int = 16) -> torch.nn.Module:
    return get_model(
        {
            "type": "standard",
            "type_map": ["H", "O"],
            "descriptor": {
                "type": "mace",
                "model_path": str(checkpoint),
                "sel": sel,
            },
            "fitting_net": {
                "type": "mace_ener",
                "model_path": str(checkpoint),
            },
        },
    )


def _sample_coord(*, pbc: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    if pbc:
        coord = torch.tensor(
            [[[0.1, 0.2, 0.3], [3.8, 0.2, 0.3], [0.2, 3.7, 0.4]]],
            dtype=torch.float64,
            device=env.DEVICE,
        )
        box = (torch.eye(3, dtype=torch.float64, device=env.DEVICE) * 4.0).reshape(1, 9)
        return coord, box
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-0.2, 1.0, 0.3]]],
        dtype=torch.float64,
        device=env.DEVICE,
    )
    return coord, None


def _atype() -> torch.Tensor:
    return torch.tensor([[1, 0, 0]], dtype=torch.int64, device=env.DEVICE)


def _native_atomic_energy(
    checkpoint: Path,
    descriptor: MaceDescriptor,
    coord: torch.Tensor,
    box: torch.Tensor | None,
) -> torch.Tensor:
    native = load_native_mace_checkpoint(checkpoint, device=env.DEVICE)
    head = MaceEnergyHead(native)
    coord_ext, atype_ext, nlist, mapping = _inputs(descriptor, coord, box)
    last_layer, packed, pair_energy = descriptor._mace_graph_outputs(  # noqa: SLF001
        coord_ext,
        atype_ext,
        nlist,
        mapping,
    )
    del last_layer
    nf, nloc, _ = nlist.shape
    source_dtype = next(native.parameters()).dtype
    node_attrs = torch.zeros(
        (nf * nloc, descriptor.ntypes),
        dtype=source_dtype,
        device=env.DEVICE,
    )
    node_attrs.scatter_(
        -1,
        atype_ext[:, :nloc].reshape(-1, 1),
        1,
    )
    layers = split_packed_layer_features(packed, descriptor.layer_feature_dims)
    atom_energy = head.node_energy(
        node_attrs,
        layers,
        pair_energy.reshape(nf * nloc),
    )
    return atom_energy.view(nf, nloc, 1)


@pytest.fixture
def mace_checkpoint(tmp_path: Path) -> Path:
    """Create a tiny native MACE checkpoint."""
    return _write_mace_checkpoint(
        tmp_path / "small_mace.model",
        keep_last_layer_irreps=False,
    )


@pytest.fixture
def pair_mace_checkpoint(tmp_path: Path) -> Path:
    """Create a tiny checkpoint that includes pair repulsion."""
    return _write_mace_checkpoint(
        tmp_path / "pair_mace.model",
        keep_last_layer_irreps=False,
        pair_repulsion=True,
    )


@pytest.mark.parametrize("pbc", [False, True])
@pytest.mark.parametrize("fixture_name", ["mace_checkpoint", "pair_mace_checkpoint"])
def test_energy_matches_native_head(
    pbc: bool,
    fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    """EnergyModel(mace + mace_ener) matches the original readout on the same graph."""
    checkpoint = request.getfixturevalue(fixture_name)
    model = _energy_model(checkpoint)
    assert isinstance(model.get_fitting_net(), MaceEnergyFitting)
    coord, box = _sample_coord(pbc=pbc)
    atype = _atype()
    pred = model(coord.reshape(1, -1), atype, box=box)
    descriptor = model.get_descriptor()
    native_atomic = _native_atomic_energy(checkpoint, descriptor, coord, box)
    torch.testing.assert_close(
        pred["atom_energy"].reshape_as(native_atomic),
        native_atomic,
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(
        pred["energy"].reshape_as(native_atomic.sum(dim=1)),
        native_atomic.sum(dim=1),
        rtol=1e-6,
        atol=1e-7,
    )


def test_energy_matches_mace_model_nopbc(mace_checkpoint: Path) -> None:
    """No-PBC total energy matches native ``MaceModel`` on the same checkpoint."""
    energy_model = _energy_model(mace_checkpoint)
    native = load_native_mace_checkpoint(mace_checkpoint, device=env.DEVICE)
    mace_model = MaceModel(
        type_map=["H", "O"],
        sel=16,
        r_max=3.0,
        num_radial_basis=4,
        num_cutoff_basis=5,
        max_ell=1,
        num_interactions=2,
        hidden_irreps="2x0e + 2x1o",
        correlation=2,
        MLP_irreps="4x0e",
        radial_MLP=[8, 8],
        std=1.0,
        avg_num_neighbors=4.0,
        precision="float64",
    )
    mace_model.model = native
    coord, box = _sample_coord(pbc=False)
    atype = _atype()
    split_energy = energy_model(coord.reshape(1, -1), atype, box=box)["energy"]
    native_energy = mace_model(coord.reshape(1, -1), atype, box=box)["energy"]
    torch.testing.assert_close(split_energy, native_energy, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("pbc", [False, True])
def test_force_matches_energy_autograd(mace_checkpoint: Path, pbc: bool) -> None:
    """Forces are derivatives of the original energy, not a new force head."""
    model = _energy_model(mace_checkpoint)
    coord, box = _sample_coord(pbc=pbc)
    coord = coord.clone().requires_grad_(requires_grad=True)
    atype = _atype()
    energy = model(coord.reshape(1, -1), atype, box=box)["energy"]
    (force,) = torch.autograd.grad(energy.sum(), coord, create_graph=False)
    pred_force = model(
        coord.detach().reshape(1, -1),
        atype,
        box=box,
    )["force"]
    torch.testing.assert_close(
        pred_force.reshape_as(force),
        -force,
        rtol=1e-5,
        atol=1e-6,
    )


def test_property_zeroe_unchanged(mace_checkpoint: Path) -> None:
    """Packed energy extras do not change the property 0e channels."""
    energy_model = _energy_model(mace_checkpoint)
    property_model = get_model(
        {
            "type": "standard",
            "type_map": ["H", "O"],
            "descriptor": {
                "type": "mace",
                "model_path": str(mace_checkpoint),
                "sel": 16,
            },
            "fitting_net": {
                "type": "property",
                "property_name": "band_gap",
                "task_dim": 1,
                "neuron": [8],
                "precision": "float64",
            },
        },
    )
    coord, box = _sample_coord(pbc=False)
    energy_desc = energy_model.get_descriptor()
    property_desc = property_model.get_descriptor()
    energy_ext, energy_atype, energy_nlist, energy_mapping = _inputs(
        energy_desc,
        coord,
        box,
    )
    property_ext, property_atype, property_nlist, property_mapping = _inputs(
        property_desc,
        coord,
        box,
    )
    energy_zeroe = energy_desc(
        energy_ext,
        energy_atype,
        energy_nlist,
        mapping=energy_mapping,
    )[0]
    property_zeroe = property_desc(
        property_ext,
        property_atype,
        property_nlist,
        mapping=property_mapping,
    )[0]
    torch.testing.assert_close(energy_zeroe, property_zeroe)
    packed = energy_desc(
        energy_ext,
        energy_atype,
        energy_nlist,
        mapping=energy_mapping,
    )[2]
    assert packed is not None
    assert packed.shape[-1] == sum(energy_desc.layer_feature_dims)


def test_share_params_aliases_backbone_not_energy_head(mace_checkpoint: Path) -> None:
    """Level-0 sharing aliases the backbone and leaves energy heads distinct."""
    energy_desc = MaceDescriptor(
        model_path=mace_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    property_desc = MaceDescriptor(
        model_path=mace_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    energy_fit = MaceEnergyFitting(
        ntypes=2,
        dim_descrpt=energy_desc.get_dim_out(),
        type_map=["H", "O"],
        model_path=mace_checkpoint,
    )
    extra_fit = MaceEnergyFitting(
        ntypes=2,
        dim_descrpt=energy_desc.get_dim_out(),
        type_map=["H", "O"],
        model_path=mace_checkpoint,
    )
    assert energy_desc.backbone is not property_desc.backbone
    property_desc.share_params(energy_desc, shared_level=0)
    assert property_desc.backbone is energy_desc.backbone
    backbone_param = next(energy_desc.backbone.parameters())
    backbone_param.data.add_(0.25)
    shared_param = next(property_desc.backbone.parameters())
    torch.testing.assert_close(shared_param, backbone_param)
    assert extra_fit.head is not energy_fit.head
    energy_param = next(energy_fit.head.parameters())
    extra_param = next(extra_fit.head.parameters())
    assert energy_param.data_ptr() != extra_param.data_ptr()


def test_get_model_multitask_shares_descriptor(mace_checkpoint: Path) -> None:
    """shared_dict plus share_params aliases the MACE backbone across branches."""
    model_config = {
        "shared_dict": {
            "type_map": ["H", "O"],
            "mace_descriptor": {
                "type": "mace",
                "model_path": str(mace_checkpoint),
                "sel": 16,
            },
        },
        "model_dict": {
            "force_field": {
                "type_map": "type_map",
                "descriptor": "mace_descriptor",
                "fitting_net": {
                    "type": "mace_ener",
                    "model_path": str(mace_checkpoint),
                },
            },
            "band_gap": {
                "type_map": "type_map",
                "descriptor": "mace_descriptor",
                "fitting_net": {
                    "type": "property",
                    "property_name": "band_gap",
                    "task_dim": 1,
                    "intensive": True,
                    "neuron": [8],
                    "precision": "float64",
                },
            },
        },
    }
    expanded, shared_links = preprocess_shared_params(model_config)
    models = get_model_for_wrapper(expanded)
    wrapper = ModelWrapper(models)
    wrapper.share_params(
        shared_links,
        model_key_prob_map={"force_field": 1.0, "band_gap": 1.0},
        resume=True,
    )
    energy_desc = wrapper.model["force_field"].get_descriptor()
    property_desc = wrapper.model["band_gap"].get_descriptor()
    assert isinstance(energy_desc, MaceDescriptor)
    assert isinstance(property_desc, MaceDescriptor)
    assert energy_desc.backbone is property_desc.backbone
    fitting = wrapper.model["force_field"].get_fitting_net()
    assert isinstance(fitting, MaceEnergyFitting)
    coord, box = _sample_coord(pbc=False)
    atype = _atype()
    energy = wrapper.model["force_field"](coord.reshape(1, -1), atype, box=box)[
        "energy"
    ]
    prop = wrapper.model["band_gap"](coord.reshape(1, -1), atype, box=box)["band_gap"]
    assert energy.shape == (1, 1)
    assert prop.shape[0] == 1


def test_serialized_energy_model_roundtrip(
    mace_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Saved energy models restore the original head without the native pickle."""
    original = _energy_model(mace_checkpoint)
    serialized = original.serialize()
    restored = type(original).deserialize(serialized)
    coord, box = _sample_coord(pbc=True)
    atype = _atype()
    torch.testing.assert_close(
        restored(coord.reshape(1, -1), atype, box=box)["energy"],
        original(coord.reshape(1, -1), atype, box=box)["energy"],
        rtol=1e-6,
        atol=1e-7,
    )
    missing = tmp_path / "missing_source.model"
    assert not missing.exists()
    restored_from_config = get_model(
        {
            "type": "standard",
            "type_map": ["H", "O"],
            "descriptor": {
                "type": "mace",
                "sel": 16,
                "model_path": str(missing),
                "config": original.get_descriptor().config,
            },
            "fitting_net": {
                "type": "mace_ener",
                "model_path": str(missing),
                "config": original.get_fitting_net().config,
            },
        },
    )
    assert restored_from_config.get_descriptor().config["num_interactions"] == 2


def _energy_sample(
    coord: torch.Tensor,
    atype: torch.Tensor,
    box: torch.Tensor | None,
    energy: torch.Tensor,
) -> dict[str, torch.Tensor]:
    nloc = int(atype.shape[1])
    n_h = int((atype == 0).sum())
    n_o = int((atype == 1).sum())
    sample: dict[str, torch.Tensor] = {
        "coord": coord.reshape(1, -1).detach(),
        "atype": atype.detach(),
        "energy": energy.detach().reshape(1, 1),
        "natoms": torch.tensor(
            [[nloc, nloc, n_h, n_o]],
            dtype=torch.int64,
            device=atype.device,
        ),
        "find_energy": torch.tensor(1.0, device=atype.device),
    }
    if box is not None:
        sample["box"] = box.detach()
    return sample


def test_out_bias_starts_zero(mace_checkpoint: Path) -> None:
    """Pretrained e0 is not copied into DeePMD out_bias."""
    model = _energy_model(mace_checkpoint)
    assert torch.count_nonzero(model.get_out_bias()) == 0


def test_apply_out_stat_adds_residual_not_second_e0(mace_checkpoint: Path) -> None:
    """Nonzero out_bias is a residual on top of native MACE energy."""
    model = _energy_model(mace_checkpoint)
    coord, box = _sample_coord(pbc=False)
    atype = _atype()
    before = model(coord.reshape(1, -1), atype, box=box)["energy"]
    bias = model.get_out_bias().clone()
    bias[0, 0, 0] = 1.0
    bias[0, 1, 0] = 2.0
    model.set_out_bias(bias)
    after = model(coord.reshape(1, -1), atype, box=box)["energy"]
    # atype is O, H, H → 2 + 1 + 1
    torch.testing.assert_close(after, before + 4.0, rtol=1e-6, atol=1e-7)


def test_compute_or_load_out_stat_does_not_stack_data_bias(
    mace_checkpoint: Path,
) -> None:
    """Training stats must not LS-fit a second vacuum energy on top of e0."""
    model = _energy_model(mace_checkpoint)
    coord, box = _sample_coord(pbc=True)
    atype = _atype()
    before = model(coord.reshape(1, -1), atype, box=box)["energy"]
    e0_before = (
        model.get_fitting_net().head.atomic_energies_fn.atomic_energies.detach().clone()
    )
    sample = _energy_sample(coord, atype, box, before + 100.0)
    model.compute_or_load_stat(lambda: [sample])
    after = model(coord.reshape(1, -1), atype, box=box)["energy"]
    torch.testing.assert_close(after, before, rtol=1e-6, atol=1e-7)
    assert torch.count_nonzero(model.get_out_bias()) == 0
    torch.testing.assert_close(
        model.get_fitting_net().head.atomic_energies_fn.atomic_energies,
        e0_before,
    )


def test_change_by_statistic_is_residual_shift(mace_checkpoint: Path) -> None:
    """Finetune change-bias fits E_data - E_mace, it does not add e0 twice."""
    model = _energy_model(mace_checkpoint)
    coord, box = _sample_coord(pbc=True)
    atype = _atype()
    predicted = model(coord.reshape(1, -1), atype, box=box)["energy"].detach()
    e0_before = (
        model.get_fitting_net().head.atomic_energies_fn.atomic_energies.detach().clone()
    )
    model.change_out_bias(
        [_energy_sample(coord, atype, box, predicted)],
        bias_adjust_mode="change-by-statistic",
    )
    torch.testing.assert_close(
        model(coord.reshape(1, -1), atype, box=box)["energy"],
        predicted,
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        model.get_fitting_net().head.atomic_energies_fn.atomic_energies,
        e0_before,
    )
    shifted = predicted + 3.0
    model.change_out_bias(
        [_energy_sample(coord, atype, box, shifted)],
        bias_adjust_mode="change-by-statistic",
    )
    after = model(coord.reshape(1, -1), atype, box=box)["energy"]
    torch.testing.assert_close(after, shifted, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        model.get_fitting_net().head.atomic_energies_fn.atomic_energies,
        e0_before,
    )
    assert torch.count_nonzero(model.get_out_bias()) > 0


def test_set_by_statistic_replaces_e0_not_out_bias(mace_checkpoint: Path) -> None:
    """Explicit set-by-statistic writes e0, matching MaceModel, not a second bias."""
    model = _energy_model(mace_checkpoint)
    coord, box = _sample_coord(pbc=True)
    atype = _atype()
    predicted = model(coord.reshape(1, -1), atype, box=box)["energy"].detach()
    e0_before = (
        model.get_fitting_net().head.atomic_energies_fn.atomic_energies.detach().clone()
    )
    model.change_out_bias(
        [_energy_sample(coord, atype, box, predicted + 6.0)],
        bias_adjust_mode="set-by-statistic",
    )
    assert torch.count_nonzero(model.get_out_bias()) == 0
    assert not torch.allclose(
        model.get_fitting_net().head.atomic_energies_fn.atomic_energies,
        e0_before,
    )


def test_mace_ener_rejects_invalid_construction(mace_checkpoint: Path) -> None:
    """Constructor guards keep the original head aligned with the descriptor."""
    with pytest.raises(ValueError, match="type_map"):
        MaceEnergyFitting(ntypes=2, dim_descrpt=2)
    with pytest.raises(ValueError, match="ntypes"):
        MaceEnergyFitting(ntypes=1, dim_descrpt=2, type_map=["H", "O"])
    with pytest.raises(ValueError, match="mixed_types"):
        MaceEnergyFitting(
            ntypes=2,
            dim_descrpt=2,
            type_map=["H", "O"],
            mixed_types=False,
        )
    with pytest.raises(ValueError, match="atomic-number"):
        MaceEnergyFitting(
            ntypes=2,
            dim_descrpt=2,
            type_map=["O", "H"],
            model_path=mace_checkpoint,
        )
    with pytest.raises(FileNotFoundError, match="not found"):
        MaceEnergyFitting(
            ntypes=2,
            dim_descrpt=2,
            type_map=["H", "O"],
            model_path="/no/such/mace.model",
        )
    with pytest.raises(ValueError, match="Exactly one"):
        MaceEnergyFitting(ntypes=2, dim_descrpt=2, type_map=["H", "O"])
    filled: dict = {}
    fitting = MaceEnergyFitting(
        ntypes=2,
        dim_descrpt=2,
        type_map=["H", "O"],
        model_path=mace_checkpoint,
        config=filled,
        trainable=False,
    )
    assert filled["num_interactions"] == 2
    assert fitting.get_type_map() == ["H", "O"]
    assert fitting.get_dim_fparam() == 0
    assert fitting.has_default_fparam() is False
    assert fitting.get_default_fparam() is None
    assert fitting.get_dim_aparam() == 0
    assert fitting.get_sel_type() == []
    fitting.set_case_embd(0)
    fitting.compute_input_stats([])
    fitting.compute_output_stats([])
    with pytest.raises(NotImplementedError, match="type_map"):
        fitting.change_type_map(["H"])
    with pytest.raises(ValueError, match="g2"):
        fitting.forward(torch.zeros(1, 1, 2), torch.zeros(1, 1, dtype=torch.int64))
    with pytest.raises(ValueError, match="width"):
        split_packed_layer_features(torch.zeros(1, 1, 3), [1, 1])
    with pytest.raises(ValueError, match="type_map mismatch"):
        MaceEnergyFitting(
            ntypes=2,
            dim_descrpt=2,
            type_map=["H", "O"],
            config={**fitting.config, "type_map": ["O", "H"]},
        )
    with pytest.raises(ValueError, match="serialized"):
        MaceEnergyFitting.deserialize({"@class": "Nope", "type": "mace_ener"})
    with pytest.raises(ValueError, match="serialized"):
        MaceEnergyFitting.deserialize({"@class": "Fitting", "type": "ener"})
    serialized = fitting.serialize()
    energy_key = "atomic_energies_fn.atomic_energies"
    serialized["@variables"][energy_key] = serialized["@variables"][energy_key].reshape(
        -1,
    )
    restored = MaceEnergyFitting.deserialize(serialized)
    assert restored.head.atomic_energies_fn.atomic_energies.numel() == 2
    with pytest.raises(ValueError, match="no product"):
        product_layer_feature_dims(SimpleNamespace(products=None))
    with pytest.raises(ValueError, match="no product"):
        product_layer_feature_dims(SimpleNamespace(products=[]))
    with pytest.raises(ValueError, match="does not expose"):
        product_layer_feature_dims(
            SimpleNamespace(products=[SimpleNamespace(linear=None)]),
        )


def test_plugin_patches_cover_non_mace_and_hessian(mace_checkpoint: Path) -> None:
    """Plugin wrappers keep vanilla EnergyModel paths and optional hessian."""
    pt_mod.load()
    install_atomic = "_install_mace_ener_energy_atomic_model"
    install_standard = "_install_mace_ener_standard_model"
    getattr(pt_mod, install_atomic)()
    getattr(pt_mod, install_standard)()
    property_model = get_model(
        {
            "type": "standard",
            "type_map": ["H", "O"],
            "descriptor": {
                "type": "mace",
                "model_path": str(mace_checkpoint),
                "sel": 16,
            },
            "fitting_net": {
                "type": "property",
                "property_name": "band_gap",
                "task_dim": 1,
                "neuron": [8],
            },
        },
    )
    assert property_model.get_fitting_net().task_dim == 1
    with pytest.raises(
        (TypeError, ValueError, RuntimeError, AttributeError, AssertionError),
    ):
        DPEnergyAtomicModel.__init__(object(), None, object(), ["H"])
    dummy = type("Dummy", (), {"fitting_net": object()})()
    with pytest.raises(
        (TypeError, ValueError, RuntimeError, AttributeError, AssertionError),
    ):
        DPEnergyAtomicModel.compute_or_load_out_stat(dummy)
    with pytest.raises(
        (TypeError, ValueError, RuntimeError, AttributeError, AssertionError),
    ):
        DPEnergyAtomicModel.change_out_bias(dummy, [])
    energy_model = _energy_model(mace_checkpoint)
    with pytest.raises(RuntimeError, match="Unknown bias_adjust_mode"):
        energy_model.atomic_model.change_out_bias(
            [],
            bias_adjust_mode="not-a-mode",
        )
    with patch(
        "deepmd.pt.utils.stat.compute_output_stats",
        return_value=({}, {}),
    ):
        energy_model.atomic_model.change_out_bias(
            [],
            bias_adjust_mode="set-by-statistic",
        )
    hessian_model = get_model(
        {
            "type": "standard",
            "type_map": ["H", "O"],
            "hessian_mode": True,
            "descriptor": {
                "type": "mace",
                "model_path": str(mace_checkpoint),
                "sel": 16,
            },
            "fitting_net": {
                "type": "mace_ener",
                "model_path": str(mace_checkpoint),
            },
        },
    )
    assert hessian_model is not None
    energy_model.atomic_model.compute_or_load_out_stat([])
    one_layer = _write_mace_checkpoint(
        mace_checkpoint.parent / "one_layer.model",
        keep_last_layer_irreps=False,
        num_interactions=1,
    )
    one_layer_model = _energy_model(one_layer)
    coord, box = _sample_coord(pbc=True)
    energy = one_layer_model(coord.reshape(1, -1), _atype(), box=box)["energy"]
    assert energy.shape[-1] == 1
    zeros = ZeroPairRepulsion()(
        torch.ones(2, device=env.DEVICE),
        torch.zeros(3, 2, device=env.DEVICE),
        torch.zeros(2, 2, dtype=torch.long, device=env.DEVICE),
        torch.tensor([1, 8], device=env.DEVICE),
    )
    assert torch.equal(zeros, torch.zeros(3, device=env.DEVICE, dtype=zeros.dtype))
    name = "_deepmd_gnn_partial_mod"
    sys.modules[name] = types.ModuleType(name)
    helper = "_is_partially_initialized"
    try:
        assert getattr(pt_mod, helper)("no_such_mod", "x") is False
        assert getattr(pt_mod, helper)(name, "missing") is True
        register = "_register"
        with patch.object(pt_mod, helper, return_value=True):
            getattr(pt_mod, register)()
    finally:
        del sys.modules[name]
