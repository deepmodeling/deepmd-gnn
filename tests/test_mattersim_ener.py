# SPDX-License-Identifier: LGPL-3.0-or-later
"""Correctness tests for the original MatterSim energy head as a DeePMD fitting."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch
from deepmd.pt.model.model import get_model
from deepmd.pt.train.training import get_model_for_wrapper
from deepmd.pt.train.wrapper import ModelWrapper
from deepmd.pt.utils import env
from deepmd.pt.utils.multi_task import preprocess_shared_params

pytest.importorskip("mattersim")

import deepmd_gnn.pt  # noqa: F401
from deepmd_gnn.mattersim_checkpoint import load_native_mattersim_energy_head
from deepmd_gnn.mattersim_descriptor import MatterSimDescriptor
from deepmd_gnn.mattersim_ener import MatterSimEnergyFitting
from tests.test_mattersim_descriptor import (
    TINY_ARGS,
    _inputs,
    _write_mattersim_checkpoint,
)

if TYPE_CHECKING:
    from pathlib import Path


def _energy_model(checkpoint: Path, *, sel: int = 16) -> torch.nn.Module:
    return get_model(
        {
            "type": "standard",
            "type_map": ["H", "O"],
            "descriptor": {
                "type": "mattersim",
                "model_path": str(checkpoint),
                "sel": sel,
            },
            "fitting_net": {
                "type": "mattersim_ener",
                "model_path": str(checkpoint),
            },
        },
    )


def _sample_coord(*, pbc: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    if pbc:
        coord = torch.tensor(
            [[[0.1, 0.2, 0.3], [1.8, 0.2, 0.3], [0.2, 1.7, 0.4]]],
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
    descriptor: MatterSimDescriptor,
    coord: torch.Tensor,
    box: torch.Tensor | None,
) -> torch.Tensor:
    head, _ = load_native_mattersim_energy_head(checkpoint, device=env.DEVICE)
    coord_ext, atype_ext, nlist, mapping = _inputs(descriptor, coord, box)
    features = descriptor(coord_ext, atype_ext, nlist, mapping=mapping)[0]
    nf, nloc, width = features.shape
    z_table = torch.tensor(
        descriptor.type_to_z,
        dtype=torch.int64,
        device=atype_ext.device,
    )
    atomic_numbers = z_table[atype_ext[:, :nloc].reshape(-1)]
    source_dtype = next(head.parameters()).dtype
    atom_energy = head.node_energy(
        features.reshape(nf * nloc, width).to(source_dtype),
        atomic_numbers,
    )
    return atom_energy.view(nf, nloc, 1).to(env.GLOBAL_PT_FLOAT_PRECISION)


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


@pytest.fixture
def mattersim_energy_checkpoint(tmp_path: Path) -> Path:
    """Create a tiny MatterSim checkpoint that includes the GatedMLP readout."""
    return _write_mattersim_checkpoint(
        tmp_path / "small_mattersim_energy.pth",
        extra_energy=True,
    )


@pytest.mark.parametrize("pbc", [False, True])
def test_energy_matches_native_head(
    pbc: bool,
    mattersim_energy_checkpoint: Path,
) -> None:
    """EnergyModel(mattersim + mattersim_ener) matches the original readout."""
    model = _energy_model(mattersim_energy_checkpoint)
    assert isinstance(model.get_fitting_net(), MatterSimEnergyFitting)
    coord, box = _sample_coord(pbc=pbc)
    atype = _atype()
    pred = model(coord.reshape(1, -1), atype, box=box)
    native_atomic = _native_atomic_energy(
        mattersim_energy_checkpoint,
        model.get_descriptor(),
        coord,
        box,
    )
    torch.testing.assert_close(
        pred["atom_energy"].reshape_as(native_atomic),
        native_atomic,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        pred["energy"].reshape_as(native_atomic.sum(dim=1)),
        native_atomic.sum(dim=1),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("pbc", [False, True])
def test_force_matches_energy_autograd(
    mattersim_energy_checkpoint: Path,
    pbc: bool,
) -> None:
    """Forces are derivatives of the original energy, not a new force head."""
    model = _energy_model(mattersim_energy_checkpoint)
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


def test_property_atom_attr_unchanged(mattersim_energy_checkpoint: Path) -> None:
    """The energy branch does not change last-layer atom_attr used by property."""
    energy_model = _energy_model(mattersim_energy_checkpoint)
    property_model = get_model(
        {
            "type": "standard",
            "type_map": ["H", "O"],
            "descriptor": {
                "type": "mattersim",
                "model_path": str(mattersim_energy_checkpoint),
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
    energy_features = energy_desc(
        energy_ext,
        energy_atype,
        energy_nlist,
        mapping=energy_mapping,
    )[0]
    property_features = property_desc(
        property_ext,
        property_atype,
        property_nlist,
        mapping=property_mapping,
    )[0]
    torch.testing.assert_close(energy_features, property_features)


def test_share_params_aliases_backbone_not_energy_head(
    mattersim_energy_checkpoint: Path,
) -> None:
    """Level-0 sharing aliases the backbone and leaves energy heads distinct."""
    energy_desc = MatterSimDescriptor(
        model_path=mattersim_energy_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    property_desc = MatterSimDescriptor(
        model_path=mattersim_energy_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    energy_fit = MatterSimEnergyFitting(
        ntypes=2,
        dim_descrpt=energy_desc.get_dim_out(),
        type_map=["H", "O"],
        model_path=mattersim_energy_checkpoint,
    )
    extra_fit = MatterSimEnergyFitting(
        ntypes=2,
        dim_descrpt=energy_desc.get_dim_out(),
        type_map=["H", "O"],
        model_path=mattersim_energy_checkpoint,
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


def test_get_model_multitask_shares_descriptor(
    mattersim_energy_checkpoint: Path,
) -> None:
    """shared_dict plus share_params aliases the MatterSim backbone across branches."""
    model_config = {
        "shared_dict": {
            "type_map": ["H", "O"],
            "mattersim_descriptor": {
                "type": "mattersim",
                "model_path": str(mattersim_energy_checkpoint),
                "sel": 16,
            },
        },
        "model_dict": {
            "force_field": {
                "type_map": "type_map",
                "descriptor": "mattersim_descriptor",
                "fitting_net": {
                    "type": "mattersim_ener",
                    "model_path": str(mattersim_energy_checkpoint),
                },
            },
            "band_gap": {
                "type_map": "type_map",
                "descriptor": "mattersim_descriptor",
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
    assert isinstance(energy_desc, MatterSimDescriptor)
    assert isinstance(property_desc, MatterSimDescriptor)
    assert energy_desc.backbone is property_desc.backbone
    fitting = wrapper.model["force_field"].get_fitting_net()
    assert isinstance(fitting, MatterSimEnergyFitting)
    coord, box = _sample_coord(pbc=False)
    atype = _atype()
    energy = wrapper.model["force_field"](coord.reshape(1, -1), atype, box=box)[
        "energy"
    ]
    prop = wrapper.model["band_gap"](coord.reshape(1, -1), atype, box=box)["band_gap"]
    assert energy.shape == (1, 1)
    assert prop.shape[0] == 1


def test_serialized_energy_model_roundtrip(
    mattersim_energy_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Saved energy models restore the original head without the native checkpoint."""
    original = _energy_model(mattersim_energy_checkpoint)
    serialized = original.serialize()
    restored = type(original).deserialize(serialized)
    coord, box = _sample_coord(pbc=True)
    atype = _atype()
    torch.testing.assert_close(
        restored(coord.reshape(1, -1), atype, box=box)["energy"],
        original(coord.reshape(1, -1), atype, box=box)["energy"],
        rtol=1e-5,
        atol=1e-6,
    )
    missing = tmp_path / "missing_source.pth"
    assert not missing.exists()
    restored_from_config = get_model(
        {
            "type": "standard",
            "type_map": ["H", "O"],
            "descriptor": {
                "type": "mattersim",
                "sel": 16,
                "model_path": str(missing),
                "config": original.get_descriptor().config,
            },
            "fitting_net": {
                "type": "mattersim_ener",
                "model_path": str(missing),
                "config": original.get_fitting_net().config,
            },
        },
    )
    assert restored_from_config.get_descriptor().config["units"] == TINY_ARGS["units"]


def test_checkpoint_reloads_without_native_pickle(
    mattersim_energy_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Training checkpoints restore after the original MatterSim file is moved."""
    model_params = {
        "type": "standard",
        "type_map": ["H", "O"],
        "descriptor": {
            "type": "mattersim",
            "model_path": str(mattersim_energy_checkpoint),
            "sel": 16,
        },
        "fitting_net": {
            "type": "mattersim_ener",
            "model_path": str(mattersim_energy_checkpoint),
        },
    }
    original = get_model(model_params)
    script = json.loads(original.get_model_def_script())
    assert script["descriptor"]["config"]["type_map"] == ["H", "O"]
    assert script["fitting_net"]["config"]["type_map"] == ["H", "O"]
    json.dumps(script)
    wrapper = ModelWrapper(original, model_params=model_params)
    extra_params = wrapper.state_dict()["_extra_state"]["model_params"]
    assert extra_params["descriptor"]["config"]["units"] == TINY_ARGS["units"]
    assert extra_params["fitting_net"]["config"]["units"] == TINY_ARGS["units"]
    assert extra_params["descriptor"]["model_path"] is None
    assert extra_params["fitting_net"]["model_path"] is None
    coord, box = _sample_coord(pbc=True)
    atype = _atype()
    before = original(coord.reshape(1, -1), atype, box=box)["energy"]
    saved = tmp_path / "model.ckpt.pt"
    torch.save({"model": wrapper.state_dict()}, saved)
    hidden = mattersim_energy_checkpoint.with_name("hidden_source.pth")
    mattersim_energy_checkpoint.rename(hidden)
    from deepmd.pt.infer.inference import Tester  # noqa: PLC0415

    tester = Tester(str(saved))
    prediction, _, _ = tester.wrapper(coord.reshape(1, -1), atype, box=box)
    torch.testing.assert_close(prediction["energy"], before, rtol=1e-5, atol=1e-6)


def test_out_bias_starts_zero(mattersim_energy_checkpoint: Path) -> None:
    """Pretrained AtomScaling shift is not copied into DeePMD out_bias."""
    model = _energy_model(mattersim_energy_checkpoint)
    assert torch.count_nonzero(model.get_out_bias()) == 0


def test_apply_out_stat_adds_residual_not_second_e0(
    mattersim_energy_checkpoint: Path,
) -> None:
    """Nonzero out_bias is a residual on top of native MatterSim energy."""
    model = _energy_model(mattersim_energy_checkpoint)
    coord, box = _sample_coord(pbc=False)
    atype = _atype()
    before = model(coord.reshape(1, -1), atype, box=box)["energy"]
    bias = model.get_out_bias().clone()
    bias[0, 0, 0] = 1.0
    bias[0, 1, 0] = 2.0
    model.set_out_bias(bias)
    after = model(coord.reshape(1, -1), atype, box=box)["energy"]
    torch.testing.assert_close(after, before + 4.0, rtol=1e-5, atol=1e-6)


def test_compute_or_load_out_stat_does_not_stack_data_bias(
    mattersim_energy_checkpoint: Path,
) -> None:
    """Training stats must not LS-fit a second vacuum energy on top of AtomScaling."""
    model = _energy_model(mattersim_energy_checkpoint)
    coord, box = _sample_coord(pbc=True)
    atype = _atype()
    before = model(coord.reshape(1, -1), atype, box=box)["energy"]
    shift_before = model.get_fitting_net().head.normalizer.shift.detach().clone()
    sample = _energy_sample(coord, atype, box, before + 100.0)
    model.compute_or_load_stat(lambda: [sample])
    after = model(coord.reshape(1, -1), atype, box=box)["energy"]
    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(model.get_out_bias()) == 0
    torch.testing.assert_close(
        model.get_fitting_net().head.normalizer.shift,
        shift_before,
    )


def test_change_by_statistic_is_residual_shift(
    mattersim_energy_checkpoint: Path,
) -> None:
    """Finetune change-bias fits E_data - E_mattersim, it does not add e0 twice."""
    model = _energy_model(mattersim_energy_checkpoint)
    coord, box = _sample_coord(pbc=True)
    atype = _atype()
    predicted = model(coord.reshape(1, -1), atype, box=box)["energy"].detach()
    shift_before = model.get_fitting_net().head.normalizer.shift.detach().clone()
    model.change_out_bias(
        [_energy_sample(coord, atype, box, predicted)],
        bias_adjust_mode="change-by-statistic",
    )
    torch.testing.assert_close(
        model(coord.reshape(1, -1), atype, box=box)["energy"],
        predicted,
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(
        model.get_fitting_net().head.normalizer.shift,
        shift_before,
    )
    shifted = predicted + 3.0
    model.change_out_bias(
        [_energy_sample(coord, atype, box, shifted)],
        bias_adjust_mode="change-by-statistic",
    )
    after = model(coord.reshape(1, -1), atype, box=box)["energy"]
    torch.testing.assert_close(after, shifted, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        model.get_fitting_net().head.normalizer.shift,
        shift_before,
    )
    assert torch.count_nonzero(model.get_out_bias()) > 0


def test_set_by_statistic_replaces_atomscaling_shift(
    mattersim_energy_checkpoint: Path,
) -> None:
    """Explicit set-by-statistic writes per-Z AtomScaling shift, not out_bias."""
    model = _energy_model(mattersim_energy_checkpoint)
    coord, box = _sample_coord(pbc=True)
    atype = _atype()
    predicted = model(coord.reshape(1, -1), atype, box=box)["energy"].detach()
    shift_before = model.get_fitting_net().head.normalizer.shift.detach().clone()
    model.change_out_bias(
        [_energy_sample(coord, atype, box, predicted + 6.0)],
        bias_adjust_mode="set-by-statistic",
    )
    assert torch.count_nonzero(model.get_out_bias()) == 0
    assert not torch.allclose(
        model.get_fitting_net().head.normalizer.shift,
        shift_before,
    )


def test_mattersim_ener_rejects_invalid_construction(
    mattersim_energy_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Constructor guards keep the original head aligned with the descriptor."""
    with pytest.raises(ValueError, match="type_map"):
        MatterSimEnergyFitting(ntypes=2, dim_descrpt=8)
    with pytest.raises(ValueError, match="ntypes"):
        MatterSimEnergyFitting(ntypes=1, dim_descrpt=8, type_map=["H", "O"])
    with pytest.raises(ValueError, match="mixed_types"):
        MatterSimEnergyFitting(
            ntypes=2,
            dim_descrpt=8,
            type_map=["H", "O"],
            mixed_types=False,
        )
    with pytest.raises(FileNotFoundError, match="not found"):
        MatterSimEnergyFitting(
            ntypes=2,
            dim_descrpt=8,
            type_map=["H", "O"],
            model_path="/no/such/mattersim.pth",
        )
    with pytest.raises(ValueError, match="Exactly one"):
        MatterSimEnergyFitting(ntypes=2, dim_descrpt=8, type_map=["H", "O"])
    with pytest.raises(ValueError, match="max_z"):
        MatterSimEnergyFitting(
            ntypes=2,
            dim_descrpt=8,
            type_map=["H", "Na"],
            model_path=mattersim_energy_checkpoint,
        )
    with pytest.raises(ValueError, match="dim_descrpt"):
        MatterSimEnergyFitting(
            ntypes=2,
            dim_descrpt=2,
            type_map=["H", "O"],
            model_path=mattersim_energy_checkpoint,
        )
    filled: dict = {}
    fitting = MatterSimEnergyFitting(
        ntypes=2,
        dim_descrpt=8,
        type_map=["H", "O"],
        model_path=mattersim_energy_checkpoint,
        config=filled,
        trainable=False,
    )
    assert filled["units"] == 8
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
    with pytest.raises(ValueError, match="width"):
        fitting.forward(
            torch.zeros(1, 1, 4, dtype=torch.float32, device=env.DEVICE),
            torch.zeros(1, 1, dtype=torch.int64, device=env.DEVICE),
        )
    with pytest.raises(ValueError, match="type_map mismatch"):
        MatterSimEnergyFitting(
            ntypes=2,
            dim_descrpt=8,
            type_map=["H", "O"],
            config={**fitting.config, "type_map": ["O", "H"]},
        )
    with pytest.raises(ValueError, match="serialized"):
        MatterSimEnergyFitting.deserialize(
            {"@class": "Nope", "type": "mattersim_ener"},
        )
    with pytest.raises(ValueError, match="serialized"):
        MatterSimEnergyFitting.deserialize({"@class": "Fitting", "type": "ener"})
    restored = MatterSimEnergyFitting.deserialize(fitting.serialize())
    torch.testing.assert_close(
        restored.native_atomic_energies(),
        fitting.native_atomic_energies(),
    )
    no_energy = _write_mattersim_checkpoint(tmp_path / "no_energy.pth")
    with pytest.raises(ValueError, match="energy readout"):
        load_native_mattersim_energy_head(no_energy, device=env.DEVICE)


def test_hessian_mode_builds(mattersim_energy_checkpoint: Path) -> None:
    """Optional hessian_mode still routes through EnergyModel."""
    hessian_model = get_model(
        {
            "type": "standard",
            "type_map": ["H", "O"],
            "hessian_mode": True,
            "descriptor": {
                "type": "mattersim",
                "model_path": str(mattersim_energy_checkpoint),
                "sel": 16,
            },
            "fitting_net": {
                "type": "mattersim_ener",
                "model_path": str(mattersim_energy_checkpoint),
            },
        },
    )
    assert hessian_model is not None
