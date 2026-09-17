# SPDX-License-Identifier: LGPL-3.0-or-later
"""Correctness tests for the original NequIP energy head as a DeePMD fitting."""

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

import deepmd_gnn.pt  # noqa: F401
from deepmd_gnn.nequip import NequipModel
from deepmd_gnn.nequip_descriptor import NequipDescriptor
from deepmd_gnn.nequip_ener import NequipEnergyFitting
from tests.test_nequip_descriptor import PARAMS, _inputs

if TYPE_CHECKING:
    from pathlib import Path


def _energy_model(artifact: Path) -> torch.nn.Module:
    return get_model(
        {
            "type": "standard",
            "type_map": PARAMS["type_map"],
            "descriptor": {
                "type": "nequip",
                "sel": PARAMS["sel"],
                "model_file": str(artifact),
            },
            "fitting_net": {
                "type": "nequip_ener",
                "model_file": str(artifact),
            },
        },
    )


def _coord_atype() -> tuple[torch.Tensor, torch.Tensor]:
    coord, atype, _nlist = _inputs()
    return coord, atype


def _energy_sample(
    coord: torch.Tensor,
    atype: torch.Tensor,
    box: torch.Tensor | None,
    energy: torch.Tensor,
) -> dict[str, torch.Tensor]:
    nloc = int(atype.shape[1])
    n_o = int((atype == 0).sum())
    n_h = int((atype == 1).sum())
    sample: dict[str, torch.Tensor] = {
        "coord": coord.reshape(1, -1).detach(),
        "atype": atype.detach(),
        "energy": energy.detach().reshape(1, 1),
        "natoms": torch.tensor(
            [[nloc, nloc, n_o, n_h]],
            dtype=torch.int64,
            device=atype.device,
        ),
        "find_energy": torch.tensor(1.0, device=atype.device),
    }
    if box is not None:
        sample["box"] = box.detach()
    return sample


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    """Create a serialized NequipModel artifact."""
    model = NequipModel(**PARAMS)
    path = tmp_path / "nequip.pt"
    torch.save(model.serialize(), path)
    return path


def test_energy_matches_nequip_model(artifact: Path) -> None:
    """EnergyModel(nequip + nequip_ener) matches native NequipModel energy."""
    split = _energy_model(artifact)
    native = NequipModel.deserialize(torch.load(artifact, weights_only=False))
    coord, atype = _coord_atype()
    split_energy = split(coord.reshape(1, -1), atype)["energy"]
    native_energy = native(coord.reshape(1, -1), atype)["energy"]
    torch.testing.assert_close(
        split_energy,
        native_energy.to(dtype=split_energy.dtype),
        rtol=1e-5,
        atol=1e-6,
    )


def test_force_matches_energy_autograd(artifact: Path) -> None:
    """Forces are derivatives of the original energy, not a new force head."""
    model = _energy_model(artifact)
    coord, atype = _coord_atype()
    coord = coord.clone().requires_grad_(requires_grad=True)
    energy = model(coord.reshape(1, -1), atype)["energy"]
    (force,) = torch.autograd.grad(energy.sum(), coord, create_graph=False)
    pred_force = model(coord.detach().reshape(1, -1), atype)["force"]
    torch.testing.assert_close(
        pred_force.reshape_as(force),
        -force,
        rtol=1e-5,
        atol=1e-6,
    )


def test_property_zeroe_unchanged(artifact: Path) -> None:
    """Energy extras are unused; property 0e stays conv_to_output_hidden."""
    energy_model = _energy_model(artifact)
    property_model = get_model(
        {
            "type": "standard",
            "type_map": PARAMS["type_map"],
            "descriptor": {
                "type": "nequip",
                "sel": PARAMS["sel"],
                "model_file": str(artifact),
            },
            "fitting_net": {
                "type": "property",
                "property_name": "band_gap",
                "task_dim": 1,
                "neuron": [4],
                "precision": "float32",
            },
        },
    )
    coord, atype, nlist = _inputs()
    energy_zeroe = energy_model.get_descriptor()(coord, atype, nlist)[0]
    property_zeroe = property_model.get_descriptor()(coord, atype, nlist)[0]
    torch.testing.assert_close(energy_zeroe, property_zeroe)


def test_share_params_aliases_backbone_not_energy_head(artifact: Path) -> None:
    """Level-0 sharing aliases the backbone and leaves energy heads distinct."""
    energy_desc = NequipDescriptor(
        sel=PARAMS["sel"],
        type_map=PARAMS["type_map"],
        model_file=str(artifact),
    )
    property_desc = NequipDescriptor(
        sel=PARAMS["sel"],
        type_map=PARAMS["type_map"],
        model_file=str(artifact),
    )
    energy_fit = NequipEnergyFitting(
        ntypes=2,
        dim_descrpt=energy_desc.get_dim_out(),
        type_map=PARAMS["type_map"],
        model_file=artifact,
    )
    extra_fit = NequipEnergyFitting(
        ntypes=2,
        dim_descrpt=energy_desc.get_dim_out(),
        type_map=PARAMS["type_map"],
        model_file=artifact,
    )
    assert energy_desc.model is not property_desc.model
    property_desc.share_params(energy_desc, shared_level=0)
    assert property_desc.model is energy_desc.model
    assert extra_fit.head is not energy_fit.head
    energy_param = next(energy_fit.head.parameters())
    extra_param = next(extra_fit.head.parameters())
    assert energy_param.data_ptr() != extra_param.data_ptr()


def test_get_model_multitask_shares_descriptor(artifact: Path) -> None:
    """shared_dict plus share_params aliases the NequIP backbone across branches."""
    model_config = {
        "shared_dict": {
            "type_map": PARAMS["type_map"],
            "nequip_descriptor": {
                "type": "nequip",
                "sel": PARAMS["sel"],
                "model_file": str(artifact),
            },
        },
        "model_dict": {
            "force_field": {
                "type_map": "type_map",
                "descriptor": "nequip_descriptor",
                "fitting_net": {
                    "type": "nequip_ener",
                    "model_file": str(artifact),
                },
            },
            "band_gap": {
                "type_map": "type_map",
                "descriptor": "nequip_descriptor",
                "fitting_net": {
                    "type": "property",
                    "property_name": "band_gap",
                    "task_dim": 1,
                    "intensive": True,
                    "neuron": [4],
                    "precision": "float32",
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
    assert energy_desc.model is property_desc.model
    fitting = wrapper.model["force_field"].get_fitting_net()
    assert isinstance(fitting, NequipEnergyFitting)
    coord, atype = _coord_atype()
    energy = wrapper.model["force_field"](coord.reshape(1, -1), atype)["energy"]
    prop = wrapper.model["band_gap"](coord.reshape(1, -1), atype)["band_gap"]
    assert energy.shape == (1, 1)
    assert prop.shape[0] == 1


def test_serialized_energy_model_roundtrip(artifact: Path, tmp_path: Path) -> None:
    """Saved energy models restore the original head without the native artifact."""
    original = _energy_model(artifact)
    restored = type(original).deserialize(original.serialize())
    coord, atype = _coord_atype()
    torch.testing.assert_close(
        restored(coord.reshape(1, -1), atype)["energy"],
        original(coord.reshape(1, -1), atype)["energy"],
        rtol=1e-6,
        atol=1e-7,
    )
    missing = tmp_path / "missing_source.pt"
    restored_from_config = get_model(
        {
            "type": "standard",
            "type_map": PARAMS["type_map"],
            "descriptor": {
                "type": "nequip",
                "sel": PARAMS["sel"],
                "model_file": str(missing),
                "config": original.get_descriptor().params,
            },
            "fitting_net": {
                "type": "nequip_ener",
                "model_file": str(missing),
                "config": original.get_fitting_net().config,
            },
        },
    )
    assert restored_from_config.get_descriptor().params["num_layers"] == 2


def test_checkpoint_reloads_without_native_artifact(
    artifact: Path,
    tmp_path: Path,
) -> None:
    """Training checkpoints restore after the original NequIP artifact is moved."""
    model_params = {
        "type": "standard",
        "type_map": PARAMS["type_map"],
        "descriptor": {
            "type": "nequip",
            "sel": PARAMS["sel"],
            "model_file": str(artifact),
        },
        "fitting_net": {
            "type": "nequip_ener",
            "model_file": str(artifact),
        },
    }
    original = get_model(model_params)
    script = json.loads(original.get_model_def_script())
    assert script["descriptor"]["config"]["type_map"] == PARAMS["type_map"]
    assert script["fitting_net"]["config"]["type_map"] == PARAMS["type_map"]
    json.dumps(script)
    wrapper = ModelWrapper(original, model_params=model_params)
    extra_params = wrapper.state_dict()["_extra_state"]["model_params"]
    assert extra_params["descriptor"]["config"]["num_layers"] == 2
    assert extra_params["fitting_net"]["config"]["num_layers"] == 2
    coord, atype = _coord_atype()
    before = original(coord.reshape(1, -1), atype)["energy"]
    saved = tmp_path / "model.ckpt.pt"
    torch.save({"model": wrapper.state_dict()}, saved)
    hidden = artifact.with_name("hidden_source.pt")
    artifact.rename(hidden)
    from deepmd.pt.infer.inference import Tester  # noqa: PLC0415

    tester = Tester(str(saved))
    prediction, _, _ = tester.wrapper(coord.reshape(1, -1), atype)
    torch.testing.assert_close(prediction["energy"], before, rtol=1e-6, atol=1e-7)


def test_out_bias_and_change_bias(artifact: Path) -> None:
    """Training skips a second e0; finetune residual uses out_bias."""
    model = _energy_model(artifact)
    assert torch.count_nonzero(model.get_out_bias()) == 0
    coord, atype = _coord_atype()
    coord = coord.to(dtype=torch.float64)
    box = (torch.eye(3, dtype=torch.float64, device=env.DEVICE) * 6.0).reshape(1, 9)
    before = model(coord.reshape(1, -1), atype, box=box)["energy"]
    e0_before = model.get_fitting_net().native_atomic_energies().detach().clone()
    sample = _energy_sample(coord, atype, box, before + 100.0)
    model.compute_or_load_stat(lambda: [sample])
    torch.testing.assert_close(
        model(coord.reshape(1, -1), atype, box=box)["energy"],
        before,
        rtol=1e-6,
        atol=1e-7,
    )
    predicted = before.detach()
    model.change_out_bias(
        [_energy_sample(coord, atype, box, predicted + 3.0)],
        bias_adjust_mode="change-by-statistic",
    )
    torch.testing.assert_close(
        model(coord.reshape(1, -1), atype, box=box)["energy"],
        predicted + 3.0,
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(
        model.get_fitting_net().native_atomic_energies(),
        e0_before,
    )
    model.change_out_bias(
        [_energy_sample(coord, atype, box, predicted + 6.0)],
        bias_adjust_mode="set-by-statistic",
    )
    assert not torch.allclose(
        model.get_fitting_net().native_atomic_energies(),
        e0_before,
    )


def test_nequip_ener_rejects_invalid_construction(artifact: Path) -> None:
    """Constructor guards keep the original head aligned with the descriptor."""
    with pytest.raises(ValueError, match="type_map"):
        NequipEnergyFitting(ntypes=2, dim_descrpt=3)
    with pytest.raises(ValueError, match="ntypes"):
        NequipEnergyFitting(ntypes=1, dim_descrpt=3, type_map=["O", "H"])
    with pytest.raises(ValueError, match="mixed_types"):
        NequipEnergyFitting(
            ntypes=2,
            dim_descrpt=3,
            type_map=["O", "H"],
            mixed_types=False,
        )
    with pytest.raises(ValueError, match="type_map"):
        NequipEnergyFitting(
            ntypes=2,
            dim_descrpt=3,
            type_map=["H", "O"],
            model_file=artifact,
        )
    with pytest.raises(FileNotFoundError, match="not found"):
        NequipEnergyFitting(
            ntypes=2,
            dim_descrpt=3,
            type_map=["O", "H"],
            model_file="/no/such/nequip.pt",
        )
    with pytest.raises(ValueError, match="Exactly one"):
        NequipEnergyFitting(ntypes=2, dim_descrpt=3, type_map=["O", "H"])
    with pytest.raises(ValueError, match="dim_descrpt"):
        NequipEnergyFitting(
            ntypes=2,
            dim_descrpt=99,
            type_map=["O", "H"],
            model_file=artifact,
        )
    fitting = NequipEnergyFitting(
        ntypes=2,
        dim_descrpt=3,
        type_map=["O", "H"],
        model_file=artifact,
        trainable=False,
    )
    assert fitting.get_type_map() == ["O", "H"]
    fitting.compute_input_stats([])
    fitting.compute_output_stats([])
    fitting.set_case_embd(0)
    with pytest.raises(NotImplementedError, match="type_map"):
        fitting.change_type_map(["O"])
    with pytest.raises(ValueError, match="serialized"):
        NequipEnergyFitting.deserialize({"@class": "Nope", "type": "nequip_ener"})
    restored = NequipEnergyFitting.deserialize(fitting.serialize())
    assert restored.dim_descrpt == 3
