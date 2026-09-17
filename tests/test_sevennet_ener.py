# SPDX-License-Identifier: LGPL-3.0-or-later
"""Correctness tests for the original SevenNet energy head as a DeePMD fitting."""

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

pytest.importorskip("sevenn")

import deepmd_gnn.pt  # noqa: F401
from deepmd_gnn.sevennet_checkpoint import load_native_sevennet_energy_modules
from deepmd_gnn.sevennet_descriptor import SevenNetDescriptor
from deepmd_gnn.sevennet_ener import SevenNetEnergyFitting, SevenNetEnergyHead
from tests.test_sevennet_descriptor import _inputs, _write_sevennet_checkpoint

if TYPE_CHECKING:
    from pathlib import Path


def _energy_model(checkpoint: Path, *, sel: int = 16) -> torch.nn.Module:
    return get_model(
        {
            "type": "standard",
            "type_map": ["H", "O"],
            "descriptor": {
                "type": "sevennet",
                "model_path": str(checkpoint),
                "sel": sel,
            },
            "fitting_net": {
                "type": "sevennet_ener",
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


def _native_atomic_energy(
    checkpoint: Path,
    descriptor: SevenNetDescriptor,
    coord: torch.Tensor,
    box: torch.Tensor | None,
) -> torch.Tensor:
    modules, _config = load_native_sevennet_energy_modules(
        checkpoint,
        device=str(env.DEVICE),
    )
    head = SevenNetEnergyHead(modules)
    coord_ext, atype_ext, nlist, mapping = _inputs(descriptor, coord, box)
    last_layer = descriptor._last_layer_features(  # noqa: SLF001
        coord_ext,
        atype_ext,
        nlist,
        mapping,
    )
    nf, nloc, _ = nlist.shape
    atom_energy = head.node_energy(
        last_layer.reshape(nf * nloc, last_layer.shape[-1]),
        atype_ext[:, :nloc],
    )
    return atom_energy.view(nf, nloc, 1)


@pytest.fixture
def sevennet_checkpoint(tmp_path: Path) -> Path:
    """Create a tiny trusted local SevenNet checkpoint."""
    return _write_sevennet_checkpoint(tmp_path / "small_sevennet.pth")


@pytest.mark.parametrize("pbc", [False, True])
def test_energy_matches_native_head(sevennet_checkpoint: Path, pbc: bool) -> None:
    """EnergyModel(sevennet + sevennet_ener) matches the original readout."""
    model = _energy_model(sevennet_checkpoint)
    assert isinstance(model.get_fitting_net(), SevenNetEnergyFitting)
    coord, box = _sample_coord(pbc=pbc)
    atype = _atype()
    pred = model(coord.reshape(1, -1), atype, box=box)
    native_atomic = _native_atomic_energy(
        sevennet_checkpoint,
        model.get_descriptor(),
        coord,
        box,
    )
    torch.testing.assert_close(
        pred["atom_energy"].reshape_as(native_atomic),
        native_atomic.to(pred["atom_energy"].dtype),
        rtol=1e-5,
        atol=1e-6,
    )


def test_force_matches_energy_autograd(sevennet_checkpoint: Path) -> None:
    """Forces are derivatives of the original energy, not a new force head."""
    model = _energy_model(sevennet_checkpoint)
    coord, box = _sample_coord(pbc=False)
    coord = coord.clone().requires_grad_(requires_grad=True)
    atype = _atype()
    energy = model(coord.reshape(1, -1), atype, box=box)["energy"]
    (force,) = torch.autograd.grad(energy.sum(), coord, create_graph=False)
    pred_force = model(coord.detach().reshape(1, -1), atype, box=box)["force"]
    torch.testing.assert_close(
        pred_force.reshape_as(force),
        -force,
        rtol=1e-5,
        atol=1e-6,
    )


def test_property_zeroe_unchanged(sevennet_checkpoint: Path) -> None:
    """Packed last-layer features do not change the property 0e channels."""
    energy_model = _energy_model(sevennet_checkpoint)
    property_model = get_model(
        {
            "type": "standard",
            "type_map": ["H", "O"],
            "descriptor": {
                "type": "sevennet",
                "model_path": str(sevennet_checkpoint),
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
    energy_out = energy_desc(
        energy_ext,
        energy_atype,
        energy_nlist,
        mapping=energy_mapping,
    )
    property_zeroe = property_desc(
        property_ext,
        property_atype,
        property_nlist,
        mapping=property_mapping,
    )[0]
    torch.testing.assert_close(energy_out[0], property_zeroe)
    packed = energy_out[2]
    assert packed is not None
    assert packed.shape[-1] == energy_model.get_fitting_net().head.feature_dim


def test_share_params_aliases_backbone_not_energy_head(
    sevennet_checkpoint: Path,
) -> None:
    """Level-0 sharing aliases the backbone and leaves energy heads distinct."""
    energy_desc = SevenNetDescriptor(
        model_path=sevennet_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    property_desc = SevenNetDescriptor(
        model_path=sevennet_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    energy_fit = SevenNetEnergyFitting(
        ntypes=2,
        dim_descrpt=energy_desc.get_dim_out(),
        type_map=["H", "O"],
        model_path=sevennet_checkpoint,
    )
    extra_fit = SevenNetEnergyFitting(
        ntypes=2,
        dim_descrpt=energy_desc.get_dim_out(),
        type_map=["H", "O"],
        model_path=sevennet_checkpoint,
    )
    assert energy_desc.backbone is not property_desc.backbone
    property_desc.share_params(energy_desc, shared_level=0)
    assert property_desc.backbone is energy_desc.backbone
    assert extra_fit.head is not energy_fit.head
    energy_param = next(energy_fit.head.parameters())
    extra_param = next(extra_fit.head.parameters())
    assert energy_param.data_ptr() != extra_param.data_ptr()


def test_get_model_multitask_shares_descriptor(sevennet_checkpoint: Path) -> None:
    """shared_dict plus share_params aliases the SevenNet backbone across branches."""
    model_config = {
        "shared_dict": {
            "type_map": ["H", "O"],
            "sevennet_descriptor": {
                "type": "sevennet",
                "model_path": str(sevennet_checkpoint),
                "sel": 16,
            },
        },
        "model_dict": {
            "force_field": {
                "type_map": "type_map",
                "descriptor": "sevennet_descriptor",
                "fitting_net": {
                    "type": "sevennet_ener",
                    "model_path": str(sevennet_checkpoint),
                },
            },
            "band_gap": {
                "type_map": "type_map",
                "descriptor": "sevennet_descriptor",
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
    assert energy_desc.backbone is property_desc.backbone
    assert isinstance(
        wrapper.model["force_field"].get_fitting_net(),
        SevenNetEnergyFitting,
    )
    coord, box = _sample_coord(pbc=False)
    atype = _atype()
    energy = wrapper.model["force_field"](coord.reshape(1, -1), atype, box=box)[
        "energy"
    ]
    prop = wrapper.model["band_gap"](coord.reshape(1, -1), atype, box=box)["band_gap"]
    assert energy.shape == (1, 1)
    assert prop.shape[0] == 1


def test_serialized_energy_model_roundtrip(
    sevennet_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Saved energy models restore the original head without the native pickle."""
    original = _energy_model(sevennet_checkpoint)
    restored = type(original).deserialize(original.serialize())
    coord, box = _sample_coord(pbc=True)
    atype = _atype()
    torch.testing.assert_close(
        restored(coord.reshape(1, -1), atype, box=box)["energy"],
        original(coord.reshape(1, -1), atype, box=box)["energy"],
        rtol=1e-6,
        atol=1e-7,
    )
    missing = tmp_path / "missing_source.pth"
    restored_from_config = get_model(
        {
            "type": "standard",
            "type_map": ["H", "O"],
            "descriptor": {
                "type": "sevennet",
                "sel": 16,
                "model_path": str(missing),
                "config": original.get_descriptor().config,
            },
            "fitting_net": {
                "type": "sevennet_ener",
                "model_path": str(missing),
                "config": original.get_fitting_net().config,
            },
        },
    )
    assert restored_from_config.get_descriptor().config["num_convolution_layer"] == 2


def test_checkpoint_reloads_without_native_pickle(
    sevennet_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Training checkpoints restore after the original SevenNet pickle is moved."""
    model_params = {
        "type": "standard",
        "type_map": ["H", "O"],
        "descriptor": {
            "type": "sevennet",
            "model_path": str(sevennet_checkpoint),
            "sel": 16,
        },
        "fitting_net": {
            "type": "sevennet_ener",
            "model_path": str(sevennet_checkpoint),
        },
    }
    original = get_model(model_params)
    script = json.loads(original.get_model_def_script())
    assert script["descriptor"]["config"]["type_map"] == ["H", "O"]
    assert script["fitting_net"]["config"]["type_map"] == ["H", "O"]
    json.dumps(script)
    wrapper = ModelWrapper(original, model_params=model_params)
    extra_params = wrapper.state_dict()["_extra_state"]["model_params"]
    assert extra_params["descriptor"]["config"]["num_convolution_layer"] == 2
    assert extra_params["fitting_net"]["config"]["num_convolution_layer"] == 2
    coord, box = _sample_coord(pbc=True)
    atype = _atype()
    before = original(coord.reshape(1, -1), atype, box=box)["energy"]
    saved = tmp_path / "model.ckpt.pt"
    torch.save({"model": wrapper.state_dict()}, saved)
    hidden = sevennet_checkpoint.with_name("hidden_source.pth")
    sevennet_checkpoint.rename(hidden)
    from deepmd.pt.infer.inference import Tester  # noqa: PLC0415

    tester = Tester(str(saved))
    prediction, _, _ = tester.wrapper(coord.reshape(1, -1), atype, box=box)
    torch.testing.assert_close(prediction["energy"], before, rtol=1e-6, atol=1e-7)


def test_out_bias_and_change_bias(sevennet_checkpoint: Path) -> None:
    """Training skips a second shift; finetune residual uses out_bias."""
    model = _energy_model(sevennet_checkpoint)
    assert torch.count_nonzero(model.get_out_bias()) == 0
    coord, box = _sample_coord(pbc=True)
    atype = _atype()
    before = model(coord.reshape(1, -1), atype, box=box)["energy"]
    shift_before = model.get_fitting_net().native_atomic_energies().detach().clone()
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
        shift_before,
    )
    model.change_out_bias(
        [_energy_sample(coord, atype, box, predicted + 6.0)],
        bias_adjust_mode="set-by-statistic",
    )
    assert not torch.allclose(
        model.get_fitting_net().native_atomic_energies(),
        shift_before,
    )


def test_sevennet_ener_rejects_invalid_construction(sevennet_checkpoint: Path) -> None:
    """Constructor guards keep the original head aligned with the descriptor."""
    with pytest.raises(ValueError, match="type_map"):
        SevenNetEnergyFitting(ntypes=2, dim_descrpt=4)
    with pytest.raises(ValueError, match="ntypes"):
        SevenNetEnergyFitting(ntypes=1, dim_descrpt=4, type_map=["H", "O"])
    with pytest.raises(ValueError, match="mixed_types"):
        SevenNetEnergyFitting(
            ntypes=2,
            dim_descrpt=4,
            type_map=["H", "O"],
            mixed_types=False,
        )
    with pytest.raises(ValueError, match="exactly match checkpoint"):
        SevenNetEnergyFitting(
            ntypes=2,
            dim_descrpt=4,
            type_map=["O", "H"],
            model_path=sevennet_checkpoint,
        )
    with pytest.raises(FileNotFoundError, match="not found"):
        SevenNetEnergyFitting(
            ntypes=2,
            dim_descrpt=4,
            type_map=["H", "O"],
            model_path="/no/such/sevennet.pth",
        )
    with pytest.raises(ValueError, match="Exactly one"):
        SevenNetEnergyFitting(ntypes=2, dim_descrpt=4, type_map=["H", "O"])
    fitting = SevenNetEnergyFitting(
        ntypes=2,
        dim_descrpt=4,
        type_map=["H", "O"],
        model_path=sevennet_checkpoint,
        trainable=False,
    )
    fitting.compute_input_stats([])
    fitting.compute_output_stats([])
    fitting.set_case_embd(0)
    with pytest.raises(NotImplementedError, match="type_map"):
        fitting.change_type_map(["H"])
    with pytest.raises(ValueError, match="g2"):
        fitting.forward(torch.zeros(1, 1, 4), torch.zeros(1, 1, dtype=torch.int64))
    with pytest.raises(ValueError, match="width"):
        fitting.forward(
            torch.zeros(1, 1, 4),
            torch.zeros(1, 1, dtype=torch.int64),
            g2=torch.zeros(1, 1, 3),
        )
    with pytest.raises(ValueError, match="serialized"):
        SevenNetEnergyFitting.deserialize({"@class": "Nope", "type": "sevennet_ener"})
    restored = SevenNetEnergyFitting.deserialize(fitting.serialize())
    assert restored.head.feature_dim == fitting.head.feature_dim
