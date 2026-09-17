# SPDX-License-Identifier: LGPL-3.0-or-later
"""Focused tests for the MatterSim property descriptor."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from deepmd.pt.model.descriptor.base_descriptor import BaseDescriptor
from deepmd.pt.model.model import get_model
from deepmd.pt.utils import env
from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list

pytest.importorskip("mattersim")

from deepmd_gnn.mattersim_checkpoint import (
    ENERGY_STATE_PREFIXES,
    MatterSimFeatureBackbone,
    _dtype_from_name,
    _infer_state_dtype,
    compute_threebody_indices,
    compute_threebody_torch,
    load_mattersim_checkpoint_config,
    persistable_checkpoint_config,
    reject_unsupported_mattersim,
    resolve_mattersim_checkpoint_path,
    validate_mattersim_state_dict_load,
)
from deepmd_gnn.mattersim_descriptor import MatterSimDescriptor

TINY_ARGS = {
    "num_blocks": 2,
    "units": 8,
    "max_l": 2,
    "max_n": 2,
    "cutoff": 3.0,
    "max_z": 10,
    "threebody_cutoff": 2.5,
}


def _write_mattersim_checkpoint(
    path: Path,
    *,
    model_name: str = "m3gnet",
    extra_energy: bool = False,
) -> Path:
    backbone = MatterSimFeatureBackbone(TINY_ARGS)
    state = dict(backbone.state_dict())
    if extra_energy:
        state["final.g.0.linear.weight"] = torch.zeros(8, 8)
        state["normalizer.scale"] = torch.ones(11)
    torch.save(
        {
            "model_name": model_name,
            "model_args": dict(TINY_ARGS),
            "model": state,
        },
        path,
    )
    return path


@pytest.fixture
def mattersim_checkpoint(tmp_path: Path) -> Path:
    """Create a tiny trusted local MatterSim M3GNet checkpoint."""
    return _write_mattersim_checkpoint(tmp_path / "small_mattersim.pth")


def _inputs(
    descriptor: MatterSimDescriptor,
    coord: torch.Tensor | None = None,
    box: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if coord is None:
        coord = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-0.2, 1.0, 0.3]]],
            dtype=torch.float64,
        )
    coord = coord.to(env.DEVICE)
    atype = torch.tensor([[1, 0, 0]], dtype=torch.int64, device=env.DEVICE)
    if box is not None:
        box = box.to(env.DEVICE)
    coord_ext, atype_ext, mapping, nlist = extend_input_and_build_neighbor_list(
        coord.reshape(1, -1),
        atype,
        descriptor.get_rcut(),
        descriptor.get_sel(),
        mixed_types=True,
        box=box,
    )
    return coord_ext, atype_ext, nlist, mapping


def _descriptor_output(
    descriptor: MatterSimDescriptor,
    coord: torch.Tensor | None = None,
    box: torch.Tensor | None = None,
) -> torch.Tensor:
    coord_ext, atype_ext, nlist, mapping = _inputs(descriptor, coord, box)
    return descriptor(coord_ext, atype_ext, nlist, mapping=mapping)[0]


def _brute_force_graph(
    coord: torch.Tensor,
    cutoff: float,
    threebody_cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a MatterSim pair/triple graph from all pairs within cutoff."""
    nloc = coord.shape[0]
    device = coord.device
    dtype = coord.dtype
    pairs: list[tuple[int, int]] = []
    for center in range(nloc):
        for neighbor in range(nloc):
            if center == neighbor:
                continue
            dist = torch.linalg.norm(coord[center] - coord[neighbor])
            if float(dist) <= cutoff:
                pairs.append((center, neighbor))
    pairs.sort(key=lambda pair: (pair[0], pair[1]))
    if pairs:
        edge_index = torch.tensor(pairs, dtype=torch.int64, device=device).T
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.int64, device=device)
    pbc_offsets = torch.zeros((edge_index.shape[1], 3), dtype=dtype, device=device)
    distances = torch.linalg.norm(
        coord[edge_index[0]] - (coord[edge_index[1]] + pbc_offsets),
        dim=1,
    )
    num_atoms = torch.tensor([nloc], dtype=torch.int64, device=device)
    triples, num_triple_ij = compute_threebody_indices(
        edge_index,
        distances,
        num_atoms,
        nloc,
        threebody_cutoff,
    )
    return edge_index, pbc_offsets, triples, num_triple_ij


def test_constructor_registry_and_metadata(mattersim_checkpoint: Path) -> None:
    """Descriptor registration and metadata follow DeePMD's contract."""
    descriptor = BaseDescriptor(
        type="mattersim",
        model_path=str(mattersim_checkpoint),
        sel=16,
        type_map=["H", "O"],
        ntypes=2,
    )
    assert isinstance(descriptor, MatterSimDescriptor)
    assert descriptor.get_type_map() == ["H", "O"]
    assert descriptor.get_sel() == [16]
    assert descriptor.get_nsel() == 16
    assert descriptor.get_rcut() == pytest.approx(3.0)
    assert descriptor.get_dim_out() == 8
    assert descriptor.get_ntypes() == 2
    assert descriptor.get_rcut_smth() == pytest.approx(3.0)
    assert descriptor.get_dim_emb() == 8
    assert descriptor.get_env_protection() == 0.0
    assert descriptor.mixed_types()
    assert descriptor.has_message_passing()
    assert not descriptor.has_message_passing_across_ranks()
    assert not descriptor.supports_edge_parallel()
    assert not descriptor.dense_lower_supports_comm()
    assert not descriptor.need_sorted_nlist_for_lower()
    assert descriptor.has_default_chg_spin() is False
    descriptor.get_default_chg_spin()
    descriptor.compute_input_stats([])
    assert descriptor.get_stats() == {}
    assert all(parameter.requires_grad for parameter in descriptor.parameters())
    descriptor.set_stat_mean_and_stddev(torch.ones(1), torch.ones(1))
    mean, stddev = descriptor.get_stat_mean_and_stddev()
    assert mean.numel() == 0
    assert stddev.numel() == 0
    assert descriptor.type_to_z == [1, 8]

    local_config = {
        "type": "mattersim",
        "model_path": str(mattersim_checkpoint),
        "sel": 16,
    }
    updated, min_distance = BaseDescriptor.update_sel(
        None,
        ["H", "O"],
        local_config,
    )
    assert updated["model_path"] == str(mattersim_checkpoint)
    assert updated["sel"] == 16
    assert updated["config"]["cutoff"] == pytest.approx(3.0)
    assert updated["config"]["type_map"] == ["H", "O"]
    json.dumps(updated)
    assert min_distance is None


def test_descriptor_supports_first_import_in_fresh_process() -> None:
    """Importing the descriptor first must not recurse through the PT entry point."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from deepmd_gnn.mattersim_descriptor import MatterSimDescriptor; "
                "assert MatterSimDescriptor.__name__ == 'MatterSimDescriptor'"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_constructor_and_runtime_error_paths(
    mattersim_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Cover constructor, update_sel, share, and forward rejection paths."""
    with pytest.raises(ValueError, match="positive integer"):
        MatterSimDescriptor(
            model_path=mattersim_checkpoint,
            sel=0,
            type_map=["H", "O"],
        )
    with pytest.raises(ValueError, match="requires the model-level type_map"):
        MatterSimDescriptor(model_path=mattersim_checkpoint, sel=16)
    with pytest.raises(ValueError, match="ntypes="):
        MatterSimDescriptor(
            model_path=mattersim_checkpoint,
            sel=16,
            type_map=["H", "O"],
            ntypes=3,
        )
    with pytest.raises(ValueError, match="not an element"):
        MatterSimDescriptor(
            model_path=mattersim_checkpoint,
            sel=16,
            type_map=["H", "OW"],
        )
    with pytest.raises(TypeError, match="Unsupported MatterSim descriptor arguments"):
        MatterSimDescriptor(
            model_path=mattersim_checkpoint,
            sel=16,
            type_map=["H", "O"],
            unknown=True,
        )
    with pytest.raises(FileNotFoundError, match="not found"):
        MatterSimDescriptor(
            model_path=tmp_path / "missing.pth",
            sel=16,
            type_map=["H", "O"],
        )
    with pytest.raises(FileNotFoundError, match="not found"):
        load_mattersim_checkpoint_config("not-a-real-mattersim-keyword")
    with pytest.raises(ValueError, match="Exactly one of model_path"):
        MatterSimDescriptor(sel=16, type_map=["H", "O"])
    with pytest.raises(ValueError, match="Serialized MatterSim descriptor type_map"):
        MatterSimDescriptor(
            sel=16,
            type_map=["O", "H"],
            config=load_mattersim_checkpoint_config(mattersim_checkpoint)
            | {"type_map": ["H", "O"]},
        )
    frozen = MatterSimDescriptor(
        model_path=mattersim_checkpoint,
        sel=16,
        type_map=["H", "O"],
        trainable=False,
    )
    assert all(not parameter.requires_grad for parameter in frozen.parameters())
    inferred_config: dict = {"stale": True}
    MatterSimDescriptor(
        model_path=mattersim_checkpoint,
        sel=16,
        type_map=["H", "O"],
        config=inferred_config,
    )
    assert "stale" not in inferred_config
    assert inferred_config["type_map"] == ["H", "O"]

    descriptor = MatterSimDescriptor(
        model_path=mattersim_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    shared = MatterSimDescriptor(
        model_path=mattersim_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    shared.share_params(descriptor, 0)
    assert shared.backbone is descriptor.backbone
    with pytest.raises(TypeError, match="can only share"):
        shared.share_params(object(), 0)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="level 0"):
        shared.share_params(descriptor, 1)
    with pytest.raises(NotImplementedError, match="changing or subsetting type_map"):
        descriptor.change_type_map(["H"])

    coord_ext, atype_ext, nlist, mapping = _inputs(descriptor)
    with pytest.raises(NotImplementedError, match="MPI communication"):
        descriptor(
            coord_ext,
            atype_ext,
            nlist,
            mapping=mapping,
            comm_dict={"send": torch.zeros(1)},
        )
    coord = torch.tensor(
        [[[0.1, 0.2, 0.3], [3.8, 0.2, 0.3], [0.2, 3.7, 0.4]]],
        dtype=torch.float64,
    )
    box = (torch.eye(3, dtype=torch.float64) * 4.0).reshape(1, 9)
    coord_ext, atype_ext, nlist, mapping = _inputs(descriptor, coord, box)
    assert atype_ext.shape[1] > nlist.shape[1]
    with pytest.raises(ValueError, match="requires mapping"):
        descriptor(coord_ext, atype_ext, nlist, mapping=None)

    with pytest.raises(ValueError, match="serialized MatterSimDescriptor"):
        MatterSimDescriptor.deserialize(
            {
                "@class": "Descriptor",
                "type": "mace",
                "@version": 1,
            },
        )
    with pytest.raises(ValueError, match="explicit positive integer sel"):
        MatterSimDescriptor.update_sel(None, ["H", "O"], {"sel": True})
    updated, min_distance = MatterSimDescriptor.update_sel(
        None,
        None,
        {"sel": 16, "model_path": str(mattersim_checkpoint)},
    )
    assert updated["config"]["units"] == 8
    assert min_distance is None
    unchanged, _ = MatterSimDescriptor.update_sel(None, ["H", "O"], {"sel": 16})
    assert "config" not in unchanged


def test_checkpoint_helpers_cover_unsupported_and_json_paths(
    tmp_path: Path,
) -> None:
    """Reject Graphormer payloads and keep persisted configs JSON-safe."""
    with pytest.raises(ValueError, match="Only MatterSim-v1 M3GNet"):
        reject_unsupported_mattersim({"model_name": "graphormer"})
    graphormer = _write_mattersim_checkpoint(
        tmp_path / "graphormer.pth",
        model_name="graphormer",
    )
    with pytest.raises(ValueError, match="Only MatterSim-v1 M3GNet"):
        load_mattersim_checkpoint_config(graphormer)

    persistable = persistable_checkpoint_config(
        {
            **TINY_ARGS,
            "source_dtype": "float16",
            "type_map": ["H", "O"],
        },
    )
    assert persistable["source_dtype"] == "float32"
    json.dumps(persistable)
    assert _dtype_from_name("float64") == torch.float64
    assert _dtype_from_name("float32") == torch.float32
    assert _infer_state_dtype({"n": torch.tensor(1)}) == torch.float32
    assert (
        _infer_state_dtype({"w": torch.zeros(1, dtype=torch.float64)}) == torch.float64
    )
    assert ENERGY_STATE_PREFIXES == ("final.", "normalizer.")

    broken = tmp_path / "not_a_checkpoint.pth"
    torch.save([1, 2, 3], broken)
    with pytest.raises(TypeError, match="dictionary"):
        load_mattersim_checkpoint_config(broken)
    assert resolve_mattersim_checkpoint_path(broken) == broken
    assert resolve_mattersim_checkpoint_path("missing-keyword") is None

    class _LoadResult:
        missing_keys = ("weight",)
        unexpected_keys = ()

    with pytest.raises(RuntimeError, match="Failed to load"):
        validate_mattersim_state_dict_load(_LoadResult())

    class _EnergyKeys:
        missing_keys: tuple[str, ...] = ()
        unexpected_keys = ("final.g.0.linear.weight", "normalizer.scale")

    validate_mattersim_state_dict_load(_EnergyKeys(), allow_energy_keys=True)


def test_energy_keys_in_native_checkpoint_are_dropped(tmp_path: Path) -> None:
    """The original GatedMLP / AtomScaling readout is not part of the backbone."""
    checkpoint = _write_mattersim_checkpoint(
        tmp_path / "with_energy.pth",
        extra_energy=True,
    )
    descriptor = MatterSimDescriptor(
        model_path=checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    names = descriptor.backbone.state_dict()
    assert not any(name.startswith("final.") for name in names)
    assert not any(name.startswith("normalizer.") for name in names)


def test_forward_shape_rotation_invariance_and_gradient(
    mattersim_checkpoint: Path,
) -> None:
    """Last-layer features have local-atom shape, are invariant, and train."""
    descriptor = MatterSimDescriptor(
        model_path=mattersim_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    output = _descriptor_output(descriptor)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-0.2, 1.0, 0.3]]],
        dtype=torch.float64,
    )
    rotated = _descriptor_output(descriptor, coord @ rotation.T)
    assert output.shape == (1, 3, 8)
    torch.testing.assert_close(output, rotated, rtol=2e-5, atol=2e-6)
    output.square().sum().backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in descriptor.backbone.parameters()
    )


def test_nlist_graph_matches_brute_force_pairs(mattersim_checkpoint: Path) -> None:
    """DeePMD nlist conversion preserves last-layer features of the pair graph."""
    descriptor = MatterSimDescriptor(
        model_path=mattersim_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    coord = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-0.2, 1.0, 0.3]],
        dtype=torch.float32,
        device=env.DEVICE,
    )
    edge_index, pbc_offsets, triples, num_triple_ij = _brute_force_graph(
        coord,
        cutoff=3.0,
        threebody_cutoff=2.5,
    )
    nloc = coord.shape[0]
    cell = torch.eye(3, dtype=coord.dtype, device=coord.device).unsqueeze(0)
    atomic_numbers = torch.tensor(
        [[8.0], [1.0], [1.0]],
        dtype=coord.dtype,
        device=coord.device,
    )
    batch = torch.zeros((nloc,), dtype=torch.int64, device=coord.device)
    num_atoms = torch.tensor([nloc], dtype=torch.int64, device=coord.device)
    num_bonds = torch.tensor(
        [edge_index.shape[1]],
        dtype=torch.int64,
        device=coord.device,
    )
    native = descriptor.backbone(
        coord,
        cell,
        pbc_offsets,
        atomic_numbers,
        edge_index,
        triples,
        num_bonds,
        num_triple_ij,
        num_atoms,
        batch,
    ).view(1, nloc, -1)
    actual = _descriptor_output(descriptor, coord.unsqueeze(0).to(torch.float64))
    torch.testing.assert_close(
        actual,
        native.to(env.GLOBAL_PT_FLOAT_PRECISION),
        rtol=1e-5,
        atol=1e-6,
    )


def test_periodic_mapping_and_coordinate_gradient(
    mattersim_checkpoint: Path,
) -> None:
    """Periodic images follow compact mapping without breaking gradients."""
    descriptor = MatterSimDescriptor(
        model_path=mattersim_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    coord = torch.tensor(
        [[[0.1, 0.2, 0.3], [3.8, 0.2, 0.3], [0.2, 3.7, 0.4]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    box = (torch.eye(3, dtype=torch.float64) * 4.0).reshape(1, 9)
    coord_ext, atype_ext, nlist, mapping = _inputs(descriptor, coord, box)
    assert atype_ext.shape[1] > nlist.shape[1]
    output = descriptor(coord_ext, atype_ext, nlist, mapping=mapping)[0]
    output.square().sum().backward()
    assert coord.grad is not None
    assert torch.count_nonzero(coord.grad)

    translated = coord.detach().clone()
    translated[:, 1, 0] -= 4.0
    translated_output = _descriptor_output(descriptor, translated, box)
    torch.testing.assert_close(output, translated_output, rtol=2e-5, atol=2e-6)


def test_compact_periodic_mapping_handles_multiple_frames(
    mattersim_checkpoint: Path,
) -> None:
    """Compact node indices preserve independent frames with periodic images."""
    descriptor = MatterSimDescriptor(
        model_path=mattersim_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    coord = torch.tensor(
        [
            [[0.1, 0.2, 0.3], [3.8, 0.2, 0.3], [0.2, 3.7, 0.4]],
            [[0.2, 0.1, 0.4], [3.7, 0.3, 0.2], [0.3, 3.6, 0.5]],
        ],
        dtype=torch.float64,
        device=env.DEVICE,
    )
    atype = torch.tensor(
        [[1, 0, 0], [1, 0, 0]],
        dtype=torch.int64,
        device=env.DEVICE,
    )
    box = (
        torch.eye(3, dtype=torch.float64, device=env.DEVICE)
        .mul(4.0)
        .reshape(1, 9)
        .expand(2, 9)
    )
    coord_ext, atype_ext, mapping, nlist = extend_input_and_build_neighbor_list(
        coord.reshape(2, -1),
        atype,
        descriptor.get_rcut(),
        descriptor.get_sel(),
        mixed_types=True,
        box=box,
    )
    batched = descriptor(coord_ext, atype_ext, nlist, mapping=mapping)[0]
    per_frame = [
        _descriptor_output(descriptor, coord[frame : frame + 1], box[frame : frame + 1])
        for frame in range(2)
    ]
    torch.testing.assert_close(batched, torch.cat(per_frame))


def test_threebody_helper_matches_sorted_pair_counts() -> None:
    """Three-body indices enumerate ordered neighbor pairs around each center."""
    edge_indices = torch.tensor(
        [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]],
        dtype=torch.int64,
    )
    n_atoms = torch.tensor([3], dtype=torch.int64)
    triples, n_triple_ij, n_triple_i, n_triple_s = compute_threebody_torch(
        edge_indices,
        n_atoms,
        3,
    )
    assert triples.shape == (6, 2)
    assert n_triple_ij.tolist() == [1, 1, 1, 1, 1, 1]
    assert n_triple_i.tolist() == [2, 2, 2]
    assert n_triple_s.tolist() == [6]


def test_property_model_runs_under_get_model(mattersim_checkpoint: Path) -> None:
    """Property composition stays eager; official M3GNet layers are not TorchScriptable."""
    model = get_model(
        {
            "type": "standard",
            "type_map": ["H", "O"],
            "descriptor": {
                "type": "mattersim",
                "model_path": str(mattersim_checkpoint),
                "sel": 16,
            },
            "fitting_net": {
                "type": "property",
                "property_name": "band_gap",
                "task_dim": 1,
                "neuron": [8, 8],
                "precision": "float64",
            },
        },
    )
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-0.2, 1.0, 0.3]]],
        dtype=torch.float64,
        device=env.DEVICE,
    ).reshape(1, -1)
    atype = torch.tensor([[1, 0, 0]], dtype=torch.int64, device=env.DEVICE)
    prediction = model(coord, atype)["band_gap"]
    assert prediction.shape[0] == 1
    assert torch.isfinite(prediction).all()


def test_property_model_composition_and_optimizer_step(
    mattersim_checkpoint: Path,
) -> None:
    """Property fitting updates both backbone and fitting parameters."""
    model = get_model(
        {
            "type": "standard",
            "type_map": ["H", "O"],
            "descriptor": {
                "type": "mattersim",
                "model_path": str(mattersim_checkpoint),
                "sel": 16,
            },
            "fitting_net": {
                "type": "property",
                "property_name": "band_gap",
                "task_dim": 2,
                "neuron": [8, 8],
                "precision": "float64",
            },
        },
    )
    descriptor = model.get_descriptor()
    fitting = model.get_fitting_net()
    backbone_parameter = next(descriptor.backbone.parameters())
    fitting_parameter = next(fitting.parameters())
    backbone_before = backbone_parameter.detach().clone()
    fitting_before = fitting_parameter.detach().clone()

    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-0.2, 1.0, 0.3]]],
        dtype=torch.float64,
        device=env.DEVICE,
    ).reshape(1, -1)
    atype = torch.tensor([[1, 0, 0]], dtype=torch.int64, device=env.DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss = model(coord, atype)["band_gap"].square().sum()
    optimizer.zero_grad()
    loss.backward()
    assert backbone_parameter.grad is not None
    assert torch.count_nonzero(backbone_parameter.grad)
    assert fitting_parameter.grad is not None
    assert torch.count_nonzero(fitting_parameter.grad)
    optimizer.step()
    assert not torch.equal(backbone_before, backbone_parameter)
    assert not torch.equal(fitting_before, fitting_parameter)


def test_get_model_restores_without_source_checkpoint(
    mattersim_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Saved model definitions reconstruct the backbone without the native file."""
    updated, _ = MatterSimDescriptor.update_sel(
        None,
        ["H", "O"],
        {
            "type": "mattersim",
            "model_path": str(mattersim_checkpoint),
            "sel": 16,
        },
    )
    original = MatterSimDescriptor(
        model_path=mattersim_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    missing = tmp_path / "missing_source.pth"
    restored = get_model(
        {
            "type": "standard",
            "type_map": ["H", "O"],
            "descriptor": {
                **updated,
                "model_path": str(missing),
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
    validate = restored.get_descriptor().backbone.load_state_dict(
        original.backbone.state_dict(),
        strict=False,
    )
    assert not validate.missing_keys
    torch.testing.assert_close(
        _descriptor_output(restored.get_descriptor()),
        _descriptor_output(original),
    )


def test_checkpoint_feature_parity_and_serialization_roundtrip(
    mattersim_checkpoint: Path,
) -> None:
    """Checkpoint state and features survive DeePMD serialization."""
    descriptor = MatterSimDescriptor(
        model_path=mattersim_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    expected = _descriptor_output(descriptor)
    serialized = descriptor.serialize()
    assert serialized["model_path"] is None
    json.dumps({key: value for key, value in serialized.items() if key != "@variables"})
    restored = BaseDescriptor.deserialize(serialized)
    actual = _descriptor_output(restored)
    torch.testing.assert_close(actual, expected)
    for name, value in descriptor.backbone.state_dict().items():
        torch.testing.assert_close(
            value.cpu(),
            restored.backbone.state_dict()[name].cpu(),
        )


def test_dp_property_training_smoke(
    mattersim_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Run one real ``dp --pt train`` property step and restore without the source."""
    system = tmp_path / "property_data"
    data_set = system / "set.000"
    data_set.mkdir(parents=True)
    (system / "type.raw").write_text("1\n0\n0\n")
    (system / "type_map.raw").write_text("H\nO\n")
    coordinates = np.array(
        [
            [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-0.2, 1.0, 0.3]],
            [[0.0, 0.0, 0.0], [0.8, 0.2, 0.1], [-0.1, 0.9, 0.4]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.2], [-0.3, 1.1, 0.2]],
            [[0.0, 0.0, 0.0], [0.7, 0.3, 0.0], [-0.2, 0.8, 0.5]],
        ],
        dtype=np.float64,
    )
    np.save(data_set / "coord.npy", coordinates.reshape(4, -1))
    np.save(
        data_set / "box.npy",
        np.tile((np.eye(3) * 8.0).reshape(1, 9), (4, 1)),
    )
    np.save(
        data_set / "band_gap.npy",
        np.array([[0.2], [0.7], [-0.4], [1.1]], dtype=np.float64),
    )

    input_data = {
        "model": {
            "type": "standard",
            "type_map": ["H", "O"],
            "data_stat_nbatch": 1,
            "descriptor": {
                "type": "mattersim",
                "model_path": str(mattersim_checkpoint.resolve()),
                "sel": 16,
            },
            "fitting_net": {
                "type": "property",
                "property_name": "band_gap",
                "task_dim": 1,
                "neuron": [8],
                "precision": "float64",
                "seed": 11,
            },
        },
        "learning_rate": {
            "type": "exp",
            "decay_steps": 1,
            "start_lr": 1e-2,
            "stop_lr": 1e-3,
        },
        "loss": {
            "type": "property",
            "loss_func": "mse",
        },
        "optimizer": {
            "type": "Adam",
        },
        "training": {
            "training_data": {
                "systems": [str(system.resolve())],
                "batch_size": 2,
            },
            "numb_steps": 1,
            "seed": 17,
            "disp_freq": 1,
            "save_freq": 1,
            "save_ckpt": "model.ckpt",
        },
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(input_data))
    env_vars = os.environ.copy()
    env_vars.setdefault("OMP_NUM_THREADS", "1")
    subprocess.run(
        [sys.executable, "-m", "deepmd", "--pt", "train", input_path.name],
        cwd=tmp_path,
        env=env_vars,
        check=True,
        timeout=180,
    )

    checkpoint_pointer = tmp_path / "checkpoint"
    assert checkpoint_pointer.is_file()
    saved_checkpoint = Path(checkpoint_pointer.read_text().strip())
    if not saved_checkpoint.is_absolute():
        saved_checkpoint = tmp_path / saved_checkpoint
    assert saved_checkpoint.is_file()

    extra_params = torch.load(
        saved_checkpoint,
        map_location="cpu",
        weights_only=False,
    )["model"]["_extra_state"]["model_params"]
    assert extra_params["descriptor"]["config"]["type_map"] == ["H", "O"]
    hidden_source = mattersim_checkpoint.with_name("hidden_source.pth")
    mattersim_checkpoint.rename(hidden_source)
    from deepmd.pt.infer.inference import Tester  # noqa: PLC0415
    from deepmd.pt.utils.env import DEVICE  # noqa: PLC0415

    tester = Tester(str(saved_checkpoint))
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-0.2, 1.0, 0.3]]],
        dtype=torch.float64,
        device=DEVICE,
    ).reshape(1, -1)
    atype = torch.tensor([[1, 0, 0]], dtype=torch.int64, device=DEVICE)
    prediction, _, _ = tester.wrapper(coord, atype)
    assert torch.isfinite(prediction["band_gap"]).all()


def test_persistable_config_roundtrip_is_json_safe(mattersim_checkpoint: Path) -> None:
    """update_sel config can be written into a DeePMD JSON model definition."""
    persistable = load_mattersim_checkpoint_config(mattersim_checkpoint)
    json.dumps(persistable)
    again = persistable_checkpoint_config(persistable)
    assert again["cutoff"] == pytest.approx(3.0)
    assert again["threebody_cutoff"] == pytest.approx(2.5)
