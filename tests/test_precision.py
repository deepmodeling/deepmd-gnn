"""Tests for model construction precision and global dtype isolation."""

from __future__ import annotations

import json

import pytest
import torch

from deepmd_gnn.mace import MaceModel
from deepmd_gnn.nequip import NequipModel


def test_mace_get_model_respects_precision_without_global_leak() -> None:
    """The config constructor should build float64 MACE under a local context."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        model = MaceModel.get_model(
            {
                "type": "mace",
                "type_map": ["H"],
                "sel": 4,
                "precision": "float64",
            },
        )
        assert torch.get_default_dtype() == torch.float32
        assert model.model.atomic_energies_fn.atomic_energies.dtype == torch.float64
        assert model.serialize()["precision"] == "float64"
        assert json.loads(model.model_def_script)["precision"] == "float64"
    finally:
        torch.set_default_dtype(old_dtype)


def test_nequip_constructor_respects_precision_without_global_leak() -> None:
    """Direct NequIP construction should use and preserve the requested dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        model = NequipModel(
            type_map=["H"],
            sel=4,
            num_layers=1,
            precision="float64",
        )
        assert torch.get_default_dtype() == torch.float32
        assert next(model.model.parameters()).dtype == torch.float64
        assert model.serialize()["precision"] == "float64"
    finally:
        torch.set_default_dtype(old_dtype)


@pytest.mark.parametrize("model_cls", [MaceModel, NequipModel])
def test_model_rejects_unknown_precision(
    model_cls: type[MaceModel | NequipModel],
) -> None:
    """Both backends should reject unsupported precision names explicitly."""
    with pytest.raises(ValueError, match="precision float16 not supported"):
        model_cls(type_map=["H"], sel=4, precision="float16")
