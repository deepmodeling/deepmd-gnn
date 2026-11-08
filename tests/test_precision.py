"""Tests for model construction precision and global dtype isolation."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
import torch

from deepmd_gnn.mace import MaceModel
from deepmd_gnn.nequip import NequipModel
from deepmd_gnn.precision import temporary_default_dtype


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


def test_temporary_dtype_context_serializes_concurrent_construction() -> None:
    """Two concurrent builders must keep their dtypes and restore the global default."""
    original = torch.get_default_dtype()
    first_entered = Event()
    second_attempting = Event()
    second_entered = Event()
    release_first = Event()

    def first_builder() -> torch.dtype:
        with temporary_default_dtype(torch.float64):
            first_entered.set()
            assert release_first.wait(5), "timed out releasing first builder"
            return torch.nn.Linear(1, 1).weight.dtype

    def second_builder() -> torch.dtype:
        assert first_entered.wait(5), "first builder did not enter"
        second_attempting.set()
        with temporary_default_dtype(torch.float32):
            second_entered.set()
            return torch.nn.Linear(1, 1).weight.dtype

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(first_builder)
        assert first_entered.wait(5)
        second = executor.submit(second_builder)
        try:
            assert second_attempting.wait(5)
            assert not second_entered.wait(0.05), (
                "another builder changed the process-global dtype concurrently"
            )
        finally:
            release_first.set()
        assert first.result(timeout=5) == torch.float64
        assert second.result(timeout=5) == torch.float32

    assert torch.get_default_dtype() == original
