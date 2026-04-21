"""Test training."""

import inspect
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def test_compute_or_load_stat_accepts_preset_observed_type() -> None:
    """The plugin models should accept new deepmd-kit compatibility kwargs."""
    from deepmd_gnn.mace import MaceModel
    from deepmd_gnn.nequip import NequipModel

    for model_cls in [MaceModel, NequipModel]:
        signature = inspect.signature(model_cls.compute_or_load_stat)
        assert "preset_observed_type" in signature.parameters


@pytest.mark.parametrize(
    "input_fn",
    [
        "mace.json",
        "nequip.json",
    ],
)
def test_e2e_training(input_fn) -> None:
    """Test training the model."""
    model_fn = "model.pth"
    # create temp directory and copy example files
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        this_dir = Path(__file__).parent
        data_path = this_dir / "data"
        input_path = this_dir / input_fn
        # copy data to tmpdir
        shutil.copytree(data_path, tmpdir / "data")
        # copy input.json to tmpdir
        shutil.copy(input_path, tmpdir / input_fn)

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "deepmd",
                "--pt",
                "train",
                input_fn,
            ],
            cwd=tmpdir,
        )
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "deepmd",
                "--pt",
                "freeze",
                "-o",
                model_fn,
            ],
            cwd=tmpdir,
        )
        assert (tmpdir / model_fn).exists()
