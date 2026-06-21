"""Test training."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tests._pt_expt import (
    pt_expt_cli_env,
)


def _run_e2e_training(input_fn, backend_flag: str, model_fn: str) -> None:
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        this_dir = Path(__file__).parent
        data_path = this_dir / "data"
        input_path = this_dir / input_fn
        # copy data to tmpdir
        shutil.copytree(data_path, tmpdir / "data")
        # copy input.json to tmpdir
        shutil.copy(input_path, tmpdir / input_fn)
        env = None
        if backend_flag == "--pt-expt":
            env = pt_expt_cli_env(tmpdir)

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "deepmd",
                backend_flag,
                "train",
                input_fn,
            ],
            cwd=tmpdir,
            env=env,
        )
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "deepmd",
                backend_flag,
                "freeze",
                "-o",
                model_fn,
            ],
            cwd=tmpdir,
            env=env,
        )
        assert (tmpdir / model_fn).exists()


@pytest.mark.parametrize(
    "input_fn",
    [
        "mace.json",
        "nequip.json",
    ],
)
def test_e2e_training(input_fn) -> None:
    """Test training the model."""
    _run_e2e_training(input_fn, "--pt", "model.pth")


@pytest.mark.parametrize(
    "input_fn",
    [
        "mace.json",
    ],
)
def test_e2e_pt_expt_training_exports_pt2(input_fn) -> None:
    """Test training and freezing exportable models for LAMMPS."""
    _run_e2e_training(input_fn, "--pt-expt", "model.pt2")
