"""Test training."""

import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

from tests._pt_expt import (
    pt_expt_cli_env,
)


def _assert_pt2_comm_artifact(model_path: Path, *, expected: bool) -> None:
    with zipfile.ZipFile(model_path) as archive:
        names = archive.namelist()
        metadata_name = next(
            name for name in names if name.endswith("extra/metadata.json")
        )
        metadata = json.loads(archive.read(metadata_name))

    has_artifact = any(
        name.endswith("extra/forward_lower_with_comm.pt2") for name in names
    )
    assert metadata["has_comm_artifact"] is expected
    assert has_artifact is expected


def _run_e2e_training(input_fn, backend_flag: str, model_fn: str) -> None:
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        this_dir = Path(__file__).parent
        data_path = this_dir / "data"
        input_path = this_dir / input_fn
        model_data = json.loads(input_path.read_text())
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
        model_path = tmpdir / model_fn
        assert model_path.exists()
        if backend_flag == "--pt-expt":
            model_config = model_data["model"]
            _assert_pt2_comm_artifact(
                model_path,
                expected=(
                    model_config["type"] == "mace"
                    and model_config.get("num_interactions", 2) > 1
                ),
            )


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
        "nequip.json",
    ],
)
def test_e2e_pt_expt_training_exports_pt2(input_fn) -> None:
    """Test training and freezing exportable models for LAMMPS."""
    _run_e2e_training(input_fn, "--pt-expt", "model.pt2")
