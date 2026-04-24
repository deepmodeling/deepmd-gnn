# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for conservative MACE-OFF loading helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from deepmd_gnn.mace import MaceModel
from deepmd_gnn.mace_off import (
    MACE_OFF_MODELS,
    _infer_deepmd_config,
    _load_mace_checkpoint,
    convert_mace_off_to_deepmd,
    download_mace_off_model,
    load_mace_off_model,
)


def test_mace_off_model_urls_are_raw_github_paths() -> None:
    assert MACE_OFF_MODELS["off23_small"].endswith("mace_off23/MACE-OFF23_small.model")
    assert MACE_OFF_MODELS["off23_medium"].endswith(
        "mace_off23/MACE-OFF23_medium.model",
    )
    assert MACE_OFF_MODELS["off23_large"].endswith("mace_off23/MACE-OFF23_large.model")
    assert MACE_OFF_MODELS["off24_medium"].endswith(
        "mace_off24/MACE-OFF24_medium.model",
    )


@pytest.mark.slow
def test_download_real_model_uses_existing_url(tmp_path: Path) -> None:
    model_path = download_mace_off_model("off23_small", cache_dir=tmp_path)
    assert model_path.exists()
    assert model_path.stat().st_size > 0


@pytest.mark.slow
def test_infer_config_from_real_off23_small_checkpoint(tmp_path: Path) -> None:
    model_path = download_mace_off_model("off23_small", cache_dir=tmp_path)
    mace_model = _load_mace_checkpoint(model_path, device="cpu")
    config = _infer_deepmd_config(mace_model)

    assert config["type_map"] == ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
    assert config["r_max"] == pytest.approx(4.5)
    assert config["num_radial_basis"] == 8
    assert config["num_cutoff_basis"] == 5
    assert config["max_ell"] == 3
    assert config["interaction"] == "RealAgnosticResidualInteractionBlock"
    assert config["num_interactions"] == 2
    assert config["hidden_irreps"] == "96x0e"
    assert config["pair_repulsion"] is False
    assert config["distance_transform"] == "None"
    assert config["correlation"] == 3
    assert config["gate"] == "silu"
    assert config["MLP_irreps"] == "16x0e"
    assert config["radial_type"] == "bessel"
    assert config["radial_MLP"] == [64, 64, 64]
    assert config["std"] == pytest.approx(1.0757331948055126)


@pytest.mark.slow
def test_load_mace_off_model_requires_explicit_sel_and_uses_plain_elements(
    tmp_path: Path,
) -> None:
    model = load_mace_off_model(model_name="off23_small", cache_dir=tmp_path, sel=64)
    assert isinstance(model, MaceModel)
    assert model.get_sel() == [64]
    assert model.get_type_map() == ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
    assert all(not atom_type.startswith("m") for atom_type in model.get_type_map())
    assert "HW" not in model.get_type_map()
    assert "OW" not in model.get_type_map()


@pytest.mark.slow
def test_convert_mace_off_to_deepmd_scripts_model(tmp_path: Path) -> None:
    output_file = tmp_path / "mace_off23_small_dp.pt"
    result = convert_mace_off_to_deepmd(
        output_file=str(output_file),
        model_name="off23_small",
        cache_dir=tmp_path,
        sel=64,
    )
    assert result == output_file
    assert output_file.exists()
    loaded = torch.jit.load(str(output_file))
    assert loaded is not None
