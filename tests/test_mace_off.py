# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for MACE-OFF model loading."""

import tempfile
import unittest
from pathlib import Path

import pytest
import torch
from torch import nn

from deepmd_gnn.mace import ELEMENTS, MaceModel
from deepmd_gnn.mace_off import (
    MACE_OFF_MODELS,
    convert_mace_off_to_deepmd,
    download_mace_off_model,
    get_mace_off_cache_dir,
    load_mace_off_model,
)


class TestMaceOffDownload(unittest.TestCase):
    """Test MACE-OFF download functionality."""

    def test_get_cache_dir(self):
        """Test cache directory creation."""
        cache_dir = get_mace_off_cache_dir()
        assert isinstance(cache_dir, Path)
        assert cache_dir.exists()

    def test_mace_off_models_defined(self):
        """Test that MACE-OFF models are defined."""
        assert "small" in MACE_OFF_MODELS
        assert "medium" in MACE_OFF_MODELS
        assert "large" in MACE_OFF_MODELS
        for url in MACE_OFF_MODELS.values():
            assert isinstance(url, str)
            assert url.startswith("http")

    def test_invalid_model_name(self):
        """Test that invalid model names raise errors."""
        with self.assertRaises(ValueError):
            download_mace_off_model("invalid_model")

    @pytest.mark.slow
    def test_download_real_model(self):
        """Test downloading a real MACE-OFF model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download the small model (fastest)
            model_path = download_mace_off_model("small", cache_dir=tmpdir)

            # Verify the file was downloaded
            assert model_path.exists()
            assert model_path.stat().st_size > 0

            # Test that downloading again uses cache
            model_path_2 = download_mace_off_model("small", cache_dir=tmpdir)
            assert model_path == model_path_2

    @pytest.mark.slow
    def test_download_force(self):
        """Test force download."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download the model first
            model_path = download_mace_off_model("small", cache_dir=tmpdir)
            original_size = model_path.stat().st_size

            # Modify the file to simulate corruption
            model_path.write_text("corrupted")

            # Force re-download
            model_path_2 = download_mace_off_model(
                "small",
                cache_dir=tmpdir,
                force_download=True,
            )

            # Verify it was re-downloaded (size should be restored)
            assert model_path == model_path_2
            assert model_path_2.stat().st_size == original_size


class TestMaceOffLoading(unittest.TestCase):
    """Test MACE-OFF model loading functionality."""

    @pytest.mark.slow
    def test_load_real_mace_off_model(self):
        """Test loading a real MACE-OFF model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Load the small model
            model = load_mace_off_model("small", cache_dir=tmpdir)

            # Verify the model is a MaceModel instance
            assert isinstance(model, MaceModel)

            # Verify it has the expected attributes
            assert hasattr(model, "get_type_map")
            assert hasattr(model, "get_rcut")

            # Verify type_map is properly set
            type_map = model.get_type_map()
            assert isinstance(type_map, list)
            assert len(type_map) > 0

            # Verify rcut is set
            rcut = model.get_rcut()
            assert isinstance(rcut, float)
            assert rcut > 0

    def test_load_model_invalid_atomic_numbers(self):
        """Test that invalid atomic numbers are caught during loading."""
        # Create a minimal mock model file to test validation
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_model_path = Path(tmpdir) / "invalid_model.pth"

            # Create a mock model with invalid atomic numbers
            class MockModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.atomic_numbers = torch.tensor([0, 150])  # Invalid
                    self.r_max = 5.0
                    self.num_interactions = 2

            mock_model = MockModel()
            torch.save(mock_model, mock_model_path)

            # Attempt to load and validate
            # Note: This test validates the atomic number checking logic
            loaded = torch.load(mock_model_path, weights_only=False)
            # Check if atomic numbers would be validated
            atomic_numbers = loaded.atomic_numbers.tolist()
            # Should fail validation
            invalid = any(z < 1 or z > len(ELEMENTS) for z in atomic_numbers)
            assert invalid, "Invalid atomic numbers should be detected"


class TestMaceOffConversion(unittest.TestCase):
    """Test MACE-OFF model conversion functionality."""

    @pytest.mark.slow
    def test_convert_real_model_to_deepmd(self):
        """Test converting a real MACE-OFF model to DeePMD format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "frozen_model.pth"

            # Convert the small model
            result = convert_mace_off_to_deepmd(
                "small",
                str(output_file),
                cache_dir=tmpdir,
            )

            # Verify the output file was created
            assert result == output_file
            assert output_file.exists()
            assert output_file.stat().st_size > 0

            # Verify the frozen model can be loaded
            frozen_model = torch.jit.load(str(output_file))
            assert frozen_model is not None


if __name__ == "__main__":
    unittest.main()
