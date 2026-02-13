# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for MACE-OFF model loading."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

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

    @patch("deepmd_gnn.mace_off.urlretrieve")
    def test_download_mock(self, mock_urlretrieve):
        """Test download function (mocked)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake model file
            model_path = Path(tmpdir) / "mace_off_small.model"
            model_path.touch()

            # Mock the download
            mock_urlretrieve.return_value = None

            # Download should use cache if file exists
            result = download_mace_off_model("small", cache_dir=tmpdir)
            assert result == model_path
            assert result.exists()

            # Should not download if file exists
            mock_urlretrieve.assert_not_called()

    @patch("deepmd_gnn.mace_off.urlretrieve")
    def test_download_force(self, mock_urlretrieve):
        """Test force download."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake model file
            model_path = Path(tmpdir) / "mace_off_small.model"
            model_path.touch()

            # Mock the download
            mock_urlretrieve.return_value = None

            # Force download
            result = download_mace_off_model(
                "small",
                cache_dir=tmpdir,
                force_download=True,
            )
            assert result == model_path

            # Should download even if file exists
            mock_urlretrieve.assert_called_once()


class TestMaceOffLoading(unittest.TestCase):
    """Test MACE-OFF model loading functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock MACE model
        self.mock_mace_model = MagicMock()
        self.mock_mace_model.atomic_numbers = torch.tensor([1, 6, 7, 8])
        self.mock_mace_model.r_max = 5.0
        self.mock_mace_model.num_interactions = 2
        self.mock_mace_model.num_bessel = 8
        self.mock_mace_model.num_polynomial_cutoff = 5
        self.mock_mace_model.max_ell = 3
        self.mock_mace_model.hidden_irreps = "128x0e + 128x1o"
        self.mock_mace_model.correlation = 3
        self.mock_mace_model.radial_MLP = [64, 64, 64]
        self.mock_mace_model.interactions = [MagicMock()]
        self.mock_mace_model.interactions[
            0
        ].__class__.__name__ = "RealAgnosticResidualInteractionBlock"
        self.mock_mace_model.state_dict.return_value = {}

    @patch("deepmd_gnn.mace_off.download_mace_off_model")
    @patch("deepmd_gnn.mace_off.torch.load")
    @patch("deepmd_gnn.mace_off.MaceModel")
    def test_load_mace_off_model(
        self,
        mock_mace_model_class,
        mock_torch_load,
        mock_download,
    ):
        """Test loading MACE-OFF model."""
        # Setup mocks
        mock_download.return_value = Path("/fake/path/model.pth")
        mock_torch_load.return_value = self.mock_mace_model
        mock_deepmd_model = MagicMock()
        mock_mace_model_class.return_value = mock_deepmd_model

        # Call the function
        result = load_mace_off_model("small")

        # Verify
        mock_download.assert_called_once()
        mock_torch_load.assert_called_once()
        mock_mace_model_class.assert_called_once()
        assert result == mock_deepmd_model

    @patch("deepmd_gnn.mace_off.download_mace_off_model")
    @patch("deepmd_gnn.mace_off.torch.load")
    def test_load_invalid_model_structure(self, mock_torch_load, mock_download):
        """Test loading model with invalid structure."""
        # Setup mocks
        mock_download.return_value = Path("/fake/path/model.pth")
        invalid_model = MagicMock()
        # Missing atomic_numbers attribute
        del invalid_model.atomic_numbers
        mock_torch_load.return_value = invalid_model

        # Should raise ValueError
        with self.assertRaises(ValueError):
            load_mace_off_model("small")

    @patch("deepmd_gnn.mace_off.download_mace_off_model")
    @patch("deepmd_gnn.mace_off.torch.load")
    def test_load_model_missing_required_attrs(self, mock_torch_load, mock_download):
        """Test loading model missing required attributes."""
        # Setup mocks
        mock_download.return_value = Path("/fake/path/model.pth")
        invalid_model = MagicMock()
        invalid_model.atomic_numbers = torch.tensor([1, 6])
        # Missing r_max
        del invalid_model.r_max
        mock_torch_load.return_value = invalid_model

        # Should raise ValueError
        with self.assertRaises(ValueError):
            load_mace_off_model("small")

    @patch("deepmd_gnn.mace_off.download_mace_off_model")
    @patch("deepmd_gnn.mace_off.torch.load")
    def test_load_model_invalid_atomic_numbers(
        self,
        mock_torch_load,
        mock_download,
    ):
        """Test loading model with invalid atomic numbers."""
        # Setup mocks
        mock_download.return_value = Path("/fake/path/model.pth")
        invalid_model = MagicMock()
        # Invalid atomic number (0 or > 118)
        invalid_model.atomic_numbers = torch.tensor([0, 150])
        invalid_model.r_max = 5.0
        invalid_model.num_interactions = 2
        mock_torch_load.return_value = invalid_model

        # Should raise ValueError
        with self.assertRaises(ValueError):
            load_mace_off_model("small")


class TestMaceOffConversion(unittest.TestCase):
    """Test MACE-OFF model conversion functionality."""

    @patch("deepmd_gnn.mace_off.load_mace_off_model")
    @patch("deepmd_gnn.mace_off.torch.jit.script")
    @patch("deepmd_gnn.mace_off.torch.jit.save")
    def test_convert_to_deepmd(self, mock_jit_save, mock_jit_script, mock_load):
        """Test conversion to DeePMD format."""
        # Setup mocks
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        mock_scripted = MagicMock()
        mock_jit_script.return_value = mock_scripted

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_model.pth"

            # Call the function
            result = convert_mace_off_to_deepmd(
                "small",
                str(output_file),
                cache_dir=None,
            )

            # Verify
            mock_load.assert_called_once()
            mock_jit_script.assert_called_once_with(mock_model)
            mock_jit_save.assert_called_once()
            assert result == output_file

    @patch("deepmd_gnn.mace_off.load_mace_off_model")
    @patch("deepmd_gnn.mace_off.torch.jit.script")
    @patch("deepmd_gnn.mace_off.torch.save")
    def test_convert_fallback_on_script_fail(
        self,
        mock_torch_save,
        mock_jit_script,
        mock_load,
    ):
        """Test fallback to torch.save when scripting fails."""
        # Setup mocks
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        mock_jit_script.side_effect = RuntimeError("Scripting failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_model.pth"

            # Call the function
            result = convert_mace_off_to_deepmd(
                "small",
                str(output_file),
                cache_dir=None,
            )

            # Verify fallback was used
            mock_load.assert_called_once()
            mock_jit_script.assert_called_once()
            mock_torch_save.assert_called_once()
            assert result == output_file


if __name__ == "__main__":
    unittest.main()
