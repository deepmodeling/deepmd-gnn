# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for sander interface and MACE-OFF support."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from deepmd_gnn.mace_off import (
    MACE_OFF_MODELS,
    convert_mace_off_to_deepmd,
    download_mace_off_model,
    get_mace_off_cache_dir,
    load_mace_off_model,
)
from deepmd_gnn.sander import SanderInterface, compute_qm_energy_sander


class TestMaceOff(unittest.TestCase):
    """Test MACE-OFF model loading and conversion."""

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
        assert all(
            isinstance(url, str) and url.startswith("http")
            for url in MACE_OFF_MODELS.values()
        )

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


class TestSanderInterface(unittest.TestCase):
    """Test sander interface functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock MACE model
        self.mock_model = MagicMock()
        self.mock_model.get_type_map.return_value = ["C", "H", "O", "mH", "mO"]
        self.mock_model.get_rcut.return_value = 5.0
        self.mock_model.eval.return_value = None

        # Create temporary model file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = Path(self.temp_dir.name) / "test_model.pth"

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_interface_init_file_not_found(self):
        """Test that FileNotFoundError is raised for missing model."""
        with self.assertRaises(FileNotFoundError):
            SanderInterface("nonexistent_model.pth")

    def test_interface_init_invalid_dtype(self):
        """Test that invalid dtype raises ValueError."""
        # Create a dummy model file
        torch.save({"dummy": "data"}, self.model_path)

        with self.assertRaises(ValueError):
            SanderInterface(self.model_path, dtype="invalid")

    @patch("torch.jit.load")
    def test_interface_initialization(self, mock_load):
        """Test interface initialization."""
        # Create dummy model file
        torch.save({"dummy": "data"}, self.model_path)

        mock_load.return_value = self.mock_model

        interface = SanderInterface(self.model_path, dtype="float64")

        assert interface.type_map == ["C", "H", "O", "mH", "mO"]
        assert interface.rcut == 5.0
        assert 3 in interface.mm_type_indices  # mH
        assert 4 in interface.mm_type_indices  # mO
        assert 0 not in interface.mm_type_indices  # C

    @patch("torch.jit.load")
    def test_compute_qm_correction(self, mock_load):
        """Test QM correction computation."""
        # Setup
        torch.save({"dummy": "data"}, self.model_path)
        mock_load.return_value = self.mock_model

        # Mock model output
        self.mock_model.return_value = {
            "energy": torch.tensor([[1.0]]),
            "force": torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]),
        }

        interface = SanderInterface(self.model_path)

        # Test inputs
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        types = np.array([0, 1])

        energy, forces = interface.compute_qm_correction(coords, types)

        assert isinstance(energy, float)
        assert isinstance(forces, np.ndarray)
        assert forces.shape == (2, 3)

    @patch("torch.jit.load")
    def test_compute_qm_correction_with_box(self, mock_load):
        """Test QM correction with periodic box."""
        torch.save({"dummy": "data"}, self.model_path)
        mock_load.return_value = self.mock_model

        self.mock_model.return_value = {
            "energy": torch.tensor([[1.0]]),
            "force": torch.tensor([[[0.1, 0.2, 0.3]]]),
        }

        interface = SanderInterface(self.model_path)

        coords = np.array([[0.0, 0.0, 0.0]])
        types = np.array([0])
        box = np.eye(3) * 10.0

        energy, forces = interface.compute_qm_correction(coords, types, box)

        assert isinstance(energy, float)
        assert forces.shape == (1, 3)

    @patch("torch.jit.load")
    def test_invalid_coordinates_shape(self, mock_load):
        """Test that invalid coordinate shape raises error."""
        torch.save({"dummy": "data"}, self.model_path)
        mock_load.return_value = self.mock_model

        interface = SanderInterface(self.model_path)

        # Wrong shape: 1D instead of 2D
        coords = np.array([0.0, 0.0, 0.0])
        types = np.array([0])

        with self.assertRaises(ValueError):
            interface.compute_qm_correction(coords, types)

    @patch("torch.jit.load")
    def test_invalid_box_shape(self, mock_load):
        """Test that invalid box shape raises error."""
        torch.save({"dummy": "data"}, self.model_path)
        mock_load.return_value = self.mock_model

        interface = SanderInterface(self.model_path)

        coords = np.array([[0.0, 0.0, 0.0]])
        types = np.array([0])
        box = np.array([1.0, 2.0, 3.0])  # Wrong shape

        with self.assertRaises(ValueError):
            interface.compute_qm_correction(coords, types, box)

    @patch("torch.jit.load")
    def test_get_qm_atoms_from_types(self, mock_load):
        """Test QM atom identification."""
        torch.save({"dummy": "data"}, self.model_path)
        mock_load.return_value = self.mock_model

        interface = SanderInterface(self.model_path)

        # Types: [C, H, O, mH, mO]
        types = np.array([0, 1, 2, 3, 4])

        qm_atoms = interface.get_qm_atoms_from_types(types)

        # First 3 are QM (C, H, O), last 2 are MM (mH, mO)
        assert qm_atoms == [0, 1, 2]

    def test_from_config(self):
        """Test loading from configuration file."""
        # Create config file
        config_path = Path(self.temp_dir.name) / "test.conf"
        with config_path.open("w") as f:
            f.write(f"model_file={self.model_path}\n")
            f.write("dtype=float32\n")
            f.write("qm_atoms=0,1,2\n")

        # Create dummy model
        torch.save({"dummy": "data"}, self.model_path)

        with patch("torch.jit.load") as mock_load:
            mock_load.return_value = self.mock_model

            interface = SanderInterface.from_config(config_path)

            assert interface.qm_atoms == [0, 1, 2]

    def test_from_config_missing_file(self):
        """Test error when config file is missing."""
        with self.assertRaises(FileNotFoundError):
            SanderInterface.from_config("nonexistent.conf")

    def test_from_config_missing_model_file(self):
        """Test error when model_file not in config."""
        config_path = Path(self.temp_dir.name) / "test.conf"
        with config_path.open("w") as f:
            f.write("dtype=float64\n")

        with self.assertRaises(ValueError):
            SanderInterface.from_config(config_path)

    @patch("torch.jit.load")
    def test_compute_qm_energy_sander(self, mock_load):
        """Test convenience function."""
        torch.save({"dummy": "data"}, self.model_path)
        mock_load.return_value = self.mock_model

        self.mock_model.return_value = {
            "energy": torch.tensor([[2.5]]),
            "force": torch.tensor([[[0.1, 0.2, 0.3]]]),
        }

        coords = np.array([[0.0, 0.0, 0.0]])
        types = np.array([0])

        energy, forces = compute_qm_energy_sander(
            str(self.model_path), coords, types
        )

        assert isinstance(energy, float)
        assert forces.shape == (1, 3)


if __name__ == "__main__":
    unittest.main()
