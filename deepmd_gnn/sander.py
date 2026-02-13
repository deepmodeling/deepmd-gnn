# SPDX-License-Identifier: LGPL-3.0-or-later
"""Sander interface for QM/MM simulations with MACE models."""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from deepmd.pt.model.model import get_model


class SanderInterface:
    """Interface for sander QM/MM calculations using MACE models.

    This class provides methods to compute QM internal energy corrections
    for QM/MM simulations in sander using MACE models.

    Parameters
    ----------
    model_file : str or Path
        Path to the frozen MACE model file (.pth format)
    qm_atoms : list[int], optional
        List of atom indices in the QM region. If None, will be determined
        from atom types (non-MM atoms are QM)
    dtype : str, optional
        Data type for calculations, either "float32" or "float64"
        Default is "float64"

    Examples
    --------
    >>> interface = SanderInterface("frozen_model.pth")
    >>> energy, forces = interface.compute_qm_correction(
    ...     coordinates=coords,
    ...     atom_types=types,
    ...     box=box
    ... )
    """

    def __init__(
        self,
        model_file: str | Path,
        qm_atoms: Optional[list[int]] = None,
        dtype: str = "float64",
    ) -> None:
        """Initialize the sander interface."""
        self.model_file = Path(model_file)
        if not self.model_file.exists():
            msg = f"Model file not found: {model_file}"
            raise FileNotFoundError(msg)

        # Set default dtype
        if dtype == "float32":
            torch.set_default_dtype(torch.float32)
            self.np_dtype = np.float32
        elif dtype == "float64":
            torch.set_default_dtype(torch.float64)
            self.np_dtype = np.float64
        else:
            msg = f"Unsupported dtype: {dtype}"
            raise ValueError(msg)

        # Load the model
        self.model = torch.jit.load(str(self.model_file))
        self.model.eval()

        # Get model metadata
        self.type_map = self.model.get_type_map()
        self.rcut = self.model.get_rcut()

        # Identify MM types (types starting with 'm' or HW/OW)
        self.mm_type_indices = []
        for i, type_name in enumerate(self.type_map):
            if type_name.startswith("m") or type_name in {"HW", "OW"}:
                self.mm_type_indices.append(i)

        # Store QM atom indices if provided
        self.qm_atoms = qm_atoms

    def compute_qm_correction(
        self,
        coordinates: np.ndarray,
        atom_types: np.ndarray,
        box: Optional[np.ndarray] = None,
    ) -> tuple[float, np.ndarray]:
        """Compute QM internal energy correction for sander QM/MM.

        This method computes the energy and forces for the QM region
        using the MACE model. The results should be used as a correction
        to the QM energy in sander.

        Parameters
        ----------
        coordinates : np.ndarray
            Atomic coordinates in Angstroms, shape (natoms, 3)
        atom_types : np.ndarray
            Atom types as integers, shape (natoms,)
        box : np.ndarray, optional
            Simulation box vectors in Angstroms, shape (3, 3)
            If None, non-periodic boundary conditions are assumed

        Returns
        -------
        energy : float
            Total QM energy in the model's native units (typically eV for MACE models)
        forces : np.ndarray
            Atomic forces in the model's native units (typically eV/Angstrom for MACE),
            shape (natoms, 3). Only QM atoms will have non-zero forces in QM/MM mode

        Notes
        -----
        For QM/MM calculations:
        - MM atoms (types starting with 'm' or HW/OW) get zero energy bias
        - Forces are computed for all atoms but MM-MM interactions are excluded
        - The energy includes QM-QM and QM-MM interactions only
        - Energy and force units depend on the model; MACE models typically use
          eV for energy and eV/Angstrom for forces
        """
        # Ensure inputs are numpy arrays
        coordinates = np.asarray(coordinates, dtype=self.np_dtype)
        atom_types = np.asarray(atom_types, dtype=np.int32)

        # Validate input shapes
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            msg = f"coordinates must have shape (natoms, 3), got {coordinates.shape}"
            raise ValueError(msg)
        if atom_types.ndim != 1:
            msg = f"atom_types must be 1D array, got shape {atom_types.shape}"
            raise ValueError(msg)
        if len(coordinates) != len(atom_types):
            msg = "coordinates and atom_types must have same length"
            raise ValueError(msg)

        # Convert to torch tensors
        coord = torch.from_numpy(coordinates).unsqueeze(0)  # (1, natoms, 3)
        atype = torch.from_numpy(atom_types).unsqueeze(0)  # (1, natoms)

        # Handle box
        if box is not None:
            box = np.asarray(box, dtype=self.np_dtype)
            if box.shape != (3, 3):
                msg = f"box must have shape (3, 3), got {box.shape}"
                raise ValueError(msg)
            box_tensor = torch.from_numpy(box).unsqueeze(0)  # (1, 3, 3)
        else:
            box_tensor = None

        # Forward pass
        with torch.no_grad():
            result = self.model(
                coord=coord,
                atype=atype,
                box=box_tensor,
            )

        # Extract energy and forces
        energy = result["energy"].item()  # Scalar
        forces = result["force"].squeeze(0).numpy()  # (natoms, 3)

        # For QM/MM: zero out forces on MM atoms if qm_atoms is specified
        if self.qm_atoms is not None:
            mm_atoms = [i for i in range(len(atom_types)) if i not in self.qm_atoms]
            forces[mm_atoms] = 0.0

        return energy, forces

    def get_qm_atoms_from_types(self, atom_types: np.ndarray) -> list[int]:
        """Determine QM atoms from atom types.

        QM atoms are those that are NOT MM types (i.e., not starting with 'm'
        and not HW/OW).

        Parameters
        ----------
        atom_types : np.ndarray
            Atom types as integers, shape (natoms,)

        Returns
        -------
        qm_atoms : list[int]
            List of QM atom indices
        """
        qm_atoms = []
        for i, atype in enumerate(atom_types):
            if atype not in self.mm_type_indices:
                qm_atoms.append(i)
        return qm_atoms

    @classmethod
    def from_config(cls, config_file: str | Path) -> "SanderInterface":
        """Create interface from configuration file.

        The configuration file should be a simple text file with key-value pairs:
        model_file=/path/to/model.pth
        qm_atoms=0,1,2,3
        dtype=float64

        Parameters
        ----------
        config_file : str or Path
            Path to configuration file

        Returns
        -------
        interface : SanderInterface
            Initialized sander interface
        """
        config_file = Path(config_file)
        if not config_file.exists():
            msg = f"Config file not found: {config_file}"
            raise FileNotFoundError(msg)

        # Parse config
        config = {}
        with config_file.open() as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    config[key.strip()] = value.strip()

        # Extract parameters
        model_file = config.get("model_file")
        if not model_file:
            msg = "model_file must be specified in config"
            raise ValueError(msg)

        qm_atoms_str = config.get("qm_atoms")
        qm_atoms = (
            [int(x) for x in qm_atoms_str.split(",")]
            if qm_atoms_str
            else None
        )

        dtype = config.get("dtype", "float64")

        return cls(model_file=model_file, qm_atoms=qm_atoms, dtype=dtype)


def compute_qm_energy_sander(
    model_file: str,
    coordinates: np.ndarray,
    atom_types: np.ndarray,
    box: Optional[np.ndarray] = None,
) -> tuple[float, np.ndarray]:
    """Convenience function for sander QM energy calculation.

    This is a simple wrapper around SanderInterface for direct use
    from sander or other programs.

    Parameters
    ----------
    model_file : str
        Path to frozen MACE model file
    coordinates : np.ndarray
        Atomic coordinates in Angstroms, shape (natoms, 3)
    atom_types : np.ndarray
        Atom types as integers, shape (natoms,)
    box : np.ndarray, optional
        Simulation box vectors in Angstroms, shape (3, 3)

    Returns
    -------
    energy : float
        Total QM energy in the model's native units (typically eV for MACE)
    forces : np.ndarray
        Atomic forces in the model's native units (typically eV/Angstrom for MACE),
        shape (natoms, 3)
    """
    interface = SanderInterface(model_file)
    return interface.compute_qm_correction(coordinates, atom_types, box)
