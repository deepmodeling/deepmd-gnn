#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Example wrapper script for sander QM/MM calculations with MACE-OFF.

This script demonstrates how to use the SanderInterface for computing
QM energy corrections in sander QM/MM simulations.
"""

import sys
from pathlib import Path

import numpy as np

from deepmd_gnn import SanderInterface


def main():
    """Run example QM/MM energy calculation."""
    # Configuration file
    config_file = "sander_mace.conf"

    # Check if model exists
    if not Path(config_file).exists():
        print(f"Error: Configuration file not found: {config_file}")
        print("Please create the config file first.")
        return 1

    # Initialize the sander interface
    print("Initializing Sander-MACE interface...")
    try:
        interface = SanderInterface.from_config(config_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease download the MACE-OFF model first:")
        print("  python3 -c \"from deepmd_gnn import download_mace_off_model;")
        print('               download_mace_off_model("small")"')
        return 1

    print(f"Model loaded successfully!")
    print(f"  Type map: {interface.type_map}")
    print(f"  Cutoff radius: {interface.rcut} Å")
    print(f"  MM type indices: {interface.mm_type_indices}")

    # Example system: small organic molecule in water
    # QM region: 3 atoms (C-H-H fragment)
    # MM region: 3 water molecules
    print("\n" + "=" * 60)
    print("Example: QM/MM calculation for organic molecule in water")
    print("=" * 60)

    # Example coordinates (in Angstroms)
    # Atoms 0-2: QM region (C-H-H)
    # Atoms 3-11: MM region (3 water molecules)
    coordinates = np.array(
        [
            # QM atoms (C, H, H)
            [0.0, 0.0, 0.0],  # C
            [1.09, 0.0, 0.0],  # H
            [-0.36, 1.03, 0.0],  # H
            # MM water 1 (HW, OW, HW)
            [3.0, 0.0, 0.0],
            [3.76, 0.59, 0.0],
            [3.24, -0.76, 0.59],
            # MM water 2
            [-3.0, 0.0, 0.0],
            [-3.76, -0.59, 0.0],
            [-3.24, 0.76, -0.59],
            # MM water 3
            [0.0, 3.0, 0.0],
            [0.59, 3.76, 0.0],
            [-0.76, 3.24, 0.59],
        ],
        dtype=np.float64,
    )

    # Atom types (assuming type_map: [C, H, HW, OW])
    # 0=C, 1=H (QM types)
    # 2=HW, 3=OW (MM types)
    atom_types = np.array(
        [
            0,  # C (QM)
            1,  # H (QM)
            1,  # H (QM)
            2,  # HW (MM)
            3,  # OW (MM)
            2,  # HW (MM)
            2,  # HW (MM)
            3,  # OW (MM)
            2,  # HW (MM)
            2,  # HW (MM)
            3,  # OW (MM)
            2,  # HW (MM)
        ],
        dtype=np.int32,
    )

    # Simulation box (10 Angstrom cubic box)
    box = np.eye(3) * 10.0

    print(f"\nSystem details:")
    print(f"  Total atoms: {len(atom_types)}")
    # Count QM atoms (those not in MM type indices)
    qm_count = np.sum(~np.isin(atom_types, interface.mm_type_indices))
    mm_count = np.sum(np.isin(atom_types, interface.mm_type_indices))
    print(f"  QM atoms: {qm_count}")
    print(f"  MM atoms: {mm_count}")

    # Compute QM energy correction
    print("\nComputing QM energy correction...")
    try:
        energy, forces = interface.compute_qm_correction(
            coordinates=coordinates,
            atom_types=atom_types,
            box=box,
        )

        print("\nResults:")
        print(f"  QM Energy: {energy:.6f} eV")
        print(f"  Forces shape: {forces.shape}")
        print(f"  Max force magnitude: {np.max(np.abs(forces)):.6f} eV/Å")

        # Show forces on QM atoms
        print("\n  Forces on QM atoms:")
        for i in range(3):
            print(f"    Atom {i}: [{forces[i, 0]:8.4f}, "
                  f"{forces[i, 1]:8.4f}, {forces[i, 2]:8.4f}] eV/Å")

    except Exception as e:
        print(f"Error during calculation: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("Calculation completed successfully!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
