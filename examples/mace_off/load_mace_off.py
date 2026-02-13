#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Example script to load and convert MACE-OFF models."""

import sys

from deepmd_gnn import convert_mace_off_to_deepmd, load_mace_off_model


def main():
    """Load and convert MACE-OFF model."""
    print("=" * 70)
    print("MACE-OFF Model Loading Example")
    print("=" * 70)
    print()

    # Example 1: Load model programmatically
    print("Example 1: Loading MACE-OFF model into DeePMD-GNN wrapper")
    print("-" * 70)
    try:
        model = load_mace_off_model("small")
        print(f"✓ Model loaded successfully!")
        print(f"  Type map: {model.get_type_map()}")
        print(f"  Cutoff radius: {model.get_rcut()} Å")
        print()
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print()

    # Example 2: Convert to frozen format
    print("Example 2: Converting to frozen DeePMD format")
    print("-" * 70)
    try:
        frozen_path = convert_mace_off_to_deepmd(
            "small", "mace_off_frozen.pth"
        )
        print(f"✓ Model converted and saved to: {frozen_path}")
        print()
        print("You can now use this model with:")
        print("  • LAMMPS (with DP_PLUGIN_PATH set)")
        print("  • AMBER/sander (through DeePMD-kit interface)")
        print("  • Other MD packages supported by DeePMD-kit")
        print()
    except Exception as e:
        print(f"✗ Failed to convert model: {e}")
        print()

    print("=" * 70)
    print("For QM/MM simulations:")
    print("  - Use standard elements (H, C, N, O) for QM atoms")
    print("  - Use 'm' prefix (mH, mC) or HW/OW for MM atoms")
    print("  - The DPRc mechanism handles QM/MM separation automatically")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
