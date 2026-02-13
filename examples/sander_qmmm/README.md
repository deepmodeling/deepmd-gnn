# Sander QM/MM with MACE-OFF Example

This example demonstrates how to use MACE-OFF models for QM internal energy correction in sander QM/MM simulations.

## Overview

This setup shows how to:
1. Download and prepare a MACE-OFF pretrained model
2. Configure sander for QM/MM calculations with MACE
3. Run QM/MM simulations with energy corrections from MACE-OFF

## Prerequisites

- DeePMD-GNN with MACE support installed
- AMBER/AmberTools with sander
- Python 3.9 or later

## Quick Start

### 1. Download MACE-OFF Model

```python
from deepmd_gnn import download_mace_off_model, convert_mace_off_to_deepmd

# Download MACE-OFF small model (recommended for QM/MM)
model_path = download_mace_off_model("small")

# Convert to DeePMD format if needed
deepmd_model = convert_mace_off_to_deepmd("small", "mace_off_qmmm.pth")
```

### 2. Prepare Configuration

Create a configuration file `sander_mace.conf`:

```
model_file=mace_off_qmmm.pth
dtype=float64
```

### 3. Use in Python Scripts

```python
from deepmd_gnn import SanderInterface
import numpy as np

# Initialize interface
interface = SanderInterface.from_config("sander_mace.conf")

# Example coordinates (3 atoms)
coords = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0]
])

# Atom types (0=H, 1=C, etc. according to type_map)
types = np.array([0, 1, 0])

# Optional: simulation box for periodic systems
box = np.eye(3) * 10.0  # 10 Angstrom cubic box

# Compute QM energy correction
energy, forces = interface.compute_qm_correction(coords, types, box)

print(f"QM Energy: {energy:.6f} eV")
print(f"Forces shape: {forces.shape}")
```

## Type Map for QM/MM

When defining your system, use the following convention:
- **QM atoms**: Use standard element symbols (H, C, N, O, etc.)
- **MM atoms**: Prefix with 'm' (mH, mC, mN, etc.) or use HW/OW for water

Example type_map for a QM/MM system:
```json
{
  "type_map": ["C", "H", "O", "N", "mC", "mH", "mO", "HW", "OW"]
}
```

In this example:
- C, H, O, N are QM atoms (will be treated with MACE-OFF)
- mC, mH, mO, HW, OW are MM atoms (classical force field)

## Integration with Sander

For direct integration with sander, you can create a Python wrapper that:
1. Reads coordinates from sander
2. Calls MACE-OFF for QM region energy/forces
3. Returns corrections to sander

See `sander_wrapper.py` for an example implementation.

## Model Selection

MACE-OFF provides three model sizes:

| Model | Parameters | Speed | Accuracy | Best For |
|-------|-----------|-------|----------|----------|
| small | ~1M | Fast | Good | QM/MM, screening |
| medium | ~5M | Medium | Better | Production runs |
| large | ~20M | Slow | Best | High-accuracy calculations |

For QM/MM simulations, the **small** model is recommended for a good balance of speed and accuracy.

## Notes

- MACE-OFF models are pretrained on diverse molecular datasets
- QM/MM boundary is automatically handled by the DPRc mechanism
- Forces on MM atoms from QM interactions are computed automatically
- Energy units: typically eV (check model documentation)
- Force units: typically eV/Angstrom

## References

- MACE: https://github.com/ACEsuit/mace
- MACE-OFF: Foundation model for molecular simulation
- DeePMD-GNN: https://gitlab.com/RutgersLBSR/deepmd-gnn
- DPRc in AMBER: AmberTools documentation

## Troubleshooting

**Issue**: Model file not found
- **Solution**: Run `download_mace_off_model()` first to download the model

**Issue**: Type mapping errors
- **Solution**: Ensure all atom types in your system are defined in the model's type_map

**Issue**: Forces seem incorrect
- **Solution**: Check units (MACE uses eV and Angstrom by default)
