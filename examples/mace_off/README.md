# Using MACE-OFF Pretrained Models with DeePMD-GNN

This example demonstrates how to load MACE-OFF foundation models and use them with DeePMD-kit for MD simulations, including QM/MM simulations with AMBER/sander.

## Overview

MACE-OFF models are pretrained foundation models for molecular systems. This integration allows you to:
1. Download MACE-OFF models (small, medium, large)
2. Load them into DeePMD-GNN's MaceModel wrapper
3. Use them with MD packages (LAMMPS, AMBER/sander) through DeePMD-kit

## Architecture

```
MACE-OFF pretrained model
    ↓
DeePMD-GNN MaceModel wrapper
    ↓
DeePMD-kit interface
    ↓
MD packages (LAMMPS, AMBER/sander, etc.)
```

## Quick Start

### 1. Download and Convert MACE-OFF Model

```python
from deepmd_gnn import convert_mace_off_to_deepmd

# Download and convert MACE-OFF small model to frozen DeePMD format
model_path = convert_mace_off_to_deepmd("small", "frozen_model.pth")
```

Available models:
- `"small"`: Fast, good for screening and QM/MM
- `"medium"`: Balanced speed and accuracy
- `"large"`: Best accuracy, slower

### 2. Use with LAMMPS

```bash
# Set plugin path
export DP_PLUGIN_PATH=/path/to/libdeepmd_gnn.so

# In LAMMPS input:
pair_style deepmd frozen_model.pth
pair_coeff * *
```

### 3. Use with AMBER/sander for QM/MM

The MACE-OFF model integrates with AMBER through DeePMD-kit's existing interface.

#### QM/MM Type Map Convention

For QM/MM simulations, use the DPRc mechanism:
- **QM atoms**: Standard element symbols (H, C, N, O, etc.)
- **MM atoms**: Prefix with 'm' (mH, mC, etc.) or use HW/OW for water

#### Integration with sander

The frozen_model.pth works with sander through DeePMD-kit's AMBER interface. See DeePMD-kit and AMBER documentation for setup details.

## Programmatic Usage

### Load Model Directly

```python
from deepmd_gnn import load_mace_off_model

# Load MACE-OFF model as DeePMD-GNN MaceModel
model = load_mace_off_model("small")

# This is now a MaceModel instance that can be used with DeePMD-kit
print(f"Type map: {model.get_type_map()}")
print(f"Cutoff: {model.get_rcut()}")
```

## Model Details

### MACE-OFF Models

| Model | Parameters | Speed | Best For |
|-------|-----------|-------|----------|
| small | ~1M | Fast | QM/MM, screening, quick simulations |
| medium | ~5M | Medium | Production MD runs |
| large | ~20M | Slow | High-accuracy calculations |

### Energy and Force Units

MACE models use:
- **Energy**: eV
- **Forces**: eV/Angstrom  
- **Coordinates**: Angstrom

## References

- MACE: https://github.com/ACEsuit/mace
- MACE-OFF: Foundation models for molecular simulation
- DeePMD-kit: https://github.com/deepmodeling/deepmd-kit
- DeePMD-GNN: https://gitlab.com/RutgersLBSR/deepmd-gnn
