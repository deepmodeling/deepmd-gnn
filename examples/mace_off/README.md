# MACE-OFF examples

`deepmd_gnn.mace_off` supports a conservative conversion path for selected
official MACE-OFF checkpoints.

Current scope:

- downloads selected official checkpoints such as `off23_small`
- reconstructs a DeePMD-GNN `MaceModel` only when the checkpoint structure can
  be recovered conservatively from the real model object
- requires an explicit `sel` neighbor cap
- does **not** infer DPRc/QM/MM atom types such as `mH`, `HW`, or `OW`

Example:

```python
from deepmd_gnn.mace_off import convert_mace_off_to_deepmd

convert_mace_off_to_deepmd(
    output_file="mace_off23_small_dp.pt",
    model_name="off23_small",
    sel=64,
)
```

This produces a scripted DeePMD-GNN wrapper. Validate your final downstream
engine workflow separately.
