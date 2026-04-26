# MACE-OFF examples

`deepmd_gnn.mace_off` is meant for the narrow case where you already have a
supported official MACE-OFF checkpoint and want a DeePMD-GNN wrapper without
silently guessing wrapper-only semantics.

The two main entry points have slightly different roles:

- `load_mace_off_model(...)` reconstructs the DeePMD-GNN wrapper
  conservatively from the real checkpoint object and keeps the original eager
  MACE model internally for closer native/wrapper parity
- `convert_mace_off_to_deepmd(...)` performs the export step by scripting the
  wrapped model only at serialization time

Current scope and assumptions:

- supports selected official checkpoints such as `off23_small`
- requires an explicit `sel` neighbor cap
- treats `sel` and MACE's internal `avg_num_neighbors` as different concepts,
  and recovers `avg_num_neighbors` from the checkpoint instead of substituting
  `sel`
- only infers ordinary element `type_map` entries from `atomic_numbers`
- does **not** infer DPRc/QM/MM atom types such as `mH`, `HW`, or `OW`
- relies on `torch.load(..., weights_only=False)` to recover the original
  `ScaleShiftMACE` object, so only trusted checkpoints should be used

Example:

```sh
deepmd-gnn mace-off convert mace_off23_small_dp.pt --model off23_small --sel 64
```

This produces a scripted DeePMD-GNN wrapper for downstream use. It is still a
good idea to validate the final engine-specific workflow separately.
