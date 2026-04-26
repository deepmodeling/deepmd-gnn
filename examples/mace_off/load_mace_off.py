"""Example script for converting an official MACE-OFF checkpoint."""

from deepmd_gnn.mace_off import convert_mace_off_to_deepmd


def main() -> None:
    """Convert the off23_small checkpoint into a TorchScript DeePMD-GNN model."""
    convert_mace_off_to_deepmd(
        output_file="mace_off23_small_dp.pt",
        model_name="off23_small",
        sel=64,
    )


if __name__ == "__main__":
    main()
