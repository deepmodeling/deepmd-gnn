"""Main entry point for the command line interface."""

import argparse
import logging
import sys
from pathlib import Path

from deepmd_gnn.mace_off import convert_mace_off_to_deepmd, download_mace_off_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Run the main CLI."""
    parser = argparse.ArgumentParser(
        description="DeePMD-GNN utilities",
        prog="deepmd-gnn",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # MACE-OFF download command
    download_parser = subparsers.add_parser(
        "download-mace-off",
        help="Download MACE-OFF pretrained models",
    )
    download_parser.add_argument(
        "model",
        choices=["small", "medium", "large"],
        help="MACE-OFF model size to download",
    )
    download_parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache the downloaded model (default: ~/.cache/deepmd-gnn/mace-off/)",
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists",
    )

    # MACE-OFF convert command
    convert_parser = subparsers.add_parser(
        "convert-mace-off",
        help="Convert MACE-OFF model to DeePMD format",
    )
    convert_parser.add_argument(
        "model",
        choices=["small", "medium", "large"],
        help="MACE-OFF model size to convert",
    )
    convert_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="frozen_model.pth",
        help="Output file name for the frozen model (default: frozen_model.pth)",
    )
    convert_parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory where MACE-OFF models are cached",
    )

    args = parser.parse_args()

    if args.command == "download-mace-off":
        cache_dir = Path(args.cache_dir) if args.cache_dir else None
        model_path = download_mace_off_model(
            model_name=args.model,
            cache_dir=cache_dir,
            force_download=args.force,
        )
        logger.info("Model downloaded to: %s", model_path)
        return 0

    if args.command == "convert-mace-off":
        cache_dir = Path(args.cache_dir) if args.cache_dir else None
        output_path = convert_mace_off_to_deepmd(
            model_name=args.model,
            output_file=args.output,
            cache_dir=cache_dir,
        )
        logger.info("Converted model saved to: %s", output_path)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
