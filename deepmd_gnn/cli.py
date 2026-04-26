"""Command-line interface for DeePMD-GNN."""

from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

from deepmd_gnn.mace_off_cli import (
    MACE_OFF_MODEL_CHOICES,
    download_mace_off_model,
    get_mace_off_cache_dir,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command-line parser."""
    parser = argparse.ArgumentParser(
        prog="deepmd-gnn",
        description="Command-line tools for DeePMD-GNN.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    mace_off_parser = subparsers.add_parser(
        "mace-off",
        help="Utilities for selected official MACE-OFF checkpoints.",
    )
    mace_off_subparsers = mace_off_parser.add_subparsers(
        dest="mace_off_command",
        required=True,
    )

    convert_parser = mace_off_subparsers.add_parser(
        "convert",
        help="Convert a supported MACE-OFF checkpoint to a DeePMD-GNN TorchScript model.",
    )
    convert_parser.add_argument(
        "output_file",
        help="Output path for the exported DeePMD-GNN TorchScript model.",
    )
    convert_parser.add_argument(
        "--sel",
        type=int,
        required=True,
        help="Required DeePMD runtime neighbor cap.",
    )
    convert_source_group = convert_parser.add_mutually_exclusive_group(required=True)
    convert_source_group.add_argument(
        "--model",
        choices=MACE_OFF_MODEL_CHOICES,
        help="Official MACE-OFF model name or alias to download.",
    )
    convert_source_group.add_argument(
        "--model-path",
        type=Path,
        help="Use a trusted local checkpoint path instead of downloading an official model.",
    )
    convert_parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Cache directory for downloaded official checkpoints.",
    )

    download_parser = mace_off_subparsers.add_parser(
        "download",
        help="Download a supported official MACE-OFF checkpoint.",
    )
    download_parser.add_argument(
        "--model",
        required=True,
        choices=MACE_OFF_MODEL_CHOICES,
        help="Official MACE-OFF model name or alias to download.",
    )
    download_parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Cache directory for downloaded official checkpoints.",
    )

    mace_off_subparsers.add_parser(
        "cache-dir",
        help="Print the cache directory used for MACE-OFF checkpoints.",
    )

    return parser


def _parser_error(parser: argparse.ArgumentParser, message: str) -> NoReturn:
    parser.error(message)
    raise AssertionError("unreachable")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "mace-off":
        if args.mace_off_command == "convert":
            convert_mace_off_to_deepmd = import_module(
                "deepmd_gnn.mace_off",
            ).convert_mace_off_to_deepmd

            output_path = convert_mace_off_to_deepmd(
                output_file=args.output_file,
                sel=args.sel,
                model_name=args.model,
                model_path=args.model_path,
                cache_dir=args.cache_dir,
            )
            sys.stdout.write(f"{output_path}\n")
            return 0
        if args.mace_off_command == "download":
            model_path = download_mace_off_model(
                model_name=args.model,
                cache_dir=args.cache_dir,
            )
            sys.stdout.write(f"{model_path}\n")
            return 0
        if args.mace_off_command == "cache-dir":
            sys.stdout.write(f"{get_mace_off_cache_dir()}\n")
            return 0

    _parser_error(parser, "Unhandled command")
