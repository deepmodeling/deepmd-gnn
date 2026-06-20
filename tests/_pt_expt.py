"""Helpers for pt_expt integration tests."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import TYPE_CHECKING

import pytest
from packaging.version import InvalidVersion, Version

if TYPE_CHECKING:
    from pathlib import Path

DEEPMD_KIT_PT2_MIN_VERSION = Version("3.2.0b0")


def require_deepmd_kit_pt2_support() -> None:
    """Skip when deepmd-kit is too old for the pt2 path."""
    try:
        installed = Version(package_version("deepmd-kit"))
    except PackageNotFoundError:
        pytest.skip("Cannot determine deepmd-kit version")
    except InvalidVersion as exc:
        pytest.skip(f"Cannot parse deepmd-kit version: {exc}")

    if installed < DEEPMD_KIT_PT2_MIN_VERSION:
        pytest.skip(
            "deepmd-kit >= "
            f"{DEEPMD_KIT_PT2_MIN_VERSION} is required for pt2 tests; "
            f"found {installed}",
        )


def register_pt_expt_plugin_for_cli(work_dir: Path) -> None:
    """Load deepmd-gnn's pt_expt entry point before the deepmd CLI runs."""
    (work_dir / "sitecustomize.py").write_text(
        "# Temporary workaround for deepmodeling/deepmd-kit#5559.\ntry:\n    import deepmd_gnn.pt_expt  # noqa: F401\nexcept Exception as exc:\n    raise SystemExit(\n        f'Failed to register deepmd-gnn pt_expt plugin: {exc}'\n    ) from exc\n",
    )
