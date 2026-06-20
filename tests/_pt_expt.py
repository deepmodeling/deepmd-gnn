"""Helpers for pt_expt integration tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from packaging.version import Version

if TYPE_CHECKING:
    from pathlib import Path

DEEPMD_KIT_PT2_MIN_VERSION = Version("3.2.0b0")


def register_pt_expt_plugin_for_cli(work_dir: Path) -> None:
    """Load deepmd-gnn's pt_expt entry point before the deepmd CLI runs."""
    (work_dir / "sitecustomize.py").write_text(
        "# Temporary workaround for deepmodeling/deepmd-kit#5559.\ntry:\n    import deepmd_gnn.pt_expt  # noqa: F401\nexcept Exception as exc:\n    raise SystemExit(\n        f'Failed to register deepmd-gnn pt_expt plugin: {exc}'\n    ) from exc\n",
    )
