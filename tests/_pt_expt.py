"""Helpers for pt_expt integration tests."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def skip_cuda_pt_expt_aoti_if_requested() -> None:
    """Skip pt_expt AOTI packaging in CUDA CI when the runner cannot compile it."""
    if os.environ.get("DEEPMD_GNN_SKIP_CUDA_PT_EXPT_AOTI") == "1":
        pytest.skip("pt_expt AOTI packaging is covered by the CPU workflow")


def register_pt_expt_plugin_for_cli(work_dir: Path) -> None:
    """Load deepmd-gnn's pt_expt entry point before the deepmd CLI runs."""
    (work_dir / "sitecustomize.py").write_text(
        """# Temporary workaround for deepmodeling/deepmd-kit#5559.
try:
    import deepmd_gnn.argcheck  # noqa: F401
    import deepmd_gnn.pt_expt  # noqa: F401
except Exception as exc:
    raise SystemExit(
        f'Failed to register deepmd-gnn pt_expt plugin: {exc}'
    ) from exc
""",
    )


def pt_expt_cli_env(
    work_dir: Path,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment that imports the temporary CLI registration hook."""
    register_pt_expt_plugin_for_cli(work_dir)
    if env is None:
        env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    entries = [str(work_dir)]
    if pythonpath:
        entries.append(pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env
