"""Test version."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.metadata import version

from deepmd_gnn import __version__


def test_version() -> None:
    """Test version."""
    assert version("deepmd-gnn") == __version__


def test_import_does_not_disable_weights_only_loading() -> None:
    """Importing the plugin must not change process-wide torch.load policy."""
    env = os.environ.copy()
    env.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; import deepmd_gnn; import deepmd_gnn.mace; "
                "import deepmd_gnn.nequip; "
                "assert 'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD' not in os.environ"
            ),
        ],
        check=True,
        env=env,
    )
