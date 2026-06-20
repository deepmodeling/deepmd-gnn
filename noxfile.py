"""Nox configuration file."""

from __future__ import annotations

import nox

nox.options.sessions = ["tests"]

PYTORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"


def install_editable(session: nox.Session) -> None:
    """Install this package against the PyTorch in the nox environment."""
    cmake_prefix_path = session.run(
        "python",
        "-c",
        "import torch;print(torch.utils.cmake_prefix_path)",
        silent=True,
    ).strip()
    session.log(f"{cmake_prefix_path=}")
    session.install(
        "-e.[test]",
        "deepmd-kit[torch]>=3.2.0b0",
        env={"CMAKE_PREFIX_PATH": cmake_prefix_path},
    )


@nox.session
def tests(session: nox.Session) -> None:
    """Run test suite with pytest."""
    session.install(
        "numpy",
        "deepmd-kit[torch]>=3.2.0b0",
        "--extra-index-url",
        PYTORCH_CPU_INDEX_URL,
    )
    install_editable(session)
    session.run(
        "pytest",
        "--cov",
        "--cov-config",
        "pyproject.toml",
        "--cov-report",
        "term",
        "--cov-report",
        "xml",
    )


@nox.session
def lammps(session: nox.Session) -> None:
    """Run LAMMPS integration tests."""
    session.install(
        "numpy",
        "deepmd-kit[torch,lmp,cpu]>=3.2.0b0",
        "mpi4py",
        "mpich",
        "--extra-index-url",
        PYTORCH_CPU_INDEX_URL,
    )
    install_editable(session)
    session.run(
        "pytest",
        "-m",
        "lammps",
        "tests/test_lammps.py",
        env={"DEEPMD_GNN_REQUIRE_LAMMPS": "1"},
    )
