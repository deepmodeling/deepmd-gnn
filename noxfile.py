"""Nox configuration file."""

from __future__ import annotations

import nox

nox.options.sessions = ["tests"]

PYTORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
# MACE pins e3nn 0.4.4, while deepmd-kit 3.2 declares e3nn>=0.5.9.
MACE_E3NN = "e3nn==0.4.4"
PROJECT_RUNTIME_DEPS = [
    "mace-torch>=0.3.5",
    "nequip",
    MACE_E3NN,
    "dargs",
]
PROJECT_TEST_DEPS = [
    "pytest",
    "pytest-cov",
    "dargs>=0.4.8",
]


def install_project_deps(session: nox.Session) -> None:
    """Install project dependencies without re-solving deepmd-kit."""
    session.install(
        *PROJECT_RUNTIME_DEPS,
        *PROJECT_TEST_DEPS,
        "--extra-index-url",
        PYTORCH_CPU_INDEX_URL,
    )


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
        "--no-deps",
        "-e.",
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
    install_project_deps(session)
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
    install_project_deps(session)
    install_editable(session)
    session.run(
        "pytest",
        "-m",
        "lammps",
        "tests/test_lammps.py",
        env={"DEEPMD_GNN_REQUIRE_LAMMPS": "1"},
    )
