"""Nox configuration file."""

from __future__ import annotations

import nox

nox.options.sessions = ["tests"]

PYTORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
UV_OVERRIDES = "requirements-overrides.txt"
MACE_E3NN = "e3nn==0.4.4"
RUNTIME_DEPS = [
    "mace-torch>=0.3.5",
    "nequip<0.7",
    MACE_E3NN,
    "dargs",
]
TEST_DEPS = [
    "pytest",
    "pytest-cov",
    "dargs>=0.4.8",
]


def install_deps(
    session: nox.Session,
    deepmd_requirement: str,
    *extra_requirements: str,
) -> None:
    """Install all dependencies with MACE's e3nn override in one resolution."""
    session.install(
        "numpy",
        deepmd_requirement,
        *RUNTIME_DEPS,
        *TEST_DEPS,
        *extra_requirements,
        "--overrides",
        UV_OVERRIDES,
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
    install_deps(session, "deepmd-kit[torch]>=3.2.0b0")
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
    install_deps(
        session,
        "deepmd-kit[torch,lmp,cpu]>=3.2.0b0",
        "mpi4py",
        "mpich",
    )
    install_editable(session)
    session.run(
        "pytest",
        "-m",
        "lammps",
        "tests/test_lammps.py",
        env={"DEEPMD_GNN_REQUIRE_LAMMPS": "1"},
    )
