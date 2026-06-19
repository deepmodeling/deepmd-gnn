"""LAMMPS integration tests."""

from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WATER_DATA = REPO_ROOT / "tests" / "data"
MACE_INPUT = REPO_ROOT / "tests" / "mace.json"
SIX_ATOM_BOX = np.array([0.0, 13.0, 0.0, 13.0, 0.0, 13.0, 0.0, 0.0, 0.0])
SIX_ATOM_COORD = np.array(
    [
        [12.83, 2.56, 2.18],
        [12.09, 2.87, 2.74],
        [0.25, 3.32, 1.68],
        [3.36, 3.00, 1.81],
        [3.51, 2.51, 2.60],
        [4.27, 3.22, 1.56],
    ],
)
SIX_ATOM_TYPES = np.array([1, 1, 1, 2, 2, 2])


def _missing_lammps(message: str) -> NoReturn:
    if os.environ.get("DEEPMD_GNN_REQUIRE_LAMMPS") == "1":
        pytest.fail(message)
    pytest.skip(message)
    raise RuntimeError(message)


def _prepend_env_path(env: dict[str, str], name: str, *paths: Path | str) -> None:
    entries = [str(path) for path in paths if str(path)]
    current = env.get(name)
    if current:
        entries.append(current)
    env[name] = os.pathsep.join(entries)


def _deepmd_lib_dir() -> Path:
    try:
        deepmd_env = importlib.import_module("deepmd.env")
    except ImportError as exc:  # pragma: no cover - depends on optional install
        _missing_lammps(f"Cannot import deepmd.env: {exc}")
    return Path(deepmd_env.SHARED_LIB_DIR)


def _deepmd_gnn_plugin() -> Path:
    try:
        deepmd_gnn_lib = importlib.import_module("deepmd_gnn.lib")
    except ImportError as exc:  # pragma: no cover - depends on local build
        _missing_lammps(f"Cannot import deepmd_gnn.lib: {exc}")

    shared_lib_dir = Path(deepmd_gnn_lib.__path__[0])
    if platform.system() == "Windows":
        candidates = [shared_lib_dir / "deepmd_gnn.dll"]
    else:
        candidates = [shared_lib_dir / "libdeepmd_gnn.so"]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    _missing_lammps(f"Cannot find deepmd-gnn C++ plugin in {shared_lib_dir}")
    msg = "unreachable"
    raise AssertionError(msg)


def _ensure_lammps_available() -> None:
    if shutil.which("lmp") is not None or shutil.which("lmp_serial") is not None:
        return
    if importlib.util.find_spec("lammps") is None:
        _missing_lammps("Cannot find lmp executable or import lammps")


def _ensure_mpi_lammps_available() -> tuple[str, str]:
    mpi = shutil.which("mpirun") or shutil.which("mpiexec")
    if mpi is None:
        _missing_lammps("Cannot find mpirun or mpiexec")
    lmp = shutil.which("lmp")
    if lmp is None:
        _missing_lammps("Cannot find MPI-capable lmp executable")
    return mpi, lmp


def _lammps_env() -> dict[str, str]:
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - depends on optional install
        _missing_lammps(f"Cannot import torch: {exc}")
    try:
        tf = importlib.import_module("tensorflow")
    except ImportError as exc:  # pragma: no cover - depends on optional install
        _missing_lammps(f"Cannot import tensorflow runtime required by LAMMPS: {exc}")

    env = os.environ.copy()
    deepmd_lib_dir = _deepmd_lib_dir()
    deepmd_gnn_plugin = _deepmd_gnn_plugin()
    torch_lib_dir = Path(torch.__path__[0]) / "lib"
    tensorflow_lib_dir = Path(tf.sysconfig.get_lib())

    env["LAMMPS_PLUGIN_PATH"] = str(deepmd_lib_dir)
    env["DP_PLUGIN_PATH"] = str(deepmd_gnn_plugin)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")

    if platform.system() == "Darwin":
        lib_env = "DYLD_FALLBACK_LIBRARY_PATH"
    else:
        lib_env = "LD_LIBRARY_PATH"
    _prepend_env_path(
        env,
        lib_env,
        deepmd_lib_dir,
        deepmd_gnn_plugin.parent,
        torch_lib_dir,
        tensorflow_lib_dir,
        tensorflow_lib_dir / "python",
    )
    return env


def _run_command(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _freeze_mace_model(tmp_path: Path) -> Path:
    work_dir = tmp_path / "model"
    shutil.copytree(WATER_DATA, work_dir / "data")
    shutil.copy(MACE_INPUT, work_dir / "mace.json")

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    _run_command(
        [sys.executable, "-m", "deepmd", "--pt", "train", "mace.json"],
        work_dir,
        env,
    )

    model_file = work_dir / "model.pth"
    _run_command(
        [sys.executable, "-m", "deepmd", "--pt", "freeze", "-o", str(model_file)],
        work_dir,
        env,
    )
    return model_file


def _write_lammps_data(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# generated by tests/test_lammps.py",
                "",
                f"{SIX_ATOM_COORD.shape[0]} atoms",
                f"{int(SIX_ATOM_TYPES.max())} atom types",
                "",
                f"{SIX_ATOM_BOX[0]:.10e} {SIX_ATOM_BOX[1]:.10e} xlo xhi",
                f"{SIX_ATOM_BOX[2]:.10e} {SIX_ATOM_BOX[3]:.10e} ylo yhi",
                f"{SIX_ATOM_BOX[4]:.10e} {SIX_ATOM_BOX[5]:.10e} zlo zhi",
                (
                    f"{SIX_ATOM_BOX[6]:.10e} {SIX_ATOM_BOX[7]:.10e} "
                    f"{SIX_ATOM_BOX[8]:.10e} xy xz yz"
                ),
                "",
                "Masses",
                "",
                "1 15.999",
                "2 1.008",
                "",
                "Atoms # atomic",
                "",
                *(
                    f"{idx} {atom_type} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}"
                    for idx, (atom_type, xyz) in enumerate(
                        zip(SIX_ATOM_TYPES, SIX_ATOM_COORD, strict=True),
                        start=1,
                    )
                ),
                "",
            ],
        ),
    )


def _write_lammps_input(
    path: Path,
    data_file: Path,
    model_file: Path,
    processors: str | None = None,
) -> None:
    processor_commands = [] if processors is None else [f"processors {processors}"]
    path.write_text(
        "\n".join(
            [
                "units metal",
                "boundary p p p",
                "atom_style atomic",
                "neighbor 2.0 bin",
                "neigh_modify every 1 delay 0 check yes",
                *processor_commands,
                f"read_data {data_file}",
                f"pair_style deepmd {model_file}",
                "pair_coeff * * O H",
                "thermo_style custom step pe etotal temp press",
                "thermo 1",
                "run 0",
                "",
            ],
        ),
    )


def _run_lammps(input_file: Path, log_file: Path, env: dict[str, str]) -> None:
    lmp = shutil.which("lmp") or shutil.which("lmp_serial")
    if lmp is not None:
        result = subprocess.run(
            [lmp, "-in", str(input_file), "-log", str(log_file), "-screen", "none"],
            cwd=input_file.parent,
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return

    try:
        lammps_module = importlib.import_module("lammps")
    except ImportError as exc:  # pragma: no cover - depends on optional install
        _missing_lammps(f"Cannot find lmp executable or import lammps: {exc}")

    old_env = os.environ.copy()
    os.environ.update(env)
    try:
        lmp_instance = lammps_module.lammps(
            cmdargs=["-log", str(log_file), "-screen", "none"],
        )
        try:
            lmp_instance.file(str(input_file))
        finally:
            lmp_instance.close()
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def _run_lammps_mpi(
    input_file: Path,
    log_file: Path,
    env: dict[str, str],
    *,
    nprocs: int = 2,
) -> None:
    mpi, lmp = _ensure_mpi_lammps_available()
    result = subprocess.run(
        [
            mpi,
            "-np",
            str(nprocs),
            lmp,
            "-in",
            str(input_file),
            "-log",
            str(log_file),
            "-screen",
            "none",
        ],
        cwd=input_file.parent,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _read_step_zero_pe(log_file: Path) -> float:
    log_text = log_file.read_text()
    assert "Loop time" in log_text
    thermo_match = re.search(
        r"^\s*0\s+(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)",
        log_text,
        re.MULTILINE,
    )
    assert thermo_match is not None
    pe = float(thermo_match.group(1))
    assert np.isfinite(pe)
    return pe


@pytest.mark.lammps
def test_lammps_runs_six_atom_mace_model(tmp_path: Path) -> None:
    """Run one LAMMPS step with a frozen MACE model through DeePMD-kit."""
    _ensure_lammps_available()
    lammps_env = _lammps_env()
    model_file = _freeze_mace_model(tmp_path)

    data_file = tmp_path / "water.lmp"
    input_file = tmp_path / "in.lammps"
    log_file = tmp_path / "log.lammps"

    _write_lammps_data(data_file)
    _write_lammps_input(input_file, data_file, model_file.resolve())
    _run_lammps(input_file, log_file, lammps_env)

    _read_step_zero_pe(log_file)


@pytest.mark.lammps
@pytest.mark.mpi
def test_lammps_mpi_matches_single_rank_six_atom_mace_model(tmp_path: Path) -> None:
    """Run frozen MACE through DeePMD-kit with two MPI ranks."""
    _ensure_lammps_available()
    _ensure_mpi_lammps_available()
    lammps_env = _lammps_env()
    model_file = _freeze_mace_model(tmp_path)

    data_file = tmp_path / "water.lmp"
    single_input = tmp_path / "in.single.lammps"
    mpi_input = tmp_path / "in.mpi.lammps"
    single_log = tmp_path / "log.single.lammps"
    mpi_log = tmp_path / "log.mpi.lammps"

    _write_lammps_data(data_file)
    _write_lammps_input(single_input, data_file, model_file.resolve())
    _write_lammps_input(mpi_input, data_file, model_file.resolve(), processors="2 1 1")

    _run_lammps(single_input, single_log, lammps_env)
    _run_lammps_mpi(mpi_input, mpi_log, lammps_env, nprocs=2)

    np.testing.assert_allclose(
        _read_step_zero_pe(mpi_log),
        _read_step_zero_pe(single_log),
        atol=1e-8,
        rtol=0.0,
    )
