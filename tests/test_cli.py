"""Tests for DeePMD-GNN CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from deepmd_gnn import cli


def test_cli_cache_dir_prints_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The cache-dir subcommand should print the configured cache path."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    exit_code = cli.main(["mace-off", "cache-dir"])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "deepmd-gnn" / "mace-off")


def test_cli_download_invokes_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The download subcommand should pass parsed arguments to the helper."""
    expected_model = tmp_path / "MACE-OFF23_small.model"
    recorded: dict[str, object] = {}

    def fake_download(
        model_name: str,
        cache_dir: Path | None,
    ) -> Path:
        recorded["model_name"] = model_name
        recorded["cache_dir"] = cache_dir
        return expected_model

    monkeypatch.setattr(cli, "download_mace_off_model", fake_download)

    exit_code = cli.main(
        [
            "mace-off",
            "download",
            "--model",
            "off23_small",
            "--cache-dir",
            str(tmp_path),
        ],
    )

    assert exit_code == 0
    assert recorded == {
        "model_name": "off23_small",
        "cache_dir": tmp_path,
    }
    assert capsys.readouterr().out.strip() == str(expected_model)


def test_cli_convert_invokes_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The convert subcommand should pass parsed arguments to the helper."""
    output_file = tmp_path / "converted.pt"
    model_path = tmp_path / "checkpoint.model"
    cache_dir = tmp_path / "cache"
    recorded: dict[str, object] = {}

    def fake_convert(
        output_file: str,
        *,
        sel: int,
        model_name: str | None,
        model_path: Path | None,
        cache_dir: Path | None,
        device: str = "cpu",
    ) -> Path:
        recorded["output_file"] = output_file
        recorded["sel"] = sel
        recorded["model_name"] = model_name
        recorded["model_path"] = model_path
        recorded["cache_dir"] = cache_dir
        recorded["device"] = device
        return Path(output_file)

    import deepmd_gnn.mace_off as mace_off

    monkeypatch.setattr(mace_off, "convert_mace_off_to_deepmd", fake_convert)

    exit_code = cli.main(
        [
            "mace-off",
            "convert",
            str(output_file),
            "--sel",
            "64",
            "--model-path",
            str(model_path),
            "--cache-dir",
            str(cache_dir),
        ],
    )

    assert exit_code == 0
    assert recorded == {
        "output_file": str(output_file),
        "sel": 64,
        "model_name": None,
        "model_path": model_path,
        "cache_dir": cache_dir,
        "device": "cpu",
    }
    assert capsys.readouterr().out.strip() == str(output_file)


def test_cli_convert_requires_explicit_model_or_model_path() -> None:
    """The convert subcommand should require exactly one source argument."""
    with pytest.raises(SystemExit, match="2"):
        cli.main(["mace-off", "convert", "out.pt", "--sel", "64"])


def test_cli_download_rejects_unknown_model_choice() -> None:
    """The download subcommand should reject unsupported model names at parse time."""
    with pytest.raises(SystemExit, match="2"):
        cli.main(["mace-off", "download", "--model", "not-a-model"])
