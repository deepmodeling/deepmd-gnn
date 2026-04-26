# SPDX-License-Identifier: LGPL-3.0-or-later
"""Lightweight helpers for official MACE-OFF checkpoint selection and download."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

_MACE_OFF_BASE_URL = "https://raw.githubusercontent.com/ACEsuit/mace-off/v0.2"

MACE_OFF_MODELS = {
    "off23_small": f"{_MACE_OFF_BASE_URL}/mace_off23/MACE-OFF23_small.model",
    "off23_medium": f"{_MACE_OFF_BASE_URL}/mace_off23/MACE-OFF23_medium.model",
    "off23_large": f"{_MACE_OFF_BASE_URL}/mace_off23/MACE-OFF23_large.model",
    "off24_medium": f"{_MACE_OFF_BASE_URL}/mace_off24/MACE-OFF24_medium.model",
}

MACE_OFF_MODEL_SHA256 = {
    "off23_small": "165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f",
    "off23_medium": "4842c52ad210d6e1f84d6cf1ffa70fae25a7e0d755ed55cf223f43913f587db7",
    "off23_large": "a29e397dbf3e7a24ac50a9b0dfc919bd5a62efa346f5895a6237b0950c1d76f4",
    "off24_medium": "e5ccf5837f685899811a68754e7c994393bfd1a81720393b03c643b46c70bc69",
}

_MACE_OFF_MODEL_ALIASES = {
    "small": "off23_small",
    "medium": "off23_medium",
    "large": "off23_large",
}

MACE_OFF_MODEL_CHOICES = tuple(
    sorted({*MACE_OFF_MODELS.keys(), *_MACE_OFF_MODEL_ALIASES.keys()}),
)


def _canonical_model_name(model_name: str) -> str:
    return _MACE_OFF_MODEL_ALIASES.get(model_name, model_name)


def _sha256sum(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _validate_model_file(model_path: Path, expected_sha256: str) -> bool:
    return model_path.exists() and _sha256sum(model_path) == expected_sha256


def get_mace_off_cache_dir() -> Path:
    """Get the cache directory for MACE-OFF models."""
    if "XDG_CACHE_HOME" in os.environ:
        cache_dir = Path(os.environ["XDG_CACHE_HOME"]) / "deepmd-gnn" / "mace-off"
    else:
        cache_dir = Path.home() / ".cache" / "deepmd-gnn" / "mace-off"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_mace_off_model(
    model_name: str,
    cache_dir: Path | None = None,
) -> Path:
    """Download a selected official MACE-OFF model file.

    Parameters
    ----------
    model_name
        Canonical model name (for example ``off23_small``) or one of the
        compatibility aliases ``small`` / ``medium`` / ``large``.
    cache_dir
        Cache directory. Defaults to :func:`get_mace_off_cache_dir`.
    """
    canonical_name = _canonical_model_name(model_name)
    if canonical_name not in MACE_OFF_MODELS:
        msg = (
            f"Unknown MACE-OFF model: {model_name}. Available models: "
            f"{sorted(MACE_OFF_MODELS)}"
        )
        raise ValueError(msg)

    if cache_dir is None:
        cache_dir = get_mace_off_cache_dir()
    else:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    url = MACE_OFF_MODELS[canonical_name]
    expected_sha256 = MACE_OFF_MODEL_SHA256[canonical_name]
    filename = Path(url).name
    model_path = cache_dir / filename

    if _validate_model_file(model_path, expected_sha256):
        logger.info("Using cached model: %s", model_path)
        return model_path

    if model_path.exists():
        logger.warning(
            "Cached model SHA256 mismatch for %s; re-downloading %s",
            model_path,
            canonical_name,
        )
    else:
        logger.info("Downloading MACE-OFF model %s", canonical_name)

    logger.info("URL: %s", url)
    with NamedTemporaryFile(dir=cache_dir, delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    logger.info("Temporary download path: %s", tmp_path)
    logger.info("Final destination after SHA256 validation: %s", model_path)
    try:
        urlretrieve(url, tmp_path)  # noqa: S310
        actual_sha256 = _sha256sum(tmp_path)
        if actual_sha256 != expected_sha256:
            msg = (
                f"SHA256 mismatch for downloaded model {canonical_name}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
            raise ValueError(msg)
        tmp_path.replace(model_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return model_path


__all__ = [
    "MACE_OFF_MODEL_CHOICES",
    "MACE_OFF_MODELS",
    "MACE_OFF_MODEL_SHA256",
    "download_mace_off_model",
    "get_mace_off_cache_dir",
]
