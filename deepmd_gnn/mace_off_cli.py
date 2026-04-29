# SPDX-License-Identifier: LGPL-3.0-or-later
"""Lightweight helpers for official MACE-OFF checkpoint selection and download."""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT_SECONDS = 120
SOURCE_PROBE_TIMEOUT_SECONDS = 8
_MACE_OFF_BASE_URL = "https://raw.githubusercontent.com/ACEsuit/mace-off/v0.2"
_GHFAST_PREFIX = "https://ghfast.top/"

MACE_OFF_MODELS = {
    "off23_small": f"{_MACE_OFF_BASE_URL}/mace_off23/MACE-OFF23_small.model",
    "off23_medium": f"{_MACE_OFF_BASE_URL}/mace_off23/MACE-OFF23_medium.model",
    "off23_large": f"{_MACE_OFF_BASE_URL}/mace_off23/MACE-OFF23_large.model",
    "off24_medium": f"{_MACE_OFF_BASE_URL}/mace_off24/MACE-OFF24_medium.model",
}

MACE_OFF_MODEL_URLS = {
    model_name: [
        url,
        f"{_GHFAST_PREFIX}{url}",
    ]
    for model_name, url in MACE_OFF_MODELS.items()
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


def _validate_download_url(url: str) -> None:
    """Validate that download URL uses HTTPS scheme."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        msg = f"Unsupported URL scheme for download: {parsed.scheme}"
        raise ValueError(msg)


def _model_download_urls(model_name: str) -> list[str]:
    """Return candidate download URLs for a canonical model name."""
    urls = MACE_OFF_MODEL_URLS[model_name]
    seen: set[str] = set()
    unique_urls: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    return unique_urls


def _sha256sum(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _validate_model_file(model_path: Path, expected_sha256: str) -> bool:
    return model_path.exists() and _sha256sum(model_path) == expected_sha256


def _probe_download_url(url: str) -> float | None:
    """Probe one URL and return latency seconds if reachable; else None."""
    _validate_download_url(url)
    request = urllib.request.Request(  # noqa: S310
        url,
        headers={"Range": "bytes=0-0"},
        method="GET",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(  # noqa: S310
            request,
            timeout=SOURCE_PROBE_TIMEOUT_SECONDS,
        ):
            pass
    except (urllib.error.URLError, OSError, ValueError):
        return None

    return time.monotonic() - start


def _rank_download_urls(urls: list[str]) -> list[str]:
    """Rank candidate URLs by probe latency (fastest first)."""
    if len(urls) <= 1:
        return urls

    results: dict[str, float] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(urls))) as exe:
        future_to_url = {exe.submit(_probe_download_url, url): url for url in urls}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            latency = future.result()
            if latency is not None:
                results[url] = latency

    ranked_ok = sorted(results, key=lambda url: results[url])
    ranked_fail = [url for url in urls if url not in results]
    return ranked_ok + ranked_fail


def _download_file(url: str, destination: Path) -> None:
    """Download URL content to destination atomically."""
    _validate_download_url(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".part",
            delete=False,
        ) as out_file:
            tmp_path = Path(out_file.name)
            with urllib.request.urlopen(  # noqa: S310
                url,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            ) as response:
                shutil.copyfileobj(response, out_file)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise

    tmp_path.replace(destination)


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

    urls = _model_download_urls(canonical_name)
    expected_sha256 = MACE_OFF_MODEL_SHA256[canonical_name]
    filename = Path(MACE_OFF_MODELS[canonical_name]).name
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
        model_path.unlink(missing_ok=True)
    else:
        logger.info("Downloading MACE-OFF model %s", canonical_name)

    ranked_urls = _rank_download_urls(urls)
    if len(ranked_urls) > 1:
        logger.info(
            "Selecting fastest source among %d candidates...",
            len(ranked_urls),
        )

    last_error: Exception | None = None
    for idx, url in enumerate(ranked_urls, start=1):
        logger.info(
            "Downloading MACE-OFF model %s (source %d/%d): %s",
            canonical_name,
            idx,
            len(ranked_urls),
            url,
        )
        try:
            _download_file(url, model_path)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last_error = exc
            logger.warning("Download attempt failed from %s: %s", url, exc)
            continue

        actual_sha256 = _sha256sum(model_path)
        if actual_sha256 != expected_sha256:
            model_path.unlink(missing_ok=True)
            msg = (
                f"SHA256 mismatch for downloaded model {canonical_name}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
            last_error = ValueError(msg)
            logger.warning("SHA256 verification failed from source: %s", url)
            logger.warning("Expected: %s", expected_sha256)
            logger.warning("Actual:   %s", actual_sha256)
            continue

        return model_path

    if isinstance(last_error, ValueError):
        raise last_error
    msg = f"Failed to download model '{canonical_name}' from all sources"
    raise RuntimeError(msg) from last_error


__all__ = [
    "MACE_OFF_MODELS",
    "MACE_OFF_MODEL_CHOICES",
    "MACE_OFF_MODEL_SHA256",
    "MACE_OFF_MODEL_URLS",
    "download_mace_off_model",
    "get_mace_off_cache_dir",
]
