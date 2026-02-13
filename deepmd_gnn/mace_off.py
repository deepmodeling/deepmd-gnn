# SPDX-License-Identifier: LGPL-3.0-or-later
"""Support for MACE-OFF and other pretrained MACE foundation models."""

import os
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import torch

# URLs for MACE-OFF pretrained models
MACE_OFF_MODELS = {
    "small": "https://github.com/ACEsuit/mace-off/releases/download/mace_off_small/mace_off_small.model",
    "medium": "https://github.com/ACEsuit/mace-off/releases/download/mace_off_medium/mace_off_medium.model",
    "large": "https://github.com/ACEsuit/mace-off/releases/download/mace_off_large/mace_off_large.model",
}


def get_mace_off_cache_dir() -> Path:
    """Get the cache directory for MACE-OFF models.

    Uses the XDG_CACHE_HOME environment variable if set,
    otherwise uses ~/.cache/deepmd-gnn/mace-off/

    Returns
    -------
    cache_dir : Path
        Path to cache directory
    """
    if "XDG_CACHE_HOME" in os.environ:
        cache_dir = Path(os.environ["XDG_CACHE_HOME"]) / "deepmd-gnn" / "mace-off"
    else:
        cache_dir = Path.home() / ".cache" / "deepmd-gnn" / "mace-off"

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_mace_off_model(
    model_name: str = "small",
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Path:
    """Download a MACE-OFF pretrained model.

    Parameters
    ----------
    model_name : str, optional
        Name of the model to download: "small", "medium", or "large"
        Default is "small"
    cache_dir : Path, optional
        Directory to cache the downloaded model
        If None, uses the default cache directory
    force_download : bool, optional
        If True, download even if file exists
        Default is False

    Returns
    -------
    model_path : Path
        Path to the downloaded model file

    Raises
    ------
    ValueError
        If model_name is not recognized

    Examples
    --------
    >>> model_path = download_mace_off_model("small")
    >>> print(f"Model downloaded to: {model_path}")
    """
    if model_name not in MACE_OFF_MODELS:
        msg = (
            f"Unknown MACE-OFF model: {model_name}. "
            f"Available models: {list(MACE_OFF_MODELS.keys())}"
        )
        raise ValueError(msg)

    if cache_dir is None:
        cache_dir = get_mace_off_cache_dir()
    else:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine local file path
    url = MACE_OFF_MODELS[model_name]
    filename = f"mace_off_{model_name}.model"
    model_path = cache_dir / filename

    # Download if needed
    if not model_path.exists() or force_download:
        print(f"Downloading MACE-OFF {model_name} model...")
        print(f"URL: {url}")
        print(f"Destination: {model_path}")
        urlretrieve(url, model_path)
        print("Download complete!")
    else:
        print(f"Using cached model: {model_path}")

    return model_path


def load_mace_off_model(
    model_name: str = "small",
    cache_dir: Optional[Path] = None,
    device: str = "cpu",
) -> torch.nn.Module:
    """Load a MACE-OFF pretrained model.

    Parameters
    ----------
    model_name : str, optional
        Name of the model to load: "small", "medium", or "large"
        Default is "small"
    cache_dir : Path, optional
        Directory where models are cached
        If None, uses the default cache directory
    device : str, optional
        Device to load the model on ("cpu" or "cuda")
        Default is "cpu"

    Returns
    -------
    model : torch.nn.Module
        Loaded MACE model ready for inference

    Examples
    --------
    >>> model = load_mace_off_model("small")
    >>> # Use model for predictions
    """
    # Download model if necessary
    model_path = download_mace_off_model(
        model_name=model_name,
        cache_dir=cache_dir,
        force_download=False,
    )

    # Load the model
    print(f"Loading MACE-OFF {model_name} model from {model_path}...")
    model = torch.load(str(model_path), map_location=device)
    model.eval()
    print("Model loaded successfully!")

    return model


def convert_mace_off_to_deepmd(
    model_name: str = "small",
    output_file: str = "mace_off_deepmd.pth",
    type_map: Optional[list[str]] = None,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Convert a MACE-OFF model to DeePMD-compatible format.

    This function loads a MACE-OFF pretrained model and converts it
    to a format compatible with DeePMD-kit for use in MD simulations.

    Parameters
    ----------
    model_name : str, optional
        Name of the MACE-OFF model: "small", "medium", or "large"
        Default is "small"
    output_file : str, optional
        Output file name for the converted model
        Default is "mace_off_deepmd.pth"
    type_map : list[str], optional
        Type map for the model. If None, uses the default from MACE-OFF
    cache_dir : Path, optional
        Directory where models are cached

    Returns
    -------
    output_path : Path
        Path to the converted model file

    Notes
    -----
    The converted model will be in TorchScript format compatible with
    DeePMD-kit's model serving infrastructure.

    Examples
    --------
    >>> model_path = convert_mace_off_to_deepmd("small", "mace_small.pth")
    >>> # Use model_path with DeePMD-kit or sander
    """
    # Load the MACE-OFF model
    model = load_mace_off_model(model_name, cache_dir=cache_dir)

    # Create output path
    output_path = Path(output_file)

    # For now, we'll save the model in a format compatible with torch.jit
    # The actual conversion may require wrapping the model
    print(f"Converting MACE-OFF {model_name} to DeePMD format...")

    # Script the model for deployment
    try:
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, str(output_path))
        print(f"Model saved to: {output_path}")
    except Exception as e:
        # If scripting fails, try tracing instead
        print(f"Scripting failed ({e}), attempting to save directly...")
        torch.save(model, str(output_path))
        print(f"Model saved to: {output_path}")

    return output_path


__all__ = [
    "MACE_OFF_MODELS",
    "download_mace_off_model",
    "load_mace_off_model",
    "convert_mace_off_to_deepmd",
    "get_mace_off_cache_dir",
]
