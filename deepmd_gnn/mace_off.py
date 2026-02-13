# SPDX-License-Identifier: LGPL-3.0-or-later
"""Support for loading pretrained MACE models into DeePMD-GNN."""

import logging
import os
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import torch

from deepmd_gnn.mace import ELEMENTS, MaceModel

# Setup logger for this module
logger = logging.getLogger(__name__)

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
        logger.info("Downloading MACE-OFF %s model...", model_name)
        logger.info("URL: %s", url)
        logger.info("Destination: %s", model_path)
        urlretrieve(url, model_path)  # noqa: S310
        logger.info("Download complete!")
    else:
        logger.info("Using cached model: %s", model_path)

    return model_path


def load_mace_off_model(
    model_name: str = "small",
    cache_dir: Optional[Path] = None,
    device: str = "cpu",
) -> MaceModel:
    """Load a MACE-OFF pretrained model as a DeePMD-GNN MaceModel.

    This function downloads a MACE-OFF pretrained model and wraps it
    in the DeePMD-GNN MaceModel interface, making it compatible with
    DeePMD-kit and AMBER/sander through the existing integration.

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
    model : MaceModel
        Loaded MACE model wrapped in DeePMD-GNN's MaceModel interface,
        ready for use with DeePMD-kit and MD packages

    Examples
    --------
    >>> from deepmd_gnn.mace_off import load_mace_off_model
    >>> model = load_mace_off_model("small")
    >>> # Now use with DeePMD-kit: dp freeze, then use in LAMMPS/Amber
    """
    # Download model if necessary
    model_path = download_mace_off_model(
        model_name=model_name,
        cache_dir=cache_dir,
        force_download=False,
    )

    # Load the pretrained MACE model
    logger.info("Loading MACE-OFF %s model from %s...", model_name, model_path)
    # Note: weights_only=False is required for MACE models as they contain
    # custom objects. Only use with trusted MACE-OFF models from official sources.
    mace_model = torch.load(str(model_path), map_location=device, weights_only=False)

    # Extract configuration from the pretrained model
    # MACE models have attributes we need to extract
    if not hasattr(mace_model, "atomic_numbers"):
        msg = "Loaded model does not appear to be a valid MACE model (missing atomic_numbers)"
        raise ValueError(msg)

    atomic_numbers = mace_model.atomic_numbers.tolist()

    # Validate atomic numbers are in valid range
    if any(z < 1 or z > len(ELEMENTS) for z in atomic_numbers):
        msg = f"Invalid atomic numbers found: {atomic_numbers}. Must be between 1 and {len(ELEMENTS)}"
        raise ValueError(msg)

    # Convert atomic numbers to element symbols
    type_map = [ELEMENTS[z - 1] for z in atomic_numbers]

    # Extract model hyperparameters
    # These are stored in the MACE model's configuration
    if not hasattr(mace_model, "r_max") or not hasattr(mace_model, "num_interactions"):
        msg = "Loaded model missing required attributes (r_max, num_interactions)"
        raise ValueError(msg)

    r_max = float(mace_model.r_max)
    num_interactions = int(mace_model.num_interactions)

    # Helper function to get attribute with default and warning
    def get_attr_with_default(
        obj: object,
        attr: str,
        default: object,
        warn: bool = True,
    ) -> object:
        """Get attribute from object with default value and optional warning.

        Parameters
        ----------
        obj : object
            Object to get attribute from
        attr : str
            Attribute name
        default : object
            Default value if attribute not found
        warn : bool, optional
            Whether to print warning when using default

        Returns
        -------
        object
            Attribute value or default
        """
        value = getattr(obj, attr, None)
        if value is None:
            if warn:
                logger.warning(
                    "Using default %s=%s (not found in model)", attr, default,
                )
            return default
        return value

    # Get other parameters with defaults if not available
    num_radial_basis = get_attr_with_default(mace_model, "num_bessel", 8)
    num_cutoff_basis = get_attr_with_default(mace_model, "num_polynomial_cutoff", 5)
    max_ell = get_attr_with_default(mace_model, "max_ell", 3)
    correlation = get_attr_with_default(mace_model, "correlation", 3)
    radial_mlp = get_attr_with_default(mace_model, "radial_MLP", [64, 64, 64])

    # Get hidden_irreps with validation
    if hasattr(mace_model, "hidden_irreps") and mace_model.hidden_irreps is not None:
        hidden_irreps = str(mace_model.hidden_irreps)
    else:
        hidden_irreps = "128x0e + 128x1o"
        logger.warning("Using default hidden_irreps (not found in model)")

    # Determine interaction class name
    interaction_cls_name = (
        mace_model.interactions[0].__class__.__name__
        if hasattr(mace_model, "interactions") and len(mace_model.interactions) > 0
        else "RealAgnosticResidualInteractionBlock"
    )

    # Create MaceModel with the extracted configuration
    logger.info("Creating DeePMD-GNN MaceModel wrapper...")
    logger.debug("Type map: %s", type_map)
    logger.debug("r_max: %s", r_max)
    logger.debug("num_interactions: %s", num_interactions)

    deepmd_model = MaceModel(
        type_map=type_map,
        sel=100,  # This will be auto-determined during training/usage
        r_max=r_max,
        num_radial_basis=num_radial_basis,
        num_cutoff_basis=num_cutoff_basis,
        max_ell=max_ell,
        interaction=interaction_cls_name,
        num_interactions=num_interactions,
        hidden_irreps=hidden_irreps,
        correlation=correlation,
        radial_MLP=radial_mlp,
    )

    # Load the pretrained weights into the DeePMD model
    # The MaceModel.model is a ScaleShiftMACE instance, same as MACE-OFF
    logger.info("Loading pretrained weights...")
    try:
        deepmd_model.model.load_state_dict(mace_model.state_dict(), strict=True)
    except RuntimeError as e:
        msg = f"Failed to load pretrained weights: {e}. Model architectures may not match."
        raise RuntimeError(msg) from e

    deepmd_model.eval()

    logger.info("MACE-OFF model successfully loaded into DeePMD-GNN wrapper!")
    logger.info("You can now use this with DeePMD-kit (dp freeze) and MD packages.")

    return deepmd_model


def convert_mace_off_to_deepmd(
    model_name: str = "small",
    output_file: str = "frozen_model.pth",
    cache_dir: Optional[Path] = None,
) -> Path:
    """Convert a MACE-OFF model to frozen DeePMD format for use with MD packages.

    This function loads a MACE-OFF pretrained model, wraps it in the
    DeePMD-GNN MaceModel interface, and saves it as a frozen model
    that can be used directly with LAMMPS, AMBER/sander, and other
    MD packages through the DeePMD-kit interface.

    Parameters
    ----------
    model_name : str, optional
        Name of the MACE-OFF model: "small", "medium", or "large"
        Default is "small"
    output_file : str, optional
        Output file name for the frozen model
        Default is "frozen_model.pth"
    cache_dir : Path, optional
        Directory where MACE-OFF models are cached

    Returns
    -------
    output_path : Path
        Path to the frozen model file

    Notes
    -----
    The frozen model can be used with:
    - LAMMPS: Set DP_PLUGIN_PATH and use pair_style deepmd
    - AMBER/sander: Use through DeePMD-kit's AMBER interface
    - Other MD packages: Through DeePMD-kit's C++ interface

    For QM/MM simulations with sander, use MM type prefixes ('m', 'HW', 'OW')
    in your type_map to designate MM atoms.

    Examples
    --------
    >>> from deepmd_gnn.mace_off import convert_mace_off_to_deepmd
    >>> model_path = convert_mace_off_to_deepmd("small", "mace_small.pth")
    >>> # Now use in LAMMPS, AMBER, etc. through DeePMD-kit
    """
    # Load the MACE-OFF model as a MaceModel
    model = load_mace_off_model(model_name, cache_dir=cache_dir)

    # Create output path
    output_path = Path(output_file)

    # Try to freeze the model using TorchScript
    logger.info("Freezing model to %s...", output_path)
    try:
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, str(output_path))
        logger.info("Model successfully frozen and saved to: %s", output_path)
        logger.info("\nYou can now use this model with:")
        logger.info("  - LAMMPS: Set DP_PLUGIN_PATH and use pair_style deepmd")
        logger.info("  - AMBER/sander: Use DeePMD-kit's AMBER interface")
        logger.info("  - For QM/MM: Use 'm' prefix or HW/OW for MM atom types")
    except (RuntimeError, torch.jit.Error) as e:
        logger.warning("TorchScript compilation failed (%s)", e)
        logger.warning("Saving model in PyTorch format instead...")
        torch.save(model, str(output_path))
        logger.info("Model saved to: %s", output_path)

    return output_path


__all__ = [
    "MACE_OFF_MODELS",
    "convert_mace_off_to_deepmd",
    "download_mace_off_model",
    "get_mace_off_cache_dir",
    "load_mace_off_model",
]
