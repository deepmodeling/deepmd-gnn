# SPDX-License-Identifier: LGPL-3.0-or-later
"""Conservative helpers for loading selected MACE-OFF checkpoints.

The helpers in this module intentionally support a narrower scope than the
upstream discussion in PR #96: they load ordinary MACE-OFF checkpoints into a
DeePMD-GNN ``MaceModel`` wrapper for standard MD use, but they do *not* infer
DPRc/QM/MM atom-type semantics from the checkpoint.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict
from urllib.request import urlretrieve

import torch
from deepmd.pt import model as _deepmd_pt_model  # noqa: F401
from mace.modules import ScaleShiftMACE, gate_dict

from deepmd_gnn.mace import ELEMENTS, MaceModel

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

_MACE_OFF_BASE_URL = "https://raw.githubusercontent.com/ACEsuit/mace-off/v0.2"

MACE_OFF_MODELS = {
    "off23_small": f"{_MACE_OFF_BASE_URL}/mace_off23/MACE-OFF23_small.model",
    "off23_medium": f"{_MACE_OFF_BASE_URL}/mace_off23/MACE-OFF23_medium.model",
    "off23_large": f"{_MACE_OFF_BASE_URL}/mace_off23/MACE-OFF23_large.model",
    "off24_medium": f"{_MACE_OFF_BASE_URL}/mace_off24/MACE-OFF24_medium.model",
}

_MACE_OFF_MODEL_ALIASES = {
    "small": "off23_small",
    "medium": "off23_medium",
    "large": "off23_large",
}

_ALLOWED_MISSING_STATE_DICT_SUFFIXES = ("_zeroed",)

_SUPPORTED_DISTANCE_TRANSFORMS = {
    None: "None",
    "AgnesiTransform": "Agnesi",
    "SoftTransform": "Soft",
}

_SUPPORTED_RADIAL_BASES = {
    "BesselBasis": "bessel",
}


class _InferredMaceConfig(TypedDict):
    type_map: list[str]
    r_max: float
    num_radial_basis: int
    num_cutoff_basis: int
    max_ell: int
    interaction: str
    num_interactions: int
    hidden_irreps: str
    pair_repulsion: bool
    distance_transform: str
    correlation: int
    gate: str
    MLP_irreps: str
    radial_type: str
    radial_MLP: list[int]
    std: float


@contextmanager
def _temporary_default_dtype(dtype: torch.dtype) -> Iterator[None]:
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def _canonical_model_name(model_name: str) -> str:
    return _MACE_OFF_MODEL_ALIASES.get(model_name, model_name)


def get_mace_off_cache_dir() -> Path:
    """Get the cache directory for MACE-OFF models."""
    if "XDG_CACHE_HOME" in os.environ:
        cache_dir = Path(os.environ["XDG_CACHE_HOME"]) / "deepmd-gnn" / "mace-off"
    else:
        cache_dir = Path.home() / ".cache" / "deepmd-gnn" / "mace-off"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_mace_off_model(
    model_name: str = "small",
    cache_dir: Path | None = None,
    force_download: bool = False,
) -> Path:
    """Download a selected official MACE-OFF model file.

    Parameters
    ----------
    model_name
        Canonical model name (for example ``off23_small``) or one of the
        compatibility aliases ``small`` / ``medium`` / ``large``.
    cache_dir
        Cache directory. Defaults to :func:`get_mace_off_cache_dir`.
    force_download
        Re-download even if the cached file already exists.
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
    filename = Path(url).name
    model_path = cache_dir / filename

    if not model_path.exists() or force_download:
        logger.info("Downloading MACE-OFF model %s", canonical_name)
        logger.info("URL: %s", url)
        logger.info("Destination: %s", model_path)
        urlretrieve(url, model_path)  # noqa: S310
    else:
        logger.info("Using cached model: %s", model_path)

    return model_path


def _validate_atomic_numbers(atomic_numbers: list[int]) -> None:
    if not atomic_numbers:
        msg = "Loaded model has empty atomic_numbers"
        raise ValueError(msg)
    if any(z < 1 or z > len(ELEMENTS) for z in atomic_numbers):
        msg = (
            f"Invalid atomic numbers found: {atomic_numbers}. "
            f"Must be between 1 and {len(ELEMENTS)}"
        )
        raise ValueError(msg)


def _infer_gate_name(mace_model: ScaleShiftMACE) -> str:
    if not mace_model.readouts:
        return "None"

    last_readout = mace_model.readouts[-1]
    if not hasattr(last_readout, "non_linearity"):
        return "None"

    acts = getattr(last_readout.non_linearity, "acts", None)
    if acts is None or len(acts) != 1 or not hasattr(acts[0], "f"):
        msg = "Unsupported MACE non-linearity structure"
        raise ValueError(msg)

    gate_fn = acts[0].f
    for gate_name, candidate in gate_dict.items():
        if candidate is not None and candidate is gate_fn:
            return gate_name

    msg = f"Unsupported MACE gate function: {gate_fn}"
    raise ValueError(msg)


def _infer_distance_transform(mace_model: ScaleShiftMACE) -> str:
    transform = getattr(mace_model.radial_embedding, "distance_transform", None)
    transform_name = None if transform is None else transform.__class__.__name__
    if transform_name not in _SUPPORTED_DISTANCE_TRANSFORMS:
        msg = (
            "Unsupported MACE distance transform for conservative loader: "
            f"{transform_name}"
        )
        raise ValueError(msg)
    return _SUPPORTED_DISTANCE_TRANSFORMS[transform_name]


def _infer_radial_type(mace_model: ScaleShiftMACE) -> str:
    basis_name = mace_model.radial_embedding.bessel_fn.__class__.__name__
    if basis_name not in _SUPPORTED_RADIAL_BASES:
        msg = f"Unsupported radial basis for conservative loader: {basis_name}"
        raise ValueError(msg)
    return _SUPPORTED_RADIAL_BASES[basis_name]


def _infer_radial_mlp(mace_model: ScaleShiftMACE) -> list[int]:
    layers = [
        layer
        for layer in mace_model.interactions[0].conv_tp_weights
        if hasattr(layer, "weight")
    ]
    if len(layers) < 2:
        msg = "Unsupported radial MLP structure in MACE checkpoint"
        raise ValueError(msg)
    return [int(layer.weight.shape[1]) for layer in layers[:-1]]


def _infer_interaction_name(mace_model: ScaleShiftMACE) -> str:
    interaction_names = [
        interaction.__class__.__name__ for interaction in mace_model.interactions
    ]
    if not interaction_names:
        msg = "Loaded MACE model has no interaction blocks"
        raise ValueError(msg)
    if interaction_names[0] != "RealAgnosticInteractionBlock":
        msg = (
            "Conservative MACE-OFF loader only supports checkpoints whose first "
            f"interaction block is RealAgnosticInteractionBlock, got {interaction_names[0]}"
        )
        raise ValueError(msg)
    later_names = interaction_names[1:]
    if later_names and len(set(later_names)) != 1:
        msg = f"Mixed later interaction blocks are unsupported: {interaction_names}"
        raise ValueError(msg)
    return later_names[0] if later_names else interaction_names[0]


def _infer_hidden_irreps(mace_model: ScaleShiftMACE) -> str:
    return str(mace_model.interactions[0].hidden_irreps)


def _infer_max_ell(mace_model: ScaleShiftMACE) -> int:
    return max(ir.l for _, ir in mace_model.spherical_harmonics.irreps_out)


def _infer_correlation(mace_model: ScaleShiftMACE) -> int:
    correlations = [
        int(product.symmetric_contractions.contractions[0].correlation)
        for product in mace_model.products
    ]
    if len(set(correlations)) != 1:
        msg = f"Layer-dependent correlation is unsupported: {correlations}"
        raise ValueError(msg)
    return correlations[0]


def _infer_mlp_irreps(mace_model: ScaleShiftMACE) -> str:
    last_readout = mace_model.readouts[-1]
    if not hasattr(last_readout, "hidden_irreps"):
        msg = "Checkpoint does not expose non-linear readout hidden_irreps"
        raise ValueError(msg)
    return str(last_readout.hidden_irreps)


def _infer_scale(mace_model: ScaleShiftMACE) -> float:
    scale_state = mace_model.scale_shift.state_dict()
    return float(scale_state["scale"])


def _validate_checkpoint_scope(mace_model: ScaleShiftMACE) -> None:
    atomic_numbers = mace_model.atomic_numbers.tolist()
    _validate_atomic_numbers(atomic_numbers)

    heads = getattr(mace_model, "heads", None)
    if heads not in (None, ["Default"]):
        msg = f"Multi-head checkpoints are unsupported: heads={heads}"
        raise ValueError(msg)

    if (
        hasattr(mace_model, "embedding_specs")
        and mace_model.embedding_specs is not None
    ):
        msg = "Joint-embedding checkpoints are unsupported by the conservative loader"
        raise ValueError(msg)

    if bool(getattr(mace_model, "pair_repulsion", False)):
        msg = "Pair-repulsion checkpoints are unsupported by the conservative loader"
        raise ValueError(msg)


def _infer_deepmd_config(mace_model: ScaleShiftMACE) -> _InferredMaceConfig:
    _validate_checkpoint_scope(mace_model)
    atomic_numbers = mace_model.atomic_numbers.tolist()

    return {
        "type_map": [ELEMENTS[z - 1] for z in atomic_numbers],
        "r_max": float(mace_model.r_max),
        "num_radial_basis": len(mace_model.radial_embedding.bessel_fn.bessel_weights),
        "num_cutoff_basis": int(mace_model.radial_embedding.cutoff_fn.p.item()),
        "max_ell": _infer_max_ell(mace_model),
        "interaction": _infer_interaction_name(mace_model),
        "num_interactions": int(mace_model.num_interactions),
        "hidden_irreps": _infer_hidden_irreps(mace_model),
        "pair_repulsion": False,
        "distance_transform": _infer_distance_transform(mace_model),
        "correlation": _infer_correlation(mace_model),
        "gate": _infer_gate_name(mace_model),
        "MLP_irreps": _infer_mlp_irreps(mace_model),
        "radial_type": _infer_radial_type(mace_model),
        "radial_MLP": _infer_radial_mlp(mace_model),
        "std": _infer_scale(mace_model),
    }


def _load_mace_checkpoint(model_path: Path, device: str) -> ScaleShiftMACE:
    model = torch.load(str(model_path), map_location=device, weights_only=False)
    if not isinstance(model, ScaleShiftMACE):
        msg = (
            "Loaded checkpoint is not a ScaleShiftMACE model: "
            f"{model.__class__.__module__}.{model.__class__.__name__}"
        )
        raise TypeError(msg)
    return model


def _validate_load_result(load_result: object) -> None:
    missing_keys = list(getattr(load_result, "missing_keys", []))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []))

    disallowed_missing_keys = [
        key
        for key in missing_keys
        if not key.endswith(_ALLOWED_MISSING_STATE_DICT_SUFFIXES)
    ]
    if disallowed_missing_keys or unexpected_keys:
        msg = (
            "Failed to load MACE checkpoint into DeePMD-GNN wrapper. "
            f"missing={disallowed_missing_keys}, unexpected={unexpected_keys}"
        )
        raise RuntimeError(msg)


def load_mace_off_model(
    model_name: str | None = "small",
    *,
    sel: int,
    model_path: Path | None = None,
    cache_dir: Path | None = None,
    device: str = "cpu",
) -> MaceModel:
    """Load a supported MACE-OFF checkpoint as a DeePMD-GNN ``MaceModel``.

    Parameters
    ----------
    model_name
        Name of the official MACE-OFF checkpoint to download. Ignored when
        ``model_path`` is provided.
    sel
        Neighbor-list cap for the DeePMD-GNN wrapper. This value is *required*:
        MACE-OFF checkpoints do not store DeePMD's runtime neighbor-list limit,
        and guessing it can silently truncate neighbors.
    model_path
        Local checkpoint path. If omitted, download the selected official model.
    cache_dir
        Cache directory for downloaded checkpoints.
    device
        Device used when loading the original checkpoint.

    Notes
    -----
    This helper intentionally supports a narrower scope than DeePMD-GNN's full
    DPRc machinery. It only infers ordinary element ``type_map`` entries from
    ``atomic_numbers`` and does not infer ``mH`` / ``HW`` / ``OW`` or other
    QM/MM-specific type semantics.
    """
    if sel <= 0:
        msg = f"sel must be positive, got {sel}"
        raise ValueError(msg)

    if model_path is None:
        if model_name is None:
            msg = "Either model_name or model_path must be provided"
            raise ValueError(msg)
        model_path = download_mace_off_model(model_name, cache_dir=cache_dir)
    else:
        model_path = Path(model_path)

    mace_model = _load_mace_checkpoint(model_path, device=device)
    config = _infer_deepmd_config(mace_model)

    source_dtype = mace_model.atomic_energies_fn.atomic_energies.dtype
    with _temporary_default_dtype(source_dtype):
        deepmd_model = MaceModel(
            type_map=config["type_map"],
            sel=sel,
            r_max=config["r_max"],
            num_radial_basis=config["num_radial_basis"],
            num_cutoff_basis=config["num_cutoff_basis"],
            max_ell=config["max_ell"],
            interaction=config["interaction"],
            num_interactions=config["num_interactions"],
            hidden_irreps=config["hidden_irreps"],
            pair_repulsion=config["pair_repulsion"],
            distance_transform=config["distance_transform"],
            correlation=config["correlation"],
            gate=config["gate"],
            MLP_irreps=config["MLP_irreps"],
            radial_type=config["radial_type"],
            radial_MLP=config["radial_MLP"],
            std=config["std"],
        )

    load_result = deepmd_model.model.load_state_dict(
        mace_model.state_dict(),
        strict=False,
    )
    _validate_load_result(load_result)
    deepmd_model.eval()
    return deepmd_model


def convert_mace_off_to_deepmd(
    output_file: str,
    *,
    sel: int,
    model_name: str | None = "small",
    model_path: Path | None = None,
    cache_dir: Path | None = None,
    device: str = "cpu",
) -> Path:
    """Serialize a loaded MACE-OFF wrapper as a TorchScript DeePMD-GNN model.

    This helper scripts the DeePMD-GNN wrapper and writes it to ``output_file``.
    It does not, by itself, prove end-to-end downstream deployment in external
    engines such as LAMMPS or AMBER; callers should still validate the final
    deployment path they care about.
    """
    model = load_mace_off_model(
        model_name=model_name,
        sel=sel,
        model_path=model_path,
        cache_dir=cache_dir,
        device=device,
    )
    output_path = Path(output_file)
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, str(output_path))
    return output_path


__all__ = [
    "MACE_OFF_MODELS",
    "convert_mace_off_to_deepmd",
    "download_mace_off_model",
    "get_mace_off_cache_dir",
    "load_mace_off_model",
]
