"""MACE plugin for DeePMD-kit."""

import os

from ._version import __version__
from .argcheck import mace_model_args
from .mace_off import (
    download_mace_off_model,
    load_mace_off_model,
    convert_mace_off_to_deepmd,
)
from .sander import SanderInterface, compute_qm_energy_sander

__email__ = "jinzhe.zeng@ustc.edu.cn"

__all__ = [
    "__version__",
    "mace_model_args",
    "download_mace_off_model",
    "load_mace_off_model",
    "convert_mace_off_to_deepmd",
    "SanderInterface",
    "compute_qm_energy_sander",
]

# make compatible with mace & e3nn & pytorch 2.6
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
