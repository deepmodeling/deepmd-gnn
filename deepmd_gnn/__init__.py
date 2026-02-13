"""MACE plugin for DeePMD-kit."""

import os

from ._version import __version__
from .argcheck import mace_model_args
from .mace_off import (
    convert_mace_off_to_deepmd,
    download_mace_off_model,
    load_mace_off_model,
)

__email__ = "jinzhe.zeng@ustc.edu.cn"

__all__ = [
    "__version__",
    "mace_model_args",
    "download_mace_off_model",
    "load_mace_off_model",
    "convert_mace_off_to_deepmd",
]

# make compatible with mace & e3nn & pytorch 2.6
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
