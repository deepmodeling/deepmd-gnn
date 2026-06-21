"""PyTorch exportable backend plugin registration."""

import sys


def load() -> None:
    """Entry point placeholder; importing this module registers plugins."""


def _is_partially_initialized(module_name: str, attr_name: str) -> bool:
    module = sys.modules.get(module_name)
    return module is not None and not hasattr(module, attr_name)


def _register() -> None:
    if _is_partially_initialized(
        "deepmd_gnn.mace",
        "MaceModel",
    ):
        return

    from deepmd.pt_expt.model.model import (  # noqa: PLC0415
        BaseModel as ExportableBaseModel,
    )

    from deepmd_gnn.mace import MaceModel  # noqa: PLC0415

    # NeQuIP 0.6/e3nn specializes atom and edge counts during torch.export.
    # Keep pt_expt registration limited to models with dynamic-shape export.
    ExportableBaseModel.register("mace")(MaceModel)


_register()
