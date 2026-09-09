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

    # DeepMD's pt_expt capability protocol gained supports_edge_parallel after
    # this plugin's self-wrapped MACE model was implemented. MACE already has
    # the layer-wise ghost communication required for domain decomposition, so
    # advertise that capability when running against a newer DeepMD version.
    if not hasattr(MaceModel, "supports_edge_parallel"):

        def supports_edge_parallel(_model: object) -> bool:
            """Report MACE domain-decomposition support."""
            return True

        MaceModel.supports_edge_parallel = supports_edge_parallel  # type: ignore[attr-defined]

    # NeQuIP 0.6/e3nn specializes atom and edge counts during torch.export.
    # Keep pt_expt registration limited to models with dynamic-shape export.
    ExportableBaseModel.register("mace")(MaceModel)


_register()
