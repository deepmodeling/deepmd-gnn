"""PyTorch backend plugin registration."""

from __future__ import annotations

import contextlib
import copy
import json
import sys
from typing import Any


def load() -> None:
    """Entry point placeholder; importing this module registers plugins."""


def _is_partially_initialized(module_name: str, attr_name: str) -> bool:
    module = sys.modules.get(module_name)
    return module is not None and not hasattr(module, attr_name)


_MACE_ENER_ATOMIC_PATCHED = False
_MACE_ENER_STANDARD_PATCHED = False
_MACE_ENER_WRAPPER_PATCHED = False


def _persist_one_mace_model(model: Any, model_params: dict[str, Any]) -> None:  # noqa: ANN401
    """Write inferred MACE constructor configs into checkpoint model parameters."""
    from deepmd_gnn.mace_checkpoint import (  # noqa: PLC0415
        persistable_checkpoint_config,
    )
    from deepmd_gnn.mace_descriptor import MaceDescriptor  # noqa: PLC0415
    from deepmd_gnn.mace_ener import MaceEnergyFitting  # noqa: PLC0415

    get_descriptor = getattr(model, "get_descriptor", None)
    get_fitting = getattr(model, "get_fitting_net", None)
    descriptor = get_descriptor() if callable(get_descriptor) else None
    fitting = get_fitting() if callable(get_fitting) else None
    if isinstance(descriptor, MaceDescriptor):
        model_params.setdefault("descriptor", {})["config"] = (
            persistable_checkpoint_config(descriptor.config)
        )
    if isinstance(fitting, MaceEnergyFitting):
        model_params.setdefault("fitting_net", {})["config"] = (
            persistable_checkpoint_config(fitting.config)
        )


def _persist_mace_runtime_configs(
    model: Any,  # noqa: ANN401
    model_params: dict[str, Any] | None,
) -> None:
    """Copy live MACE configs into the dict DeePMD stores on the wrapper."""
    if not model_params:
        return
    named_models = (
        dict(model.items())
        if hasattr(model, "items") and not hasattr(model, "get_descriptor")
        else None
    )
    if "model_dict" in model_params:
        for key, sub_params in model_params["model_dict"].items():
            live = None if named_models is None else named_models.get(key)
            if live is not None:
                _persist_mace_runtime_configs(live, sub_params)
        return
    live = model
    if named_models is not None:
        live = named_models.get("Default") or next(iter(named_models.values()), model)
    _persist_one_mace_model(live, model_params)


def _install_mace_ener_energy_atomic_model() -> None:
    """Let EnergyModel accept the original MACE head and skip extra e0 bias."""
    global _MACE_ENER_ATOMIC_PATCHED  # noqa: PLW0603
    from deepmd.pt.model.atomic_model.dp_atomic_model import (  # noqa: PLC0415
        DPAtomicModel,
    )
    from deepmd.pt.model.atomic_model.energy_atomic_model import (  # noqa: PLC0415
        DPEnergyAtomicModel,
    )

    from deepmd_gnn.mace_ener import MaceEnergyFitting  # noqa: PLC0415

    if _MACE_ENER_ATOMIC_PATCHED:
        return

    original_init = DPEnergyAtomicModel.__init__

    def init_with_mace_ener(
        self: Any,  # noqa: ANN401
        descriptor: Any,  # noqa: ANN401
        fitting: Any,  # noqa: ANN401
        type_map: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        if isinstance(fitting, MaceEnergyFitting):
            DPAtomicModel.__init__(self, descriptor, fitting, type_map, **kwargs)
            return
        original_init(self, descriptor, fitting, type_map, **kwargs)

    original_out_stat = DPEnergyAtomicModel.compute_or_load_out_stat

    def compute_or_load_out_stat_skip_mace_ener(
        self: Any,  # noqa: ANN401
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Keep pretrained MACE e0; do not LS-fit a second vacuum energy."""
        if isinstance(self.fitting_net, MaceEnergyFitting):
            return
        original_out_stat(self, *args, **kwargs)

    original_change = DPEnergyAtomicModel.change_out_bias

    def change_out_bias_mace_ener(
        self: Any,  # noqa: ANN401
        sample_merged: Any,  # noqa: ANN401
        stat_file_path: Any = None,  # noqa: ANN401
        bias_adjust_mode: str = "change-by-statistic",
    ) -> None:
        """Finetune residual goes to ``out_bias``; set-by-statistic replaces e0."""
        if (
            not isinstance(
                self.fitting_net,
                MaceEnergyFitting,
            )
            or bias_adjust_mode == "change-by-statistic"
        ):
            original_change(
                self,
                sample_merged,
                stat_file_path=stat_file_path,
                bias_adjust_mode=bias_adjust_mode,
            )
        elif bias_adjust_mode != "set-by-statistic":
            msg = "Unknown bias_adjust_mode mode: " + bias_adjust_mode
            raise RuntimeError(msg)
        else:
            from deepmd.pt.utils.stat import compute_output_stats  # noqa: PLC0415

            bias_out, _std = compute_output_stats(
                sample_merged,
                self.get_ntypes(),
                keys=self.bias_keys,
                stat_file_path=stat_file_path,
                rcond=self.rcond,
                preset_bias=self.preset_out_bias,
                stats_distinguish_types=self.get_compute_stats_distinguish_types(),
                intensive=self.get_intensive(),
            )
            if "energy" not in bias_out:
                return
            energies = self.fitting_net.head.atomic_energies_fn.atomic_energies
            energies.copy_(
                bias_out["energy"]
                .reshape(energies.shape)
                .to(
                    dtype=energies.dtype,
                    device=energies.device,
                ),
            )

    DPEnergyAtomicModel.__init__ = init_with_mace_ener  # type: ignore[method-assign]
    DPEnergyAtomicModel.compute_or_load_out_stat = (  # type: ignore[method-assign]
        compute_or_load_out_stat_skip_mace_ener
    )
    DPEnergyAtomicModel.change_out_bias = change_out_bias_mace_ener  # type: ignore[method-assign]
    _MACE_ENER_ATOMIC_PATCHED = True


def _install_mace_ener_standard_model() -> None:
    """Route ``fitting_net.type=mace_ener`` through DeePMD's EnergyModel."""
    global _MACE_ENER_STANDARD_PATCHED  # noqa: PLW0603
    import deepmd.pt.model.model as model_mod  # noqa: PLC0415

    if _MACE_ENER_STANDARD_PATCHED:
        return
    original = model_mod.get_standard_model

    def get_standard_model_with_mace_ener(model_params: dict[str, Any]) -> Any:  # noqa: ANN401
        fitting_type = (model_params.get("fitting_net") or {}).get("type", "ener")
        if fitting_type != "mace_ener":
            return original(model_params)
        model_params_old = model_params
        model_params = copy.deepcopy(model_params)
        ntypes = len(model_params["type_map"])
        descriptor, fitting, _fitting_net_type = (
            model_mod._get_standard_model_components(  # noqa: SLF001
                model_params,
                ntypes,
            )
        )
        atom_exclude_types = model_params.get("atom_exclude_types", [])
        pair_exclude_types = model_params.get("pair_exclude_types", [])
        preset_out_bias = model_mod._convert_preset_out_bias_to_array(  # noqa: SLF001
            model_params.get("preset_out_bias"),
            model_params["type_map"],
        )
        data_stat_protect = model_params.get("data_stat_protect", 1e-2)
        model = model_mod.EnergyModel(
            descriptor=descriptor,
            fitting=fitting,
            type_map=model_params["type_map"],
            atom_exclude_types=atom_exclude_types,
            pair_exclude_types=pair_exclude_types,
            preset_out_bias=preset_out_bias,
            data_stat_protect=data_stat_protect,
        )
        if model_params.get("hessian_mode"):
            model.enable_hessian()
        _persist_mace_runtime_configs(model, model_params_old)
        model.model_def_script = json.dumps(model_params_old)
        return model

    model_mod.get_standard_model = get_standard_model_with_mace_ener
    _MACE_ENER_STANDARD_PATCHED = True


def _install_mace_ener_model_wrapper() -> None:
    """Keep wrapper/checkpoint model_params self-contained after native pickle load."""
    global _MACE_ENER_WRAPPER_PATCHED  # noqa: PLW0603
    from deepmd.pt.train.wrapper import ModelWrapper  # noqa: PLC0415

    if _MACE_ENER_WRAPPER_PATCHED:
        return

    original_init = ModelWrapper.__init__
    original_extra = ModelWrapper.get_extra_state

    def init_with_mace_configs(
        self: Any,  # noqa: ANN401
        model: Any,  # noqa: ANN401
        loss: Any = None,  # noqa: ANN401
        model_params: dict[str, Any] | None = None,
        shared_links: dict[str, Any] | None = None,
        modifier: Any = None,  # noqa: ANN401
    ) -> None:
        original_init(self, model, loss, model_params, shared_links, modifier)
        _persist_mace_runtime_configs(self.model, self.model_params)

    def extra_state_with_mace_configs(self: Any) -> dict[str, Any]:  # noqa: ANN401
        _persist_mace_runtime_configs(self.model, self.model_params)
        return original_extra(self)

    ModelWrapper.__init__ = init_with_mace_configs  # type: ignore[method-assign]
    ModelWrapper.get_extra_state = extra_state_with_mace_configs  # type: ignore[method-assign]
    _MACE_ENER_WRAPPER_PATCHED = True


def _register() -> None:
    if _is_partially_initialized(
        "deepmd_gnn.mace",
        "MaceModel",
    ) or _is_partially_initialized(
        "deepmd_gnn.nequip",
        "NequipModel",
    ):
        return

    from deepmd.pt.model.model.model import (  # noqa: PLC0415
        BaseModel as PyTorchBaseModel,
    )

    import deepmd_gnn.mace_descriptor  # noqa: PLC0415
    import deepmd_gnn.nequip_descriptor  # noqa: PLC0415

    with contextlib.suppress(ImportError):
        import deepmd_gnn.sevennet_descriptor  # noqa: F401, PLC0415
    from deepmd_gnn.mace import MaceModel  # noqa: PLC0415
    from deepmd_gnn.nequip import NequipModel  # noqa: PLC0415

    PyTorchBaseModel.register("mace")(MaceModel)
    PyTorchBaseModel.register("nequip")(NequipModel)
    _install_mace_ener_energy_atomic_model()
    _install_mace_ener_standard_model()
    _install_mace_ener_model_wrapper()


_register()
