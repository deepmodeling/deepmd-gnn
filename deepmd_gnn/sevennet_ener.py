# SPDX-License-Identifier: LGPL-3.0-or-later
"""Original SevenNet energy head as a DeePMD fitting."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import torch
from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
    fitting_check_output,
)
from deepmd.pt.model.task.fitting import Fitting
from deepmd.pt.utils import env
from deepmd.pt.utils.utils import to_numpy_array, to_torch_tensor
from deepmd.utils.version import check_version_compatibility

from deepmd_gnn.sevennet_checkpoint import (
    build_sevennet_energy_modules,
    energy_input_dim,
    load_native_sevennet_energy_modules,
    persistable_checkpoint_config,
    resolve_sevennet_checkpoint_path,
    validate_sevennet_state_dict_load,
)

if TYPE_CHECKING:
    from collections import OrderedDict
    from collections.abc import Callable
    from pathlib import Path

    from deepmd.utils.path import DPPath


class SevenNetEnergyHead(torch.nn.Module):
    """Original SevenNet energy readout, rescale, and per-element shift."""

    def __init__(self, modules: OrderedDict[str, torch.nn.Module]) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleDict(modules)
        self.layer_names = list(modules)
        self.feature_dim = energy_input_dim(modules)

    def native_shift(self) -> torch.nn.Parameter:
        """Return the original energy shift parameter."""
        rescale = self.layers["rescale_atomic_energy"]
        return rescale.shift

    def node_energy(
        self,
        features: torch.Tensor,
        atype: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-atom energies from last-layer node features."""
        source_dtype = next(self.parameters()).dtype
        data: dict[str, torch.Tensor] = {
            "x": features.to(source_dtype),
            "atom_type": atype.reshape(-1).to(torch.int64),
        }
        for name in self.layer_names:
            data = self.layers[name](data)
        atomic_energy = data["atomic_energy"]
        return atomic_energy.reshape(-1)


@Fitting.register("sevennet_ener")
@fitting_check_output
class SevenNetEnergyFitting(Fitting):
    """Wrap the native SevenNet energy readout rather than a new MLP on ``0e``."""

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        type_map: list[str] | None = None,
        mixed_types: bool = True,
        model_path: str | Path | None = None,
        config: dict[str, Any] | None = None,
        trainable: bool = True,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__()
        del kwargs
        if type_map is None:
            msg = "sevennet_ener fitting requires the model-level type_map"
            raise ValueError(msg)
        if ntypes != len(type_map):
            msg = f"ntypes={ntypes} does not match type_map length {len(type_map)}"
            raise ValueError(msg)
        if not mixed_types:
            msg = "sevennet_ener fitting requires mixed_types=True"
            raise ValueError(msg)

        self.ntypes = int(ntypes)
        self.dim_descrpt = int(dim_descrpt)
        self.type_map = list(type_map)
        self.mixed_types = True
        self.trainable = bool(trainable)
        self.numb_fparam = 0
        self.numb_aparam = 0
        self.dim_case_embd = 0
        self.exclude_types: list[int] = []
        self.model_path: str | None = None
        resolved = (
            None if model_path is None else resolve_sevennet_checkpoint_path(model_path)
        )
        if resolved is not None:
            modules, inferred = load_native_sevennet_energy_modules(
                resolved,
                device=str(env.DEVICE),
            )
            checkpoint_type_map = inferred["type_map"]
            if self.type_map != checkpoint_type_map:
                msg = (
                    "Model-level type_map must exactly match checkpoint "
                    "chemical_species ordering: "
                    f"expected {checkpoint_type_map}, got {self.type_map}"
                )
                raise ValueError(msg)
            self.config = persistable_checkpoint_config(inferred)
            if config is not None:
                config.clear()
                config.update(self.config)
            self.model_path = str(model_path)
            self.head = SevenNetEnergyHead(modules)
        elif config is not None:
            self.config = persistable_checkpoint_config(config)
            checkpoint_type_map = self.config.get("type_map", self.type_map)
            if self.type_map != checkpoint_type_map:
                msg = (
                    "Serialized sevennet_ener type_map mismatch: "
                    f"expected {checkpoint_type_map}, got {self.type_map}"
                )
                raise ValueError(msg)
            self.model_path = None
            self.head = SevenNetEnergyHead(
                build_sevennet_energy_modules(
                    self.config,
                    device=str(env.DEVICE),
                ),
            )
        elif model_path is not None:
            msg = f"SevenNet checkpoint not found: {model_path}"
            raise FileNotFoundError(msg)
        else:
            msg = (
                "Exactly one of model_path (initialization) or config "
                "(deserialization) must be provided"
            )
            raise ValueError(msg)

        for parameter in self.head.parameters():
            parameter.requires_grad_(self.trainable)

    def output_def(self) -> FittingOutputDef:
        """Declare a conservative atomic energy."""
        return FittingOutputDef(
            [
                OutputVariableDef(
                    "energy",
                    [1],
                    reducible=True,
                    r_differentiable=True,
                    c_differentiable=True,
                ),
            ],
        )

    def get_type_map(self) -> list[str]:
        """Return checkpoint elements in chemical_species order."""
        return self.type_map

    def change_type_map(
        self,
        type_map: list[str],
        model_with_new_type_stat: SevenNetEnergyFitting | None = None,
    ) -> None:
        """Reject type-map changes and subsets."""
        del type_map, model_with_new_type_stat
        msg = "sevennet_ener fitting does not support changing or subsetting type_map"
        raise NotImplementedError(msg)

    def compute_input_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        protection: float = 1e-2,
        stat_file_path: DPPath | None = None,
    ) -> None:
        """Skip fitting-net statistics; SevenNet already stores shift/scale."""
        del merged, protection, stat_file_path

    def compute_output_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Skip DeePMD energy bias; the original rescale shift is the bias."""
        del merged, kwargs

    def get_dim_fparam(self) -> int:
        """Return zero frame-parameter width."""
        return 0

    def has_default_fparam(self) -> bool:
        """Return that no default frame parameters exist."""
        return False

    def get_default_fparam(self) -> torch.Tensor | None:
        """Return no default frame parameters."""
        return None

    def get_dim_aparam(self) -> int:
        """Return zero atomic-parameter width."""
        return 0

    def get_sel_type(self) -> list[int]:
        """Return that every type contributes atomic energy."""
        return []

    def set_case_embd(self, case_idx: int) -> None:
        """Ignore case embeddings; the original head has no case FiLM."""
        del case_idx

    def native_atomic_energies(self) -> torch.Tensor:
        """Return the original per-element energy shift."""
        return self.head.native_shift()

    def forward(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        gr: torch.Tensor | None = None,
        g2: torch.Tensor | None = None,
        h2: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Apply the original SevenNet energy head to last-layer features."""
        del descriptor, gr, h2, fparam, aparam
        if g2 is None:
            msg = "sevennet_ener fitting requires last-layer features in g2"
            raise ValueError(msg)
        nf, nloc = atype.shape
        width = g2.shape[-1]
        if width != self.head.feature_dim:
            msg = (
                "Packed SevenNet last-layer features have width "
                f"{width}, expected {self.head.feature_dim}"
            )
            raise ValueError(msg)
        atom_energy = self.head.node_energy(
            g2.reshape(nf * nloc, width),
            atype,
        )
        return {
            "energy": atom_energy.view(nf, nloc, 1).to(env.GLOBAL_PT_FLOAT_PRECISION),
        }

    def serialize(self) -> dict:
        """Serialize architecture and the original energy-head weights."""
        config = deepcopy(self.config)
        config["type_map"] = self.type_map
        return {
            "@class": "Fitting",
            "@version": 1,
            "type": "sevennet_ener",
            "ntypes": self.ntypes,
            "dim_descrpt": self.dim_descrpt,
            "type_map": self.type_map,
            "mixed_types": True,
            "model_path": None,
            "trainable": self.trainable,
            "config": config,
            "@variables": {
                name: to_numpy_array(value)
                for name, value in self.head.state_dict().items()
            },
        }

    @classmethod
    def deserialize(cls, data: dict) -> SevenNetEnergyFitting:
        """Restore a self-contained serialized SevenNet energy head."""
        data = data.copy()
        if data.pop("@class") != "Fitting" or data.pop("type") != "sevennet_ener":
            msg = "data is not a serialized SevenNetEnergyFitting"
            raise ValueError(msg)
        check_version_compatibility(data.pop("@version"), 1, 1)
        variables = {
            name: to_torch_tensor(value)
            for name, value in data.pop("@variables").items()
        }
        fitting = cls(**data)
        validate_sevennet_state_dict_load(
            fitting.head.load_state_dict(variables, strict=False),
        )
        return fitting


__all__ = ["SevenNetEnergyFitting", "SevenNetEnergyHead"]
