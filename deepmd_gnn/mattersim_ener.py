# SPDX-License-Identifier: LGPL-3.0-or-later
"""Original MatterSim M3GNet energy head as a DeePMD fitting."""

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

from deepmd_gnn.mattersim_checkpoint import (
    atomic_numbers_from_type_map,
    build_mattersim_energy_head,
    load_native_mattersim_energy_head,
    persistable_checkpoint_config,
    resolve_mattersim_checkpoint_path,
    validate_mattersim_state_dict_load,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from deepmd.utils.path import DPPath


@Fitting.register("mattersim_ener")
@fitting_check_output
class MatterSimEnergyFitting(Fitting):
    """Wrap the native M3GNet energy readout rather than a new MLP on atom_attr."""

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
            msg = "mattersim_ener fitting requires the model-level type_map"
            raise ValueError(msg)
        if ntypes != len(type_map):
            msg = f"ntypes={ntypes} does not match type_map length {len(type_map)}"
            raise ValueError(msg)
        if not mixed_types:
            msg = "mattersim_ener fitting requires mixed_types=True"
            raise ValueError(msg)

        self.ntypes = int(ntypes)
        self.dim_descrpt = int(dim_descrpt)
        self.type_map = list(type_map)
        self.type_to_z = atomic_numbers_from_type_map(self.type_map)
        self.mixed_types = True
        self.trainable = bool(trainable)
        self.numb_fparam = 0
        self.numb_aparam = 0
        self.dim_case_embd = 0
        self.exclude_types: list[int] = []
        self.model_path: str | None = None
        resolved = (
            None
            if model_path is None
            else resolve_mattersim_checkpoint_path(model_path)
        )
        if resolved is not None:
            head, inferred = load_native_mattersim_energy_head(
                resolved,
                device=str(env.DEVICE),
            )
            inferred["type_map"] = self.type_map
            self.config = persistable_checkpoint_config(inferred)
            if config is not None:
                config.clear()
                config.update(self.config)
            self.model_path = str(model_path)
            self.head = head
        elif config is not None:
            persistable = persistable_checkpoint_config(config)
            persistable["type_map"] = persistable["type_map"] or self.type_map
            if persistable["type_map"] != self.type_map:
                msg = (
                    "Serialized mattersim_ener type_map mismatch: "
                    f"expected {persistable['type_map']}, got {self.type_map}"
                )
                raise ValueError(msg)
            self.config = persistable
            self.model_path = None
            self.head = build_mattersim_energy_head(
                self.config,
                device=str(env.DEVICE),
            )
        elif model_path is not None:
            msg = f"MatterSim checkpoint not found: {model_path}"
            raise FileNotFoundError(msg)
        else:
            msg = (
                "Exactly one of model_path (initialization) or config "
                "(deserialization) must be provided"
            )
            raise ValueError(msg)

        max_z = int(self.config["max_z"])
        if max(self.type_to_z) > max_z:
            msg = (
                f"type_map {self.type_map} has Z>{max_z}, which exceeds the "
                "MatterSim checkpoint max_z"
            )
            raise ValueError(msg)
        if self.dim_descrpt != int(self.config["units"]):
            msg = (
                "mattersim_ener dim_descrpt "
                f"{self.dim_descrpt} does not match backbone units "
                f"{self.config['units']}"
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
        """Return model-level elements."""
        return self.type_map

    def change_type_map(
        self,
        type_map: list[str],
        model_with_new_type_stat: MatterSimEnergyFitting | None = None,
    ) -> None:
        """Reject type-map changes and subsets."""
        del type_map, model_with_new_type_stat
        msg = "mattersim_ener fitting does not support changing or subsetting type_map"
        raise NotImplementedError(msg)

    def compute_input_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        protection: float = 1e-2,
        stat_file_path: DPPath | None = None,
    ) -> None:
        """Skip fitting-net statistics; MatterSim already stores scale/shift."""
        del merged, protection, stat_file_path

    def compute_output_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Skip DeePMD energy bias; the original AtomScaling shift is the bias."""
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
        """Return pretrained per-type shifts gathered from per-Z AtomScaling."""
        z = torch.tensor(self.type_to_z, dtype=torch.int64, device=env.DEVICE)
        return self.head.normalizer.shift[z]

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
        """Apply the original M3GNet energy head to last-layer atom_attr."""
        del gr, g2, h2, fparam, aparam
        nf, nloc = atype.shape
        width = descriptor.shape[-1]
        if width != self.head.units:
            msg = (
                "MatterSim last-layer features have width "
                f"{width}, expected {self.head.units}"
            )
            raise ValueError(msg)
        source_dtype = next(self.head.parameters()).dtype
        z_table = torch.tensor(
            self.type_to_z,
            dtype=torch.int64,
            device=atype.device,
        )
        atomic_numbers = z_table[atype.to(torch.int64).reshape(-1)]
        atom_energy = self.head.node_energy(
            descriptor.reshape(nf * nloc, width).to(source_dtype),
            atomic_numbers,
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
            "type": "mattersim_ener",
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
    def deserialize(cls, data: dict) -> MatterSimEnergyFitting:
        """Restore a self-contained serialized MatterSim energy head."""
        data = data.copy()
        if data.pop("@class") != "Fitting" or data.pop("type") != "mattersim_ener":
            msg = "data is not a serialized MatterSimEnergyFitting"
            raise ValueError(msg)
        check_version_compatibility(data.pop("@version"), 1, 1)
        variables = {
            name: to_torch_tensor(value)
            for name, value in data.pop("@variables").items()
        }
        fitting = cls(**data)
        validate_mattersim_state_dict_load(
            fitting.head.load_state_dict(variables, strict=False),
        )
        return fitting


__all__ = ["MatterSimEnergyFitting"]
