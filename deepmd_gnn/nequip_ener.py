# SPDX-License-Identifier: LGPL-3.0-or-later
"""Original NequIP energy readout as a DeePMD fitting."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
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

from deepmd_gnn.nequip_descriptor import (
    _artifact_params,
    _load_serialized_nequip,
    _make_nequip_network,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from deepmd.utils.path import DPPath


def persistable_nequip_config(params: dict[str, Any]) -> dict[str, Any]:
    """Return architecture metadata needed to rebuild the NequIP energy readout."""
    return deepcopy(params)


def _readout_linear_from_network(
    params: dict[str, Any],
    ntypes: int,
) -> torch.nn.Module:
    """Build a standalone copy of ``output_hidden_to_scalar.linear``."""
    full = _make_nequip_network(params, ntypes)
    linear = full.model.output_hidden_to_scalar.linear
    clone = type(linear)(linear.irreps_in, linear.irreps_out)
    clone.load_state_dict(linear.state_dict())
    return clone.to(env.DEVICE)


def _load_readout_from_artifact(
    payload: dict[str, Any],
    params: dict[str, Any],
    ntypes: int,
) -> tuple[torch.nn.Module, torch.Tensor]:
    """Restore the original scalar readout and DeePMD ``e0`` from an artifact."""
    linear = _readout_linear_from_network(params, ntypes)
    prefix = "model.output_hidden_to_scalar.linear."
    variables = payload["@variables"]
    source = {
        key.removeprefix(prefix): to_torch_tensor(value)
        for key, value in variables.items()
        if key.startswith(prefix)
    }
    missing = sorted(set(linear.state_dict()) - set(source))
    unexpected = sorted(set(source) - set(linear.state_dict()))
    if missing or unexpected:
        msg = (
            "NequIP energy readout state mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
        raise ValueError(msg)
    linear.load_state_dict(source, strict=True)
    if "e0" not in variables:
        msg = "Serialized NequipModel artifact has no e0"
        raise ValueError(msg)
    e0 = to_torch_tensor(variables["e0"]).to(
        dtype=env.GLOBAL_PT_ENER_FLOAT_PRECISION,
        device=env.DEVICE,
    )
    return linear, e0


class NequipEnergyHead(torch.nn.Module):
    """Original NequIP ``output_hidden_to_scalar`` plus DeePMD ``e0``."""

    def __init__(self, linear: torch.nn.Module, e0: torch.Tensor) -> None:
        super().__init__()
        self.linear = linear
        self.register_buffer(
            "e0",
            e0.to(
                dtype=env.GLOBAL_PT_ENER_FLOAT_PRECISION,
                device=e0.device,
            ),
        )

    def node_energy(
        self,
        features: torch.Tensor,
        atype: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-atom energies from projected invariants."""
        source_dtype = next(self.linear.parameters()).dtype
        scalar = self.linear(features.to(source_dtype)).reshape(-1)
        shift = self.e0[atype.reshape(-1).to(torch.int64)].to(scalar.dtype)
        return scalar + shift


@Fitting.register("nequip_ener")
@fitting_check_output
class NequipEnergyFitting(Fitting):
    """Wrap the native NequIP energy readout rather than a new MLP on ``0e``."""

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        type_map: list[str] | None = None,
        mixed_types: bool = True,
        model_file: str | Path | None = None,
        config: dict[str, Any] | None = None,
        trainable: bool = True,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__()
        del kwargs
        if type_map is None:
            msg = "nequip_ener fitting requires the model-level type_map"
            raise ValueError(msg)
        if ntypes != len(type_map):
            msg = f"ntypes={ntypes} does not match type_map length {len(type_map)}"
            raise ValueError(msg)
        if not mixed_types:
            msg = "nequip_ener fitting requires mixed_types=True"
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
        self.model_file: str | None = None
        source_path = None if model_file is None else Path(model_file)
        if source_path is not None and source_path.is_file():
            payload = _load_serialized_nequip(source_path)
            artifact_params = _artifact_params(payload)
            checkpoint_type_map = artifact_params.get("type_map")
            if self.type_map != checkpoint_type_map:
                msg = (
                    "Model-level type_map must exactly match artifact type_map: "
                    f"expected {checkpoint_type_map}, got {self.type_map}"
                )
                raise ValueError(msg)
            self.config = persistable_nequip_config(artifact_params)
            if config is not None:
                config.clear()
                config.update(self.config)
            linear, e0 = _load_readout_from_artifact(
                payload,
                self.config,
                self.ntypes,
            )
            self.model_file = str(source_path)
            self.head = NequipEnergyHead(linear, e0)
        elif config is not None:
            self.config = persistable_nequip_config(config)
            checkpoint_type_map = self.config.get("type_map", self.type_map)
            if self.type_map != checkpoint_type_map:
                msg = (
                    "Serialized nequip_ener type_map mismatch: "
                    f"expected {checkpoint_type_map}, got {self.type_map}"
                )
                raise ValueError(msg)
            if self.config.get("sel") is None:
                msg = "Serialized nequip_ener config is missing sel"
                raise ValueError(msg)
            self.model_file = None
            linear = _readout_linear_from_network(self.config, self.ntypes)
            e0 = torch.zeros(
                self.ntypes,
                dtype=env.GLOBAL_PT_ENER_FLOAT_PRECISION,
                device=env.DEVICE,
            )
            self.head = NequipEnergyHead(linear, e0)
        elif source_path is not None:
            msg = f"NequIP artifact not found: {model_file}"
            raise FileNotFoundError(msg)
        else:
            msg = (
                "Exactly one of model_file (initialization) or config "
                "(deserialization) must be provided"
            )
            raise ValueError(msg)

        expected_dim = int(self.head.linear.irreps_in.dim)
        if self.dim_descrpt != expected_dim:
            msg = (
                "nequip_ener dim_descrpt must match conv_to_output_hidden width: "
                f"expected {expected_dim}, got {self.dim_descrpt}"
            )
            raise ValueError(msg)
        if self.head.e0.numel() != self.ntypes:
            msg = (
                "nequip_ener e0 length must match ntypes: "
                f"expected {self.ntypes}, got {self.head.e0.numel()}"
            )
            raise ValueError(msg)

        for parameter in self.head.parameters():
            parameter.requires_grad_(self.trainable)
        self.head.e0.requires_grad_(self.trainable)

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
        """Return artifact elements in stored order."""
        return self.type_map

    def change_type_map(
        self,
        type_map: list[str],
        model_with_new_type_stat: NequipEnergyFitting | None = None,
    ) -> None:
        """Reject type-map changes and subsets."""
        del type_map, model_with_new_type_stat
        msg = "nequip_ener fitting does not support changing or subsetting type_map"
        raise NotImplementedError(msg)

    def compute_input_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        protection: float = 1e-2,
        stat_file_path: DPPath | None = None,
    ) -> None:
        """Skip fitting-net statistics; NequIP already stores e0."""
        del merged, protection, stat_file_path

    def compute_output_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Skip DeePMD energy bias; the original e0 is the bias."""
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
        return self.head.e0

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
        """Apply the original NequIP scalar readout to last-layer invariants."""
        del gr, g2, h2, fparam, aparam
        nf, nloc = atype.shape
        atom_energy = self.head.node_energy(
            descriptor.reshape(nf * nloc, self.dim_descrpt),
            atype,
        )
        return {
            "energy": atom_energy.view(nf, nloc, 1).to(env.GLOBAL_PT_FLOAT_PRECISION),
        }

    def serialize(self) -> dict:
        """Serialize architecture and the original energy-head weights."""
        config = persistable_nequip_config(self.config)
        config["type_map"] = self.type_map
        return {
            "@class": "Fitting",
            "@version": 1,
            "type": "nequip_ener",
            "ntypes": self.ntypes,
            "dim_descrpt": self.dim_descrpt,
            "type_map": self.type_map,
            "mixed_types": True,
            "model_file": None,
            "trainable": self.trainable,
            "config": config,
            "@variables": {
                name: to_numpy_array(value)
                for name, value in self.head.state_dict().items()
            },
        }

    @classmethod
    def deserialize(cls, data: dict) -> NequipEnergyFitting:
        """Restore a self-contained serialized NequIP energy head."""
        data = data.copy()
        if data.pop("@class") != "Fitting" or data.pop("type") != "nequip_ener":
            msg = "data is not a serialized NequipEnergyFitting"
            raise ValueError(msg)
        check_version_compatibility(data.pop("@version"), 1, 1)
        variables = {
            name: to_torch_tensor(value)
            for name, value in data.pop("@variables").items()
        }
        fitting = cls(**data)
        fitting.head.load_state_dict(variables, strict=True)
        return fitting


__all__ = [
    "NequipEnergyFitting",
    "NequipEnergyHead",
    "persistable_nequip_config",
]
