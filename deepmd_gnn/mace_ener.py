# SPDX-License-Identifier: LGPL-3.0-or-later
"""Original MACE energy head as a DeePMD fitting."""

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

from deepmd_gnn.mace_checkpoint import (
    MaceEnergyHead,
    build_mace_energy_head,
    inspect_native_mace_checkpoint,
    load_native_mace_checkpoint,
    persistable_checkpoint_config,
    validate_mace_state_dict_load,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from deepmd.utils.path import DPPath


def split_packed_layer_features(
    packed: torch.Tensor,
    layer_feature_dims: list[int],
) -> list[torch.Tensor]:
    """Split concatenated per-layer node features back into layer tensors."""
    nf, nloc, width = packed.shape
    expected = sum(layer_feature_dims)
    if width != expected:
        msg = (
            "Packed MACE layer features have width "
            f"{width}, expected {expected} from {layer_feature_dims}"
        )
        raise ValueError(msg)
    layers: list[torch.Tensor] = []
    offset = 0
    for dim in layer_feature_dims:
        layer = packed[:, :, offset : offset + dim].reshape(nf * nloc, dim)
        layers.append(layer)
        offset += dim
    return layers


@Fitting.register("mace_ener")
@fitting_check_output
class MaceEnergyFitting(Fitting):
    """Wrap the native MACE energy readout rather than a new MLP on ``0e``."""

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
            msg = "mace_ener fitting requires the model-level type_map"
            raise ValueError(msg)
        if ntypes != len(type_map):
            msg = f"ntypes={ntypes} does not match type_map length {len(type_map)}"
            raise ValueError(msg)
        if not mixed_types:
            msg = "mace_ener fitting requires mixed_types=True"
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
        source_path = None if model_path is None else Path(model_path)
        if source_path is not None and source_path.is_file():
            model = load_native_mace_checkpoint(
                source_path,
                device=str(env.DEVICE),
            )
            inferred = inspect_native_mace_checkpoint(model)
            checkpoint_type_map = inferred["type_map"]
            if self.type_map != checkpoint_type_map:
                msg = (
                    "Model-level type_map must exactly match checkpoint atomic-number "
                    f"ordering: expected {checkpoint_type_map}, got {self.type_map}"
                )
                raise ValueError(msg)
            self.config = persistable_checkpoint_config(inferred)
            if config is not None:
                config.clear()
                config.update(self.config)
            self.model_path = str(source_path)
            self.head = MaceEnergyHead(model)
        elif config is not None:
            self.config = persistable_checkpoint_config(config)
            checkpoint_type_map = self.config.get("type_map", self.type_map)
            if self.type_map != checkpoint_type_map:
                msg = (
                    "Serialized mace_ener type_map mismatch: "
                    f"expected {checkpoint_type_map}, got {self.type_map}"
                )
                raise ValueError(msg)
            self.model_path = None
            self.head = build_mace_energy_head(self.config)
        elif source_path is not None:
            msg = f"MACE checkpoint not found: {model_path}"
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
        """Return checkpoint elements in atomic-number order."""
        return self.type_map

    def change_type_map(
        self,
        type_map: list[str],
        model_with_new_type_stat: MaceEnergyFitting | None = None,
    ) -> None:
        """Reject type-map changes and subsets."""
        del type_map, model_with_new_type_stat
        msg = "mace_ener fitting does not support changing or subsetting type_map"
        raise NotImplementedError(msg)

    def compute_input_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        protection: float = 1e-2,
        stat_file_path: DPPath | None = None,
    ) -> None:
        """Skip fitting-net statistics; MACE already stores e0 and scale/shift."""
        del merged, protection, stat_file_path

    def compute_output_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Skip DeePMD energy bias; the original atomic_energies_fn is the bias."""
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
        """Apply the original MACE energy head to packed layer features."""
        del descriptor, gr, fparam, aparam
        if g2 is None or h2 is None:
            msg = (
                "mace_ener fitting requires packed layer features in g2 "
                "and pair energy in h2"
            )
            raise ValueError(msg)
        nf, nloc = atype.shape
        source_dtype = next(self.head.parameters()).dtype
        node_attrs = torch.zeros(
            (nf * nloc, self.ntypes),
            dtype=source_dtype,
            device=atype.device,
        )
        node_attrs.scatter_(
            -1,
            atype.to(torch.int64).reshape(-1, 1),
            1,
        )
        layer_features = split_packed_layer_features(
            g2.to(source_dtype),
            self.head.layer_feature_dims,
        )
        atom_energy = self.head.node_energy(
            node_attrs,
            layer_features,
            h2.to(source_dtype).reshape(nf * nloc),
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
            "type": "mace_ener",
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
    def deserialize(cls, data: dict) -> MaceEnergyFitting:
        """Restore a self-contained serialized MACE energy head."""
        data = data.copy()
        if data.pop("@class") != "Fitting" or data.pop("type") != "mace_ener":
            msg = "data is not a serialized MaceEnergyFitting"
            raise ValueError(msg)
        check_version_compatibility(data.pop("@version"), 1, 1)
        variables = {
            name: to_torch_tensor(value)
            for name, value in data.pop("@variables").items()
        }
        data.pop("layer_feature_dims", None)
        fitting = cls(**data)
        current = fitting.head.state_dict()
        aligned = {}
        for name, tensor in variables.items():
            target = current.get(name)
            if (
                target is not None
                and target.shape != tensor.shape
                and target.numel() == tensor.numel()
            ):
                aligned[name] = tensor.reshape(target.shape)
            else:
                aligned[name] = tensor
        validate_mace_state_dict_load(
            fitting.head.load_state_dict(aligned, strict=False),
        )
        return fitting


__all__ = ["MaceEnergyFitting", "split_packed_layer_features"]
