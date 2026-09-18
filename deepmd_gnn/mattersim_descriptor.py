# SPDX-License-Identifier: LGPL-3.0-or-later
"""MatterSim M3GNet backbone descriptor for DeePMD property fitting."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import torch
from ase.data import atomic_numbers as ase_atomic_numbers
from deepmd.pt.model.descriptor.base_descriptor import BaseDescriptor
from deepmd.pt.utils import env
from deepmd.pt.utils.utils import to_numpy_array, to_torch_tensor
from deepmd.utils.version import check_version_compatibility

import deepmd_gnn.op  # noqa: F401
from deepmd_gnn.mattersim_checkpoint import (
    build_mattersim_feature_backbone,
    compute_threebody_indices,
    load_mattersim_checkpoint_config,
    load_native_mattersim_feature_backbone,
    persistable_checkpoint_config,
    resolve_mattersim_checkpoint_path,
    validate_mattersim_state_dict_load,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from deepmd.utils.data_system import DeepmdDataSystem
    from deepmd.utils.path import DPPath


def _atomic_numbers_from_type_map(type_map: list[str]) -> list[int]:
    """Map DeePMD type symbols onto MatterSim nuclear charges."""
    numbers: list[int] = []
    for symbol in type_map:
        if symbol not in ase_atomic_numbers:
            msg = f"MatterSim descriptor type_map entry {symbol!r} is not an element"
            raise ValueError(msg)
        numbers.append(int(ase_atomic_numbers[symbol]))
    return numbers


@BaseDescriptor.register("mattersim")
class MatterSimDescriptor(BaseDescriptor, torch.nn.Module):
    """Expose last-layer MatterSim M3GNet ``atom_attr`` as a DeePMD descriptor.

    ``model_path`` is loaded with native Python pickle semantics and therefore
    must point to a trusted local MatterSim M3GNet checkpoint or pretrained
    keyword. The original GatedMLP energy readout is not used.
    """

    def __init__(
        self,
        sel: int,
        model_path: str | Path | None = None,
        *,
        type_map: list[str] | None = None,
        ntypes: int | None = None,
        trainable: bool = True,
        config: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__()
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            msg = f"Unsupported MatterSim descriptor arguments: {unknown}"
            raise TypeError(msg)
        if not isinstance(sel, int) or isinstance(sel, bool) or sel <= 0:
            msg = f"sel must be an explicit positive integer, got {sel!r}"
            raise ValueError(msg)
        if type_map is None:
            msg = "MatterSim descriptor requires the model-level type_map"
            raise ValueError(msg)
        if ntypes is not None and ntypes != len(type_map):
            msg = f"ntypes={ntypes} does not match type_map length {len(type_map)}"
            raise ValueError(msg)

        self.sel = int(sel)
        self.type_map = list(type_map)
        self.ntypes = len(self.type_map)
        self.trainable = bool(trainable)
        self.type_to_z = _atomic_numbers_from_type_map(self.type_map)
        self.model_path: str | None = None
        resolved = (
            None
            if model_path is None
            else resolve_mattersim_checkpoint_path(model_path)
        )
        if resolved is not None:
            backbone, inferred = load_native_mattersim_feature_backbone(
                resolved,
                device=str(env.DEVICE),
            )
            inferred["type_map"] = self.type_map
            self.config = persistable_checkpoint_config(inferred)
            if config is not None:
                config.clear()
                config.update(self.config)
            self.model_path = str(model_path)
            self.backbone = backbone
        elif config is not None:
            persistable = persistable_checkpoint_config(config)
            persistable["type_map"] = persistable["type_map"] or self.type_map
            if persistable["type_map"] != self.type_map:
                msg = (
                    "Serialized MatterSim descriptor type_map mismatch: "
                    f"expected {persistable['type_map']}, got {self.type_map}"
                )
                raise ValueError(msg)
            self.config = persistable
            self.model_path = None
            self.backbone = build_mattersim_feature_backbone(
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
        self.rcut = float(self.config["cutoff"])
        self.threebody_cutoff = float(self.config["threebody_cutoff"])
        self.num_blocks = int(self.config["num_blocks"])
        self.backbone_float64 = next(self.backbone.parameters()).dtype == torch.float64
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(self.trainable)

    def has_default_chg_spin(self) -> bool:
        """Declare absent charge/spin defaults for newer DeePMD model exports."""
        return False

    def get_default_chg_spin(self) -> None:
        """Return no charge/spin defaults with a concrete TorchScript type."""
        return None  # noqa: RET501

    def get_rcut(self) -> float:
        """Return the checkpoint pair cutoff radius."""
        return self.rcut

    def get_rcut_smth(self) -> float:
        """Return the effective smooth cutoff radius."""
        return self.rcut

    def get_sel(self) -> list[int]:
        """Return the mixed-type neighbor-list capacity."""
        return [self.sel]

    def get_ntypes(self) -> int:
        """Return the number of model-level elements."""
        return self.ntypes

    def get_type_map(self) -> list[str]:
        """Return model-level elements."""
        return self.type_map

    def get_dim_out(self) -> int:
        """Return the last-layer M3GNet atom-feature width."""
        return int(self.config["units"])

    def get_dim_emb(self) -> int:
        """Return the invariant feature width."""
        return self.get_dim_out()

    def mixed_types(self) -> bool:
        """Declare use of a mixed-type neighbor list."""
        return True

    def has_message_passing(self) -> bool:
        """Return whether the backbone has multiple interaction blocks."""
        return self.num_blocks > 1

    def has_message_passing_across_ranks(self) -> bool:
        """Declare that cross-rank feature exchange is unsupported."""
        return False

    def supports_edge_parallel(self) -> bool:
        """Declare MPI edge-parallel execution unsupported."""
        return False

    def dense_lower_supports_comm(self) -> bool:
        """Declare that the dense lower does not accept communication data."""
        return False

    def need_sorted_nlist_for_lower(self) -> bool:
        """Return whether lower neighbor lists need sorting."""
        return False

    def get_env_protection(self) -> float:
        """Return the unused environment-matrix protection value."""
        return 0.0

    def compute_input_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        path: DPPath | None = None,
    ) -> None:
        """Skip input statistics because MatterSim embeds nuclear charges."""
        del merged, path

    def get_stats(self) -> dict:
        """Return the empty input-statistics collection."""
        return {}

    def set_stat_mean_and_stddev(
        self,
        mean: torch.Tensor,
        stddev: torch.Tensor,
    ) -> None:
        """Ignore external input statistics because MatterSim uses none."""
        del mean, stddev

    def get_stat_mean_and_stddev(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return empty compatibility statistics."""
        empty = torch.empty(0, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)
        return empty, empty.clone()

    def share_params(
        self,
        base_class: MatterSimDescriptor,
        shared_level: int,
        resume: bool = False,
    ) -> None:
        """Share a complete MatterSim backbone at level zero."""
        del resume
        if not isinstance(base_class, MatterSimDescriptor):
            msg = "MatterSim descriptors can only share with MatterSim descriptors"
            raise TypeError(msg)
        if shared_level != 0:
            msg = "MatterSim descriptor only supports full-backbone sharing at level 0"
            raise NotImplementedError(msg)
        self.backbone = base_class.backbone
        self.type_to_z = base_class.type_to_z
        self.backbone_float64 = base_class.backbone_float64
        self.threebody_cutoff = base_class.threebody_cutoff

    def change_type_map(
        self,
        type_map: list[str],
        model_with_new_type_stat: MatterSimDescriptor | None = None,
    ) -> None:
        """Reject type-map changes and subsets."""
        del type_map, model_with_new_type_stat
        msg = "MatterSim descriptor does not support changing or subsetting type_map"
        raise NotImplementedError(msg)

    def _last_layer_features(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None,
    ) -> torch.Tensor:
        nf, nloc, _ = nlist.shape
        nall = extended_atype.shape[1]
        source_dtype = torch.float64 if self.backbone_float64 else torch.float32
        positions = extended_coord.view(nf, nall, 3).to(source_dtype)
        positions_flat = positions.flatten(0, 1)
        atype = extended_atype.to(torch.int64)
        # DeePMD op rows are (neighbor, center); MatterSim stores (center, neighbor).
        edge_index = torch.ops.deepmd_gnn.edge_index(
            nlist.to(torch.int64),
            atype,
            torch.empty(0, dtype=torch.int64, device="cpu"),
        ).T[[1, 0]]
        n_local_total = nf * nloc
        local_pos = positions[:, :nloc, :].reshape(n_local_total, 3)
        identity = torch.eye(3, dtype=source_dtype, device=positions.device)
        cell = identity.unsqueeze(0).expand(nf, -1, -1).contiguous()
        if nloc < nall:
            if mapping is None:
                msg = "MatterSim descriptor requires mapping for extended atoms"
                raise ValueError(msg)
            compact_mapping = (
                mapping.to(torch.int64)
                + torch.arange(
                    nf,
                    dtype=torch.int64,
                    device=mapping.device,
                ).unsqueeze(-1)
                * nloc
            ).reshape(-1)
            neighbor_ext = edge_index[1]
            pbc_offsets = (
                positions_flat[neighbor_ext] - local_pos[compact_mapping[neighbor_ext]]
            )
            edge_index = compact_mapping[edge_index]
        else:
            pbc_offsets = torch.zeros(
                (edge_index.shape[1], 3),
                dtype=source_dtype,
                device=positions.device,
            )

        if edge_index.shape[1] > 0:
            order = torch.argsort(edge_index[0], stable=True)
            edge_index = edge_index[:, order]
            pbc_offsets = pbc_offsets[order]

        local_atype = atype[:, :nloc].reshape(-1)
        type_to_z = torch.tensor(
            self.type_to_z,
            dtype=torch.int64,
            device=atype.device,
        )
        atomic_numbers = type_to_z[local_atype].to(source_dtype).unsqueeze(-1)
        batch = (
            torch.arange(nf, dtype=torch.int64, device=atype.device)
            .unsqueeze(-1)
            .expand(nf, nloc)
            .reshape(-1)
        )
        num_atoms = torch.full(
            (nf,),
            nloc,
            dtype=torch.int64,
            device=atype.device,
        )
        if edge_index.shape[1] == 0:
            num_bonds = torch.zeros((nf,), dtype=torch.int64, device=atype.device)
            distances = torch.zeros((0,), dtype=source_dtype, device=positions.device)
        else:
            num_bonds = torch.bincount(batch[edge_index[0]], minlength=nf)
            edge_vector = local_pos[edge_index[0]] - (
                local_pos[edge_index[1]] + pbc_offsets
            )
            distances = torch.norm(edge_vector, p=2, dim=1)

        three_body_indices, num_triple_ij = compute_threebody_indices(
            edge_index,
            distances,
            num_atoms,
            n_local_total,
            self.threebody_cutoff,
        )
        features = self.backbone(
            local_pos,
            cell,
            pbc_offsets,
            atomic_numbers,
            edge_index,
            three_body_indices,
            num_bonds,
            num_triple_ij,
            num_atoms,
            batch,
        )
        return features.view(nf, nloc, -1)

    def forward(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        comm_dict: dict[str, torch.Tensor] | None = None,
        fparam: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Compute local-atom invariant last-layer features."""
        if comm_dict is not None:
            msg = "MPI communication is out of scope for the MatterSim descriptor"
            raise NotImplementedError(msg)
        del fparam, charge_spin
        features = self._last_layer_features(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
        )
        return (
            features.to(env.GLOBAL_PT_FLOAT_PRECISION),
            None,
            None,
            None,
            None,
        )

    def serialize(self) -> dict:
        """Serialize architecture and all trained backbone state."""
        config = deepcopy(self.config)
        config["type_map"] = self.type_map
        return {
            "@class": "Descriptor",
            "@version": 1,
            "type": "mattersim",
            "sel": self.sel,
            "model_path": None,
            "type_map": self.type_map,
            "ntypes": self.ntypes,
            "trainable": self.trainable,
            "config": config,
            "@variables": {
                name: to_numpy_array(value)
                for name, value in self.backbone.state_dict().items()
            },
        }

    @classmethod
    def deserialize(cls, data: dict) -> MatterSimDescriptor:
        """Restore a self-contained serialized MatterSim descriptor."""
        data = data.copy()
        if data.pop("@class") != "Descriptor" or data.pop("type") != "mattersim":
            msg = "data is not a serialized MatterSimDescriptor"
            raise ValueError(msg)
        check_version_compatibility(data.pop("@version"), 1, 1)
        variables = {
            name: to_torch_tensor(value)
            for name, value in data.pop("@variables").items()
        }
        descriptor = cls(**data)
        validate_mattersim_state_dict_load(
            descriptor.backbone.load_state_dict(variables, strict=False),
        )
        return descriptor

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: list[str] | None,
        local_jdata: dict,
    ) -> tuple[dict, float | None]:
        """Persist inferred architecture and validate neighbor-list capacity."""
        del train_data
        local_jdata = local_jdata.copy()
        sel = local_jdata.get("sel")
        if not isinstance(sel, int) or isinstance(sel, bool) or sel <= 0:
            msg = "MatterSim descriptor requires an explicit positive integer sel"
            raise ValueError(msg)
        model_path = local_jdata.get("model_path")
        if model_path:
            inferred = load_mattersim_checkpoint_config(model_path)
            if type_map is not None:
                inferred["type_map"] = list(type_map)
            local_jdata["config"] = inferred
        return local_jdata, None


__all__ = ["MatterSimDescriptor"]
