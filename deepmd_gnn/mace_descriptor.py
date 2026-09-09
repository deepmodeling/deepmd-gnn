# SPDX-License-Identifier: LGPL-3.0-or-later
"""MACE backbone descriptor for DeePMD property fitting."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from deepmd.pt.model.descriptor.base_descriptor import BaseDescriptor
from deepmd.pt.utils import env
from deepmd.pt.utils.utils import to_numpy_array, to_torch_tensor
from deepmd.utils.version import check_version_compatibility
from e3nn import o3

import deepmd_gnn.op  # noqa: F401
from deepmd_gnn.mace import PeriodicTable
from deepmd_gnn.mace_network import make_mace_network
from deepmd_gnn.mace_off import (
    _infer_deepmd_config,
    _load_mace_checkpoint,
    _temporary_default_dtype,
    _validate_load_result,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from deepmd.utils.data_system import DeepmdDataSystem
    from deepmd.utils.path import DPPath


def _scalar_even_indices(irreps: o3.Irreps) -> list[int]:
    """Return flattened indices belonging to invariant, even scalar irreps."""
    indices: list[int] = []
    offset = 0
    for multiplicity, irrep in irreps:
        width = multiplicity * irrep.dim
        if irrep.l == 0 and irrep.p == 1:
            indices.extend(range(offset, offset + width))
        offset += width
    return indices


def _product_output_irreps(model: torch.nn.Module) -> o3.Irreps:
    """Read the actual output irreps of the final MACE product layer."""
    products = getattr(model, "products", None)
    if products is None or len(products) == 0:
        msg = "Loaded MACE model has no product layers"
        raise ValueError(msg)
    linear = getattr(products[-1], "linear", None)
    irreps_out = getattr(linear, "irreps_out", None)
    if irreps_out is None:
        msg = "Final MACE product layer does not expose output irreps"
        raise ValueError(msg)
    return o3.Irreps(irreps_out)


def _validate_descriptor_checkpoint(model: torch.nn.Module) -> None:
    """Reject options that cannot be reconstructed during serialization."""
    unsupported_flags = {
        "use_agnostic_product": False,
        "use_so3": False,
        "use_last_readout_only": False,
        "use_embedding_readout": False,
        "use_edge_irreps_first": False,
        "apply_cutoff": True,
    }
    for name, supported_value in unsupported_flags.items():
        value = getattr(model, name, supported_value)
        if value != supported_value:
            msg = (
                f"Unsupported MACE descriptor checkpoint option: {name}={value!r}; "
                f"only {supported_value!r} is supported"
            )
            raise ValueError(msg)
    if getattr(model, "use_reduced_cg", True) is False:
        msg = "MACE descriptor checkpoints with use_reduced_cg=False are unsupported"
        raise ValueError(msg)
    if getattr(model, "edge_irreps", None) is not None:
        msg = "MACE descriptor checkpoints with edge_irreps are unsupported"
        raise ValueError(msg)

    for module in [model, *getattr(model, "products", [])]:
        cueq_config = getattr(module, "cueq_config", None)
        if cueq_config is not None and bool(getattr(cueq_config, "enabled", False)):
            msg = "cuEquivariance MACE descriptor checkpoints are unsupported"
            raise ValueError(msg)


@BaseDescriptor.register("mace")
class MaceDescriptor(BaseDescriptor, torch.nn.Module):
    """Expose final MACE product-layer ``0e`` features as a DeePMD descriptor.

    ``model_path`` is loaded with native Python pickle semantics and therefore
    must point to a trusted local ``ScaleShiftMACE`` checkpoint.
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
        del kwargs
        if not isinstance(sel, int) or isinstance(sel, bool) or sel <= 0:
            msg = f"sel must be an explicit positive integer, got {sel!r}"
            raise ValueError(msg)
        if type_map is None:
            msg = "MACE descriptor requires the model-level type_map"
            raise ValueError(msg)
        if ntypes is not None and ntypes != len(type_map):
            msg = f"ntypes={ntypes} does not match type_map length {len(type_map)}"
            raise ValueError(msg)
        if (model_path is None) == (config is None):
            msg = (
                "Exactly one of model_path (initialization) or config "
                "(deserialization) must be provided"
            )
            raise ValueError(msg)

        self.sel = int(sel)
        self.type_map = list(type_map)
        self.ntypes = len(self.type_map)
        self.trainable = bool(trainable)
        self.model_path = None if model_path is None else str(model_path)

        if model_path is not None:
            model = _load_mace_checkpoint(
                Path(model_path),
                device=str(env.DEVICE),
            )
            _validate_descriptor_checkpoint(model)
            inferred = _infer_deepmd_config(
                model,
                allow_pair_repulsion=True,
            )
            checkpoint_type_map = inferred["type_map"]
            if self.type_map != checkpoint_type_map:
                msg = (
                    "Model-level type_map must exactly match checkpoint atomic-number "
                    f"ordering: expected {checkpoint_type_map}, got {self.type_map}"
                )
                raise ValueError(msg)
            self.config = dict(inferred)
            self.backbone = model
        else:
            self.config = deepcopy(cast("dict[str, Any]", config))
            checkpoint_type_map = self.config.pop("type_map", self.type_map)
            if self.type_map != checkpoint_type_map:
                msg = (
                    "Serialized MACE descriptor type_map mismatch: "
                    f"expected {checkpoint_type_map}, got {self.type_map}"
                )
                raise ValueError(msg)
            source_dtype_name = self.config.pop("source_dtype", "float64")
            source_dtype = getattr(torch, source_dtype_name)
            atomic_numbers = [PeriodicTable[name] for name in self.type_map]
            with _temporary_default_dtype(source_dtype):
                self.backbone = make_mace_network(
                    r_max=self.config["r_max"],
                    num_radial_basis=self.config["num_radial_basis"],
                    num_cutoff_basis=self.config["num_cutoff_basis"],
                    max_ell=self.config["max_ell"],
                    interaction=self.config["interaction"],
                    num_interactions=self.config["num_interactions"],
                    num_elements=self.ntypes,
                    hidden_irreps=self.config["hidden_irreps"],
                    atomic_numbers=atomic_numbers,
                    avg_num_neighbors=self.config["avg_num_neighbors"],
                    pair_repulsion=self.config["pair_repulsion"],
                    distance_transform=self.config["distance_transform"],
                    correlation=self.config["correlation"],
                    gate=self.config["gate"],
                    MLP_irreps=self.config["MLP_irreps"],
                    std=self.config["std"],
                    radial_MLP=self.config["radial_MLP"],
                    radial_type=self.config["radial_type"],
                    enable_cueq=False,
                    script_model=False,
                    keep_last_layer_irreps=self.config.get(
                        "keep_last_layer_irreps",
                        False,
                    ),
                )

        self.rcut = float(self.config["r_max"])
        self.num_interactions = int(self.config["num_interactions"])
        output_irreps = _product_output_irreps(self.backbone)
        scalar_indices = _scalar_even_indices(output_irreps)
        if not scalar_indices:
            msg = f"Final MACE product output has no 0e channels: {output_irreps}"
            raise ValueError(msg)
        self.output_irreps = str(output_irreps)
        self.register_buffer(
            "_scalar_indices",
            torch.tensor(scalar_indices, dtype=torch.int64, device=env.DEVICE),
            persistent=False,
        )
        empty = torch.empty(0, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)
        self.register_buffer("_stat_mean", empty.clone(), persistent=False)
        self.register_buffer("_stat_stddev", empty.clone(), persistent=False)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(self.trainable)

    def get_rcut(self) -> float:
        """Return the checkpoint cutoff radius."""
        return self.rcut

    def get_rcut_smth(self) -> float:
        """Return the effective smooth cutoff radius."""
        return self.rcut

    def get_sel(self) -> list[int]:
        """Return the mixed-type neighbor-list capacity."""
        return [self.sel]

    def get_ntypes(self) -> int:
        """Return the number of checkpoint elements."""
        return self.ntypes

    def get_type_map(self) -> list[str]:
        """Return checkpoint elements in atomic-number order."""
        return self.type_map

    def get_dim_out(self) -> int:
        """Return the number of final-layer ``0e`` channels."""
        return int(self._scalar_indices.numel())

    def get_dim_emb(self) -> int:
        """Return the invariant feature width."""
        return self.get_dim_out()

    def mixed_types(self) -> bool:
        """Declare use of a mixed-type neighbor list."""
        return True

    def has_message_passing(self) -> bool:
        """Return whether the backbone has multiple interaction layers."""
        return self.num_interactions > 1

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
        """Skip input statistics because MACE normalizes internally."""
        del merged, path

    def get_stats(self) -> dict:
        """Return the empty input-statistics collection."""
        return {}

    def set_stat_mean_and_stddev(
        self,
        mean: torch.Tensor,
        stddev: torch.Tensor,
    ) -> None:
        """Ignore external input statistics because MACE uses none."""
        del mean, stddev

    def get_stat_mean_and_stddev(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return empty compatibility statistics."""
        return self._stat_mean, self._stat_stddev

    def share_params(
        self,
        base_class: MaceDescriptor,
        shared_level: int,
        resume: bool = False,
    ) -> None:
        """Share a complete MACE backbone at level zero."""
        del resume
        if not isinstance(base_class, MaceDescriptor):
            msg = "MACE descriptors can only share with MACE descriptors"
            raise TypeError(msg)
        if shared_level != 0:
            msg = "MACE descriptor only supports full-backbone sharing at level 0"
            raise NotImplementedError(msg)
        self.backbone = base_class.backbone

    def change_type_map(
        self,
        type_map: list[str],
        model_with_new_type_stat: MaceDescriptor | None = None,
    ) -> None:
        """Reject type-map changes and subsets."""
        del type_map, model_with_new_type_stat
        msg = "MACE descriptor does not support changing or subsetting type_map"
        raise NotImplementedError(msg)

    def _final_product_features(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None,
    ) -> torch.Tensor:
        nf, nloc, _ = nlist.shape
        nall = extended_atype.shape[1]
        positions = extended_coord.view(nf, nall, 3)
        source_dtype = self.backbone.atomic_energies_fn.atomic_energies.dtype
        positions_flat = positions.to(source_dtype).flatten(0, 1)
        atype = extended_atype.to(torch.int64)
        edge_index = torch.ops.deepmd_gnn.edge_index(
            nlist.to(torch.int64),
            atype,
            torch.empty(0, dtype=torch.int64, device="cpu"),
        ).T
        shifts = torch.zeros(
            (edge_index.shape[1], 3),
            dtype=source_dtype,
            device=positions_flat.device,
        )

        if self.num_interactions > 1 and mapping is not None and nloc < nall:
            mapping_flat = mapping.reshape(-1) + torch.arange(
                0,
                nf * nall,
                nall,
                dtype=mapping.dtype,
                device=mapping.device,
            ).unsqueeze(-1).expand(nf, nall).reshape(-1)
            image_shifts = positions_flat - positions_flat[mapping_flat]
            shifts = image_shifts[edge_index[1]] - image_shifts[edge_index[0]]
            edge_index = mapping_flat[edge_index]
        elif self.num_interactions > 1 and nloc < nall:
            msg = "Multi-layer MACE descriptor requires mapping for extended atoms"
            raise ValueError(msg)

        vectors = positions_flat[edge_index[1]] - positions_flat[edge_index[0]]
        vectors = vectors + shifts
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        node_attrs = torch.zeros(
            (nf * nall, self.ntypes),
            dtype=source_dtype,
            device=positions_flat.device,
        )
        node_attrs.scatter_(
            -1,
            atype.reshape(-1, 1),
            1,
        )
        node_feats = self.backbone.node_embedding(node_attrs)
        edge_attrs = self.backbone.spherical_harmonics(vectors)
        edge_feats, cutoff = self.backbone.radial_embedding(
            lengths,
            node_attrs,
            edge_index,
            self.backbone.atomic_numbers,
        )
        for index, (interaction, product) in enumerate(
            zip(self.backbone.interactions, self.backbone.products, strict=True),
        ):
            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
                cutoff=cutoff,
                first_layer=index == 0,
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=node_attrs,
            )
        return node_feats.view(nf, nall, -1)[:, :nloc]

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
        """Compute local-atom invariant final-product features."""
        if comm_dict is not None:
            msg = "MPI communication is out of scope for the MACE descriptor"
            raise NotImplementedError(msg)
        del fparam, charge_spin
        features = self._final_product_features(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
        )
        invariant = torch.index_select(
            features,
            dim=-1,
            index=self._scalar_indices.to(features.device),
        )
        return (
            invariant.to(env.GLOBAL_PT_FLOAT_PRECISION),
            None,
            None,
            None,
            None,
        )

    def serialize(self) -> dict:
        """Serialize architecture and all trained backbone state."""
        source_dtype = self.backbone.atomic_energies_fn.atomic_energies.dtype
        config = deepcopy(self.config)
        config["type_map"] = self.type_map
        config["source_dtype"] = str(source_dtype).removeprefix("torch.")
        return {
            "@class": "Descriptor",
            "@version": 1,
            "type": "mace",
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
    def deserialize(cls, data: dict) -> MaceDescriptor:
        """Restore a self-contained serialized MACE descriptor."""
        data = data.copy()
        if data.pop("@class") != "Descriptor" or data.pop("type") != "mace":
            msg = "data is not a serialized MaceDescriptor"
            raise ValueError(msg)
        check_version_compatibility(data.pop("@version"), 1, 1)
        variables = {
            name: to_torch_tensor(value)
            for name, value in data.pop("@variables").items()
        }
        descriptor = cls(**data)
        result = descriptor.backbone.load_state_dict(variables, strict=False)
        _validate_load_result(result)
        return descriptor

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: list[str] | None,
        local_jdata: dict,
    ) -> tuple[dict, float | None]:
        """Validate the required explicit neighbor-list capacity."""
        del train_data, type_map
        local_jdata = local_jdata.copy()
        sel = local_jdata.get("sel")
        if not isinstance(sel, int) or isinstance(sel, bool) or sel <= 0:
            msg = "MACE descriptor requires an explicit positive integer sel"
            raise ValueError(msg)
        return local_jdata, None


__all__ = ["MaceDescriptor"]
