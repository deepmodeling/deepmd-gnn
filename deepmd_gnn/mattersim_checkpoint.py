# SPDX-License-Identifier: LGPL-3.0-or-later
"""Native MatterSim M3GNet checkpoint support for property descriptors."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

ENERGY_STATE_PREFIXES = ("final.", "normalizer.")
MAX_GRAPH_CONV_BLOCKS = 8
PRETRAINED_FILENAMES = {
    "mattersim-v1.0.0-1m": "mattersim-v1.0.0-1M.pth",
    "mattersim-v1.0.0-1m.pth": "mattersim-v1.0.0-1M.pth",
    "mattersim-v1.0.0-5m": "mattersim-v1.0.0-5M.pth",
    "mattersim-v1.0.0-5m.pth": "mattersim-v1.0.0-5M.pth",
}
_PRETRAINED_DIR = Path.home() / ".local" / "mattersim" / "pretrained_models"
_MODEL_ARG_KEYS = (
    "num_blocks",
    "units",
    "max_l",
    "max_n",
    "cutoff",
    "max_z",
    "threebody_cutoff",
)


def _load_forcefield_namespace() -> None:
    """Expose ``mattersim.forcefield.m3gnet`` without importing Potential/pymatgen."""
    import mattersim  # noqa: PLC0415

    forcefield_dir = Path(mattersim.__file__).resolve().parent / "forcefield"
    existing = sys.modules.get("mattersim.forcefield")
    if existing is not None and getattr(existing, "__path__", None):
        try:
            from mattersim.forcefield.m3gnet.modules import MLP  # noqa: F401, PLC0415
        except ImportError:
            pass
        else:
            return
    for name in list(sys.modules):
        if name == "mattersim.forcefield" or name.startswith("mattersim.forcefield."):
            del sys.modules[name]
    pkg = ModuleType("mattersim.forcefield")
    pkg.__path__ = [str(forcefield_dir)]  # type: ignore[attr-defined]
    pkg.__file__ = str(forcefield_dir / "__init__.py")
    pkg.__package__ = "mattersim.forcefield"
    sys.modules["mattersim.forcefield"] = pkg


def _import_m3gnet_modules() -> tuple[Any, Any, Any, Any]:
    """Import official M3GNet layers without constructing the energy model."""
    try:
        from mattersim.forcefield.m3gnet.modules import (  # noqa: PLC0415
            MLP,
            MainBlock,
            SmoothBesselBasis,
            SphericalBasisLayer,
        )
    except ImportError:
        _load_forcefield_namespace()
        try:
            from mattersim.forcefield.m3gnet.modules import (  # noqa: PLC0415
                MLP,
                MainBlock,
                SmoothBesselBasis,
                SphericalBasisLayer,
            )
        except ImportError as exc:
            msg = (
                "MatterSim descriptor requires the optional extra: "
                'pip install "deepmd-gnn[mattersim]"'
            )
            raise ImportError(msg) from exc
    return MLP, MainBlock, SmoothBesselBasis, SphericalBasisLayer


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    return torch.float32


def _infer_state_dtype(state_dict: dict[str, Any]) -> torch.dtype:
    for value in state_dict.values():
        if torch.is_tensor(value) and value.is_floating_point():
            return value.dtype
    return torch.float32


def persistable_checkpoint_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return JSON-safe metadata needed to rebuild a MatterSim feature backbone."""
    source_dtype = str(config.get("source_dtype") or "float32")
    if source_dtype not in {"float32", "float64"}:
        source_dtype = "float32"
    return {
        "type_map": list(config.get("type_map") or []),
        "num_blocks": int(config["num_blocks"]),
        "units": int(config["units"]),
        "max_l": int(config["max_l"]),
        "max_n": int(config["max_n"]),
        "cutoff": float(config["cutoff"]),
        "max_z": int(config["max_z"]),
        "threebody_cutoff": float(config["threebody_cutoff"]),
        "source_dtype": source_dtype,
    }


def reject_unsupported_mattersim(payload: dict[str, Any]) -> None:
    """Reject Graphormer and other non-M3GNet MatterSim checkpoints."""
    model_name = str(payload.get("model_name") or "m3gnet").lower()
    if model_name != "m3gnet":
        msg = (
            "Only MatterSim-v1 M3GNet checkpoints are supported as property "
            f"descriptors, got model_name={model_name!r}"
        )
        raise ValueError(msg)


def resolve_mattersim_checkpoint_path(model_path: str | Path) -> Path | None:
    """Resolve a local file or MatterSim-v1 pretrained keyword."""
    path = Path(model_path)
    if path.is_file():
        return path
    filename = PRETRAINED_FILENAMES.get(str(model_path).lower())
    if filename is None:
        return None
    local = _PRETRAINED_DIR / filename
    if local.is_file():
        return local
    try:
        from mattersim.utils.download_utils import (  # noqa: PLC0415
            download_checkpoint,
        )
    except ImportError:
        return None
    _PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    download_checkpoint(filename, save_folder=str(_PRETRAINED_DIR))
    return local if local.is_file() else None


def _load_raw_mattersim_payload(model_path: str | Path) -> dict[str, Any]:
    """Read a trusted MatterSim pickle before reconstructing the backbone."""
    resolved = resolve_mattersim_checkpoint_path(model_path)
    if resolved is None:
        msg = f"MatterSim checkpoint not found: {model_path}"
        raise FileNotFoundError(msg)
    payload = torch.load(str(resolved), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        msg = (
            "MatterSim checkpoint must be a dictionary containing model_args "
            f"and model weights, got {type(payload)!r}"
        )
        raise TypeError(msg)
    reject_unsupported_mattersim(payload)
    if "model_args" not in payload or "model" not in payload:
        msg = "MatterSim checkpoint must contain 'model_args' and 'model'"
        raise TypeError(msg)
    return payload


def _config_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    model_args = dict(payload["model_args"])
    missing = [key for key in _MODEL_ARG_KEYS if key not in model_args]
    if missing:
        msg = f"MatterSim checkpoint model_args missing keys: {missing}"
        raise ValueError(msg)
    state = payload["model"]
    if not isinstance(state, dict):
        msg = "MatterSim checkpoint 'model' must be a state_dict"
        raise TypeError(msg)
    dtype = _infer_state_dtype(state)
    return persistable_checkpoint_config(
        {
            **{key: model_args[key] for key in _MODEL_ARG_KEYS},
            "source_dtype": "float64" if dtype == torch.float64 else "float32",
        },
    )


def load_mattersim_checkpoint_config(model_path: str | Path) -> dict[str, Any]:
    """Load architecture metadata from a trusted MatterSim M3GNet checkpoint."""
    return _config_from_payload(_load_raw_mattersim_payload(model_path))


def _init_linear(module: torch.nn.Module) -> None:
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)


def _init_embedding(module: torch.nn.Module) -> None:
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.uniform_(module.weight, a=-0.05, b=0.05)


class MatterSimFeatureBackbone(torch.nn.Module):
    """M3GNet embedding and MainBlocks without the energy readout."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        mlp, main_block, smooth_bessel, spherical_basis = _import_m3gnet_modules()
        persistable = persistable_checkpoint_config(config)
        self.num_blocks = int(persistable["num_blocks"])
        if self.num_blocks < 1 or self.num_blocks > MAX_GRAPH_CONV_BLOCKS:
            msg = (
                "MatterSim descriptor supports 1 to "
                f"{MAX_GRAPH_CONV_BLOCKS} M3GNet blocks, got {self.num_blocks}"
            )
            raise ValueError(msg)
        self.units = int(persistable["units"])
        self.max_l = int(persistable["max_l"])
        self.max_n = int(persistable["max_n"])
        self.cutoff = float(persistable["cutoff"])
        self.max_z = int(persistable["max_z"])
        self.threebody_cutoff = float(persistable["threebody_cutoff"])
        self.rbf = smooth_bessel(r_max=self.cutoff, max_n=self.max_n)
        self.sbf = spherical_basis(
            max_n=self.max_n,
            max_l=self.max_l,
            cutoff=self.cutoff,
        )
        self.edge_encoder = mlp(
            in_dim=self.max_n,
            out_dims=[self.units],
            activation="swish",
            use_bias=False,
        )
        self.graph_conv = torch.nn.ModuleList(
            [
                main_block(
                    self.max_n,
                    self.max_l,
                    self.cutoff,
                    self.units,
                    self.max_n,
                    self.threebody_cutoff,
                )
                for _ in range(self.num_blocks)
            ],
        )
        self.atom_embedding = mlp(
            in_dim=self.max_z + 1,
            out_dims=[self.units],
            activation=None,
            use_bias=False,
        )
        self.apply(_init_linear)
        self.atom_embedding.apply(_init_embedding)

    def forward(
        self,
        atom_pos: torch.Tensor,
        cell: torch.Tensor,
        pbc_offsets: torch.Tensor,
        atom_attr: torch.Tensor,
        edge_index: torch.Tensor,
        three_body_indices: torch.Tensor,
        num_bonds: torch.Tensor,
        num_triple_ij: torch.Tensor,
        num_atoms: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Return last-layer invariant atom features."""
        pos = atom_pos
        offsets = pbc_offsets.to(pos.dtype)
        edges = edge_index.long()
        triples = three_body_indices.long()
        edge_batch = batch[edges[0]]
        shift = torch.bmm(
            offsets.unsqueeze(1),
            cell.index_select(0, edge_batch),
        ).squeeze(1)
        edge_vector = pos[edges[0]] - (pos[edges[1]] + shift)
        sq = (edge_vector * edge_vector).sum(dim=1)
        edge_length = torch.sqrt(sq)
        triple_i = triples[:, 0]
        triple_j = triples[:, 1]
        vij = edge_vector[triple_i]
        vik = edge_vector[triple_j]
        rij = edge_length[triple_i]
        rik = edge_length[triple_j]
        cos_jik = (vij * vik).sum(dim=1) / (rij * rik)
        cos_jik = torch.clamp(cos_jik, -1.0 + 1e-7, 1.0 - 1e-7)
        triple_edge_length = rik.view(-1)
        edge_length = edge_length.unsqueeze(-1)
        atomic_numbers = atom_attr.squeeze(1).long()
        n_classes = self.max_z + 1
        species = torch.nn.functional.one_hot(atomic_numbers, n_classes)
        features = self.atom_embedding(species.to(pos.dtype))
        edge_attr = self.rbf(edge_length.view(-1))
        edge_attr_zero = edge_attr
        edge_attr = self.edge_encoder(edge_attr)
        three_basis = self.sbf(triple_edge_length, torch.acos(cos_jik))
        for conv in self.graph_conv:
            features, edge_attr = conv(
                features,
                edge_attr,
                edge_attr_zero,
                edges,
                three_basis,
                triples,
                edge_length,
                num_bonds,
                num_triple_ij,
                num_atoms,
            )
        return features


def build_mattersim_feature_backbone(
    config: dict[str, Any],
    *,
    device: str | torch.device,
) -> MatterSimFeatureBackbone:
    """Rebuild a stripped MatterSim backbone from persisted constructor metadata."""
    persistable = persistable_checkpoint_config(config)
    dtype = _dtype_from_name(persistable["source_dtype"])
    return MatterSimFeatureBackbone(persistable).to(device=device, dtype=dtype)


def load_native_mattersim_feature_backbone(
    model_path: str | Path,
    *,
    device: str | torch.device,
) -> tuple[MatterSimFeatureBackbone, dict[str, Any]]:
    """Load a trusted MatterSim M3GNet checkpoint and drop the energy readout."""
    payload = _load_raw_mattersim_payload(model_path)
    persistable = _config_from_payload(payload)
    backbone = build_mattersim_feature_backbone(persistable, device=device)
    validate_mattersim_state_dict_load(
        backbone.load_state_dict(payload["model"], strict=False),
        allow_energy_keys=True,
    )
    return backbone, persistable


def validate_mattersim_state_dict_load(
    load_result: object,
    *,
    allow_energy_keys: bool = False,
) -> None:
    """Reject missing backbone weights; optionally ignore unused energy heads."""
    missing_keys = list(getattr(load_result, "missing_keys", []))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
    if allow_energy_keys:
        unexpected_keys = [
            key for key in unexpected_keys if not key.startswith(ENERGY_STATE_PREFIXES)
        ]
    if missing_keys or unexpected_keys:
        msg = (
            "Failed to load MatterSim checkpoint into DeePMD-GNN wrapper. "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )
        raise RuntimeError(msg)


def compute_threebody_torch(
    edge_indices: torch.Tensor,
    n_atoms: torch.Tensor,
    n_local_total: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build three-body edge pairs from a center-sorted pair graph.

    ``edge_indices`` has shape ``[n_edges, 2]`` and must be sorted by the
    central atom in column 0. ``n_local_total`` is the concatenated local-atom
    count so this helper stays TorchScript-friendly.
    """
    n_structures = n_atoms.shape[0]
    n_bond_per_atom = torch.bincount(edge_indices[:, 0], minlength=n_local_total)
    n_triple_i = n_bond_per_atom * (n_bond_per_atom - 1)
    if edge_indices.size(0) == 0:
        n_triple_ij = torch.zeros(
            (0,),
            dtype=torch.long,
            device=edge_indices.device,
        )
    else:
        n_triple_ij = n_bond_per_atom[edge_indices[:, 0]] - 1
    valid_mask = n_bond_per_atom >= 2
    valid_counts = n_bond_per_atom[valid_mask]
    valid_starts = (torch.cumsum(n_bond_per_atom, dim=0) - n_bond_per_atom)[valid_mask]
    if valid_counts.size(0) == 0:
        triple_bond_indices = torch.zeros(
            (0, 2),
            dtype=torch.long,
            device=edge_indices.device,
        )
    else:
        n_triple_per_atom = valid_counts * (valid_counts - 1)
        group_ids = torch.repeat_interleave(
            torch.arange(valid_counts.size(0), device=edge_indices.device),
            n_triple_per_atom,
        )
        cum_triples = torch.cumsum(n_triple_per_atom, dim=0)
        starts_triples = cum_triples - n_triple_per_atom
        local_idx = (
            torch.arange(group_ids.size(0), device=edge_indices.device)
            - starts_triples[group_ids]
        )
        k_vec = valid_counts[group_ids]
        k_minus_1 = k_vec - 1
        u = local_idx // k_minus_1
        v_tmp = local_idx % k_minus_1
        v = v_tmp + (v_tmp >= u).long()
        group_start_indices = valid_starts[group_ids]
        triple_bond_indices = torch.stack(
            [group_start_indices + u, group_start_indices + v],
            dim=1,
        )

    atom_to_structure = torch.repeat_interleave(
        torch.arange(n_structures, device=edge_indices.device),
        n_atoms,
    )
    n_triple_s = torch.zeros(
        n_structures,
        dtype=n_triple_i.dtype,
        device=edge_indices.device,
    )
    n_triple_s.scatter_add_(0, atom_to_structure, n_triple_i)
    return triple_bond_indices, n_triple_ij, n_triple_i, n_triple_s


def compute_threebody_indices(
    edge_index: torch.Tensor,
    distances: torch.Tensor,
    num_atoms: torch.Tensor,
    n_local_total: int,
    threebody_cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter pair edges by the three-body cutoff and return global triples."""
    n_edges = edge_index.size(1)
    if n_edges == 0:
        empty_triples = torch.zeros(
            (0, 2),
            dtype=torch.long,
            device=edge_index.device,
        )
        empty_counts = torch.zeros(
            (0, 1),
            dtype=torch.long,
            device=edge_index.device,
        )
        return empty_triples, empty_counts

    valid_three_body = distances <= threebody_cutoff
    original_index = torch.arange(n_edges, device=edge_index.device)[valid_three_body]
    valid_edge_indices = edge_index[:, valid_three_body].transpose(0, 1)
    if valid_edge_indices.size(0) == 0:
        n_triple_ij = torch.zeros(
            (n_edges, 1),
            dtype=torch.long,
            device=edge_index.device,
        )
        empty_triples = torch.zeros(
            (0, 2),
            dtype=torch.long,
            device=edge_index.device,
        )
        return empty_triples, n_triple_ij

    angle_indices, num_angles_per_edge, _, _ = compute_threebody_torch(
        valid_edge_indices,
        num_atoms,
        n_local_total,
    )
    n_triple_ij = torch.zeros((n_edges,), dtype=torch.long, device=edge_index.device)
    n_triple_ij[original_index] = num_angles_per_edge
    if angle_indices.size(0) > 0:
        angle_indices = original_index[angle_indices]
    return angle_indices, n_triple_ij.unsqueeze(-1)
