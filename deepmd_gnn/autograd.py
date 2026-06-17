"""Autograd helpers shared by GNN model wrappers."""

from __future__ import annotations

from typing import Optional

import torch


@torch.jit.script
def derive_atomic_virial_from_displacement(
    atom_energy: torch.Tensor,
    displacement: torch.Tensor,
    nloc: int,
    create_graph: bool,
) -> torch.Tensor:
    """Derive per-atom virials from atom-energy gradients w.r.t. cell strain."""
    nf = atom_energy.shape[0]
    atomic_virial = torch.zeros(
        (nf, nloc, 9),
        dtype=displacement.dtype,
        device=displacement.device,
    )
    for ii in range(nloc):
        atom_energy_ii = atom_energy[:, ii]
        atom_grad_outputs = torch.jit.annotate(
            list[Optional[torch.Tensor]],
            [torch.ones_like(atom_energy_ii)],
        )
        atom_virial_ii = torch.autograd.grad(
            outputs=[atom_energy_ii],
            inputs=[displacement],
            grad_outputs=atom_grad_outputs,
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,
        )[0]
        if atom_virial_ii is None:
            atom_virial_ii = torch.zeros_like(displacement)
        atomic_virial[:, ii, :] = (-atom_virial_ii).view(nf, 9)
    return atomic_virial
