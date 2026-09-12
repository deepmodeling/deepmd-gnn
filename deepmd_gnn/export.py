# SPDX-License-Identifier: LGPL-3.0-or-later
"""Helpers used by PyTorch-exportable model paths."""

import torch


def pad_nlist_for_export(nlist: torch.Tensor) -> torch.Tensor:
    """Append a sentinel neighbor so symbolic export keeps neighbor axes dynamic."""
    pad = -torch.ones(
        (*nlist.shape[:2], 1),
        dtype=nlist.dtype,
        device=nlist.device,
    )
    return torch.cat([nlist, pad], dim=-1)


def clear_export_guards_once(traced: torch.nn.Module) -> None:
    """Clear over-specialized guards from the next export of ``traced``.

    ``make_fx`` traces may specialize symbolic atom counts too aggressively.
    DeePMD's export path calls ``torch.export.export`` immediately after tracing,
    so this one-shot wrapper relaxes those constraints only for that export.
    """
    original_export = torch.export.export

    def strip_deferred_assertions(exported: torch.export.ExportedProgram) -> None:
        graph = exported.graph_module.graph
        for node in list(graph.nodes):
            if (
                node.op == "call_function"
                and node.target is torch.ops.aten._assert_scalar.default  # noqa: SLF001
            ):
                node.args = (True, node.args[1])
        exported.graph_module.recompile()

    def relax_range_constraints(exported: torch.export.ExportedProgram) -> None:
        relaxed = exported.range_constraints.copy()
        for symbol, value_range in exported.range_constraints.items():
            try:
                should_relax = bool(value_range.lower > 1)
            except TypeError:
                should_relax = False
            if should_relax:
                relaxed[symbol] = type(value_range)(1, value_range.upper)
        exported._range_constraints = relaxed  # noqa: SLF001

    def export_with_guard_cleanup(
        *export_args: object,
        **export_kwargs: object,
    ) -> torch.export.ExportedProgram:
        try:
            exported = original_export(*export_args, **export_kwargs)
            if export_args and export_args[0] is traced:
                exported._guards_code = []  # noqa: SLF001
                strip_deferred_assertions(exported)
                relax_range_constraints(exported)
            return exported
        finally:
            if torch.export.export is export_with_guard_cleanup:
                torch.export.export = original_export

    torch.export.export = export_with_guard_cleanup
