# SPDX-License-Identifier: LGPL-3.0-or-later
"""Small compatibility hooks for DeePMD custom PyTorch ops."""

import torch


def ensure_border_op_placeholder() -> None:
    """Install a TorchScript-visible placeholder for ``deepmd::border_op``.

    DeePMD-kit provides the real op when its PyTorch extension is loaded. During
    unit tests, documentation builds, or partial imports that extension may not
    be present yet, but TorchScript still needs an attribute with the expected
    name so model classes can be scripted.
    """
    if hasattr(torch.ops.deepmd, "border_op"):
        return

    def border_op(
        _argument0: torch.Tensor,
        _argument1: torch.Tensor,
        _argument2: torch.Tensor,
        _argument3: torch.Tensor,
        _argument4: torch.Tensor,
        _argument5: torch.Tensor,
        _argument6: torch.Tensor,
        _argument7: torch.Tensor,
        _argument8: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Runtime placeholder used when DeePMD's PT op is not loaded."""
        msg = (
            "border_op is unavailable. Build/load DeePMD-kit's PyTorch OP "
            "library before running MPI message-passing inference."
        )
        raise NotImplementedError(msg)

    torch.ops.deepmd.border_op = border_op
