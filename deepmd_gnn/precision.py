"""Helpers for constructing model backends at an explicit floating precision."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator


def dtype_from_precision(precision: str) -> torch.dtype:
    """Translate a serialized precision name to a supported PyTorch dtype."""
    if precision == "float32":
        return torch.float32
    if precision == "float64":
        return torch.float64
    msg = f"precision {precision} not supported"
    raise ValueError(msg)


def precision_from_dtype(dtype: torch.dtype) -> str:
    """Translate a supported PyTorch dtype to its serialized precision name."""
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float64:
        return "float64"
    msg = f"dtype {dtype} not supported"
    raise ValueError(msg)


@contextmanager
def temporary_default_dtype(dtype: torch.dtype) -> Iterator[None]:
    """Set PyTorch's construction dtype for one backend build and restore it."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)
