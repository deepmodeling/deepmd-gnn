# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compatibility accessors for DeePMD observed-type statistics helpers."""

import importlib
from collections.abc import Callable
from typing import cast

ObservedTypeRestore = Callable[[object], list[str] | None]
ObservedTypeSave = Callable[[object, list[str]], None]
ObservedTypeCollect = Callable[[object, list[str]], list[str]]


def load_observed_type_stat_compat() -> tuple[
    ObservedTypeRestore,
    ObservedTypeSave,
    ObservedTypeCollect,
]:
    """Return observed-type statistic helpers across supported DeePMD versions."""
    try:
        stat_mod = importlib.import_module("deepmd.dpmodel.utils.stat")
    except ImportError:

        def collect_observed_types(sampled: object, type_map: list[str]) -> list[str]:
            """Fallback for older deepmd-kit without observed-type helpers."""
            _ = sampled, type_map
            return []

        def restore_observed_type_from_file(
            stat_file_path: object,
        ) -> list[str] | None:
            """Fallback for older deepmd-kit without observed-type helpers."""
            _ = stat_file_path
            return None

        def save_observed_type_to_file(
            stat_file_path: object,
            observed_type: list[str],
        ) -> None:
            """Fallback for older deepmd-kit without observed-type helpers."""
            _ = stat_file_path, observed_type

        return (
            restore_observed_type_from_file,
            save_observed_type_to_file,
            collect_observed_types,
        )
    else:
        restore = cast(
            "ObservedTypeRestore",
            stat_mod._restore_observed_type_from_file,  # noqa: SLF001
        )
        save = cast(
            "ObservedTypeSave",
            stat_mod._save_observed_type_to_file,  # noqa: SLF001
        )
        collect = cast("ObservedTypeCollect", stat_mod.collect_observed_types)
        return (restore, save, collect)
