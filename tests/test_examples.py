"""Test examples."""

import json
from pathlib import Path

import pytest
from dargs.check import check
from deepmd.pt.utils.multi_task import preprocess_shared_params
from deepmd.utils.argcheck import gen_args

from deepmd_gnn.argcheck import (  # noqa: F401
    mace_descriptor_args,
    mace_ener_fitting_args,
    mace_model_args,
    mattersim_descriptor_args,
    nequip_descriptor_args,
    sevennet_descriptor_args,
)

example_path = Path(__file__).parent.parent / "examples"

examples = (
    example_path / "water" / "mace" / "input.json",
    example_path / "dprc" / "mace" / "input.json",
    example_path / "water" / "nequip" / "input.json",
    example_path / "dprc" / "nequip" / "input.json",
    example_path / "property" / "mace" / "input.json",
    example_path / "property" / "sevennet" / "input.json",
    example_path / "property" / "mattersim" / "input.json",
)

multitask_examples = (example_path / "property" / "mace" / "input_multitask.json",)


@pytest.mark.parametrize("example", examples)
def test_examples(example: Path) -> None:
    """Check whether examples meet arguments."""
    with example.open("r") as f:
        data = json.load(f)
    check(
        gen_args(),
        data,
    )


@pytest.mark.parametrize("example", multitask_examples)
def test_multitask_examples(example: Path) -> None:
    """Check whether multi-task examples meet arguments after sharing expands."""
    with example.open("r") as f:
        data = json.load(f)
    data["model"], _ = preprocess_shared_params(data["model"])
    check(
        gen_args(multi_task=True),
        data,
    )
