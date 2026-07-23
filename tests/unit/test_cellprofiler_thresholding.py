"""Focused contracts for CellProfiler threshold parameter declarations."""

from enum import Enum
from typing import get_args, get_type_hints

import pytest

from openhcs.processing.backends.cellprofiler.thresholding import (
    GlobalThresholdKeywordArguments,
    GlobalThresholdMethodParameters,
)


def _declared_parameter_value(annotation: object) -> object:
    for candidate in (annotation, *get_args(annotation)):
        if candidate is float:
            return 0.25
        if candidate is int:
            return 8
        if isinstance(candidate, type) and issubclass(candidate, Enum):
            return next(iter(candidate))
    raise AssertionError(f"No threshold test value for annotation {annotation!r}.")


@pytest.mark.parametrize(
    "parameter_name, annotation",
    tuple(get_type_hints(GlobalThresholdKeywordArguments).items()),
)
def test_global_threshold_parameter_acceptance_comes_from_typed_dict(
    parameter_name: str,
    annotation: object,
) -> None:
    GlobalThresholdMethodParameters.from_kwargs(
        **{parameter_name: _declared_parameter_value(annotation)}
    )


def test_global_threshold_parameters_reject_undeclared_names() -> None:
    with pytest.raises(
        TypeError,
        match="Unknown CellProfiler global threshold parameter.*undeclared_parameter",
    ):
        GlobalThresholdMethodParameters.from_kwargs(undeclared_parameter=1)
