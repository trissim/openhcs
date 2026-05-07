from __future__ import annotations

import pytest

from benchmark.adapters.openhcs import (
    OPENHCS_AXIS_FILTER_PARAM,
    OPENHCS_MAX_AXIS_COUNT_PARAM,
    OpenHCSAxisSelection,
)


def test_openhcs_axis_selection_limits_discovered_axes_in_order() -> None:
    selection = OpenHCSAxisSelection.from_pipeline_params(
        {OPENHCS_MAX_AXIS_COUNT_PARAM: 2}
    )

    assert selection.resolve(("A01", "A02", "A03")) == ("A01", "A02")


def test_openhcs_axis_selection_intersects_explicit_axes_with_discovery_order() -> None:
    selection = OpenHCSAxisSelection.from_pipeline_params(
        {
            OPENHCS_AXIS_FILTER_PARAM: ("A03", "A01"),
            OPENHCS_MAX_AXIS_COUNT_PARAM: 1,
        }
    )

    assert selection.resolve(("A01", "A02", "A03")) == ("A01",)


def test_openhcs_axis_selection_rejects_missing_axes() -> None:
    selection = OpenHCSAxisSelection.from_pipeline_params(
        {OPENHCS_AXIS_FILTER_PARAM: ("A99",)}
    )

    with pytest.raises(ValueError, match="not available"):
        selection.resolve(("A01",))


def test_openhcs_axis_selection_treats_single_string_as_one_axis() -> None:
    selection = OpenHCSAxisSelection.from_pipeline_params(
        {OPENHCS_AXIS_FILTER_PARAM: "B02"}
    )

    assert selection.resolve(("B01", "B02")) == ("B02",)
