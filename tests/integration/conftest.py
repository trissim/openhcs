"""Pytest configuration owned by the integration-test subtree."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest

from tests.integration.helpers.fixture_utils import (
    BACKEND_CONFIGS,
    DATA_TYPE_CONFIGS,
    EXECUTION_MODE_CONFIGS,
    MICROSCOPE_CONFIGS,
    SEQUENTIAL_CONFIGS,
    ZMQ_EXECUTION_MODE_CONFIGS,
)


VISUALIZER_CONFIGS = {
    "none": {"enable_napari": False, "enable_fiji": False},
    "napari": {"enable_napari": True, "enable_fiji": False},
    "fiji": {"enable_napari": False, "enable_fiji": True},
    "napari+fiji": {"enable_napari": True, "enable_fiji": True},
}


def _integration_test_config() -> dict[
    str,
    tuple[str, Sequence[str], Callable[[str], Any]],
]:
    return {
        "backend_config": ("--it-backends", BACKEND_CONFIGS, lambda value: value),
        "microscope_config": (
            "--it-microscopes",
            tuple(MICROSCOPE_CONFIGS),
            MICROSCOPE_CONFIGS.__getitem__,
        ),
        "data_type_config": (
            "--it-dims",
            tuple(DATA_TYPE_CONFIGS),
            DATA_TYPE_CONFIGS.__getitem__,
        ),
        "execution_mode": (
            "--it-exec-mode",
            EXECUTION_MODE_CONFIGS,
            lambda value: value,
        ),
        "zmq_execution_mode": (
            "--it-zmq-mode",
            ZMQ_EXECUTION_MODE_CONFIGS,
            lambda value: value,
        ),
        "processing_axis": (
            "--it-processing-axis",
            ("well",),
            lambda value: value,
        ),
        "visualizer_config": (
            "--it-visualizers",
            tuple(VISUALIZER_CONFIGS),
            VISUALIZER_CONFIGS.__getitem__,
        ),
        "sequential_config": (
            "--it-sequential",
            tuple(SEQUENTIAL_CONFIGS),
            SEQUENTIAL_CONFIGS.__getitem__,
        ),
    }


def _selected_choices(
    config: pytest.Config,
    option_name: str,
    choices: Sequence[str],
) -> tuple[str, ...]:
    option_value = config.getoption(option_name)
    if option_value == "all":
        return tuple(choices)
    selected = frozenset(value.strip() for value in option_value.split(","))
    return tuple(choice for choice in choices if choice in selected)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parameterize only fixtures owned by integration tests."""

    for fixture_name, (option, choices, mapper) in _integration_test_config().items():
        if fixture_name not in metafunc.fixturenames:
            continue
        selected = _selected_choices(metafunc.config, option, choices)
        metafunc.parametrize(
            fixture_name,
            tuple(mapper(choice) for choice in selected),
            ids=selected,
            scope="module",
        )


@pytest.fixture
def enable_napari(request: pytest.FixtureRequest, visualizer_config: dict[str, bool]) -> bool:
    return bool(
        request.config.getoption("--enable-napari")
        or visualizer_config["enable_napari"]
    )


@pytest.fixture
def enable_fiji(request: pytest.FixtureRequest, visualizer_config: dict[str, bool]) -> bool:
    return bool(
        request.config.getoption("--enable-fiji")
        or visualizer_config["enable_fiji"]
    )
