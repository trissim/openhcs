"""Early pytest option registration for OpenHCS integration parametrization.

Options listed in ``pytest.ini`` ``addopts`` must be registered before pytest
fully parses command-line arguments. Keeping these hooks in root ``conftest.py``
is too late/fragile for that path, so this plugin is loaded explicitly with
``-p tests.pytest_integration_options``.
"""

import os

import pytest


def pytest_addoption(parser):
    """Add command-line options for integration test configuration."""

    def env_default(env_var, default_value):
        return os.getenv(env_var, default_value)

    parser.addoption(
        "--it-backends",
        action="store",
        default=env_default("IT_BACKENDS", "disk,zarr"),
        help="Comma-separated list of backends to test. Use 'all' for full coverage.",
    )

    parser.addoption(
        "--it-microscopes",
        action="store",
        default=env_default("IT_MICROSCOPES", "ImageXpress,OperaPhenix,OpenHCS"),
        help="Comma-separated list of microscopes to test. Use 'all' for full coverage.",
    )

    parser.addoption(
        "--it-dims",
        action="store",
        default=env_default("IT_DIMS", "3d"),
        help="Comma-separated list of dimensions to test. Use 'all' for full coverage.",
    )

    parser.addoption(
        "--it-exec-mode",
        action="store",
        default=env_default("IT_EXEC_MODE", "multiprocessing"),
        help="Comma-separated list of execution modes. Use 'all' for full coverage.",
    )

    parser.addoption(
        "--enable-napari",
        action="store_true",
        default=False,
        help="Enable Napari streaming in tests. Deprecated: use --it-visualizers.",
    )

    parser.addoption(
        "--enable-fiji",
        action="store_true",
        default=False,
        help="Enable Fiji streaming in tests. Deprecated: use --it-visualizers.",
    )

    parser.addoption(
        "--it-visualizers",
        action="store",
        default=env_default("IT_VISUALIZERS", "none"),
        help="Comma-separated list of visualizers to enable. Use 'all' for full coverage.",
    )

    parser.addoption(
        "--it-zmq-mode",
        action="store",
        default=env_default("IT_ZMQ_MODE", "direct"),
        help="Comma-separated list of ZMQ execution modes. Use 'all' for full coverage.",
    )

    parser.addoption(
        "--it-processing-axis",
        action="store",
        default=env_default("IT_PROCESSING_AXIS", "well"),
        help="Comma-separated list of processing axis components. Use 'all' for full coverage.",
    )

    parser.addoption(
        "--it-sequential",
        action="store",
        default=env_default("IT_SEQUENTIAL", "none"),
        help="Comma-separated list of sequential processing configurations. Use 'all' for full coverage.",
    )


def pytest_configure(config):
    """Validate integration configuration options."""

    valid_choices = {
        "backends": ["disk", "zarr"],
        "microscopes": ["ImageXpress", "OperaPhenix", "OpenHCS", "OMERO"],
        "dims": ["2d", "3d"],
        "exec_modes": ["threading", "multiprocessing"],
        "zmq_modes": ["direct", "zmq"],
        "processing_axis": ["well"],
        "sequential": [
            "none",
            "valid_1_component",
            "valid_2_components",
            "invalid_overlap",
            "invalid_duplicates",
        ],
    }

    options_to_validate = [
        ("--it-backends", "backends"),
        ("--it-microscopes", "microscopes"),
        ("--it-dims", "dims"),
        ("--it-exec-mode", "exec_modes"),
        ("--it-zmq-mode", "zmq_modes"),
        ("--it-processing-axis", "processing_axis"),
        ("--it-sequential", "sequential"),
    ]

    for option_name, choice_key in options_to_validate:
        option_value = config.getoption(option_name)
        if option_value == "all":
            continue

        selected_values = [v.strip() for v in option_value.split(",")]
        valid_values = valid_choices[choice_key]

        for value in selected_values:
            if value not in valid_values:
                raise pytest.UsageError(
                    f"Invalid value '{value}' for {option_name}. "
                    f"Valid choices: {', '.join(valid_values)} or 'all'"
                )
