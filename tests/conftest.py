"""Global pytest configuration for OpenHCS integration tests."""
import os

from openhcs._source_dependencies import ensure_source_checkout_external_paths

ensure_source_checkout_external_paths()

import pytest

# Conditionally import pytest-qt only when not in CPU-only mode
CPU_ONLY_MODE = os.getenv('OPENHCS_CPU_ONLY', 'false').lower() == 'true'
if not CPU_ONLY_MODE:
    pytest_plugins = ["pytestqt"]
else:
    pytest_plugins = []


# Visualizer configurations for parametrized testing
VISUALIZER_CONFIGS = {
    "none": {"enable_napari": False, "enable_fiji": False},
    "napari": {"enable_napari": True, "enable_fiji": False},
    "fiji": {"enable_napari": False, "enable_fiji": True},
    "napari+fiji": {"enable_napari": True, "enable_fiji": True}
}


def _build_integration_test_config():
    """Load integration parametrization data only when integration fixtures are used."""
    from tests.integration.helpers.fixture_utils import (
        BACKEND_CONFIGS,
        DATA_TYPE_CONFIGS,
        EXECUTION_MODE_CONFIGS,
        MICROSCOPE_CONFIGS,
        SEQUENTIAL_CONFIGS,
        ZMQ_EXECUTION_MODE_CONFIGS,
    )

    return {
        'backend_config': {
            'option': '--it-backends',
            'choices': BACKEND_CONFIGS,
            'value_mapper': lambda x: x  # Return backend name as-is
        },
        'microscope_config': {
            'option': '--it-microscopes',
            'choices': list(MICROSCOPE_CONFIGS.keys()),
            'value_mapper': lambda name: MICROSCOPE_CONFIGS[name]  # Map name to config dict
        },
        'data_type_config': {
            'option': '--it-dims',
            'choices': list(DATA_TYPE_CONFIGS.keys()),
            'value_mapper': lambda dim: DATA_TYPE_CONFIGS[dim]  # Map dim to config dict
        },
        'execution_mode': {
            'option': '--it-exec-mode',
            'choices': EXECUTION_MODE_CONFIGS,
            'value_mapper': lambda x: x  # Return mode name as-is
        },
        'zmq_execution_mode': {
            'option': '--it-zmq-mode',
            'choices': ZMQ_EXECUTION_MODE_CONFIGS,
            'value_mapper': lambda x: x  # Return mode name as-is
        },
        'processing_axis': {
            'option': '--it-processing-axis',
            'choices': ['well'],
            'value_mapper': lambda x: x  # Return axis name as-is
        },
        'visualizer_config': {
            'option': '--it-visualizers',
            'choices': list(VISUALIZER_CONFIGS.keys()),
            'value_mapper': lambda name: VISUALIZER_CONFIGS[name]  # Map name to config dict
        },
        'sequential_config': {
            'option': '--it-sequential',
            'choices': list(SEQUENTIAL_CONFIGS.keys()),
            'value_mapper': lambda name: SEQUENTIAL_CONFIGS[name]  # Map name to config dict
        }
    }


def _get_config_option(config, option_name, all_choices):
    """Get filtered parameter list based on pytest configuration option."""
    option_value = config.getoption(option_name)
    
    if option_value == "all":
        return all_choices
        
    selected = [v.strip() for v in option_value.split(",")]
    # Filter to only include valid choices that were selected
    return [choice for choice in all_choices if choice in selected]


def pytest_generate_tests(metafunc):
    """Generate test parameters based on configuration options - fully extensible."""
    integration_fixture_names = {
        "backend_config",
        "microscope_config",
        "data_type_config",
        "execution_mode",
        "zmq_execution_mode",
        "processing_axis",
        "visualizer_config",
        "sequential_config",
    }
    if not integration_fixture_names.intersection(metafunc.fixturenames):
        return

    for fixture_name, config in _build_integration_test_config().items():
        if fixture_name in metafunc.fixturenames:
            selected_choices = _get_config_option(metafunc.config, config['option'], config['choices'])
            values = [config['value_mapper'](choice) for choice in selected_choices]
            metafunc.parametrize(fixture_name, values, ids=selected_choices, scope="module")


@pytest.fixture
def enable_napari(request, visualizer_config):
    """
    Fixture to control Napari streaming in tests.

    Supports both legacy --enable-napari flag and new --it-visualizers parametrization.
    """
    # Check legacy flag first (for backward compatibility)
    legacy_flag = request.config.getoption("--enable-napari")
    if legacy_flag:
        return True

    # Use parametrized visualizer_config
    return visualizer_config.get("enable_napari", False)


@pytest.fixture
def enable_fiji(request, visualizer_config):
    """
    Fixture to control Fiji streaming in tests.

    Supports both legacy --enable-fiji flag and new --it-visualizers parametrization.
    """
    # Check legacy flag first (for backward compatibility)
    legacy_flag = request.config.getoption("--enable-fiji")
    if legacy_flag:
        return True

    # Use parametrized visualizer_config
    return visualizer_config.get("enable_fiji", False)


@pytest.fixture(autouse=True)
def cleanup_backend_connections():
    """
    Automatically clean up backend connections after each test.

    This preserves the napari window for future tests while cleaning up
    ZeroMQ connections and shared memory to prevent test hanging.
    """
    yield  # Run the test

    # Clean up connections after test completes
    try:
        from polystore import cleanup_backend_connections
        cleanup_backend_connections()
    except Exception as e:
        # Don't fail tests due to cleanup issues
        print(f"Warning: Failed to cleanup backend connections: {e}")
