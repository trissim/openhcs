"""Global pytest configuration for OpenHCS integration tests."""
import os
import pytest


def pytest_addoption(parser):
    """Add command-line options for integration test configuration."""
    
    # Helper function to get default from environment variable
    def env_default(env_var, default_value):
        return os.getenv(env_var, default_value)
    
    parser.addoption(
        "--it-backends",
        action="store",
        default=env_default("IT_BACKENDS", "disk,zarr"),
        help="Comma-separated list of backends to test (default: disk,zarr). Use 'all' for full coverage."
    )
    
    parser.addoption(
        "--it-microscopes", 
        action="store",
        default=env_default("IT_MICROSCOPES", "ImageXpress,OperaPhenix"),
        help="Comma-separated list of microscopes to test (default: ImageXpress,OperaPhenix). Use 'all' for full coverage."
    )
    
    parser.addoption(
        "--it-dims",
        action="store", 
        default=env_default("IT_DIMS", "3d"),
        help="Comma-separated list of dimensions to test (default: 3d). Options: 2d,3d. Use 'all' for full coverage."
    )
    
    parser.addoption(
        "--it-exec-mode",
        action="store",
        default=env_default("IT_EXEC_MODE", "multiprocessing"),
        help="Comma-separated list of execution modes (default: multiprocessing). Options: threading,multiprocessing. Use 'all' for full coverage."
    )

    parser.addoption(
        "--it-processing-axis",
        action="store",
        default=env_default("IT_PROCESSING_AXIS", "well"),
        help="Comma-separated list of processing axis components (default: well). Options: well. Use 'all' for full coverage."
    )


def pytest_configure(config):
    """Validate configuration options."""
    
    # Define valid choices for each option
    valid_choices = {
        "backends": ["disk", "zarr"],
        "microscopes": ["ImageXpress", "OperaPhenix"],
        "dims": ["2d", "3d"],
        "exec_modes": ["threading", "multiprocessing"],
        "processing_axis": ["well"]
    }
    
    # Validate each option
    options_to_validate = [
        ("--it-backends", "backends"),
        ("--it-microscopes", "microscopes"),
        ("--it-dims", "dims"),
        ("--it-exec-mode", "exec_modes"),
        ("--it-processing-axis", "processing_axis")
    ]
    
    for option_name, choice_key in options_to_validate:
        option_value = config.getoption(option_name)
        if option_value == "all":
            continue  # "all" is always valid
            
        selected_values = [v.strip() for v in option_value.split(",")]
        valid_values = valid_choices[choice_key]
        
        for value in selected_values:
            if value not in valid_values:
                raise pytest.UsageError(
                    f"Invalid value '{value}' for {option_name}. "
                    f"Valid choices: {', '.join(valid_values)} or 'all'"
                )


# Import constants from fixture_utils for parametrization
from tests.integration.helpers.fixture_utils import BACKEND_CONFIGS, MICROSCOPE_CONFIGS, DATA_TYPE_CONFIGS

# Extensible configuration mapping for pytest_generate_tests
INTEGRATION_TEST_CONFIG = {
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
        'choices': ['threading', 'multiprocessing'],
        'value_mapper': lambda x: x  # Return mode name as-is
    },
    'processing_axis': {
        'option': '--it-processing-axis',
        'choices': ['well'],
        'value_mapper': lambda x: x  # Return axis name as-is
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
    for fixture_name, config in INTEGRATION_TEST_CONFIG.items():
        if fixture_name in metafunc.fixturenames:
            selected_choices = _get_config_option(metafunc.config, config['option'], config['choices'])
            values = [config['value_mapper'](choice) for choice in selected_choices]
            metafunc.parametrize(fixture_name, values, ids=selected_choices, scope="module")
