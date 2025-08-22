"""
PyQt6 GUI test configuration and fixtures.

This module provides common fixtures and configuration for PyQt6 GUI tests.
"""

import pytest
import sys


@pytest.fixture
def sample_parameters():
    """Provide sample parameters for testing parameter forms."""
    return {
        "string_param": "test_value",
        "int_param": 42,
        "float_param": 3.14,
        "bool_param": True,
        "optional_param": None
    }


@pytest.fixture
def sample_parameter_types():
    """Provide sample parameter types for testing parameter forms."""
    from typing import Optional
    return {
        "string_param": str,
        "int_param": int,
        "float_param": float,
        "bool_param": bool,
        "optional_param": Optional[str]
    }


@pytest.fixture
def test_dataclass():
    """Provide a test dataclass for nested form testing."""
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class TestNestedConfig:
        nested_field1: str = "default1"
        nested_field2: int = 10
        nested_field3: Optional[str] = None

    @dataclass
    class TestConfig:
        simple_field: str = "simple_default"
        nested_config: Optional[TestNestedConfig] = None
        required_nested: TestNestedConfig = None

    return TestConfig


@pytest.fixture
def mock_color_scheme():
    """Provide a mock color scheme for testing."""
    import os

    # Skip PyQt6 imports in CPU-only mode
    if os.getenv('OPENHCS_CPU_ONLY', 'false').lower() == 'true':
        pytest.skip("PyQt6 GUI tests skipped in CPU-only mode")

    from openhcs.pyqt_gui.shared.color_scheme import PyQt6ColorScheme
    return PyQt6ColorScheme()
