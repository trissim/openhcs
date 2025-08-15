"""
Unit tests for PyQt6 widget creation functionality.

This module tests the widget creation capabilities of the ParameterFormManager,
focusing on different widget types and their proper configuration.
"""

import pytest
from PyQt6.QtWidgets import QLineEdit, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QPushButton
from PyQt6.QtCore import Qt

from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager
from openhcs.core.config import Microscope, ZarrCompressor
from pathlib import Path
from typing import Optional
from enum import Enum


class TestWidgetCreationBasicTypes:
    """Test widget creation for basic Python types."""

    def test_string_widget_creation(self, qapp):
        """Test that string parameters create QLineEdit widgets."""
        parameters = {"string_field": "test_value"}
        parameter_types = {"string_field": str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="string_test"
        )
        
        assert "string_field" in manager.widgets
        widget = manager.widgets["string_field"]
        assert isinstance(widget, QLineEdit)
        
        # Test that the widget has the correct value
        if hasattr(widget, 'text'):
            assert widget.text() == "test_value"

    def test_integer_widget_creation(self, qapp):
        """Test that integer parameters create appropriate widgets."""
        parameters = {"int_field": 42}
        parameter_types = {"int_field": int}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="int_test"
        )
        
        assert "int_field" in manager.widgets
        widget = manager.widgets["int_field"]
        # Could be QSpinBox or QLineEdit depending on implementation
        assert isinstance(widget, (QSpinBox, QLineEdit))

    def test_float_widget_creation(self, qapp):
        """Test that float parameters create appropriate widgets."""
        parameters = {"float_field": 3.14}
        parameter_types = {"float_field": float}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="float_test"
        )
        
        assert "float_field" in manager.widgets
        widget = manager.widgets["float_field"]
        # Could be QDoubleSpinBox or QLineEdit depending on implementation
        assert isinstance(widget, (QDoubleSpinBox, QLineEdit))

    def test_boolean_widget_creation(self, qapp):
        """Test that boolean parameters create QCheckBox widgets."""
        parameters = {"bool_field": True}
        parameter_types = {"bool_field": bool}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="bool_test"
        )
        
        assert "bool_field" in manager.widgets
        widget = manager.widgets["bool_field"]
        assert isinstance(widget, QCheckBox)
        assert widget.isChecked() == True

    def test_path_widget_creation(self, qapp):
        """Test that Path parameters create appropriate widgets."""
        test_path = Path("/test/path")
        parameters = {"path_field": test_path}
        parameter_types = {"path_field": Path}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="path_test"
        )
        
        assert "path_field" in manager.widgets
        widget = manager.widgets["path_field"]
        # Path widgets might be custom or QLineEdit
        assert widget is not None


class TestWidgetCreationEnumTypes:
    """Test widget creation for enum types."""

    def test_microscope_enum_widget(self, qapp):
        """Test that Microscope enum creates QComboBox widget."""
        parameters = {"microscope": Microscope.IMAGEXPRESS}
        parameter_types = {"microscope": Microscope}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="microscope_test"
        )
        
        assert "microscope" in manager.widgets
        widget = manager.widgets["microscope"]
        assert isinstance(widget, QComboBox)
        
        # Check that enum values are available
        assert widget.count() > 0

    def test_zarr_compressor_enum_widget(self, qapp):
        """Test that ZarrCompressor enum creates QComboBox widget."""
        parameters = {"compressor": ZarrCompressor.ZSTD}
        parameter_types = {"compressor": ZarrCompressor}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="compressor_test"
        )
        
        assert "compressor" in manager.widgets
        widget = manager.widgets["compressor"]
        assert isinstance(widget, QComboBox)

    def test_custom_enum_widget(self, qapp):
        """Test widget creation for custom enum types."""
        class CustomEnum(Enum):
            OPTION1 = "option1"
            OPTION2 = "option2"
            OPTION3 = "option3"
        
        parameters = {"custom_enum": CustomEnum.OPTION1}
        parameter_types = {"custom_enum": CustomEnum}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="custom_enum_test"
        )
        
        assert "custom_enum" in manager.widgets
        widget = manager.widgets["custom_enum"]
        assert isinstance(widget, QComboBox)
        assert widget.count() == 3  # Should have all enum options


class TestWidgetCreationOptionalTypes:
    """Test widget creation for Optional types."""

    def test_optional_string_widget(self, qapp):
        """Test that Optional[str] creates appropriate widget."""
        parameters = {"optional_string": None}
        parameter_types = {"optional_string": Optional[str]}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="optional_string_test"
        )
        
        # Should still create a widget, even for None values
        assert "optional_string" in manager.widgets
        widget = manager.widgets["optional_string"]
        assert isinstance(widget, QLineEdit)

    def test_optional_enum_widget(self, qapp):
        """Test that Optional[Enum] creates appropriate widget."""
        parameters = {"optional_microscope": None}
        parameter_types = {"optional_microscope": Optional[Microscope]}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="optional_enum_test"
        )
        
        assert "optional_microscope" in manager.widgets
        widget = manager.widgets["optional_microscope"]
        assert isinstance(widget, QComboBox)


class TestWidgetCreationSpecialCases:
    """Test widget creation for special cases and edge conditions."""

    def test_none_value_widget_creation(self, qapp):
        """Test widget creation when parameter value is None."""
        parameters = {"none_param": None}
        parameter_types = {"none_param": str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="none_test"
        )
        
        assert "none_param" in manager.widgets
        widget = manager.widgets["none_param"]
        assert isinstance(widget, QLineEdit)
        # Widget should be empty for None values
        if hasattr(widget, 'text'):
            assert widget.text() == ""

    def test_empty_string_widget_creation(self, qapp):
        """Test widget creation with empty string values."""
        parameters = {"empty_string": ""}
        parameter_types = {"empty_string": str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="empty_test"
        )
        
        assert "empty_string" in manager.widgets
        widget = manager.widgets["empty_string"]
        assert isinstance(widget, QLineEdit)
        if hasattr(widget, 'text'):
            assert widget.text() == ""

    def test_widget_object_names(self, qapp):
        """Test that widgets have proper object names for identification."""
        parameters = {"test_param": "value"}
        parameter_types = {"test_param": str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="object_name_test"
        )
        
        widget = manager.widgets["test_param"]
        object_name = widget.objectName()
        
        # Object name should be set and contain field information
        assert object_name != ""
        assert "test_param" in object_name or "object_name_test" in object_name

    def test_widget_signal_connections(self, qapp):
        """Test that widgets have proper signal connections."""
        parameters = {"signal_test": "value"}
        parameter_types = {"signal_test": str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="signal_test"
        )
        
        widget = manager.widgets["signal_test"]
        
        # Widget should have change signals connected
        # This is implementation-dependent, but we can check basic connectivity
        if isinstance(widget, QLineEdit):
            # QLineEdit should have textChanged signal connected
            assert widget.receivers(widget.textChanged) > 0
        elif isinstance(widget, QCheckBox):
            # QCheckBox should have stateChanged signal connected
            assert widget.receivers(widget.stateChanged) > 0
