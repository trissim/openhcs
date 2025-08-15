"""
Unit tests for the PyQt6 ParameterFormManager.

This module tests the core functionality of the refactored ParameterFormManager,
including widget creation, parameter updates, reset functionality, and signal handling.
"""

import pytest
from unittest.mock import Mock, patch
from PyQt6.QtWidgets import QWidget, QLineEdit, QComboBox, QCheckBox, QPushButton
from PyQt6.QtCore import Qt

from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager
from openhcs.core.config import PathPlanningConfig, MaterializationPathConfig, Microscope
from openhcs.core.pipeline_config import PipelineConfig


class TestParameterFormManagerBasics:
    """Test basic ParameterFormManager functionality."""

    def test_initialization(self, qtbot, sample_parameters, sample_parameter_types):
        """Test that ParameterFormManager initializes correctly."""
        manager = ParameterFormManager(
            parameters=sample_parameters,
            parameter_types=sample_parameter_types,
            field_id="test_form"
        )

        assert manager is not None
        assert isinstance(manager, QWidget)
        assert manager.field_id == "test_form"
        assert manager.parameters == sample_parameters
        assert manager.parameter_types == sample_parameter_types

    def test_widget_creation_basic_types(self, qtbot, sample_parameters, sample_parameter_types):
        """Test widget creation for basic parameter types."""
        manager = ParameterFormManager(
            parameters=sample_parameters,
            parameter_types=sample_parameter_types,
            field_id="test_form"
        )

        # Check that widgets were created
        assert len(manager.widgets) > 0

        # Check specific widget types
        if "string_param" in manager.widgets:
            assert isinstance(manager.widgets["string_param"], QLineEdit)
        if "bool_param" in manager.widgets:
            assert isinstance(manager.widgets["bool_param"], QCheckBox)

    def test_parameter_update(self, qtbot, sample_parameters, sample_parameter_types):
        """Test parameter value updates."""
        manager = ParameterFormManager(
            parameters=sample_parameters,
            parameter_types=sample_parameter_types,
            field_id="test_form"
        )

        # Test updating a parameter
        new_value = "updated_value"
        manager.update_parameter("string_param", new_value)

        assert manager.parameters["string_param"] == new_value

    def test_signal_emission(self, qapp, qtbot, sample_parameters, sample_parameter_types):
        """Test that parameter change signals are emitted correctly."""
        manager = ParameterFormManager(
            parameters=sample_parameters,
            parameter_types=sample_parameter_types,
            field_id="test_form"
        )

        # Set up signal monitoring using pytest-qt
        with qtbot.waitSignal(manager.parameter_changed, timeout=1000) as blocker:
            # Update a parameter
            new_value = "signal_test_value"
            manager.update_parameter("string_param", new_value)

        # Check that signal was emitted with correct arguments
        assert blocker.args[0] == "string_param"  # parameter name
        assert blocker.args[1] == new_value       # parameter value


class TestParameterFormManagerReset:
    """Test reset functionality."""

    def test_reset_individual_parameter(self, qapp, sample_parameters, sample_parameter_types):
        """Test resetting individual parameters."""
        manager = ParameterFormManager(
            parameters=sample_parameters,
            parameter_types=sample_parameter_types,
            field_id="test_form"
        )
        
        # Modify a parameter
        manager.update_parameter("string_param", "modified_value")
        assert manager.parameters["string_param"] == "modified_value"
        
        # Reset the parameter
        manager.reset_parameter("string_param")
        
        # Check that parameter was reset (should be None for lazy context)
        assert manager.parameters["string_param"] is None

    def test_reset_all_parameters(self, qapp, sample_parameters, sample_parameter_types):
        """Test resetting all parameters."""
        manager = ParameterFormManager(
            parameters=sample_parameters,
            parameter_types=sample_parameter_types,
            field_id="test_form"
        )
        
        # Modify multiple parameters
        manager.update_parameter("string_param", "modified1")
        manager.update_parameter("int_param", 999)
        
        # Reset all parameters
        manager.reset_all_parameters()
        
        # Check that all parameters were reset
        for param_name in sample_parameters.keys():
            assert manager.parameters[param_name] is None

    def test_reset_with_default_value(self, qapp, sample_parameters, sample_parameter_types):
        """Test resetting parameter with explicit default value."""
        manager = ParameterFormManager(
            parameters=sample_parameters,
            parameter_types=sample_parameter_types,
            field_id="test_form"
        )
        
        # Reset with explicit default
        default_value = "explicit_default"
        manager.reset_parameter("string_param", default_value)
        
        assert manager.parameters["string_param"] == default_value


class TestParameterFormManagerDataclassIntegration:
    """Test integration with dataclass types."""

    def test_dataclass_widget_creation(self, qapp):
        """Test widget creation for dataclass parameters."""
        # Test with PathPlanningConfig
        path_config = PathPlanningConfig()
        parameters = {"path_planning": path_config}
        parameter_types = {"path_planning": PathPlanningConfig}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="test_dataclass"
        )
        
        assert manager is not None
        # Should have nested managers for dataclass parameters
        assert hasattr(manager, 'nested_managers')

    def test_materialization_config_widget_creation(self, qapp):
        """Test widget creation for MaterializationPathConfig."""
        mat_config = MaterializationPathConfig()
        parameters = {"materialization": mat_config}
        parameter_types = {"materialization": MaterializationPathConfig}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="test_mat"
        )
        
        assert manager is not None

    def test_pipeline_config_integration(self, qapp):
        """Test integration with PipelineConfig."""
        pipeline_config = PipelineConfig()
        
        # Extract parameters using the same pattern as the old tests
        from openhcs.textual_tui.widgets.shared.signature_analyzer import SignatureAnalyzer
        param_info = SignatureAnalyzer.analyze(PipelineConfig)
        
        parameters = {}
        for name in param_info.keys():
            value = getattr(pipeline_config, name, None)
            parameters[name] = value
        
        parameter_types = {name: info.param_type for name, info in param_info.items()}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="pipeline_config",
            dataclass_type=PipelineConfig
        )
        
        assert manager is not None
        assert manager.dataclass_type == PipelineConfig


class TestParameterFormManagerWidgetBehavior:
    """Test widget-specific behaviors."""

    def test_widget_value_retrieval(self, qapp, sample_parameters, sample_parameter_types):
        """Test getting values from widgets."""
        manager = ParameterFormManager(
            parameters=sample_parameters,
            parameter_types=sample_parameter_types,
            field_id="test_form"
        )
        
        # Test getting widget values
        for param_name, widget in manager.widgets.items():
            value = manager.get_widget_value(widget)
            # Value should be retrievable (exact value depends on widget type)
            assert value is not None or param_name == "optional_param"

    def test_widget_value_setting(self, qapp, sample_parameters, sample_parameter_types):
        """Test setting values on widgets."""
        manager = ParameterFormManager(
            parameters=sample_parameters,
            parameter_types=sample_parameter_types,
            field_id="test_form"
        )
        
        # Test setting widget values
        for param_name, widget in manager.widgets.items():
            if param_name == "string_param":
                manager.update_widget_value(widget, "test_widget_value")
                # Verify the value was set (exact verification depends on widget type)
                if hasattr(widget, 'text'):
                    assert widget.text() == "test_widget_value"

    def test_enum_parameter_handling(self, qapp):
        """Test handling of enum parameters like Microscope."""
        parameters = {"microscope": Microscope.IMAGEXPRESS}
        parameter_types = {"microscope": Microscope}

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="test_enum"
        )

        assert manager is not None
        # Should create appropriate widget for enum (likely QComboBox)
        if "microscope" in manager.widgets:
            widget = manager.widgets["microscope"]
            # Enum widgets are typically combo boxes
            assert isinstance(widget, QComboBox) or hasattr(widget, 'currentText')


class TestParameterFormManagerPlaceholders:
    """Test placeholder text functionality."""

    def test_placeholder_application_lazy_context(self, qapp):
        """Test that placeholders are applied in lazy context."""
        pipeline_config = PipelineConfig()

        # Create parameters with None values (lazy context)
        parameters = {
            "microscope": None,
            "plate_path": None
        }
        parameter_types = {
            "microscope": Microscope,
            "plate_path": str
        }

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="lazy_test",
            dataclass_type=PipelineConfig,
            placeholder_prefix="Pipeline default: "
        )

        assert manager is not None
        assert manager.dataclass_type == PipelineConfig
        assert manager.placeholder_prefix == "Pipeline default: "

    def test_placeholder_clearing_concrete_context(self, qapp):
        """Test that placeholders are cleared in concrete context."""
        # Test with concrete values (should not show placeholders)
        parameters = {
            "microscope": Microscope.IMAGEXPRESS,
            "plate_path": "/concrete/path"
        }
        parameter_types = {
            "microscope": Microscope,
            "plate_path": str
        }

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="concrete_test"
        )

        assert manager is not None
        # Concrete values should be displayed, not placeholders
        assert manager.parameters["microscope"] == Microscope.IMAGEXPRESS
        assert manager.parameters["plate_path"] == "/concrete/path"


class TestParameterFormManagerOptionalDataclass:
    """Test optional dataclass parameter handling."""

    def test_optional_dataclass_checkbox_creation(self, qapp, test_dataclass):
        """Test that optional dataclass parameters create checkboxes."""
        TestConfig = test_dataclass
        config = TestConfig()

        parameters = {
            "nested_config": None  # Optional dataclass
        }
        parameter_types = {
            "nested_config": type(config.nested_config) if config.nested_config else type(None)
        }

        # This test would need the actual optional type annotation
        # For now, just test basic creation
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="optional_test"
        )

        assert manager is not None

    def test_optional_dataclass_toggle_behavior(self, qapp):
        """Test enabling/disabling optional dataclass parameters."""
        # This would test the checkbox toggle functionality
        # Implementation depends on the actual optional dataclass handling
        pass


class TestParameterFormManagerErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_parameter_type(self, qapp):
        """Test handling of invalid parameter types."""
        parameters = {"invalid_param": "value"}
        parameter_types = {"invalid_param": object}  # Unsupported type

        # Should not crash, even with unsupported types
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="error_test"
        )

        assert manager is not None

    def test_missing_parameter_type(self, qapp):
        """Test handling when parameter type is missing."""
        parameters = {"param": "value"}
        parameter_types = {}  # Missing type info

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="missing_type_test"
        )

        assert manager is not None

    def test_none_parameters(self, qapp):
        """Test handling of None parameters."""
        parameters = {"none_param": None}
        parameter_types = {"none_param": str}

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="none_test"
        )

        assert manager is not None
        assert manager.parameters["none_param"] is None


class TestParameterFormManagerNewAPIFeatures:
    """Test features specific to the refactored API."""

    def test_reset_parameter_by_path(self, qapp, sample_parameters, sample_parameter_types):
        """Test the reset_parameter_by_path method."""
        manager = ParameterFormManager(
            parameters=sample_parameters,
            parameter_types=sample_parameter_types,
            field_id="path_reset_test"
        )

        # Update a parameter
        manager.update_parameter("string_param", "modified_value")
        assert manager.parameters["string_param"] == "modified_value"

        # Reset using path
        manager.reset_parameter_by_path("string_param")
        assert manager.parameters["string_param"] is None

    def test_dataclass_type_parameter(self, qapp):
        """Test the dataclass_type parameter functionality."""
        parameters = {"test_param": None}
        parameter_types = {"test_param": str}

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="dataclass_type_test",
            dataclass_type=PipelineConfig
        )

        assert manager.dataclass_type == PipelineConfig

    def test_placeholder_prefix_parameter(self, qapp):
        """Test the placeholder_prefix parameter functionality."""
        parameters = {"test_param": None}
        parameter_types = {"test_param": str}

        custom_prefix = "Custom prefix: "
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="prefix_test",
            placeholder_prefix=custom_prefix
        )

        assert manager.placeholder_prefix == custom_prefix

    def test_color_scheme_parameter(self, qapp, mock_color_scheme):
        """Test the color_scheme parameter functionality."""
        parameters = {"test_param": "value"}
        parameter_types = {"test_param": str}

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="color_scheme_test",
            color_scheme=mock_color_scheme
        )

        assert manager.color_scheme == mock_color_scheme

    def test_use_scroll_area_parameter(self, qapp, sample_parameters, sample_parameter_types):
        """Test the use_scroll_area parameter functionality."""
        # Test with scroll area enabled
        manager_with_scroll = ParameterFormManager(
            parameters=sample_parameters,
            parameter_types=sample_parameter_types,
            field_id="scroll_test",
            use_scroll_area=True
        )

        assert manager_with_scroll.use_scroll_area == True

        # Test with scroll area disabled
        manager_without_scroll = ParameterFormManager(
            parameters=sample_parameters,
            parameter_types=sample_parameter_types,
            field_id="no_scroll_test",
            use_scroll_area=False
        )

        assert manager_without_scroll.use_scroll_area == False

    def test_parameter_info_parameter(self, qapp, sample_parameters, sample_parameter_types):
        """Test the parameter_info parameter functionality."""
        param_info = {
            "string_param": {"description": "Test description"},
            "int_param": {"description": "Integer parameter"}
        }

        manager = ParameterFormManager(
            parameters=sample_parameters,
            parameter_types=sample_parameter_types,
            field_id="param_info_test",
            parameter_info=param_info
        )

        assert manager.parameter_info == param_info

    def test_function_target_parameter(self, qapp, sample_parameters, sample_parameter_types):
        """Test the function_target parameter functionality."""
        def dummy_function():
            """Dummy function for testing."""
            pass

        manager = ParameterFormManager(
            parameters=sample_parameters,
            parameter_types=sample_parameter_types,
            field_id="function_target_test",
            function_target=dummy_function
        )

        assert manager.function_target == dummy_function

    def test_widget_value_methods(self, qapp, sample_parameters, sample_parameter_types):
        """Test the update_widget_value and get_widget_value methods."""
        manager = ParameterFormManager(
            parameters=sample_parameters,
            parameter_types=sample_parameter_types,
            field_id="widget_value_test"
        )

        # Test with string parameter widget
        if "string_param" in manager.widgets:
            widget = manager.widgets["string_param"]

            # Test setting widget value
            test_value = "widget_test_value"
            manager.update_widget_value(widget, test_value)

            # Test getting widget value
            retrieved_value = manager.get_widget_value(widget)

            # The exact comparison depends on widget type, but should be retrievable
            assert retrieved_value is not None
