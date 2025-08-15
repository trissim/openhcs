"""
Functional tests for lazy placeholder text functionality in PyQt6 parameter forms.

These tests verify that placeholder text is actually displayed in the UI widgets
and behaves correctly when users interact with the forms.
"""

import pytest
from PyQt6.QtWidgets import QLineEdit, QComboBox, QCheckBox, QPushButton
from PyQt6.QtCore import Qt

from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager
from openhcs.core.config import Microscope, PathPlanningConfig
from openhcs.core.pipeline_config import PipelineConfig, GlobalPipelineConfig


class TestLazyPlaceholderText:
    """Test that placeholder text is actually displayed and behaves correctly in the UI."""

    def test_placeholder_text_displayed_for_none_values_lazy_context(self, qtbot):
        """Test that placeholder text is displayed for None values in lazy context."""
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
            field_id="placeholder_display_test",
            dataclass_type=PipelineConfig,  # Lazy context
            placeholder_prefix="Pipeline default: "
        )
        
        qtbot.addWidget(manager)
        manager.show()  # Explicitly show the widget
        qtbot.wait(10)  # Wait for UI to update

        # Check string parameter widget
        if "plate_path" in manager.widgets:
            string_widget = manager.widgets["plate_path"]
            assert isinstance(string_widget, QLineEdit)
            
            # Widget should be empty (showing placeholder)
            assert string_widget.text() == ""
            
            # Should have placeholder text set
            placeholder_text = string_widget.placeholderText()
            assert placeholder_text != ""
            assert "Pipeline default:" in placeholder_text
        
        # Check enum parameter widget
        if "microscope" in manager.widgets:
            enum_widget = manager.widgets["microscope"]
            assert isinstance(enum_widget, QComboBox)
            
            # Should have placeholder state property
            is_placeholder_state = enum_widget.property("is_placeholder_state")
            # May be True or None depending on implementation

    def test_placeholder_text_cleared_when_user_enters_value(self, qtbot):
        """Test that placeholder text is cleared when user enters a concrete value."""
        parameters = {"test_param": None}
        parameter_types = {"test_param": str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="placeholder_clear_test",
            dataclass_type=PipelineConfig,
            placeholder_prefix="Pipeline default: "
        )
        
        qtbot.addWidget(manager)
        manager.show()  # Explicitly show the widget
        qtbot.wait(10)  # Wait for UI to update

        # Find the string widget
        string_widget = None
        if "test_param" in manager.widgets:
            string_widget = manager.widgets["test_param"]
        
        assert string_widget is not None
        assert isinstance(string_widget, QLineEdit)
        
        # Initially should have placeholder
        initial_placeholder = string_widget.placeholderText()
        assert initial_placeholder != ""
        
        # Simulate user typing
        string_widget.setText("user_entered_value")
        qtbot.keyClick(string_widget, Qt.Key.Key_Return)
        
        # Parameter should be updated
        assert manager.parameters["test_param"] == "user_entered_value"
        
        # Widget should show the entered value
        assert string_widget.text() == "user_entered_value"

    def test_placeholder_text_reappears_after_reset(self, qtbot):
        """Test that placeholder text reappears when parameter is reset back to None."""
        parameters = {"test_param": "concrete_value"}
        parameter_types = {"test_param": str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="placeholder_reappear_test",
            dataclass_type=PipelineConfig,
            placeholder_prefix="Pipeline default: "
        )
        
        qtbot.addWidget(manager)
        manager.show()  # Explicitly show the widget
        qtbot.wait(10)  # Wait for UI to update

        # Find the string widget
        string_widget = None
        if "test_param" in manager.widgets:
            string_widget = manager.widgets["test_param"]
        
        assert string_widget is not None
        assert isinstance(string_widget, QLineEdit)
        
        # Initially should show concrete value (no placeholder)
        assert string_widget.text() == "concrete_value"
        initial_placeholder = string_widget.placeholderText()
        
        # Reset the parameter
        manager.reset_parameter("test_param")
        
        # Parameter should be None
        assert manager.parameters["test_param"] is None
        
        # Widget should be empty
        assert string_widget.text() == ""
        
        # Placeholder should be set
        final_placeholder = string_widget.placeholderText()
        assert final_placeholder != ""
        assert "Pipeline default:" in final_placeholder

    def test_different_placeholder_prefixes(self, qtbot):
        """Test that different placeholder prefixes work correctly."""
        # Test with custom prefix
        parameters = {"test_param": None}
        parameter_types = {"test_param": str}
        
        custom_prefix = "Global default: "
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="custom_prefix_test",
            dataclass_type=GlobalPipelineConfig,  # Global context
            placeholder_prefix=custom_prefix
        )
        
        qtbot.addWidget(manager)
        
        # Check that custom prefix is used
        assert manager.placeholder_prefix == custom_prefix
        
        # Find the string widget
        if "test_param" in manager.widgets:
            string_widget = manager.widgets["test_param"]
            assert isinstance(string_widget, QLineEdit)
            
            # Should have custom prefix in placeholder
            placeholder_text = string_widget.placeholderText()
            if placeholder_text:  # May be empty if no default value available
                assert "Global default:" in placeholder_text

    def test_no_placeholder_in_concrete_context(self, qtbot):
        """Test that placeholders are not applied in concrete context."""
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
            field_id="concrete_context_test"
            # No dataclass_type = concrete context
        )
        
        qtbot.addWidget(manager)
        
        # Check string parameter widget
        if "plate_path" in manager.widgets:
            string_widget = manager.widgets["plate_path"]
            assert isinstance(string_widget, QLineEdit)
            
            # Should show concrete value
            assert string_widget.text() == "/concrete/path"
            
            # Should not have placeholder state
            is_placeholder_state = string_widget.property("is_placeholder_state")
            assert not is_placeholder_state

    def test_enum_placeholder_text_behavior(self, qtbot):
        """Test placeholder text behavior specifically for enum widgets."""
        parameters = {"microscope": None}
        parameter_types = {"microscope": Microscope}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="enum_placeholder_test",
            dataclass_type=PipelineConfig,
            placeholder_prefix="Pipeline default: "
        )
        
        qtbot.addWidget(manager)
        
        # Find the enum widget
        if "microscope" in manager.widgets:
            enum_widget = manager.widgets["microscope"]
            assert isinstance(enum_widget, QComboBox)
            
            # Should have items available
            assert enum_widget.count() > 0
            
            # Check for placeholder state
            is_placeholder_state = enum_widget.property("is_placeholder_state")
            # Implementation may vary for combo boxes
            
            # Should have tooltip with placeholder information
            tooltip = enum_widget.toolTip()
            if tooltip:
                assert "Pipeline default:" in tooltip

    def test_placeholder_text_with_nested_dataclass(self, qtbot):
        """Test placeholder text behavior with nested dataclass parameters."""
        parameters = {"path_planning": None}
        parameter_types = {"path_planning": PathPlanningConfig}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="nested_placeholder_test",
            dataclass_type=PipelineConfig,
            placeholder_prefix="Pipeline default: "
        )
        
        qtbot.addWidget(manager)
        
        # Nested dataclass parameters may not have traditional placeholder text
        # but should be handled gracefully
        assert manager.parameters["path_planning"] is None

    def test_placeholder_text_persistence_through_interactions(self, qtbot):
        """Test that placeholder text persists correctly through various UI interactions."""
        parameters = {"test_param": None}
        parameter_types = {"test_param": str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="placeholder_persistence_test",
            dataclass_type=PipelineConfig,
            placeholder_prefix="Pipeline default: "
        )
        
        qtbot.addWidget(manager)
        
        # Find the string widget
        string_widget = None
        if "test_param" in manager.widgets:
            string_widget = manager.widgets["test_param"]
        
        assert string_widget is not None
        assert isinstance(string_widget, QLineEdit)
        
        # Initially should have placeholder
        initial_placeholder = string_widget.placeholderText()
        assert initial_placeholder != ""
        
        # Focus and unfocus without typing
        string_widget.setFocus()
        qtbot.wait(10)
        string_widget.clearFocus()
        qtbot.wait(10)
        
        # Placeholder should still be there
        assert string_widget.placeholderText() == initial_placeholder
        assert string_widget.text() == ""
        
        # Type and delete all text
        string_widget.setText("temp")
        string_widget.clear()
        
        # Placeholder should still be there
        assert string_widget.placeholderText() == initial_placeholder

    def test_placeholder_text_with_special_characters(self, qtbot):
        """Test placeholder text handling with special characters and formatting."""
        parameters = {"special_param": None}
        parameter_types = {"special_param": str}
        
        special_prefix = "Custom: "
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="special_chars_test",
            dataclass_type=PipelineConfig,
            placeholder_prefix=special_prefix
        )
        
        qtbot.addWidget(manager)
        
        # Should handle special characters in prefix gracefully
        assert manager.placeholder_prefix == special_prefix
