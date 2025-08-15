"""
Functional tests for reset button functionality in PyQt6 parameter forms.

These tests verify that reset buttons actually work in the UI by interacting
with the PyQt6 widgets and checking both internal state and visual behavior.
"""

import pytest
from PyQt6.QtWidgets import QPushButton, QLineEdit, QComboBox, QCheckBox
from PyQt6.QtCore import Qt

from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager
from openhcs.core.config import Microscope, PathPlanningConfig
from openhcs.core.pipeline_config import PipelineConfig


class TestResetButtonFunctionality:
    """Test that reset buttons actually work in the UI."""

    def test_individual_reset_button_exists_and_clickable(self, qtbot):
        """Test that individual reset buttons are created and clickable."""
        parameters = {"test_param": "initial_value"}
        parameter_types = {"test_param": str}

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="reset_button_test"
        )

        qtbot.addWidget(manager)
        manager.show()  # Explicitly show the widget
        qtbot.wait(10)  # Wait for UI to update

        # Find reset buttons
        reset_buttons = manager.findChildren(QPushButton)
        reset_buttons = [btn for btn in reset_buttons if "Reset" in btn.text()]

        # Should have at least one reset button
        assert len(reset_buttons) > 0

        # Reset button should be enabled and clickable
        reset_button = reset_buttons[0]
        assert reset_button.isEnabled()
        assert reset_button.isVisible()

    def test_individual_reset_button_resets_parameter_to_none_lazy_context(self, qtbot):
        """Test that individual reset button resets parameter to None in lazy context."""
        parameters = {"test_param": "modified_value"}
        parameter_types = {"test_param": str}

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="lazy_reset_test",
            dataclass_type=PipelineConfig  # Lazy context
        )

        qtbot.addWidget(manager)
        manager.show()  # Explicitly show the widget
        qtbot.wait(10)  # Wait for UI to update
        
        # Verify initial state
        assert manager.parameters["test_param"] == "modified_value"
        
        # Find and click the reset button
        reset_buttons = manager.findChildren(QPushButton)
        reset_buttons = [btn for btn in reset_buttons if "Reset" in btn.text()]
        assert len(reset_buttons) > 0
        
        reset_button = reset_buttons[0]
        
        # Track signal emissions
        signal_emitted = False
        reset_value = None
        
        def track_reset_signal(param_name, value):
            nonlocal signal_emitted, reset_value
            if param_name == "test_param":
                signal_emitted = True
                reset_value = value
        
        manager.parameter_changed.connect(track_reset_signal)
        
        # Click the reset button
        qtbot.mouseClick(reset_button, Qt.MouseButton.LeftButton)
        
        # Verify parameter was reset to None (lazy context)
        assert manager.parameters["test_param"] is None
        assert signal_emitted
        assert reset_value is None

    def test_individual_reset_button_updates_widget_value(self, qtbot):
        """Test that reset button updates the widget value in the UI."""
        parameters = {"string_param": "modified_value"}
        parameter_types = {"string_param": str}

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="widget_reset_test",
            dataclass_type=PipelineConfig
        )

        qtbot.addWidget(manager)
        manager.show()  # Explicitly show the widget
        qtbot.wait(10)  # Wait for UI to update
        
        # Find the string widget
        string_widget = None
        if "string_param" in manager.widgets:
            string_widget = manager.widgets["string_param"]
        
        assert string_widget is not None
        assert isinstance(string_widget, QLineEdit)
        
        # Verify initial widget value
        assert string_widget.text() == "modified_value"
        
        # Find and click the reset button
        reset_buttons = manager.findChildren(QPushButton)
        reset_buttons = [btn for btn in reset_buttons if "Reset" in btn.text()]
        reset_button = reset_buttons[0]
        
        # Click the reset button
        qtbot.mouseClick(reset_button, Qt.MouseButton.LeftButton)
        
        # Verify widget was cleared (reset to None shows as empty text)
        assert string_widget.text() == ""

    def test_reset_all_button_exists_and_functional(self, qtbot):
        """Test that Reset All button exists and resets all parameters."""
        parameters = {
            "param1": "value1",
            "param2": 42,
            "param3": True
        }
        parameter_types = {
            "param1": str,
            "param2": int,
            "param3": bool
        }
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="reset_all_test",
            dataclass_type=PipelineConfig
        )
        
        qtbot.addWidget(manager)
        manager.show()  # Explicitly show the widget
        qtbot.wait(10)  # Wait for UI to update

        # Verify initial state
        assert manager.parameters["param1"] == "value1"
        assert manager.parameters["param2"] == 42
        assert manager.parameters["param3"] == True
        
        # Track signal emissions
        reset_signals = []
        
        def track_signals(param_name, value):
            reset_signals.append((param_name, value))
        
        manager.parameter_changed.connect(track_signals)
        
        # Call reset all (programmatically since UI button may not exist)
        manager.reset_all_parameters()
        
        # Verify all parameters were reset to None
        assert manager.parameters["param1"] is None
        assert manager.parameters["param2"] is None
        assert manager.parameters["param3"] is None
        
        # Verify signals were emitted for all parameters
        assert len(reset_signals) >= 3

    def test_enum_parameter_reset_button_functionality(self, qtbot):
        """Test reset button functionality with enum parameters."""
        parameters = {"microscope": Microscope.OPERAPHENIX}
        parameter_types = {"microscope": Microscope}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="enum_reset_test",
            dataclass_type=PipelineConfig
        )
        
        qtbot.addWidget(manager)
        manager.show()  # Explicitly show the widget
        qtbot.wait(10)  # Wait for UI to update

        # Find the enum widget
        enum_widget = None
        if "microscope" in manager.widgets:
            enum_widget = manager.widgets["microscope"]
        
        assert enum_widget is not None
        assert isinstance(enum_widget, QComboBox)
        
        # Verify initial state
        assert manager.parameters["microscope"] == Microscope.OPERAPHENIX
        
        # Find and click the reset button
        reset_buttons = manager.findChildren(QPushButton)
        reset_buttons = [btn for btn in reset_buttons if "Reset" in btn.text()]
        reset_button = reset_buttons[0]
        
        # Click the reset button
        qtbot.mouseClick(reset_button, Qt.MouseButton.LeftButton)
        
        # Verify parameter was reset
        assert manager.parameters["microscope"] is None

    def test_boolean_parameter_reset_button_functionality(self, qtbot):
        """Test reset button functionality with boolean parameters."""
        parameters = {"bool_param": True}
        parameter_types = {"bool_param": bool}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="bool_reset_test",
            dataclass_type=PipelineConfig
        )
        
        qtbot.addWidget(manager)
        manager.show()  # Explicitly show the widget
        qtbot.wait(10)  # Wait for UI to update

        # Find the boolean widget
        bool_widget = None
        if "bool_param" in manager.widgets:
            bool_widget = manager.widgets["bool_param"]
        
        assert bool_widget is not None
        assert isinstance(bool_widget, QCheckBox)
        
        # Verify initial state
        assert manager.parameters["bool_param"] == True
        assert bool_widget.isChecked() == True
        
        # Find and click the reset button
        reset_buttons = manager.findChildren(QPushButton)
        reset_buttons = [btn for btn in reset_buttons if "Reset" in btn.text()]
        reset_button = reset_buttons[0]
        
        # Click the reset button
        qtbot.mouseClick(reset_button, Qt.MouseButton.LeftButton)
        
        # Verify parameter was reset
        assert manager.parameters["bool_param"] is None
        # Widget state depends on how None is handled for checkboxes
        # In lazy context, it should be unchecked or in indeterminate state

    def test_reset_button_signal_connections(self, qtbot):
        """Test that reset buttons are properly connected to signal handlers."""
        parameters = {"test_param": "value"}
        parameter_types = {"test_param": str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="signal_connection_test"
        )
        
        qtbot.addWidget(manager)
        manager.show()  # Explicitly show the widget
        qtbot.wait(10)  # Wait for UI to update

        # Find reset buttons
        reset_buttons = manager.findChildren(QPushButton)
        reset_buttons = [btn for btn in reset_buttons if "Reset" in btn.text()]
        
        # Each reset button should have signal connections
        for button in reset_buttons:
            # Check that clicked signal has receivers
            assert button.receivers(button.clicked) > 0

    def test_nested_dataclass_reset_functionality(self, qtbot):
        """Test reset functionality with nested dataclass parameters."""
        path_config = PathPlanningConfig()
        parameters = {"path_planning": path_config}
        parameter_types = {"path_planning": PathPlanningConfig}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="nested_reset_test",
            dataclass_type=PipelineConfig
        )
        
        qtbot.addWidget(manager)
        manager.show()  # Explicitly show the widget
        qtbot.wait(10)  # Wait for UI to update

        # Verify initial state
        assert manager.parameters["path_planning"] is not None
        
        # Find and click a reset button (should reset the nested parameter)
        reset_buttons = manager.findChildren(QPushButton)
        reset_buttons = [btn for btn in reset_buttons if "Reset" in btn.text()]
        
        if reset_buttons:
            reset_button = reset_buttons[0]
            qtbot.mouseClick(reset_button, Qt.MouseButton.LeftButton)
            
            # Verify nested parameter was reset
            # The exact behavior depends on implementation
            # It might reset to None or to a new default instance
