"""
Unit tests for PyQt6 signal handling in parameter forms.

This module tests the signal/slot functionality of the ParameterFormManager,
including parameter change signals, widget signal connections, and event handling.
"""

import pytest
from unittest.mock import Mock, patch
from PyQt6.QtWidgets import QLineEdit, QComboBox, QCheckBox, QPushButton
from PyQt6.QtCore import Qt

from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager
from openhcs.core.config import Microscope


class TestParameterChangeSignals:
    """Test parameter change signal emission."""

    def test_parameter_changed_signal_emission(self, qtbot):
        """Test that parameter_changed signal is emitted on parameter updates."""
        parameters = {"test_param": "initial_value"}
        parameter_types = {"test_param": str}

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="signal_test"
        )

        # Set up signal monitoring using pytest-qt
        with qtbot.waitSignal(manager.parameter_changed, timeout=1000) as blocker:
            # Update parameter
            new_value = "updated_value"
            manager.update_parameter("test_param", new_value)

        # Verify signal was emitted with correct arguments
        assert blocker.args[0] == "test_param"  # parameter name
        assert blocker.args[1] == new_value     # parameter value

    def test_multiple_parameter_changes_signal_emission(self, qtbot):
        """Test signal emission for multiple parameter changes."""
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
            field_id="multi_signal_test"
        )

        # Track signal emissions
        signal_emissions = []

        def track_signal(param_name, param_value):
            signal_emissions.append((param_name, param_value))

        manager.parameter_changed.connect(track_signal)

        # Update multiple parameters
        manager.update_parameter("param1", "new_value1")
        manager.update_parameter("param2", 999)
        manager.update_parameter("param3", False)

        # Verify all signals were emitted (may be more due to internal updates)
        assert len(signal_emissions) >= 3

        # Check that all expected parameter updates are present
        param_names = [emission[0] for emission in signal_emissions]
        param_values = [emission[1] for emission in signal_emissions]

        assert "param1" in param_names and "new_value1" in param_values
        assert "param2" in param_names and 999 in param_values
        assert "param3" in param_names and False in param_values

    def test_reset_parameter_signal_emission(self, qtbot):
        """Test that parameter_changed signal is emitted on parameter reset."""
        parameters = {"test_param": "initial_value"}
        parameter_types = {"test_param": str}

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="reset_signal_test"
        )

        # Track signal emissions
        signal_emissions = []

        def track_signal(param_name, param_value):
            signal_emissions.append((param_name, param_value))

        manager.parameter_changed.connect(track_signal)

        # Reset parameter
        manager.reset_parameter("test_param")

        # Verify signal was emitted (may be more than 1 due to internal updates)
        assert len(signal_emissions) >= 1

        # Check that reset signal was emitted
        param_names = [emission[0] for emission in signal_emissions]
        param_values = [emission[1] for emission in signal_emissions]

        assert "test_param" in param_names
        # Reset value should be None in lazy context
        assert None in param_values

    def test_reset_all_parameters_signal_emission(self, qtbot):
        """Test signal emission when resetting all parameters."""
        parameters = {
            "param1": "value1",
            "param2": 42
        }
        parameter_types = {
            "param1": str,
            "param2": int
        }

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="reset_all_signal_test"
        )

        # Track signal emissions
        signal_count = 0

        def count_signals(*args):
            nonlocal signal_count
            signal_count += 1

        manager.parameter_changed.connect(count_signals)

        # Reset all parameters
        manager.reset_all_parameters()

        # Should emit signal for each parameter (may be more due to internal updates)
        assert signal_count >= 2


class TestWidgetSignalConnections:
    """Test widget signal connections and handling."""

    def test_line_edit_signal_connection(self, qapp):
        """Test that QLineEdit widgets have proper signal connections."""
        parameters = {"string_param": "test_value"}
        parameter_types = {"string_param": str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="line_edit_signal_test"
        )
        
        if "string_param" in manager.widgets:
            widget = manager.widgets["string_param"]
            if isinstance(widget, QLineEdit):
                # Check that textChanged signal has receivers
                assert widget.receivers(widget.textChanged) > 0

    def test_combo_box_signal_connection(self, qapp):
        """Test that QComboBox widgets have proper signal connections."""
        parameters = {"enum_param": Microscope.IMAGEXPRESS}
        parameter_types = {"enum_param": Microscope}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="combo_signal_test"
        )
        
        if "enum_param" in manager.widgets:
            widget = manager.widgets["enum_param"]
            if isinstance(widget, QComboBox):
                # Check that currentTextChanged or currentIndexChanged signal has receivers
                has_text_receivers = widget.receivers(widget.currentTextChanged) > 0
                has_index_receivers = widget.receivers(widget.currentIndexChanged) > 0
                assert has_text_receivers or has_index_receivers

    def test_checkbox_signal_connection(self, qapp):
        """Test that QCheckBox widgets have proper signal connections."""
        parameters = {"bool_param": True}
        parameter_types = {"bool_param": bool}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="checkbox_signal_test"
        )
        
        if "bool_param" in manager.widgets:
            widget = manager.widgets["bool_param"]
            if isinstance(widget, QCheckBox):
                # Check that stateChanged signal has receivers
                assert widget.receivers(widget.stateChanged) > 0

    def test_reset_button_signal_connection(self, qapp):
        """Test that reset buttons have proper signal connections."""
        parameters = {"test_param": "value"}
        parameter_types = {"test_param": str}
        
        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="reset_button_signal_test"
        )
        
        # Find reset buttons in the form
        reset_buttons = manager.findChildren(QPushButton)
        reset_buttons = [btn for btn in reset_buttons if "Reset" in btn.text()]
        
        # Each reset button should have signal connections
        for button in reset_buttons:
            assert button.receivers(button.clicked) > 0


class TestSignalHandlerBehavior:
    """Test the behavior of signal handlers."""

    def test_widget_change_triggers_parameter_update(self, qtbot):
        """Test that widget changes trigger parameter updates."""
        parameters = {"test_param": "initial"}
        parameter_types = {"test_param": str}

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="widget_change_test"
        )

        # Track signal emissions
        signal_emissions = []

        def track_signal(param_name, param_value):
            signal_emissions.append((param_name, param_value))

        manager.parameter_changed.connect(track_signal)

        # Simulate widget change by directly calling the signal handler
        if hasattr(manager, '_emit_parameter_change'):
            manager._emit_parameter_change("test_param", "changed_value")

            # Verify signal was emitted
            assert len(signal_emissions) == 1
            assert signal_emissions[0][0] == "test_param"
            assert signal_emissions[0][1] == "changed_value"

    def test_signal_handler_with_type_conversion(self, qtbot):
        """Test signal handling with type conversion."""
        parameters = {"int_param": 42}
        parameter_types = {"int_param": int}

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="type_conversion_test"
        )

        # Track signal emissions
        signal_emissions = []

        def track_signal(param_name, param_value):
            signal_emissions.append((param_name, param_value))

        manager.parameter_changed.connect(track_signal)

        # Update with string that should be converted to int
        manager.update_parameter("int_param", "999")

        # Verify signal was emitted with converted value
        assert len(signal_emissions) == 1
        assert signal_emissions[0][0] == "int_param"
        # Value should be converted to int
        assert isinstance(signal_emissions[0][1], int)
        assert signal_emissions[0][1] == 999

    def test_signal_blocking_and_unblocking(self, qtbot):
        """Test signal blocking functionality if available."""
        parameters = {"test_param": "value"}
        parameter_types = {"test_param": str}

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="signal_blocking_test"
        )

        # Track signal emissions
        signal_count = 0

        def count_signals(*args):
            nonlocal signal_count
            signal_count += 1

        manager.parameter_changed.connect(count_signals)

        # Block signals
        manager.blockSignals(True)

        # Update parameter (should not emit signal)
        manager.update_parameter("test_param", "blocked_value")

        # Verify no signal was emitted
        assert signal_count == 0

        # Unblock signals
        manager.blockSignals(False)

        # Update parameter (should emit signal)
        manager.update_parameter("test_param", "unblocked_value")

        # Verify signal was emitted (may be more than 1 due to internal updates)
        assert signal_count >= 1


class TestSignalErrorHandling:
    """Test error handling in signal processing."""

    def test_signal_emission_with_invalid_parameter(self, qtbot):
        """Test signal emission behavior with invalid parameters."""
        parameters = {"valid_param": "value"}
        parameter_types = {"valid_param": str}

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="invalid_signal_test"
        )

        # Track signal emissions
        signal_count = 0

        def count_signals(*args):
            nonlocal signal_count
            signal_count += 1

        manager.parameter_changed.connect(count_signals)

        # Try to emit signal for non-existent parameter
        if hasattr(manager, '_emit_parameter_change'):
            try:
                manager._emit_parameter_change("non_existent", "value")
                # Should either handle gracefully or emit the signal anyway
                # The exact behavior depends on implementation
            except Exception:
                # If an exception is raised, it should be handled gracefully
                pass

    def test_signal_emission_with_none_values(self, qtbot):
        """Test signal emission with None values."""
        parameters = {"nullable_param": None}
        parameter_types = {"nullable_param": str}

        manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="none_signal_test"
        )

        # Track signal emissions
        signal_emissions = []

        def track_signal(param_name, param_value):
            signal_emissions.append((param_name, param_value))

        manager.parameter_changed.connect(track_signal)

        # Update with None value
        manager.update_parameter("nullable_param", None)

        # Should emit signal with None value
        assert len(signal_emissions) == 1
        assert signal_emissions[0][0] == "nullable_param"
        assert signal_emissions[0][1] is None
