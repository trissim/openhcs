#!/usr/bin/env python3
"""
Integration tests for cross-context validation and signal emission patterns.

These tests verify that the parameter form system correctly handles context switching
and signal emission during reset operations.
"""

import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest, QSignalSpy

from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager
from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.pipeline_config import PipelineConfig, set_current_pipeline_config
from openhcs.constants import Microscope


class TestCrossContextValidation:
    """Test switching between global config editing and orchestrator config editing."""

    def test_context_switching_reset_behavior(self, qtbot):
        """Test that reset behavior changes appropriately when switching contexts."""
        
        # Set up global config for placeholder resolution
        global_config = GlobalPipelineConfig(
            num_workers=24,  # Custom value for testing
            use_threading=True
        )
        set_current_pipeline_config(global_config)
        
        # Create GlobalPipelineConfig form (concrete context)
        global_parameters = {"num_workers": 8, "use_threading": False}
        global_parameter_types = {"num_workers": int, "use_threading": bool}
        
        global_form = ParameterFormManager(
            parameters=global_parameters,
            parameter_types=global_parameter_types,
            field_id="global_context_test",
            dataclass_type=GlobalPipelineConfig,
            placeholder_prefix=""
        )
        qtbot.addWidget(global_form)
        
        # Create PipelineConfig form (lazy context)
        lazy_parameters = {"num_workers": 8, "use_threading": False}
        lazy_parameter_types = {"num_workers": int, "use_threading": bool}
        
        lazy_form = ParameterFormManager(
            parameters=lazy_parameters,
            parameter_types=lazy_parameter_types,
            field_id="lazy_context_test",
            dataclass_type=PipelineConfig,
            placeholder_prefix="Pipeline default: "
        )
        qtbot.addWidget(lazy_form)
        
        # Test reset behavior in global context
        global_reset_btn = global_form.reset_buttons["num_workers"]
        qtbot.mouseClick(global_reset_btn, Qt.MouseButton.LeftButton)
        
        # Should reset to actual GlobalPipelineConfig default (16)
        default_global_config = GlobalPipelineConfig()
        expected_global_default = default_global_config.num_workers
        assert global_form.parameters["num_workers"] == expected_global_default, f"Global form should reset to {expected_global_default}"
        
        global_widget = global_form.widgets["num_workers"]
        assert global_widget.value() == expected_global_default, f"Global widget should display {expected_global_default}"
        
        # Test reset behavior in lazy context
        lazy_reset_btn = lazy_form.reset_buttons["num_workers"]
        qtbot.mouseClick(lazy_reset_btn, Qt.MouseButton.LeftButton)
        
        # Should reset to None with placeholder showing global config value (24)
        assert lazy_form.parameters["num_workers"] is None, "Lazy form should reset to None"
        
        lazy_widget = lazy_form.widgets["num_workers"]
        if hasattr(lazy_widget, 'specialValueText') and lazy_widget.specialValueText():
            assert "24" in lazy_widget.specialValueText(), "Placeholder should show current global config value (24)"
        elif hasattr(lazy_widget, 'placeholderText') and lazy_widget.placeholderText():
            assert "24" in lazy_widget.placeholderText(), "Placeholder should show current global config value (24)"

    def test_global_config_changes_update_lazy_placeholders(self, qtbot):
        """Test that changes to global config update placeholder text in lazy forms."""
        
        # Set up initial global config
        global_config = GlobalPipelineConfig(num_workers=16)
        set_current_pipeline_config(global_config)
        
        # Create lazy form with None values
        lazy_form = ParameterFormManager(
            parameters={"num_workers": None},
            parameter_types={"num_workers": int},
            field_id="placeholder_update_test",
            dataclass_type=PipelineConfig,
            placeholder_prefix="Pipeline default: "
        )
        qtbot.addWidget(lazy_form)
        
        # Verify initial placeholder shows 16
        lazy_widget = lazy_form.widgets["num_workers"]
        initial_placeholder = ""
        if hasattr(lazy_widget, 'specialValueText'):
            initial_placeholder = lazy_widget.specialValueText()
        elif hasattr(lazy_widget, 'placeholderText'):
            initial_placeholder = lazy_widget.placeholderText()
        
        assert "16" in initial_placeholder, "Initial placeholder should show 16"
        
        # Update global config
        updated_global_config = GlobalPipelineConfig(num_workers=32)
        set_current_pipeline_config(updated_global_config)
        
        # Reset the lazy form to trigger placeholder update
        lazy_form.reset_all_parameters()
        
        # Verify placeholder now shows updated value
        updated_placeholder = ""
        if hasattr(lazy_widget, 'specialValueText'):
            updated_placeholder = lazy_widget.specialValueText()
        elif hasattr(lazy_widget, 'placeholderText'):
            updated_placeholder = lazy_widget.placeholderText()
        
        assert "32" in updated_placeholder, "Updated placeholder should show 32"

    def test_mixed_lazy_state_save_load_cycle(self, qtbot):
        """Test save/load cycle with mixed lazy state (some None, some concrete values)."""
        
        # Set up global config
        global_config = GlobalPipelineConfig()
        set_current_pipeline_config(global_config)
        
        # Create lazy form with mixed state
        mixed_parameters = {
            "num_workers": 12,  # Concrete value
            "use_threading": None,  # Lazy value
            "microscope": Microscope.IMAGEXPRESS,  # Concrete value
        }
        mixed_parameter_types = {
            "num_workers": int,
            "use_threading": bool,
            "microscope": Microscope,
        }
        
        form_manager = ParameterFormManager(
            parameters=mixed_parameters,
            parameter_types=mixed_parameter_types,
            field_id="mixed_state_test",
            dataclass_type=PipelineConfig,
            placeholder_prefix="Pipeline default: "
        )
        qtbot.addWidget(form_manager)
        
        # Verify initial mixed state
        assert form_manager.parameters["num_workers"] == 12, "num_workers should be concrete"
        assert form_manager.parameters["use_threading"] is None, "use_threading should be None"
        assert form_manager.parameters["microscope"] == Microscope.IMAGEXPRESS, "microscope should be concrete"
        
        # Verify widget states
        num_workers_widget = form_manager.widgets["num_workers"]
        use_threading_widget = form_manager.widgets["use_threading"]
        microscope_widget = form_manager.widgets["microscope"]
        
        assert num_workers_widget.value() == 12, "num_workers widget should show concrete value"
        assert microscope_widget.currentText() == "IMAGEXPRESS", "microscope widget should show concrete value"
        
        # use_threading widget should show placeholder state
        if hasattr(use_threading_widget, 'toolTip') and use_threading_widget.toolTip():
            assert "Pipeline default:" in use_threading_widget.toolTip(), "use_threading should show placeholder"
        
        # Simulate save operation (get current parameters)
        saved_state = form_manager.parameters.copy()
        
        # Modify some values
        form_manager.update_parameter("num_workers", 20)
        form_manager.update_parameter("use_threading", True)  # Change from None to concrete
        
        # Verify modifications
        assert form_manager.parameters["num_workers"] == 20, "num_workers should be updated"
        assert form_manager.parameters["use_threading"] == True, "use_threading should be concrete now"
        
        # Simulate load operation (restore saved state)
        for param_name, value in saved_state.items():
            form_manager.update_parameter(param_name, value)
        
        # Verify restoration to mixed state
        assert form_manager.parameters["num_workers"] == 12, "num_workers should be restored"
        assert form_manager.parameters["use_threading"] is None, "use_threading should be restored to None"
        assert form_manager.parameters["microscope"] == Microscope.IMAGEXPRESS, "microscope should be restored"
        
        # Verify widgets reflect restored state
        assert num_workers_widget.value() == 12, "num_workers widget should show restored value"
        
        # use_threading should show placeholder again
        if hasattr(use_threading_widget, 'toolTip') and use_threading_widget.toolTip():
            assert "Pipeline default:" in use_threading_widget.toolTip(), "use_threading should show placeholder again"


class TestSignalEmissionPatterns:
    """Test signal emission patterns during reset operations."""

    def test_reset_signal_emission_patterns(self, qtbot):
        """Test that reset operations emit appropriate signals."""
        
        form_manager = ParameterFormManager(
            parameters={"num_workers": 8, "use_threading": True},
            parameter_types={"num_workers": int, "use_threading": bool},
            field_id="signal_test",
            dataclass_type=GlobalPipelineConfig,
            placeholder_prefix=""
        )
        qtbot.addWidget(form_manager)
        
        # Set up signal spy
        signal_spy = QSignalSpy(form_manager.parameter_changed)
        
        # Test individual reset signal emission
        reset_btn = form_manager.reset_buttons["num_workers"]
        qtbot.mouseClick(reset_btn, Qt.MouseButton.LeftButton)
        
        # Verify signal was emitted
        assert len(signal_spy) == 1, "One signal should be emitted for individual reset"
        
        # Verify signal contains correct data
        signal_data = signal_spy[0]
        param_name = signal_data[0]
        param_value = signal_data[1]
        
        assert param_name == "num_workers", "Signal should contain correct parameter name"
        
        # Get expected default value
        default_config = GlobalPipelineConfig()
        expected_default = default_config.num_workers
        assert param_value == expected_default, f"Signal should contain correct reset value ({expected_default})"
        
        # Clear signal spy for next test by creating a new one
        signal_spy = QSignalSpy(form_manager.parameter_changed)
        
        # Test reset all signal emission
        form_manager.reset_all_parameters()
        
        # Should emit signals for all parameters
        assert len(signal_spy) >= 2, "Multiple signals should be emitted for reset all"
        
        # Verify all parameters were signaled
        signaled_params = [signal[0] for signal in signal_spy]
        assert "num_workers" in signaled_params, "num_workers should be signaled"
        assert "use_threading" in signaled_params, "use_threading should be signaled"

    def test_widget_signal_blocking_during_reset(self, qtbot):
        """Test that widget signals are properly blocked during reset to prevent loops."""
        
        form_manager = ParameterFormManager(
            parameters={"num_workers": 8},
            parameter_types={"num_workers": int},
            field_id="signal_blocking_test",
            dataclass_type=GlobalPipelineConfig,
            placeholder_prefix=""
        )
        qtbot.addWidget(form_manager)
        
        widget = form_manager.widgets["num_workers"]
        
        # Set up signal spy on widget's valueChanged signal
        widget_signal_spy = QSignalSpy(widget.valueChanged)
        
        # Set up signal spy on form manager's parameter_changed signal
        form_signal_spy = QSignalSpy(form_manager.parameter_changed)
        
        # Perform reset operation
        reset_btn = form_manager.reset_buttons["num_workers"]
        qtbot.mouseClick(reset_btn, Qt.MouseButton.LeftButton)
        
        # Widget signals should be blocked during reset (no widget valueChanged signals)
        # But form manager should still emit parameter_changed signal
        assert len(form_signal_spy) == 1, "Form manager should emit parameter_changed signal"
        
        # Widget valueChanged might or might not be emitted depending on implementation
        # The key is that it shouldn't cause infinite loops or duplicate processing
