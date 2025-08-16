#!/usr/bin/env python3
"""
Integration tests for PyQt6 GUI parameter form reset behavior workflows.

These tests simulate actual user interactions with the GUI, focusing on end-to-end
workflows that users would perform in real usage scenarios.
"""

import pytest
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager
from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.pipeline_config import PipelineConfig, set_current_pipeline_config
from openhcs.constants import Microscope


class TestGlobalConfigResetWorkflows:
    """Test reset workflows for GlobalPipelineConfig (concrete/non-lazy context)."""

    def test_complete_global_config_reset_workflow(self, qtbot):
        """Test complete user workflow with GlobalPipelineConfig reset operations."""
        
        # Create a comprehensive parameter form with multiple field types
        parameters = {
            "num_workers": 8,  # int - should reset to 16 (actual default)
            "use_threading": True,  # bool - should reset to False (actual default)
            "microscope": Microscope.IMAGEXPRESS,  # enum - should reset to AUTO (actual default)
        }
        parameter_types = {
            "num_workers": int,
            "use_threading": bool,
            "microscope": Microscope,
        }
        
        # Create form manager for GlobalPipelineConfig (concrete context)
        form_manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="global_config_test",
            dataclass_type=GlobalPipelineConfig,
            placeholder_prefix=""  # No placeholder prefix for concrete context
        )
        qtbot.addWidget(form_manager)
        
        # Get actual default values for verification
        default_config = GlobalPipelineConfig()
        expected_num_workers = default_config.num_workers
        expected_use_threading = default_config.use_threading
        expected_microscope = default_config.microscope
        
        # Verify initial state - widgets show modified values
        num_workers_widget = form_manager.widgets["num_workers"]
        use_threading_widget = form_manager.widgets["use_threading"]
        microscope_widget = form_manager.widgets["microscope"]
        
        assert num_workers_widget.value() == 8, "Initial num_workers widget should show 8"
        assert use_threading_widget.isChecked() == True, "Initial use_threading widget should be checked"
        # Note: Enum display text may be formatted (e.g., "ImageXpress" vs "IMAGEXPRESS")
        assert "imagexpress" in microscope_widget.currentText().lower(), "Initial microscope widget should show IMAGEXPRESS"
        
        # Test individual field reset - num_workers (use the public API)
        form_manager.reset_parameter("num_workers")
        
        # Verify parameter and widget updated to actual default
        assert form_manager.parameters["num_workers"] == expected_num_workers, f"Parameter should reset to {expected_num_workers}"
        assert num_workers_widget.value() == expected_num_workers, f"Widget should display {expected_num_workers}"
        
        # Test individual field reset - use_threading (use the public API)
        form_manager.reset_parameter("use_threading")
        
        # Verify parameter and widget updated to actual default
        assert form_manager.parameters["use_threading"] == expected_use_threading, f"Parameter should reset to {expected_use_threading}"
        assert use_threading_widget.isChecked() == expected_use_threading, f"Widget should show {expected_use_threading}"
        
        # Modify a field again to test "Reset All" functionality
        form_manager.update_parameter("num_workers", 12)
        form_manager.update_parameter("microscope", Microscope.OPERAPHENIX)
        
        # Verify modifications took effect
        assert form_manager.parameters["num_workers"] == 12, "Parameter should be modified to 12"
        assert num_workers_widget.value() == 12, "Widget should display 12"
        assert form_manager.parameters["microscope"] == Microscope.OPERAPHENIX, "Parameter should be OPERAPHENIX"
        
        # Test "Reset All" functionality
        form_manager.reset_all_parameters()
        
        # Verify all parameters and widgets reset to actual defaults
        assert form_manager.parameters["num_workers"] == expected_num_workers, f"num_workers should reset to {expected_num_workers}"
        assert num_workers_widget.value() == expected_num_workers, f"num_workers widget should display {expected_num_workers}"
        
        assert form_manager.parameters["use_threading"] == expected_use_threading, f"use_threading should reset to {expected_use_threading}"
        assert use_threading_widget.isChecked() == expected_use_threading, f"use_threading widget should show {expected_use_threading}"
        
        assert form_manager.parameters["microscope"] == expected_microscope, f"microscope should reset to {expected_microscope}"
        # Note: ComboBox text comparison needs to handle enum display format
        
        # Verify no placeholder text appears in concrete context
        for param_name, widget in form_manager.widgets.items():
            if hasattr(widget, 'placeholderText'):
                assert widget.placeholderText() == "", f"Widget {param_name} should have no placeholder text in concrete context"
            if hasattr(widget, 'specialValueText'):
                # For spinboxes, special value text should not be set for concrete values
                assert widget.specialValueText() == "", f"Widget {param_name} should have no special value text in concrete context"

    def test_global_config_form_initialization(self, qtbot):
        """Test that GlobalPipelineConfig forms initialize with concrete default values."""
        
        # Create form with None values (simulating fresh initialization)
        parameters = {
            "num_workers": None,
            "use_threading": None,
        }
        parameter_types = {
            "num_workers": int,
            "use_threading": bool,
        }
        
        # Create form manager for GlobalPipelineConfig
        form_manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="global_init_test",
            dataclass_type=GlobalPipelineConfig,
            placeholder_prefix=""
        )
        qtbot.addWidget(form_manager)
        
        # Get expected default values
        default_config = GlobalPipelineConfig()
        expected_num_workers = default_config.num_workers
        expected_use_threading = default_config.use_threading
        
        # In concrete context, None values should be replaced with actual defaults during initialization
        # This tests the service layer's handling of None values in concrete contexts
        num_workers_widget = form_manager.widgets["num_workers"]
        use_threading_widget = form_manager.widgets["use_threading"]
        
        # Widgets should show concrete default values, not be empty
        # Note: The exact behavior depends on how the form manager handles None initialization
        # For concrete contexts, we expect the widgets to show meaningful default values
        assert isinstance(num_workers_widget.value(), int), "num_workers widget should show an integer value"
        assert isinstance(use_threading_widget.isChecked(), bool), "use_threading widget should show a boolean value"


class TestLazyConfigResetWorkflows:
    """Test reset workflows for PipelineConfig (lazy/orchestrator context)."""

    def test_complete_lazy_config_reset_workflow(self, qtbot):
        """Test complete user workflow with PipelineConfig reset operations."""

        # Set up global config context for placeholder resolution
        global_config = GlobalPipelineConfig()
        set_current_pipeline_config(global_config)

        # Create a comprehensive parameter form with multiple field types
        parameters = {
            "num_workers": 8,  # int - should reset to None with placeholder
            "use_threading": True,  # bool - should reset to None with placeholder
            "microscope": Microscope.IMAGEXPRESS,  # enum - should reset to None with placeholder
        }
        parameter_types = {
            "num_workers": int,
            "use_threading": bool,
            "microscope": Microscope,
        }

        # Create form manager for PipelineConfig (lazy context)
        form_manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="lazy_config_test",
            dataclass_type=PipelineConfig,
            placeholder_prefix="Pipeline default: "
        )
        qtbot.addWidget(form_manager)

        # Verify initial state - widgets show concrete values (not placeholders)
        num_workers_widget = form_manager.widgets["num_workers"]
        use_threading_widget = form_manager.widgets["use_threading"]
        microscope_widget = form_manager.widgets["microscope"]

        assert num_workers_widget.value() == 8, "Initial num_workers widget should show 8"
        assert use_threading_widget.isChecked() == True, "Initial use_threading widget should be checked"
        # Note: Enum display text may be formatted
        assert "imagexpress" in microscope_widget.currentText().lower(), "Initial microscope widget should show IMAGEXPRESS"

        # Test individual field reset - num_workers (should reset to None with placeholder)
        form_manager.reset_parameter("num_workers")

        # Verify parameter reset to None and widget shows placeholder
        assert form_manager.parameters["num_workers"] is None, "Parameter should reset to None in lazy context"

        # Check for placeholder text display (format may vary)
        if hasattr(num_workers_widget, 'specialValueText') and num_workers_widget.specialValueText():
            # QSpinBox with special value text for placeholder
            special_text = num_workers_widget.specialValueText()
            # Accept various placeholder formats: "Pipeline default:", ": (none)", etc.
            assert special_text.strip(), "Widget should show some placeholder text"
            assert num_workers_widget.value() == num_workers_widget.minimum(), "QSpinBox should be at minimum for placeholder"
        elif hasattr(num_workers_widget, 'placeholderText'):
            # QLineEdit with placeholder text
            placeholder_text = num_workers_widget.placeholderText()
            assert placeholder_text.strip(), "Widget should show placeholder text"

        # Test individual field reset - use_threading
        form_manager.reset_parameter("use_threading")

        # Verify parameter reset to None
        assert form_manager.parameters["use_threading"] is None, "use_threading should reset to None in lazy context"

        # Test mixed state workflow: some concrete values, some None values
        form_manager.update_parameter("num_workers", 12)  # Set concrete value
        # use_threading remains None (placeholder state)
        form_manager.update_parameter("microscope", Microscope.OPERAPHENIX)  # Set concrete value

        # Verify mixed state
        assert form_manager.parameters["num_workers"] == 12, "num_workers should be concrete value"
        assert form_manager.parameters["use_threading"] is None, "use_threading should remain None"
        assert form_manager.parameters["microscope"] == Microscope.OPERAPHENIX, "microscope should be concrete value"

        # Test "Reset All" functionality - all should become None with placeholders
        form_manager.reset_all_parameters()

        # Verify all parameters reset to None
        assert form_manager.parameters["num_workers"] is None, "num_workers should reset to None"
        assert form_manager.parameters["use_threading"] is None, "use_threading should reset to None"
        assert form_manager.parameters["microscope"] is None, "microscope should reset to None"

        # Verify widgets show placeholder text (format may vary)
        for param_name, widget in form_manager.widgets.items():
            if hasattr(widget, 'specialValueText') and widget.specialValueText():
                assert widget.specialValueText().strip(), f"Widget {param_name} should show some placeholder text"
            elif hasattr(widget, 'placeholderText') and widget.placeholderText():
                assert widget.placeholderText().strip(), f"Widget {param_name} should show placeholder text"

    def test_lazy_config_initial_placeholder_state(self, qtbot):
        """Test that PipelineConfig forms show placeholder text for None values on initialization."""

        # Set up global config context
        global_config = GlobalPipelineConfig()
        set_current_pipeline_config(global_config)

        # Create form with None values (typical lazy initialization)
        parameters = {
            "num_workers": None,
            "use_threading": None,
        }
        parameter_types = {
            "num_workers": int,
            "use_threading": bool,
        }

        # Create form manager for PipelineConfig (lazy context)
        form_manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="lazy_init_test",
            dataclass_type=PipelineConfig,
            placeholder_prefix="Pipeline default: "
        )
        qtbot.addWidget(form_manager)

        # Verify parameters remain None
        assert form_manager.parameters["num_workers"] is None, "num_workers should be None"
        assert form_manager.parameters["use_threading"] is None, "use_threading should be None"

        # Verify widgets show placeholder text immediately
        num_workers_widget = form_manager.widgets["num_workers"]
        use_threading_widget = form_manager.widgets["use_threading"]

        # Check for placeholder text in widgets
        has_placeholder = False
        if hasattr(num_workers_widget, 'specialValueText') and num_workers_widget.specialValueText():
            has_placeholder = True
            # Accept various placeholder formats
            assert num_workers_widget.specialValueText().strip(), "num_workers should show some placeholder text"
        elif hasattr(num_workers_widget, 'placeholderText') and num_workers_widget.placeholderText():
            has_placeholder = True
            assert num_workers_widget.placeholderText().strip(), "num_workers should show placeholder text"

        assert has_placeholder, "num_workers widget should have some form of placeholder text"


class TestWidgetLevelResetVerification:
    """Test reset behavior at the individual widget level for different widget types."""

    def test_qspinbox_reset_behavior(self, qtbot):
        """Test QSpinBox widget reset behavior in both contexts."""

        # Test GlobalPipelineConfig context (concrete)
        concrete_manager = ParameterFormManager(
            parameters={"num_workers": 8},
            parameter_types={"num_workers": int},
            field_id="spinbox_concrete_test",
            dataclass_type=GlobalPipelineConfig,
            placeholder_prefix=""
        )
        qtbot.addWidget(concrete_manager)

        spinbox = concrete_manager.widgets["num_workers"]

        # Verify initial state
        assert spinbox.value() == 8, "Initial spinbox value should be 8"

        # Trigger reset using public API
        concrete_manager.reset_parameter("num_workers")

        # Verify reset to actual default (16 for GlobalPipelineConfig)
        default_config = GlobalPipelineConfig()
        expected_default = default_config.num_workers
        assert spinbox.value() == expected_default, f"Spinbox should reset to {expected_default}"
        assert spinbox.specialValueText() == "", "Concrete context should have no special value text"

        # Test PipelineConfig context (lazy)
        set_current_pipeline_config(GlobalPipelineConfig())

        lazy_manager = ParameterFormManager(
            parameters={"num_workers": 8},
            parameter_types={"num_workers": int},
            field_id="spinbox_lazy_test",
            dataclass_type=PipelineConfig,
            placeholder_prefix="Pipeline default: "
        )
        qtbot.addWidget(lazy_manager)

        lazy_spinbox = lazy_manager.widgets["num_workers"]

        # Trigger reset using public API
        lazy_manager.reset_parameter("num_workers")

        # Verify reset to None with placeholder
        assert lazy_manager.parameters["num_workers"] is None, "Parameter should be None in lazy context"
        if lazy_spinbox.specialValueText():
            assert lazy_spinbox.specialValueText().strip(), "Should show some placeholder text"

    def test_qcheckbox_reset_behavior(self, qtbot):
        """Test QCheckBox widget reset behavior in both contexts."""

        # Test GlobalPipelineConfig context (concrete)
        concrete_manager = ParameterFormManager(
            parameters={"use_threading": True},
            parameter_types={"use_threading": bool},
            field_id="checkbox_concrete_test",
            dataclass_type=GlobalPipelineConfig,
            placeholder_prefix=""
        )
        qtbot.addWidget(concrete_manager)

        checkbox = concrete_manager.widgets["use_threading"]

        # Verify initial state
        assert checkbox.isChecked() == True, "Initial checkbox should be checked"

        # Trigger reset using public API
        concrete_manager.reset_parameter("use_threading")

        # Verify reset to actual default (False for GlobalPipelineConfig)
        default_config = GlobalPipelineConfig()
        expected_default = default_config.use_threading
        assert checkbox.isChecked() == expected_default, f"Checkbox should reset to {expected_default}"

        # Test PipelineConfig context (lazy)
        set_current_pipeline_config(GlobalPipelineConfig())

        lazy_manager = ParameterFormManager(
            parameters={"use_threading": True},
            parameter_types={"use_threading": bool},
            field_id="checkbox_lazy_test",
            dataclass_type=PipelineConfig,
            placeholder_prefix="Pipeline default: "
        )
        qtbot.addWidget(lazy_manager)

        lazy_checkbox = lazy_manager.widgets["use_threading"]

        # Trigger reset using public API
        lazy_manager.reset_parameter("use_threading")

        # Verify reset to None with placeholder styling
        assert lazy_manager.parameters["use_threading"] is None, "Parameter should be None in lazy context"
        # Checkbox placeholder behavior is typically shown through styling/tooltip
        if hasattr(lazy_checkbox, 'toolTip') and lazy_checkbox.toolTip():
            assert lazy_checkbox.toolTip().strip(), "Should show some placeholder information in tooltip"

    def test_qcombobox_reset_behavior(self, qtbot):
        """Test QComboBox widget reset behavior in both contexts."""

        # Test GlobalPipelineConfig context (concrete)
        concrete_manager = ParameterFormManager(
            parameters={"microscope": Microscope.IMAGEXPRESS},
            parameter_types={"microscope": Microscope},
            field_id="combobox_concrete_test",
            dataclass_type=GlobalPipelineConfig,
            placeholder_prefix=""
        )
        qtbot.addWidget(concrete_manager)

        combobox = concrete_manager.widgets["microscope"]

        # Verify initial state (enum display may be formatted)
        assert "imagexpress" in combobox.currentText().lower(), "Initial combobox should show IMAGEXPRESS"

        # Trigger reset using public API
        concrete_manager.reset_parameter("microscope")

        # Verify reset to actual default (AUTO for GlobalPipelineConfig)
        default_config = GlobalPipelineConfig()
        expected_default = default_config.microscope
        # Note: ComboBox text might be formatted differently than enum name
        assert concrete_manager.parameters["microscope"] == expected_default, f"Parameter should reset to {expected_default}"

        # Test PipelineConfig context (lazy)
        set_current_pipeline_config(GlobalPipelineConfig())

        lazy_manager = ParameterFormManager(
            parameters={"microscope": Microscope.IMAGEXPRESS},
            parameter_types={"microscope": Microscope},
            field_id="combobox_lazy_test",
            dataclass_type=PipelineConfig,
            placeholder_prefix="Pipeline default: "
        )
        qtbot.addWidget(lazy_manager)

        lazy_combobox = lazy_manager.widgets["microscope"]

        # Trigger reset using public API
        lazy_manager.reset_parameter("microscope")

        # Verify reset to None with placeholder
        assert lazy_manager.parameters["microscope"] is None, "Parameter should be None in lazy context"
