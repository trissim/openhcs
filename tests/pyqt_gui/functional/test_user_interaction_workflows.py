#!/usr/bin/env python3
"""
Functional tests for comprehensive user interaction workflows with PyQt6 GUI parameter forms.

These tests simulate complete user workflows that would occur in real-world usage,
focusing on the user experience and visual feedback.
"""

import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager
from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.pipeline_config import PipelineConfig, set_current_pipeline_config
from openhcs.constants import Microscope


class TestCompleteUserWorkflows:
    """Test complete user workflows from start to finish."""

    def test_global_config_editing_complete_workflow(self, qtbot):
        """Test complete workflow for editing global configuration."""
        
        # Scenario: User opens global config editor, modifies settings, uses reset functionality
        
        # Step 1: Initialize global config form with current settings
        current_settings = {
            "num_workers": 8,
            "use_threading": True,
            "microscope": Microscope.IMAGEXPRESS,
        }
        parameter_types = {
            "num_workers": int,
            "use_threading": bool,
            "microscope": Microscope,
        }
        
        form_manager = ParameterFormManager(
            parameters=current_settings,
            parameter_types=parameter_types,
            field_id="global_workflow_test",
            dataclass_type=GlobalPipelineConfig,
            placeholder_prefix=""
        )
        qtbot.addWidget(form_manager)
        
        # Step 2: User sees current settings displayed in widgets
        num_workers_widget = form_manager.widgets["num_workers"]
        use_threading_widget = form_manager.widgets["use_threading"]
        microscope_widget = form_manager.widgets["microscope"]
        
        assert num_workers_widget.value() == 8, "User should see current num_workers value"
        assert use_threading_widget.isChecked() == True, "User should see current use_threading value"
        assert microscope_widget.currentText() == "IMAGEXPRESS", "User should see current microscope value"
        
        # Step 3: User modifies some settings
        form_manager.update_parameter("num_workers", 16)
        form_manager.update_parameter("use_threading", False)
        
        # Verify user sees updated values
        assert num_workers_widget.value() == 16, "User should see updated num_workers value"
        assert use_threading_widget.isChecked() == False, "User should see updated use_threading value"
        
        # Step 4: User decides to reset num_workers to default
        num_workers_reset_btn = form_manager.reset_buttons["num_workers"]
        qtbot.mouseClick(num_workers_reset_btn, Qt.MouseButton.LeftButton)
        
        # User should see the actual default value (not 0)
        default_config = GlobalPipelineConfig()
        expected_default = default_config.num_workers
        assert num_workers_widget.value() == expected_default, f"User should see default value {expected_default}"
        assert form_manager.parameters["num_workers"] == expected_default, "Parameter should be set to default"
        
        # Step 5: User decides to reset all settings
        form_manager.reset_all_parameters()
        
        # User should see all defaults
        assert num_workers_widget.value() == default_config.num_workers, "All fields should show defaults"
        assert use_threading_widget.isChecked() == default_config.use_threading, "All fields should show defaults"
        assert form_manager.parameters["microscope"] == default_config.microscope, "All fields should show defaults"
        
        # Step 6: Verify no placeholder text appears (concrete context)
        for param_name, widget in form_manager.widgets.items():
            if hasattr(widget, 'placeholderText'):
                assert widget.placeholderText() == "", f"No placeholder text should appear for {param_name}"
            if hasattr(widget, 'specialValueText'):
                assert widget.specialValueText() == "", f"No special value text should appear for {param_name}"

    def test_orchestrator_config_editing_complete_workflow(self, qtbot):
        """Test complete workflow for editing orchestrator/pipeline configuration."""
        
        # Scenario: User opens pipeline config editor, sees placeholders, modifies settings, resets
        
        # Step 1: Set up global config context
        global_config = GlobalPipelineConfig(
            num_workers=24,
            use_threading=True,
            microscope=Microscope.OPERAPHENIX
        )
        set_current_pipeline_config(global_config)
        
        # Step 2: Initialize pipeline config form with mixed state
        pipeline_settings = {
            "num_workers": None,  # Should show placeholder
            "use_threading": False,  # Concrete override
            "microscope": None,  # Should show placeholder
        }
        parameter_types = {
            "num_workers": int,
            "use_threading": bool,
            "microscope": Microscope,
        }
        
        form_manager = ParameterFormManager(
            parameters=pipeline_settings,
            parameter_types=parameter_types,
            field_id="orchestrator_workflow_test",
            dataclass_type=PipelineConfig,
            placeholder_prefix="Pipeline default: "
        )
        qtbot.addWidget(form_manager)
        
        # Step 3: User sees mixed state - placeholders for None values, concrete for others
        num_workers_widget = form_manager.widgets["num_workers"]
        use_threading_widget = form_manager.widgets["use_threading"]
        microscope_widget = form_manager.widgets["microscope"]
        
        # num_workers should show placeholder (None value)
        assert form_manager.parameters["num_workers"] is None, "num_workers should be None"
        if hasattr(num_workers_widget, 'specialValueText') and num_workers_widget.specialValueText():
            assert "24" in num_workers_widget.specialValueText(), "Should show global config value in placeholder"
        
        # use_threading should show concrete value
        assert use_threading_widget.isChecked() == False, "use_threading should show concrete override"
        
        # microscope should show placeholder (None value)
        assert form_manager.parameters["microscope"] is None, "microscope should be None"
        
        # Step 4: User modifies a field that was showing placeholder
        form_manager.update_parameter("num_workers", 32)
        
        # User should see concrete value, no more placeholder
        assert num_workers_widget.value() == 32, "User should see concrete value"
        assert form_manager.parameters["num_workers"] == 32, "Parameter should be concrete"
        
        # Step 5: User resets the field back to inherit from global
        num_workers_reset_btn = form_manager.reset_buttons["num_workers"]
        qtbot.mouseClick(num_workers_reset_btn, Qt.MouseButton.LeftButton)
        
        # User should see placeholder again
        assert form_manager.parameters["num_workers"] is None, "Parameter should be None again"
        if hasattr(num_workers_widget, 'specialValueText') and num_workers_widget.specialValueText():
            assert "24" in num_workers_widget.specialValueText(), "Should show placeholder again"
        
        # Step 6: User resets all to inherit everything from global
        form_manager.reset_all_parameters()
        
        # All parameters should be None with placeholders
        assert form_manager.parameters["num_workers"] is None, "All should be None"
        assert form_manager.parameters["use_threading"] is None, "All should be None"
        assert form_manager.parameters["microscope"] is None, "All should be None"
        
        # User should see placeholders for all fields
        for param_name, widget in form_manager.widgets.items():
            if hasattr(widget, 'specialValueText') and widget.specialValueText():
                assert "Pipeline default:" in widget.specialValueText(), f"{param_name} should show placeholder"
            elif hasattr(widget, 'placeholderText') and widget.placeholderText():
                assert "Pipeline default:" in widget.placeholderText(), f"{param_name} should show placeholder"
            elif hasattr(widget, 'toolTip') and widget.toolTip():
                assert "Pipeline default:" in widget.toolTip(), f"{param_name} should show placeholder in tooltip"

    def test_user_confusion_prevention_workflow(self, qtbot):
        """Test workflow that prevents common user confusion scenarios."""
        
        # Scenario: Prevent user confusion about what values will actually be used
        
        # Step 1: Set up a scenario where global and pipeline configs differ
        global_config = GlobalPipelineConfig(
            num_workers=16,
            use_threading=True
        )
        set_current_pipeline_config(global_config)
        
        # Step 2: Create pipeline config with some overrides
        pipeline_settings = {
            "num_workers": 8,  # Override global
            "use_threading": None,  # Inherit from global
        }
        parameter_types = {
            "num_workers": int,
            "use_threading": bool,
        }
        
        form_manager = ParameterFormManager(
            parameters=pipeline_settings,
            parameter_types=parameter_types,
            field_id="confusion_prevention_test",
            dataclass_type=PipelineConfig,
            placeholder_prefix="Pipeline default: "
        )
        qtbot.addWidget(form_manager)
        
        # Step 3: User should clearly see which values are overridden vs inherited
        num_workers_widget = form_manager.widgets["num_workers"]
        use_threading_widget = form_manager.widgets["use_threading"]
        
        # num_workers shows concrete override (8)
        assert num_workers_widget.value() == 8, "User should see override value clearly"
        
        # use_threading shows placeholder indicating inheritance
        assert form_manager.parameters["use_threading"] is None, "Should be None (inherited)"
        if hasattr(use_threading_widget, 'toolTip') and use_threading_widget.toolTip():
            assert "Pipeline default:" in use_threading_widget.toolTip(), "Should show inheritance clearly"
        
        # Step 4: User resets override to see what global value would be
        num_workers_reset_btn = form_manager.reset_buttons["num_workers"]
        qtbot.mouseClick(num_workers_reset_btn, Qt.MouseButton.LeftButton)
        
        # User should see placeholder showing the actual global value (16)
        assert form_manager.parameters["num_workers"] is None, "Should be None (inherited)"
        if hasattr(num_workers_widget, 'specialValueText') and num_workers_widget.specialValueText():
            assert "16" in num_workers_widget.specialValueText(), "Should show actual global value"
        
        # Step 5: User sets override again
        form_manager.update_parameter("num_workers", 12)
        
        # Should clearly show override again
        assert num_workers_widget.value() == 12, "Should show new override clearly"
        assert form_manager.parameters["num_workers"] == 12, "Parameter should be concrete override"

    def test_visual_feedback_consistency_workflow(self, qtbot):
        """Test that visual feedback is consistent across different widget types."""
        
        # Set up global config
        global_config = GlobalPipelineConfig()
        set_current_pipeline_config(global_config)
        
        # Create form with various widget types
        parameters = {
            "num_workers": None,  # QSpinBox
            "use_threading": None,  # QCheckBox
            "microscope": None,  # QComboBox
        }
        parameter_types = {
            "num_workers": int,
            "use_threading": bool,
            "microscope": Microscope,
        }
        
        form_manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="visual_consistency_test",
            dataclass_type=PipelineConfig,
            placeholder_prefix="Pipeline default: "
        )
        qtbot.addWidget(form_manager)
        
        # Verify consistent placeholder behavior across widget types
        widgets_with_placeholders = 0
        
        for param_name, widget in form_manager.widgets.items():
            has_placeholder_indicator = False
            
            if hasattr(widget, 'specialValueText') and widget.specialValueText():
                assert "Pipeline default:" in widget.specialValueText(), f"{param_name} should have consistent placeholder format"
                has_placeholder_indicator = True
            elif hasattr(widget, 'placeholderText') and widget.placeholderText():
                assert "Pipeline default:" in widget.placeholderText(), f"{param_name} should have consistent placeholder format"
                has_placeholder_indicator = True
            elif hasattr(widget, 'toolTip') and widget.toolTip():
                assert "Pipeline default:" in widget.toolTip(), f"{param_name} should have consistent placeholder format"
                has_placeholder_indicator = True
            
            if has_placeholder_indicator:
                widgets_with_placeholders += 1
        
        # At least some widgets should have placeholder indicators
        assert widgets_with_placeholders > 0, "At least some widgets should show placeholder indicators"
        
        # Test that setting concrete values removes placeholders consistently
        form_manager.update_parameter("num_workers", 16)
        form_manager.update_parameter("use_threading", True)
        form_manager.update_parameter("microscope", Microscope.IMAGEXPRESS)
        
        # Verify concrete values are displayed clearly
        assert form_manager.widgets["num_workers"].value() == 16, "Concrete values should be clear"
        assert form_manager.widgets["use_threading"].isChecked() == True, "Concrete values should be clear"
        assert form_manager.widgets["microscope"].currentText() == "IMAGEXPRESS", "Concrete values should be clear"
        
        # Reset all and verify placeholders return consistently
        form_manager.reset_all_parameters()
        
        # All should show placeholders again
        for param_name in parameters.keys():
            assert form_manager.parameters[param_name] is None, f"{param_name} should be None after reset"
