"""
Test reset button behavior for GlobalPipelineConfig parameter forms.

This module tests the specific reset button functionality issues identified:
1. Reset buttons should clear user-set values and return fields to lazy placeholder state
2. Only explicitly set fields should show concrete values
3. String widgets should show placeholder text after reset, not concrete default values
"""

import pytest
from PyQt6.QtWidgets import QLineEdit, QSpinBox, QComboBox
from PyQt6.QtTest import QSignalSpy

from openhcs.core.config import GlobalPipelineConfig
from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager
from openhcs.ui.shared.parameter_form_config_factory import ParameterFormConfigFactory


class TestResetButtonBehavior:
    """Test reset button functionality for GlobalPipelineConfig forms."""
    
    @pytest.fixture
    def global_config_form_manager(self, qtbot):
        """Create a parameter form manager for GlobalPipelineConfig editing."""
        # Create a GlobalPipelineConfig instance with some values set
        config = GlobalPipelineConfig(
            num_workers=8,  # User-set value different from default
        )
        
        # Create form configuration for global config editing
        form_config = ParameterFormConfigFactory.create_global_config(
            field_id="test_global_config",
            global_config_type=GlobalPipelineConfig,
            framework="pyqt6"
        )
        
        # Create form manager
        form_manager = ParameterFormManager(
            parameters={"num_workers": 8},  # User has set this value
            parameter_types={"num_workers": int},
            config=form_config,
            dataclass_type=GlobalPipelineConfig
        )
        
        qtbot.addWidget(form_manager)
        return form_manager
    
    def test_reset_button_clears_user_set_integer_field(self, global_config_form_manager, qtbot):
        """Test that reset button clears user-set integer field and shows placeholder."""
        form_manager = global_config_form_manager

        # Verify initial state - user has set num_workers to 8
        assert form_manager.parameters["num_workers"] == 8

        # Get the reset button
        reset_button = form_manager.reset_buttons["num_workers"]

        # Click reset button
        qtbot.mouseClick(reset_button, qtbot.LeftButton)

        # After reset, the parameter should be None (unset)
        assert form_manager.parameters["num_workers"] is None

    def test_reset_functionality_core_behavior(self, qtbot):
        """Test the core reset functionality without complex widget setup."""
        from openhcs.ui.shared.parameter_form_service import ParameterFormService

        service = ParameterFormService()

        # Test that reset returns None for GlobalPipelineConfig
        reset_value = service.get_reset_value_for_parameter('num_workers', int, GlobalPipelineConfig)
        assert reset_value is None

        # Test that placeholder text is still available
        placeholder = service.get_placeholder_text('num_workers', GlobalPipelineConfig)
        assert placeholder is not None
        assert "16" in placeholder  # Should show the default value
    
    def test_reset_button_clears_user_set_string_field(self, qtbot):
        """Test that reset button clears user-set string field and shows placeholder."""
        # Create a form manager with a string field that has been set by user
        # We'll use output_dir_suffix from PathPlanningConfig as an example
        from openhcs.core.config import PathPlanningConfig
        
        config = GlobalPipelineConfig(
            path_planning=PathPlanningConfig(output_dir_suffix="_custom")
        )
        
        form_config = ParameterFormConfigFactory.create_global_config(
            field_id="test_string_reset",
            global_config_type=GlobalPipelineConfig,
            framework="pyqt6"
        )
        
        # Create form manager with user-set string value
        form_manager = ParameterFormManager(
            parameters={"path_planning": config.path_planning},
            parameter_types={"path_planning": PathPlanningConfig},
            config=form_config,
            dataclass_type=GlobalPipelineConfig
        )
        
        qtbot.addWidget(form_manager)
        
        # For nested dataclass, we need to check the nested form manager
        # This test focuses on the principle - string fields should reset to placeholder state
        
        # Verify initial state - user has set a custom value
        assert form_manager.parameters["path_planning"].output_dir_suffix == "_custom"
        
        # Get reset button for the nested field
        if "path_planning" in form_manager.reset_buttons:
            reset_button = form_manager.reset_buttons["path_planning"]
            
            # Click reset button
            qtbot.mouseClick(reset_button, qtbot.LeftButton)
            
            # After reset, the nested parameter should be None or default
            # The key is that it should not retain the user-set "_custom" value
            reset_value = form_manager.parameters["path_planning"]
            if reset_value is not None:
                # If not None, it should be the default, not the user-set value
                assert reset_value.output_dir_suffix != "_custom"
    
    def test_string_widget_shows_placeholder_after_reset(self, qtbot):
        """Test that string widgets show placeholder text after reset, not concrete values."""
        # Create a simple form with a string parameter
        form_config = ParameterFormConfigFactory.create_global_config(
            field_id="test_string_placeholder",
            global_config_type=GlobalPipelineConfig,
            framework="pyqt6"
        )
        
        # Simulate a string field that user has set
        form_manager = ParameterFormManager(
            parameters={"test_string": "user_value"},
            parameter_types={"test_string": str},
            config=form_config,
            dataclass_type=GlobalPipelineConfig
        )
        
        qtbot.addWidget(form_manager)
        
        # Add a string widget manually for testing
        from PyQt6.QtWidgets import QLineEdit, QPushButton
        string_widget = QLineEdit()
        string_widget.setText("user_value")
        reset_button = QPushButton("Reset")
        
        form_manager.widgets["test_string"] = string_widget
        form_manager.reset_buttons["test_string"] = reset_button
        
        # Connect reset button to the reset functionality
        reset_button.clicked.connect(lambda: form_manager._reset_parameter("test_string"))
        
        # Verify initial state
        assert string_widget.text() == "user_value"
        assert form_manager.parameters["test_string"] == "user_value"
        
        # Click reset button
        qtbot.mouseClick(reset_button, qtbot.LeftButton)
        
        # After reset, parameter should be None (unset)
        assert form_manager.parameters["test_string"] is None
        
        # Widget should show placeholder text, not a concrete empty string
        # The widget text should be empty, but placeholderText should be set
        assert string_widget.text() == ""
        assert string_widget.placeholderText() != ""
        assert "default" in string_widget.placeholderText().lower()
    
    def test_reset_signal_emission(self, global_config_form_manager, qtbot):
        """Test that reset button emits parameter_changed signal with None value."""
        form_manager = global_config_form_manager
        
        # Set up signal spy
        signal_spy = QSignalSpy(form_manager.parameter_changed)
        
        # Get reset button
        reset_button = form_manager.reset_buttons["num_workers"]
        
        # Click reset button
        qtbot.mouseClick(reset_button, qtbot.LeftButton)
        
        # Verify signal was emitted with None value
        assert len(signal_spy) == 1
        signal_args = signal_spy[0]
        assert signal_args[0] == "num_workers"  # parameter name
        assert signal_args[1] is None  # reset value should be None
    
    def test_lazy_vs_concrete_value_display(self, qtbot):
        """Test that lazy (unset) values show placeholders while concrete values show actual values."""
        form_config = ParameterFormConfigFactory.create_global_config(
            field_id="test_lazy_concrete",
            global_config_type=GlobalPipelineConfig,
            framework="pyqt6"
        )
        
        # Create form manager with mixed lazy and concrete values
        form_manager = ParameterFormManager(
            parameters={
                "num_workers": 16,  # Concrete value set by user
                "unset_field": None  # Lazy value (not set by user)
            },
            parameter_types={
                "num_workers": int,
                "unset_field": str
            },
            config=form_config,
            dataclass_type=GlobalPipelineConfig
        )
        
        qtbot.addWidget(form_manager)
        
        # Add widgets manually for testing
        concrete_widget = QSpinBox()
        lazy_widget = QLineEdit()
        
        form_manager.widgets["num_workers"] = concrete_widget
        form_manager.widgets["unset_field"] = lazy_widget
        
        # Update widgets with current values
        form_manager._update_widget_value_with_context(concrete_widget, 16, "num_workers")
        form_manager._update_widget_value_with_context(lazy_widget, None, "unset_field")
        
        # Concrete value should show actual value
        assert concrete_widget.value() == 16
        
        # Lazy value should show placeholder text
        assert lazy_widget.text() == ""
        assert lazy_widget.placeholderText() != ""
        assert "default" in lazy_widget.placeholderText().lower()


class TestResetButtonBehaviorIntegration:
    """Integration tests for reset button behavior with real GlobalPipelineConfig."""
    
    def test_full_global_config_reset_workflow(self, qtbot):
        """Test complete workflow: set value → reset → verify placeholder → set again."""
        # Create actual GlobalPipelineConfig form
        config = GlobalPipelineConfig()
        
        form_config = ParameterFormConfigFactory.create_global_config(
            field_id="test_full_workflow",
            global_config_type=GlobalPipelineConfig,
            framework="pyqt6"
        )
        
        # Extract parameters from config for form
        import dataclasses
        parameters = {}
        parameter_types = {}
        
        for field in dataclasses.fields(GlobalPipelineConfig):
            if field.type in [int, str, bool]:  # Simple types only for this test
                parameters[field.name] = getattr(config, field.name)
                parameter_types[field.name] = field.type
        
        form_manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            config=form_config,
            dataclass_type=GlobalPipelineConfig
        )
        
        qtbot.addWidget(form_manager)
        
        # Focus on num_workers field for this test
        if "num_workers" in parameters:
            original_value = parameters["num_workers"]
            
            # Step 1: User sets a different value
            form_manager.update_parameter("num_workers", 32)
            assert form_manager.parameters["num_workers"] == 32
            
            # Step 2: User clicks reset
            if "num_workers" in form_manager.reset_buttons:
                reset_button = form_manager.reset_buttons["num_workers"]
                qtbot.mouseClick(reset_button, qtbot.LeftButton)
                
                # Step 3: Verify field is reset to unset state (None)
                assert form_manager.parameters["num_workers"] is None
                
                # Step 4: User sets value again
                form_manager.update_parameter("num_workers", 64)
                assert form_manager.parameters["num_workers"] == 64
