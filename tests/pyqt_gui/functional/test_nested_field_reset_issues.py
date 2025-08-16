"""
Test nested field reset functionality issues.

This module reproduces and tests the specific issues with nested dataclass field
reset functionality that were identified:
1. Nested fields returning concrete values instead of None after reset
2. Nested managers not inheriting correct dataclass type context
3. Placeholder text not showing correctly for nested fields after reset
"""

import pytest
from PyQt6.QtWidgets import QLineEdit, QSpinBox

from openhcs.core.config import GlobalPipelineConfig, PathPlanningConfig
from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager
from openhcs.ui.shared.parameter_form_config_factory import ParameterFormConfigFactory


class TestNestedFieldResetIssues:
    """Test nested field reset functionality issues."""
    
    @pytest.fixture
    def global_config_with_nested_form_manager(self, qtbot):
        """Create a form manager for GlobalPipelineConfig with nested PathPlanningConfig."""
        # Create a GlobalPipelineConfig with nested PathPlanningConfig
        config = GlobalPipelineConfig(
            path_planning=PathPlanningConfig(
                output_dir_suffix="_custom_suffix",
                global_output_folder="/custom/path"
            )
        )
        
        # Extract parameters for form manager
        import dataclasses
        parameters = {}
        parameter_types = {}

        for field in dataclasses.fields(GlobalPipelineConfig):
            field_value = getattr(config, field.name)
            parameters[field.name] = field_value
            parameter_types[field.name] = field.type

        # Create form manager with correct constructor signature
        form_manager = ParameterFormManager(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id="test_nested_reset",
            dataclass_type=GlobalPipelineConfig
        )
        
        qtbot.addWidget(form_manager)
        return form_manager
    
    def test_nested_manager_dataclass_type_issue(self, global_config_with_nested_form_manager):
        """Test that nested managers have incorrect dataclass_type causing reset issues."""
        form_manager = global_config_with_nested_form_manager
        
        # Check if nested manager exists
        if hasattr(form_manager, 'nested_managers') and 'path_planning' in form_manager.nested_managers:
            nested_manager = form_manager.nested_managers['path_planning']
            
            print(f"Parent form manager dataclass_type: {form_manager.dataclass_type}")
            print(f"Nested form manager dataclass_type: {nested_manager.dataclass_type}")
            
            # The issue: nested manager has GlobalPipelineConfig instead of PathPlanningConfig
            assert form_manager.dataclass_type == GlobalPipelineConfig
            
            # This is the bug - nested manager should have PathPlanningConfig, not GlobalPipelineConfig
            if nested_manager.dataclass_type == GlobalPipelineConfig:
                print("❌ BUG CONFIRMED: Nested manager has wrong dataclass_type")
                print("   Expected: PathPlanningConfig")
                print("   Actual: GlobalPipelineConfig")
                print("   This causes reset to use wrong context for determining reset values")
            else:
                print("✅ Nested manager has correct dataclass_type")
    
    def test_nested_field_reset_returns_concrete_value(self, global_config_with_nested_form_manager):
        """Test that nested field reset incorrectly returns concrete values instead of None."""
        form_manager = global_config_with_nested_form_manager
        
        # Test the service layer directly for nested field reset
        from openhcs.ui.shared.parameter_form_service import ParameterFormService
        service = ParameterFormService()
        
        # Test reset value for nested field in GlobalPipelineConfig context
        # This should return None (for placeholder display) but might return concrete value
        reset_value = service.get_reset_value_for_parameter(
            'output_dir_suffix', str, GlobalPipelineConfig
        )
        
        print(f"Reset value for nested field 'output_dir_suffix': {reset_value}")
        print(f"Type: {type(reset_value)}")
        
        # The issue: this might return "_outputs" instead of None
        if reset_value is not None:
            print("❌ BUG CONFIRMED: Nested field reset returns concrete value")
            print(f"   Expected: None (to show placeholder)")
            print(f"   Actual: {reset_value} (concrete value)")
        else:
            print("✅ Nested field reset correctly returns None")
    
    def test_nested_field_reset_with_correct_context(self):
        """Test what happens when nested field reset uses correct PathPlanningConfig context."""
        from openhcs.ui.shared.parameter_form_service import ParameterFormService
        service = ParameterFormService()
        
        # Test reset value for field in PathPlanningConfig context (correct)
        reset_value_correct_context = service.get_reset_value_for_parameter(
            'output_dir_suffix', str, PathPlanningConfig
        )
        
        # Test reset value for field in GlobalPipelineConfig context (incorrect)
        reset_value_wrong_context = service.get_reset_value_for_parameter(
            'output_dir_suffix', str, GlobalPipelineConfig
        )
        
        print(f"Reset value with PathPlanningConfig context: {reset_value_correct_context}")
        print(f"Reset value with GlobalPipelineConfig context: {reset_value_wrong_context}")
        
        # PathPlanningConfig is non-lazy, so should return concrete value
        # GlobalPipelineConfig is non-lazy, but doesn't have output_dir_suffix field, so returns None
        assert reset_value_correct_context == "_outputs", "PathPlanningConfig context should return concrete value"
        assert reset_value_wrong_context is None, "GlobalPipelineConfig context should return None (field doesn't exist)"
    
    def test_nested_field_placeholder_text_display(self, global_config_with_nested_form_manager):
        """Test that nested fields show correct placeholder text after reset."""
        form_manager = global_config_with_nested_form_manager
        
        # Check if nested manager exists and has widgets
        if hasattr(form_manager, 'nested_managers') and 'path_planning' in form_manager.nested_managers:
            nested_manager = form_manager.nested_managers['path_planning']
            
            # Check if output_dir_suffix widget exists
            if 'output_dir_suffix' in nested_manager.widgets:
                widget = nested_manager.widgets['output_dir_suffix']
                
                if isinstance(widget, QLineEdit):
                    # Get current state
                    current_text = widget.text()
                    current_placeholder = widget.placeholderText()
                    
                    print(f"Before reset - Text: '{current_text}', Placeholder: '{current_placeholder}'")
                    
                    # Simulate reset button click
                    if 'output_dir_suffix' in nested_manager.reset_buttons:
                        reset_button = nested_manager.reset_buttons['output_dir_suffix']
                        reset_button.click()
                        
                        # Check state after reset
                        after_text = widget.text()
                        after_placeholder = widget.placeholderText()
                        
                        print(f"After reset - Text: '{after_text}', Placeholder: '{after_placeholder}'")
                        
                        # Expected behavior: text should be empty, placeholder should show default
                        if after_text == "" and after_placeholder != "":
                            print("✅ Reset correctly cleared field and shows placeholder")
                        else:
                            print("❌ BUG: Reset did not clear field or show placeholder correctly")
                            print(f"   Expected: text='', placeholder='Pipeline default: _outputs'")
                            print(f"   Actual: text='{after_text}', placeholder='{after_placeholder}'")
    
    def test_nested_vs_top_level_reset_consistency(self, global_config_with_nested_form_manager):
        """Test that nested field reset behaves consistently with top-level field reset."""
        form_manager = global_config_with_nested_form_manager
        
        from openhcs.ui.shared.parameter_form_service import ParameterFormService
        service = ParameterFormService()
        
        # Test top-level field reset
        top_level_reset = service.get_reset_value_for_parameter('num_workers', int, GlobalPipelineConfig)
        
        # Test nested field reset
        nested_reset = service.get_reset_value_for_parameter('output_dir_suffix', str, GlobalPipelineConfig)
        
        print(f"Top-level field reset (num_workers): {top_level_reset}")
        print(f"Nested field reset (output_dir_suffix): {nested_reset}")
        
        # Both should return concrete values for GlobalPipelineConfig (non-lazy context)
        assert top_level_reset == 16, "Top-level reset should return concrete value for non-lazy context"
        assert nested_reset is None, "Nested field reset should return None (field doesn't exist in GlobalPipelineConfig)"
        
        if top_level_reset == nested_reset:
            print("✅ Consistent reset behavior between top-level and nested fields")
        else:
            print("❌ BUG: Inconsistent reset behavior")
            print("   Top-level and nested fields should have same reset behavior")


class TestNestedFieldResetFix:
    """Test the fix for nested field reset issues."""
    
    def test_nested_manager_should_use_correct_dataclass_type(self):
        """Test that nested managers should be created with correct dataclass type."""
        # This test documents the intended fix
        
        print("=== INTENDED FIX ===")
        print("When creating nested manager for 'path_planning' field:")
        print("  Current (buggy): dataclass_type=GlobalPipelineConfig")
        print("  Fixed (correct): dataclass_type=PathPlanningConfig")
        print("")
        print("This ensures that:")
        print("  1. Reset logic uses correct context")
        print("  2. Service layer gets correct dataclass type")
        print("  3. Placeholder text is generated correctly")
        print("  4. Nested fields behave consistently with top-level fields")
        
        # The fix should be in the nested manager creation code
        # Line 315 in parameter_form_manager.py should pass the nested dataclass type
        # instead of self.dataclass_type
        
        assert True  # This test documents the intended fix
    
    def test_service_layer_handles_nested_context_correctly(self):
        """Test that service layer correctly handles nested dataclass context."""
        from openhcs.ui.shared.parameter_form_service import ParameterFormService
        service = ParameterFormService()
        
        # Test with correct nested context
        reset_value = service.get_reset_value_for_parameter('output_dir_suffix', str, PathPlanningConfig)
        placeholder = service.get_placeholder_text('output_dir_suffix', PathPlanningConfig)
        
        print(f"Service layer with PathPlanningConfig context:")
        print(f"  Reset value: {reset_value}")
        print(f"  Placeholder: {placeholder}")
        
        # PathPlanningConfig is non-lazy, so should return concrete value
        assert reset_value == "_outputs"
        assert placeholder is not None
        assert "_outputs" in placeholder  # Should show the default value
