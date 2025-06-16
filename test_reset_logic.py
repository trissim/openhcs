#!/usr/bin/env python3
"""
Unit test to verify reset logic in parameter form manager.
"""

from openhcs.textual_tui.widgets.shared.parameter_form_manager import ParameterFormManager
from openhcs.textual_tui.services.config_reflection_service import FieldIntrospector
from openhcs.core.config import GlobalPipelineConfig


def test_reset_logic():
    """Test the reset parameter logic."""
    print("Testing reset parameter logic...")
    
    # Create a test config
    config = GlobalPipelineConfig()
    print(f"Original num_workers: {config.num_workers}")
    
    # Analyze the config to get field specs
    field_specs = FieldIntrospector().analyze_dataclass(GlobalPipelineConfig, config)
    
    # Convert to parameter form manager format
    parameters = {}
    parameter_types = {}
    param_defaults = {}
    
    for spec in field_specs:
        parameters[spec.name] = spec.current_value
        parameter_types[spec.name] = spec.actual_type
        param_defaults[spec.name] = spec.default_value
        print(f"Field {spec.name}: current={spec.current_value}, default={spec.default_value}")
    
    # Create form manager
    form_manager = ParameterFormManager(parameters, parameter_types, "test")
    
    # Test 1: Modify a parameter
    print(f"\n=== Test 1: Modify num_workers ===")
    original_value = form_manager.parameters['num_workers']
    print(f"Original value: {original_value}")
    
    # Change the value
    form_manager.update_parameter('num_workers', 8)
    modified_value = form_manager.parameters['num_workers']
    print(f"Modified value: {modified_value}")
    
    # Test 2: Reset the parameter
    print(f"\n=== Test 2: Reset num_workers ===")
    default_value = param_defaults['num_workers']
    print(f"Default value: {default_value}")
    
    form_manager.reset_parameter('num_workers', default_value)
    reset_value = form_manager.parameters['num_workers']
    print(f"Reset value: {reset_value}")
    
    # Verify the reset worked
    if reset_value == default_value:
        print("✅ Reset logic works correctly!")
    else:
        print("❌ Reset logic failed!")
        print(f"Expected: {default_value}, Got: {reset_value}")
    
    return reset_value == default_value


if __name__ == "__main__":
    success = test_reset_logic()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")
