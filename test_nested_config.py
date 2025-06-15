#!/usr/bin/env python3
"""
Test script to verify nested dataclass config editing works.
"""

from dataclasses import dataclass
from openhcs.core.config import GlobalPipelineConfig, PathPlanningConfig
from openhcs.textual_tui.services.config_reflection_service import FieldIntrospector
from openhcs.textual_tui.widgets.shared.parameter_form_manager import ParameterFormManager

def test_nested_config_editing():
    """Test that nested dataclass editing works correctly."""
    
    print("=== Testing Nested Dataclass Config Editing ===")
    
    # Create a test config with nested dataclass
    path_config = PathPlanningConfig(
        output_dir_suffix="_outputs",
        global_output_folder="/original/path"
    )
    global_config = GlobalPipelineConfig(path_planning=path_config)
    
    print(f"Original config:")
    print(f"  global_output_folder: {global_config.path_planning.global_output_folder}")
    print(f"  output_dir_suffix: {global_config.path_planning.output_dir_suffix}")
    
    # Test 1: FieldIntrospector analysis
    print("\n=== Test 1: FieldIntrospector Analysis ===")
    field_specs = FieldIntrospector.analyze_dataclass(GlobalPipelineConfig, global_config)
    
    for spec in field_specs:
        print(f"Field: {spec.name}, Type: {spec.actual_type}, Nested: {spec.is_nested_dataclass}")
        if spec.is_nested_dataclass and spec.nested_fields:
            for nested_spec in spec.nested_fields:
                print(f"  Nested: {nested_spec.name} = {nested_spec.current_value}")
    
    # Test 2: ParameterFormManager
    print("\n=== Test 2: ParameterFormManager ===")
    
    # Convert FieldSpec to ParameterFormManager format
    parameters = {}
    parameter_types = {}
    
    for spec in field_specs:
        parameters[spec.name] = spec.current_value
        parameter_types[spec.name] = spec.actual_type
    
    # Create form manager
    form_manager = ParameterFormManager(parameters, parameter_types, "config")

    # IMPORTANT: Must manually create nested managers since build_form() requires Textual app
    print("Manually creating nested managers...")
    form_manager._create_nested_managers_for_testing()

    print("Initial parameters:")
    for name, value in form_manager.parameters.items():
        print(f"  {name}: {value}")
    
    # Test 3: Update nested parameter
    print("\n=== Test 3: Update Nested Parameter ===")
    
    # Check if nested managers were created
    print(f"Has nested_managers: {hasattr(form_manager, 'nested_managers')}")
    if hasattr(form_manager, 'nested_managers'):
        print(f"Nested managers: {list(form_manager.nested_managers.keys())}")

    # Simulate updating the global_output_folder field
    nested_field_name = "path_planning_global_output_folder"
    new_value = "/new/global/path"

    print(f"Updating {nested_field_name} to {new_value}")

    # Debug the parsing
    parts = nested_field_name.split('_')
    print(f"Parts: {parts}")
    for i in range(1, len(parts)):
        potential_nested = '_'.join(parts[:i])
        print(f"  Checking potential_nested: '{potential_nested}' in parameters: {potential_nested in form_manager.parameters}")
        if hasattr(form_manager, 'nested_managers'):
            print(f"    In nested_managers: {potential_nested in form_manager.nested_managers}")

    form_manager.update_parameter(nested_field_name, new_value)
    
    print("Updated parameters:")
    for name, value in form_manager.parameters.items():
        print(f"  {name}: {value}")
        if hasattr(value, 'global_output_folder'):
            print(f"    global_output_folder: {value.global_output_folder}")
    
    # Test 4: Get final values and reconstruct config
    print("\n=== Test 4: Reconstruct Config ===")
    
    final_values = form_manager.get_current_values()
    print("Final values:")
    for name, value in final_values.items():
        print(f"  {name}: {value}")
    
    # Reconstruct the config
    try:
        new_config = GlobalPipelineConfig(**final_values)
        print(f"\nReconstructed config:")
        print(f"  global_output_folder: {new_config.path_planning.global_output_folder}")
        print(f"  output_dir_suffix: {new_config.path_planning.output_dir_suffix}")
        
        # Verify the change was applied
        if new_config.path_planning.global_output_folder == new_value:
            print("✅ SUCCESS: Nested dataclass editing works correctly!")
        else:
            print(f"❌ FAILED: Expected {new_value}, got {new_config.path_planning.global_output_folder}")
            
    except Exception as e:
        print(f"❌ FAILED to reconstruct config: {e}")

if __name__ == "__main__":
    test_nested_config_editing()
