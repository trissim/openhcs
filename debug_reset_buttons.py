#!/usr/bin/env python3
"""
Debug script to test reset button ID parsing.
"""

from openhcs.textual_tui.widgets.shared.parameter_form_manager import ParameterFormManager
from openhcs.textual_tui.services.config_reflection_service import FieldIntrospector
from openhcs.core.config import GlobalPipelineConfig


def test_button_id_parsing():
    """Test button ID parsing for config form."""
    print("=== Testing Button ID Parsing ===")
    
    # Create a test config
    config = GlobalPipelineConfig()
    
    # Analyze the config to get field specs
    field_specs = FieldIntrospector().analyze_dataclass(GlobalPipelineConfig, config)
    
    print(f"Found {len(field_specs)} field specs:")
    for spec in field_specs:
        print(f"  - {spec.name}: {spec.actual_type}")
    
    # Convert to parameter form manager format
    parameters = {}
    parameter_types = {}
    param_defaults = {}
    
    for spec in field_specs:
        parameters[spec.name] = spec.current_value
        parameter_types[spec.name] = spec.actual_type
        param_defaults[spec.name] = spec.default_value
    
    # Create form manager
    form_manager = ParameterFormManager(parameters, parameter_types, "config")

    print(f"\nForm manager parameters:")
    for param_name in form_manager.parameters:
        print(f"  - {param_name}")

    # Use the testing method to create nested managers
    print(f"\nCreating nested managers for testing...")
    try:
        form_manager._create_nested_managers_for_testing()
        print("Nested managers created successfully")
    except Exception as e:
        import traceback
        print(f"Nested manager creation failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")

    # Check if nested managers are created
    if hasattr(form_manager, 'nested_managers'):
        print(f"\nNested managers:")
        for nested_name, nested_manager in form_manager.nested_managers.items():
            print(f"  - {nested_name}:")
            for nested_param in nested_manager.parameters:
                print(f"    - {nested_param}")

                # Test nested parameter button ID parsing
                nested_widget_id = f"config_{nested_name}_{nested_param}"
                nested_reset_button_id = f"reset_{nested_widget_id}"

                print(f"      Widget ID: {nested_widget_id}")
                print(f"      Reset Button ID: {nested_reset_button_id}")

                # Parse the nested button ID
                if nested_reset_button_id.startswith("reset_config_"):
                    parsed_field_name = nested_reset_button_id.split("_", 2)[2]
                    print(f"      Parsed Field Name: {parsed_field_name}")

                    # This should be "nested_name_nested_param"
                    expected_field_name = f"{nested_name}_{nested_param}"
                    print(f"      Expected Field Name: {expected_field_name}")
                    print(f"      Match: {parsed_field_name == expected_field_name}")
    else:
        print(f"\nNo nested managers found")
    
    # Test button ID generation and parsing
    print(f"\n=== Button ID Generation and Parsing ===")
    for param_name in form_manager.parameters:
        # Generate widget ID (same as ParameterFormManager._build_regular_parameter_form)
        widget_id = f"config_{param_name}"
        
        # Generate reset button ID (same as ParameterFormManager._build_regular_parameter_form)
        reset_button_id = f"reset_{widget_id}"
        
        # Parse button ID (same as ConfigFormWidget.on_button_pressed)
        if reset_button_id.startswith("reset_config_"):
            parsed_field_name = reset_button_id.split("_", 2)[2]
            
            print(f"  {param_name}:")
            print(f"    Widget ID: {widget_id}")
            print(f"    Reset Button ID: {reset_button_id}")
            print(f"    Parsed Field Name: {parsed_field_name}")
            print(f"    Match: {param_name == parsed_field_name}")
            
            if param_name != parsed_field_name:
                print(f"    ❌ MISMATCH! Expected: {param_name}, Got: {parsed_field_name}")
            else:
                print(f"    ✅ OK")
        else:
            print(f"  {param_name}: ❌ Button ID doesn't start with 'reset_config_'")


if __name__ == "__main__":
    test_button_id_parsing()
