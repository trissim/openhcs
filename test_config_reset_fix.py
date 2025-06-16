#!/usr/bin/env python3
"""
Test the config form reset fix.
"""

from openhcs.textual_tui.widgets.config_form import ConfigFormWidget
from openhcs.textual_tui.services.config_reflection_service import FieldIntrospector
from openhcs.core.config import GlobalPipelineConfig


def test_config_reset_fix():
    """Test the config form reset functionality."""
    print("=== Testing Config Form Reset Fix ===")
    
    # Create a test config
    config = GlobalPipelineConfig()
    print(f"Original config:")
    print(f"  num_workers: {config.num_workers}")
    print(f"  path_planning.output_dir_suffix: {config.path_planning.output_dir_suffix}")
    print(f"  vfs.default_intermediate_backend: {config.vfs.default_intermediate_backend}")
    
    # Analyze the config to get field specs
    field_specs = FieldIntrospector().analyze_dataclass(GlobalPipelineConfig, config)
    
    # Create config form widget
    config_form = ConfigFormWidget(field_specs)
    
    print(f"\nConfig form param_defaults:")
    for name, default in config_form.param_defaults.items():
        print(f"  {name}: {default}")
    
    # Create nested managers for testing
    config_form.form_manager._create_nested_managers_for_testing()
    
    print(f"\nTesting reset functionality:")
    
    # Test 1: Reset top-level parameter
    print(f"\n--- Test 1: Reset top-level parameter (num_workers) ---")
    try:
        # Modify the parameter first
        config_form.form_manager.update_parameter('num_workers', 999)
        print(f"Modified num_workers to: {config_form.form_manager.parameters['num_workers']}")
        
        # Reset it
        config_form._reset_field('num_workers')
        print(f"After reset: {config_form.form_manager.parameters['num_workers']}")
        print("✅ Top-level reset works")
    except Exception as e:
        print(f"❌ Top-level reset failed: {e}")
    
    # Test 2: Reset nested parameter
    print(f"\n--- Test 2: Reset nested parameter (path_planning_output_dir_suffix) ---")
    try:
        # Modify the nested parameter first
        config_form.form_manager.update_parameter('path_planning_output_dir_suffix', '_modified')
        
        # Check if it was updated
        if hasattr(config_form.form_manager, 'nested_managers') and 'path_planning' in config_form.form_manager.nested_managers:
            nested_value = config_form.form_manager.nested_managers['path_planning'].parameters['output_dir_suffix']
            print(f"Modified path_planning_output_dir_suffix to: {nested_value}")
        
        # Reset it
        config_form._reset_field('path_planning_output_dir_suffix')
        
        # Check if it was reset
        if hasattr(config_form.form_manager, 'nested_managers') and 'path_planning' in config_form.form_manager.nested_managers:
            reset_value = config_form.form_manager.nested_managers['path_planning'].parameters['output_dir_suffix']
            print(f"After reset: {reset_value}")
            
            # Check if it matches the default
            expected_default = config.path_planning.output_dir_suffix
            if reset_value == expected_default:
                print("✅ Nested reset works")
            else:
                print(f"❌ Nested reset failed: expected {expected_default}, got {reset_value}")
        else:
            print("❌ Nested managers not found")
            
    except Exception as e:
        import traceback
        print(f"❌ Nested reset failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
    
    # Test 3: Reset another nested parameter
    print(f"\n--- Test 3: Reset another nested parameter (vfs_default_intermediate_backend) ---")
    try:
        # Modify the nested parameter first
        config_form.form_manager.update_parameter('vfs_default_intermediate_backend', 'disk')
        
        # Check if it was updated
        if hasattr(config_form.form_manager, 'nested_managers') and 'vfs' in config_form.form_manager.nested_managers:
            nested_value = config_form.form_manager.nested_managers['vfs'].parameters['default_intermediate_backend']
            print(f"Modified vfs_default_intermediate_backend to: {nested_value}")
        
        # Reset it
        config_form._reset_field('vfs_default_intermediate_backend')
        
        # Check if it was reset
        if hasattr(config_form.form_manager, 'nested_managers') and 'vfs' in config_form.form_manager.nested_managers:
            reset_value = config_form.form_manager.nested_managers['vfs'].parameters['default_intermediate_backend']
            print(f"After reset: {reset_value}")
            
            # Check if it matches the default
            expected_default = config.vfs.default_intermediate_backend
            if reset_value == expected_default:
                print("✅ Second nested reset works")
            else:
                print(f"❌ Second nested reset failed: expected {expected_default}, got {reset_value}")
        else:
            print("❌ VFS nested managers not found")
            
    except Exception as e:
        import traceback
        print(f"❌ Second nested reset failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    test_config_reset_fix()
