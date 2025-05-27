#!/usr/bin/env python3
"""
Test script for OpenHCS Hybrid TUI.

Simple validation test to ensure the hybrid TUI can be imported and initialized
without errors. This tests the basic architecture and import resolution.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all hybrid TUI components can be imported."""
    try:
        logger.info("Testing hybrid TUI imports...")
        
        # Test main module import
        from openhcs.tui_hybrid import HybridTUIApp, run_tui
        logger.info("✅ Main module imports successful")
        
        # Test controller imports
        from openhcs.tui_hybrid.controllers import AppController, DualEditorController
        logger.info("✅ Controller imports successful")
        
        # Test component imports
        from openhcs.tui_hybrid.components import (
            FunctionPatternEditor,
            StepSettingsEditor,
            ParameterEditor,
            GroupedDropdown,
            FileManagerBrowser
        )
        logger.info("✅ Component imports successful")
        
        # Test interface imports
        from openhcs.tui_hybrid.interfaces.component_interfaces import (
            ComponentInterface,
            EditorComponentInterface,
            ControllerInterface
        )
        logger.info("✅ Interface imports successful")
        
        # Test utility imports
        from openhcs.tui_hybrid.utils.static_analysis import (
            get_function_signature,
            get_abstractstep_parameters
        )
        logger.info("✅ Utility imports successful")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during imports: {e}")
        return False

async def test_app_initialization():
    """Test that the app controller can be initialized."""
    try:
        logger.info("Testing app controller initialization...")
        
        from openhcs.tui_hybrid.controllers import AppController
        
        # Create app controller
        app_controller = AppController()
        logger.info("✅ AppController created")
        
        # Initialize controller
        await app_controller.initialize_controller()
        logger.info("✅ AppController initialized")
        
        # Test container creation
        container = app_controller.get_container()
        if container is None:
            raise ValueError("Container is None")
        logger.info("✅ Container created successfully")
        
        # Cleanup
        await app_controller.cleanup_controller()
        logger.info("✅ AppController cleanup successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ App initialization error: {e}")
        return False

async def test_component_creation():
    """Test that components can be created."""
    try:
        logger.info("Testing component creation...")
        
        from openhcs.tui_hybrid.components import (
            StepSettingsEditor,
            ParameterEditor,
            GroupedDropdown
        )
        
        # Test StepSettingsEditor
        step_editor = StepSettingsEditor()
        if step_editor.container is None:
            raise ValueError("StepSettingsEditor container is None")
        logger.info("✅ StepSettingsEditor created")
        
        # Test ParameterEditor
        param_editor = ParameterEditor()
        if param_editor.container is None:
            raise ValueError("ParameterEditor container is None")
        logger.info("✅ ParameterEditor created")
        
        # Test GroupedDropdown
        dropdown = GroupedDropdown(values=[("test", "Test Option")])
        if dropdown.container is None:
            raise ValueError("GroupedDropdown container is None")
        logger.info("✅ GroupedDropdown created")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Component creation error: {e}")
        return False

def test_demo_step_creation():
    """Test demo step creation."""
    try:
        logger.info("Testing demo step creation...")
        
        from openhcs.tui_hybrid.controllers.app_controller import AppController
        
        app_controller = AppController()
        demo_step = app_controller._create_demo_step()
        
        # Validate demo step attributes
        required_attrs = ['name', 'variable_components', 'force_disk_output', 
                         'group_by', 'input_dir', 'output_dir', 'func']
        
        for attr in required_attrs:
            if not hasattr(demo_step, attr):
                raise ValueError(f"Demo step missing attribute: {attr}")
        
        logger.info("✅ Demo step created with all required attributes")
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo step creation error: {e}")
        return False

async def run_all_tests():
    """Run all validation tests."""
    logger.info("🚀 Starting OpenHCS Hybrid TUI validation tests...")
    
    tests = [
        ("Import Tests", test_imports),
        ("App Initialization", test_app_initialization),
        ("Component Creation", test_component_creation),
        ("Demo Step Creation", test_demo_step_creation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name}...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n📊 Test Results Summary:")
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {status}: {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Hybrid TUI is ready for use.")
        return True
    else:
        logger.error("💥 Some tests failed. Please check the errors above.")
        return False

def main():
    """Main test entry point."""
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test runner error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
