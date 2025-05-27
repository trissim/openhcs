#!/usr/bin/env python3
"""
Simple test to verify TUI components are working.
"""

import asyncio
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path.cwd()))

def test_imports():
    """Test that all TUI components can be imported."""
    try:
        print("Testing TUI imports...")
        
        # Test basic prompt_toolkit
        from prompt_toolkit import Application
        from prompt_toolkit.layout import Layout, HSplit, VSplit
        from prompt_toolkit.widgets import Label, Button, Frame
        print("✅ prompt_toolkit imported successfully")
        
        # Test our TUI components
        from openhcs.tui.state import TUIState, PlateData, StepData
        print("✅ TUIState imported successfully")
        
        from openhcs.tui.commands import Command
        print("✅ Command pattern imported successfully")
        
        from openhcs.core.config import GlobalPipelineConfig
        print("✅ GlobalPipelineConfig imported successfully")
        
        from openhcs.core.context.processing_context import ProcessingContext
        print("✅ ProcessingContext imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_state():
    """Test TUI state management."""
    try:
        print("\nTesting TUI state...")
        
        from openhcs.tui.state import TUIState, PlateData, StepData
        
        # Create state
        state = TUIState()
        print(f"✅ Created TUIState: {len(state.plates)} plates, {len(state.steps)} steps")
        
        # Test adding data
        plate = PlateData(id='test1', name='Test Plate', path='/tmp/test')
        step = StepData(id='step1', name='Test Step', type='function')
        
        print("✅ Created test data objects")
        return True
        
    except Exception as e:
        print(f"❌ State test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_ui():
    """Test a simple UI layout."""
    try:
        print("\nTesting simple UI...")
        
        from prompt_toolkit import Application
        from prompt_toolkit.layout import Layout, HSplit, VSplit
        from prompt_toolkit.widgets import Label, Button, Frame
        
        # Create simple layout
        layout = Layout(
            Frame(
                HSplit([
                    Label("OpenHCS TUI Test"),
                    VSplit([
                        Label("Left Pane"),
                        Label("Right Pane")
                    ]),
                    Label("Status: Ready")
                ])
            )
        )
        
        print("✅ Created simple UI layout")
        return True
        
    except Exception as e:
        print(f"❌ UI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Testing OpenHCS TUI Components\n")
    
    tests = [
        ("Import Test", test_imports),
        ("State Test", test_state),
        ("UI Test", test_simple_ui)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name}...")
        print('='*50)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    print('='*50)
    
    if passed == total:
        print("🎉 All tests passed! TUI components are working.")
        return 0
    else:
        print("❌ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
