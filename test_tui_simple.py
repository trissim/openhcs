#!/usr/bin/env python3
"""
Simple test script to verify TUI components work.
"""

def test_imports():
    """Test basic imports."""
    print("Testing imports...")
    
    try:
        print("1. Testing prompt_toolkit...")
        from prompt_toolkit.widgets import Frame, Label, Button
        print("   ✓ prompt_toolkit widgets imported")
        
        from prompt_toolkit.layout.containers import HSplit, VSplit
        print("   ✓ prompt_toolkit containers imported")
        
        print("2. Testing OpenHCS core...")
        from openhcs.core.config import get_default_global_config
        print("   ✓ OpenHCS core config imported")
        
        print("3. Testing TUI components...")
        from openhcs.tui.components.framed_button import FramedButton
        print("   ✓ FramedButton imported")
        
        print("4. Testing TUI architecture...")
        from openhcs.tui.tui_architecture import OpenHCSTUI, TUIState
        print("   ✓ TUI architecture imported")
        
        print("5. Testing TUI launcher...")
        from openhcs.tui.tui_launcher import OpenHCSTUILauncher
        print("   ✓ TUI launcher imported")
        
        print("\nAll imports successful!")
        return True
        
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_creation():
    """Test basic object creation."""
    print("\nTesting object creation...")
    
    try:
        from openhcs.core.config import get_default_global_config
        config = get_default_global_config()
        print("   ✓ Default config created")
        
        from openhcs.tui.tui_launcher import OpenHCSTUILauncher
        launcher = OpenHCSTUILauncher(core_global_config=config)
        print("   ✓ TUI launcher created")
        
        print("\nBasic creation successful!")
        return True
        
    except Exception as e:
        print(f"   ✗ Creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("OpenHCS TUI Simple Test")
    print("=" * 40)
    
    success = True
    success &= test_imports()
    
    if success:
        success &= test_basic_creation()
    
    if success:
        print("\n🎉 All tests passed! TUI should be able to start.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
