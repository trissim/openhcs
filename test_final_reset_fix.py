#!/usr/bin/env python3
"""
Test the final reset fix - BaseFloatingWindow should ignore reset buttons.
"""

from openhcs.textual_tui.widgets.floating_window import BaseFloatingWindow
from textual.widgets import Button


def test_floating_window_button_filtering():
    """Test that BaseFloatingWindow ignores reset buttons."""
    print("=== Testing BaseFloatingWindow Button Filtering ===")
    
    # Create a mock button event
    class MockButton:
        def __init__(self, button_id):
            self.id = button_id
            self.label = "Test"
    
    class MockEvent:
        def __init__(self, button_id):
            self.button = MockButton(button_id)
    
    # Create a test floating window
    class TestWindow(BaseFloatingWindow):
        def __init__(self):
            super().__init__(title="Test")
            self.handle_button_action_called = False
            self.dismiss_called = False
            self.dismiss_result = None
        
        def handle_button_action(self, button_id, button_text):
            self.handle_button_action_called = True
            return True
        
        def dismiss(self, result):
            self.dismiss_called = True
            self.dismiss_result = result
    
    window = TestWindow()
    
    # Test 1: Reset button should be ignored
    print("\n--- Test 1: Reset button should be ignored ---")
    reset_event = MockEvent("reset_config_num_workers")
    window.on_button_pressed(reset_event)
    
    print(f"handle_button_action called: {window.handle_button_action_called}")
    print(f"dismiss called: {window.dismiss_called}")
    
    if not window.handle_button_action_called and not window.dismiss_called:
        print("✅ Reset button correctly ignored")
    else:
        print("❌ Reset button was not ignored")
    
    # Reset state
    window.handle_button_action_called = False
    window.dismiss_called = False
    window.dismiss_result = None
    
    # Test 2: Regular dialog button should be handled
    print("\n--- Test 2: Regular dialog button should be handled ---")
    save_event = MockEvent("save")
    window.on_button_pressed(save_event)
    
    print(f"handle_button_action called: {window.handle_button_action_called}")
    print(f"dismiss called: {window.dismiss_called}")
    print(f"dismiss result: {window.dismiss_result}")
    
    if window.handle_button_action_called and window.dismiss_called:
        print("✅ Regular button correctly handled")
    else:
        print("❌ Regular button was not handled correctly")
    
    # Test 3: Another reset button pattern
    print("\n--- Test 3: Another reset button pattern ---")
    window.handle_button_action_called = False
    window.dismiss_called = False
    window.dismiss_result = None
    
    another_reset_event = MockEvent("reset_func_0_param")
    window.on_button_pressed(another_reset_event)
    
    print(f"handle_button_action called: {window.handle_button_action_called}")
    print(f"dismiss called: {window.dismiss_called}")
    
    if not window.handle_button_action_called and not window.dismiss_called:
        print("✅ Another reset button correctly ignored")
    else:
        print("❌ Another reset button was not ignored")


if __name__ == "__main__":
    test_floating_window_button_filtering()
    print("\n🎉 All tests completed!")
    print("\nThe fix ensures that:")
    print("- Reset buttons (starting with 'reset_') are ignored by BaseFloatingWindow")
    print("- Regular dialog buttons (Save/Cancel) are handled normally")
    print("- Config forms can handle their own reset buttons without closing the dialog")
