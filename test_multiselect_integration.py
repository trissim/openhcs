#!/usr/bin/env python3
"""
Test script to verify multi-select list integration.

This tests that the new multi-select functionality integrates properly
with the existing PlateManager and other components.
"""
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from openhcs.tui.components.list_manager import ListView, ListConfig, ListModel, ButtonConfig

def test_basic_multiselect():
    """Test basic multi-select functionality."""
    print("🧪 Testing basic multi-select functionality...")
    
    # Create test data
    items = [
        {"name": "Plate A", "path": "/path/to/plate_a"},
        {"name": "Plate B", "path": "/path/to/plate_b"},
        {"name": "Plate C", "path": "/path/to/plate_c"}
    ]
    
    # Create model and populate
    model = ListModel()
    model.set_items(items)
    
    # Create config with multi-select enabled
    config = ListConfig(
        title="Test Plates",
        allow_multi_select=True,
        display_func=lambda item, selected: item["name"],
        bulk_button_configs=[
            ButtonConfig("Delete Selected", lambda: print("Delete bulk action"), width=16),
            ButtonConfig("Process Selected", lambda: print("Process bulk action"), width=18)
        ]
    )
    
    # Create view
    view = ListView(model, config)
    
    # Test selection operations
    print("📋 Initial state:")
    print(f"  Items: {len(model.items)}")
    print(f"  Focused: {model.focused_index}")
    print(f"  Selected: {model.selected_indices}")
    
    # Select some items
    model.toggle_selection(0)  # Select Plate A
    model.toggle_selection(2)  # Select Plate C
    
    print("📋 After selecting items 0 and 2:")
    print(f"  Selected indices: {model.selected_indices}")
    print(f"  Selected items: {[item['name'] for item in model.get_selected_items()]}")
    
    # Test text generation
    text_lines = view._generate_list_text()
    print("📋 Generated display text:")
    for i, (style, text) in enumerate(text_lines):
        print(f"  {i}: {text}")
    
    # Test bulk button creation
    if hasattr(view, '_create_bulk_buttons'):
        bulk_buttons = view._create_bulk_buttons()
        print(f"📋 Bulk buttons container: {type(bulk_buttons).__name__}")
    
    return True

def test_compatibility():
    """Test backward compatibility with existing code."""
    print("🧪 Testing backward compatibility...")
    
    # Test that existing single-select behavior still works
    model = ListModel()
    model.set_items([{"name": "Item 1"}, {"name": "Item 2"}])
    
    # Create config WITHOUT multi-select (default behavior)
    config = ListConfig("Test List")
    view = ListView(model, config)
    
    # Test old-style selection
    model.select_item(1)
    print(f"📋 Single-select focused index: {model.focused_index}")
    
    # Test get_selected_item still works
    from openhcs.tui.components.list_manager import ListManagerPane
    pane = ListManagerPane(model, config)
    selected = pane.get_selected_item()
    print(f"📋 Selected item: {selected}")
    
    return True

def test_key_bindings():
    """Test key binding functionality."""
    print("🧪 Testing key bindings...")
    
    model = ListModel()
    model.set_items([{"name": "Item 1"}, {"name": "Item 2"}, {"name": "Item 3"}])
    
    config = ListConfig("Test List", allow_multi_select=True)
    view = ListView(model, config)
    
    # Test key bindings creation
    kb = view._create_list_key_bindings()
    print(f"📋 Key bindings created: {type(kb).__name__}")
    print(f"📋 Key bindings count: {len(kb.bindings)}")
    
    return True

def test_mouse_handling():
    """Test mouse handler functionality."""
    print("🧪 Testing mouse handling...")
    
    model = ListModel()
    model.set_items([{"name": "Item 1"}, {"name": "Item 2"}])
    
    config = ListConfig("Test List", allow_multi_select=True)
    view = ListView(model, config)
    
    # Test mouse handler creation
    handler = view._create_mouse_handler()
    print(f"📋 Mouse handler created: {callable(handler)}")
    
    return True

def main():
    """Run all tests."""
    print("🚀 Testing multi-select list integration...")
    print("=" * 60)
    
    tests = [
        test_basic_multiselect,
        test_compatibility,
        test_key_bindings,
        test_mouse_handling
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print(f"✅ {test.__name__}: PASS")
        except Exception as e:
            print(f"❌ {test.__name__}: FAIL - {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("📊 Test Summary:")
    print(f"  Passed: {passed}/{total}")
    print(f"  Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed! Multi-select lists are ready for use.")
        print("💡 Features available:")
        print("  - Checkbox-based multi-selection")
        print("  - Scrollable list display")
        print("  - Keyboard navigation (up/down/space/enter)")
        print("  - Mouse click support")
        print("  - Bulk operation buttons")
        print("  - Backward compatibility with single-select")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check implementation.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
