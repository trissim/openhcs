#!/usr/bin/env python3
"""
Debug the pickle mystery - why do manual functions work but auto-discovered don't?
"""

import dill as pickle
import tempfile
import traceback

def test_exact_same_function():
    """Test the exact same function decorated in different ways."""
    
    print("🔍 Testing exact same function with different decoration paths...")
    
    # Test 1: Get a manual function from the registry
    try:
        from openhcs.processing.func_registry import get_function_by_name
        manual_func = get_function_by_name("tophat", "cupy")
        
        if manual_func:
            print(f"✅ Manual function found: {manual_func.__name__}")
            print(f"  - Module: {manual_func.__module__}")
            print(f"  - Type: {type(manual_func)}")
            print(f"  - Has __wrapped__: {hasattr(manual_func, '__wrapped__')}")
            
            # Test pickling
            with tempfile.NamedTemporaryFile() as f:
                pickle.dump(manual_func, f)
                print("✅ Manual function pickles successfully")
        else:
            print("❌ Manual function not found")
            
    except Exception as e:
        print(f"❌ Manual function failed: {e}")
        traceback.print_exc()
    
    print()
    
    # Test 2: Get an auto-discovered function from the registry
    try:
        auto_func = get_function_by_name("sobel", "pyclesperanto")
        
        if auto_func:
            print(f"✅ Auto-discovered function found: {auto_func.__name__}")
            print(f"  - Module: {auto_func.__module__}")
            print(f"  - Type: {type(auto_func)}")
            print(f"  - Has __wrapped__: {hasattr(auto_func, '__wrapped__')}")
            
            # Test pickling
            with tempfile.NamedTemporaryFile() as f:
                pickle.dump(auto_func, f)
                print("✅ Auto-discovered function pickles successfully")
        else:
            print("❌ Auto-discovered function not found")
            
    except Exception as e:
        print(f"❌ Auto-discovered function failed: {e}")
        traceback.print_exc()
    
    print()
    
    # Test 3: Compare their attributes
    if 'manual_func' in locals() and 'auto_func' in locals() and manual_func and auto_func:
        print("🔍 Comparing function attributes...")
        
        manual_attrs = set(dir(manual_func))
        auto_attrs = set(dir(auto_func))
        
        print(f"Manual function attributes: {len(manual_attrs)}")
        print(f"Auto function attributes: {len(auto_attrs)}")
        
        # Check for differences
        only_manual = manual_attrs - auto_attrs
        only_auto = auto_attrs - manual_attrs
        
        if only_manual:
            print(f"Only in manual: {only_manual}")
        if only_auto:
            print(f"Only in auto: {only_auto}")
        
        # Check specific attributes
        for attr in ['__module__', '__globals__', '__closure__', '__wrapped__']:
            if hasattr(manual_func, attr) and hasattr(auto_func, attr):
                manual_val = getattr(manual_func, attr)
                auto_val = getattr(auto_func, attr)
                print(f"{attr}:")
                print(f"  Manual: {manual_val}")
                print(f"  Auto:   {auto_val}")
                print(f"  Same:   {manual_val == auto_val}")
    
    print()
    
    # Test 4: Try to pickle their __globals__
    print("🔍 Testing function globals...")
    
    if 'manual_func' in locals() and manual_func:
        try:
            with tempfile.NamedTemporaryFile() as f:
                pickle.dump(manual_func.__globals__, f)
                print("✅ Manual function globals pickle successfully")
        except Exception as e:
            print(f"❌ Manual function globals failed: {e}")
    
    if 'auto_func' in locals() and auto_func:
        try:
            with tempfile.NamedTemporaryFile() as f:
                pickle.dump(auto_func.__globals__, f)
                print("✅ Auto function globals pickle successfully")
        except Exception as e:
            print(f"❌ Auto function globals failed: {e}")

if __name__ == "__main__":
    test_exact_same_function()
