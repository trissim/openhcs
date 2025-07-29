#!/usr/bin/env python3
"""
Test script to verify monkey patching fix for pickling issue.
"""

import dill as pickle
import tempfile
import traceback

def test_monkey_patching_fix():
    """Test that monkey patching restores pickling functionality."""
    
    print("🔍 Testing monkey patching fix...")
    
    # Test 1: Check that external library functions are monkey patched
    try:
        import pyclesperanto as cle
        
        # Check if sobel function has OpenHCS attributes
        if hasattr(cle.sobel, 'input_memory_type') and hasattr(cle.sobel, 'output_memory_type'):
            print("✅ pyclesperanto.sobel has OpenHCS attributes (monkey patched)")
            print(f"  - Input type: {cle.sobel.input_memory_type}")
            print(f"  - Output type: {cle.sobel.output_memory_type}")
        else:
            print("❌ pyclesperanto.sobel missing OpenHCS attributes")
            
        # Test pickling
        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(cle.sobel, f)
            print("✅ pyclesperanto.sobel pickles successfully")
            
    except Exception as e:
        print(f"❌ pyclesperanto test failed: {e}")
        traceback.print_exc()
    
    # Test 2: Check registry lookup still works
    try:
        from openhcs.processing.func_registry import get_function_by_name
        registry_func = get_function_by_name("sobel", "pyclesperanto")
        
        if registry_func:
            print("✅ Registry lookup still works")
            
            # Check if they're the same function (should be due to monkey patching)
            if registry_func is cle.sobel:
                print("✅ Registry function is same as module function (perfect monkey patching)")
            else:
                print("⚠️  Registry function differs from module function")
                
        else:
            print("❌ Registry lookup failed")
            
    except Exception as e:
        print(f"❌ Registry test failed: {e}")
        traceback.print_exc()
    
    # Test 3: Check function still has OpenHCS features
    try:
        import numpy as np
        
        # Create test image
        test_image = np.random.randint(0, 255, (10, 100, 100), dtype=np.uint16)
        
        # Check if function has slice_by_slice parameter (OpenHCS feature)
        import inspect
        sig = inspect.signature(cle.sobel)
        if 'slice_by_slice' in sig.parameters:
            print("✅ Function has OpenHCS slice_by_slice parameter")
        else:
            print("❌ Function missing OpenHCS parameters")
            
        print(f"Function signature: {sig}")
        
    except Exception as e:
        print(f"❌ Feature test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_monkey_patching_fix()
