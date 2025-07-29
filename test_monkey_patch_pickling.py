#!/usr/bin/env python3
"""
Test script to verify that monkey patched functions are pickleable.
"""

import dill as pickle
import tempfile
import traceback

def test_monkey_patched_pickling():
    """Test that monkey patched functions are pickleable."""
    
    print("🔍 Testing monkey patched function pickling...")
    
    # Import OpenHCS to trigger registry initialization and monkey patching
    from openhcs.processing.func_registry import get_function_by_name
    
    # Test 1: Check that pyclesperanto.sobel is monkey patched
    try:
        import pyclesperanto as cle
        
        print(f"pyclesperanto.sobel function: {cle.sobel}")
        print(f"Has input_memory_type: {hasattr(cle.sobel, 'input_memory_type')}")
        print(f"Has output_memory_type: {hasattr(cle.sobel, 'output_memory_type')}")
        
        if hasattr(cle.sobel, 'input_memory_type'):
            print(f"Input type: {cle.sobel.input_memory_type}")
            print(f"Output type: {cle.sobel.output_memory_type}")
        
        # Test pickling the monkey patched function
        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(cle.sobel, f)
            print("✅ Monkey patched pyclesperanto.sobel pickles successfully")
            
    except Exception as e:
        print(f"❌ Monkey patched function test failed: {e}")
        traceback.print_exc()
    
    # Test 2: Compare with registry function
    try:
        registry_sobel = get_function_by_name("sobel", "pyclesperanto")
        
        if registry_sobel:
            print(f"\nRegistry sobel function: {registry_sobel}")
            print(f"Same object as module function: {registry_sobel is cle.sobel}")
            
            # Test pickling registry function
            with tempfile.NamedTemporaryFile() as f:
                pickle.dump(registry_sobel, f)
                print("✅ Registry sobel function pickles successfully")
        else:
            print("❌ Registry sobel function not found")
            
    except Exception as e:
        print(f"❌ Registry function test failed: {e}")
        traceback.print_exc()
    
    # Test 3: Test other monkey patched functions
    try:
        # Test a few other functions that should be monkey patched
        test_functions = [
            ('gaussian_blur', cle.gaussian_blur),
            ('maximum_filter', cle.maximum_filter),
            ('binary_opening', cle.binary_opening)
        ]
        
        for func_name, func in test_functions:
            if hasattr(func, 'input_memory_type'):
                with tempfile.NamedTemporaryFile() as f:
                    pickle.dump(func, f)
                    print(f"✅ {func_name} pickles successfully")
            else:
                print(f"⚠️  {func_name} not monkey patched")
                
    except Exception as e:
        print(f"❌ Additional function test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_monkey_patched_pickling()
