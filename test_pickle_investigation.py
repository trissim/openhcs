#!/usr/bin/env python3
"""
Test script to investigate pickling differences between manual and auto-discovered functions.
"""

import dill as pickle
import tempfile
import traceback

def test_function_pickling():
    """Test pickling of different function types."""
    
    print("🔍 Testing function pickling...")
    
    # Test 1: Manual function from cupy_processor - FRESH IMPORT
    try:
        # Import in a completely fresh way to avoid any registry contamination
        import importlib
        import sys

        # Clear any existing imports
        if 'openhcs.processing.backends.processors.cupy_processor' in sys.modules:
            del sys.modules['openhcs.processing.backends.processors.cupy_processor']

        # Fresh import
        cupy_processor = importlib.import_module('openhcs.processing.backends.processors.cupy_processor')
        tophat = cupy_processor.tophat

        print(f"✅ Manual function imported fresh: {tophat.__name__}")

        # Check attributes
        print(f"  - Has backend attr: {hasattr(tophat, 'backend')}")
        print(f"  - Backend value: {getattr(tophat, 'backend', 'None')}")
        print(f"  - Module: {tophat.__module__}")
        print(f"  - Input type: {getattr(tophat, 'input_memory_type', 'None')}")
        print(f"  - Output type: {getattr(tophat, 'output_memory_type', 'None')}")

        # Test pickling
        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(tophat, f)
            print("✅ Manual function pickles successfully")

    except Exception as e:
        print(f"❌ Manual function failed: {e}")
        traceback.print_exc()
    
    # Test 2: Auto-discovered function from registry
    try:
        from openhcs.processing.func_registry import get_function_by_name
        auto_func = get_function_by_name("sobel", "pyclesperanto")
        
        if auto_func:
            print(f"✅ Auto-discovered function found: {auto_func.__name__}")
            
            # Check attributes
            print(f"  - Has backend attr: {hasattr(auto_func, 'backend')}")
            print(f"  - Backend value: {getattr(auto_func, 'backend', 'None')}")
            print(f"  - Module: {auto_func.__module__}")
            
            # Test pickling
            with tempfile.NamedTemporaryFile() as f:
                pickle.dump(auto_func, f)
                print("✅ Auto-discovered function pickles successfully")
        else:
            print("❌ Auto-discovered function not found in registry")
            
    except Exception as e:
        print(f"❌ Auto-discovered function failed: {e}")
        traceback.print_exc()
    
    # Test 3: Raw external library function
    try:
        import pyclesperanto as cle
        raw_func = cle.sobel
        print(f"✅ Raw external function imported: {raw_func.__name__}")
        
        # Test pickling
        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(raw_func, f)
            print("✅ Raw external function pickles successfully")
            
    except Exception as e:
        print(f"❌ Raw external function failed: {e}")
        traceback.print_exc()
    
    # Test 4: Raw function with setattr (simulating registry process)
    try:
        import pyclesperanto as cle
        raw_func_copy = cle.sobel

        # Simulate what _register_function does
        setattr(raw_func_copy, "backend", "pyclesperanto")
        print(f"✅ Raw function with setattr: {raw_func_copy.__name__}")

        # Test pickling
        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(raw_func_copy, f)
            print("✅ Raw function with setattr pickles successfully")

    except Exception as e:
        print(f"❌ Raw function with setattr failed: {e}")
        traceback.print_exc()

    # Test 5: Full decoration but keep original module
    try:
        import pyclesperanto as cle
        from openhcs.core.memory.decorators import pyclesperanto as pyclesperanto_decorator

        raw_func = cle.sobel
        print(f"✅ Raw function before decoration: {raw_func.__name__}")
        print(f"  - Original module: {raw_func.__module__}")

        # Apply full decoration but keep original module
        decorated_func = pyclesperanto_decorator(raw_func)
        print(f"✅ Function after decoration: {decorated_func.__name__}")
        print(f"  - Module after decoration: {decorated_func.__module__}")

        # Test pickling
        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(decorated_func, f)
            print("✅ Fully decorated function (original module) pickles successfully")

    except Exception as e:
        print(f"❌ Fully decorated function (original module) failed: {e}")
        traceback.print_exc()

    # Test 6: Full decoration with module change
    try:
        import pyclesperanto as cle
        from openhcs.core.memory.decorators import pyclesperanto as pyclesperanto_decorator

        raw_func = cle.gaussian_blur  # Use different function to avoid conflicts
        print(f"✅ Raw function before decoration: {raw_func.__name__}")
        print(f"  - Original module: {raw_func.__module__}")

        # Apply full decoration AND change module
        decorated_func = pyclesperanto_decorator(raw_func)
        decorated_func.__module__ = "openhcs.processing.func_registry"  # Change module like registry does
        print(f"✅ Function after decoration + module change: {decorated_func.__name__}")
        print(f"  - Module after change: {decorated_func.__module__}")

        # Test pickling
        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(decorated_func, f)
            print("✅ Fully decorated function (changed module) pickles successfully")

    except Exception as e:
        print(f"❌ Fully decorated function (changed module) failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_function_pickling()
