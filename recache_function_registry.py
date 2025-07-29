#!/usr/bin/env python3
"""
OpenHCS Function Registry Recache Script

This script forces a complete rebuild of the OpenHCS function registry,
clearing all caches and re-scanning all functions. Use this when:

1. You've made changes to decorators or function signatures
2. The TUI isn't showing updated function parameters
3. You've added new functions or modified existing ones
4. You want to ensure the registry reflects the latest code changes

Usage:
    python recache_function_registry.py

The script will:
- Clear function metadata cache
- Reset registry initialization flags
- Clear decoration tracking
- Force complete re-initialization
- Verify the registry is working correctly
"""

import sys
import logging

def recache_function_registry():
    """Force a complete recache of the OpenHCS function registry."""
    
    print("🔄 Starting OpenHCS function registry recache...")
    
    try:
        # Import required modules
        import openhcs.processing.func_registry as func_registry
        import openhcs.processing.backends.analysis.scikit_image_registry as scikit_registry
        
        # Show current status
        current_initialized = func_registry.is_registry_initialized()
        current_count = sum(len(funcs) for funcs in func_registry.FUNC_REGISTRY.values()) if current_initialized else 0
        
        print(f"📊 Current registry status: {'✅ Initialized' if current_initialized else '❌ Not initialized'}")
        print(f"📊 Current function count: {current_count}")
        
        # Step 1: Clear function metadata cache
        print("\n🧹 Clearing function metadata cache...")
        scikit_registry.clear_function_metadata_cache()
        
        # Step 2: Force clear and reset the registry
        print("🧹 Clearing and resetting function registry...")
        with func_registry._registry_lock:
            # Reset initialization flags
            func_registry._registry_initialized = False
            func_registry._registry_initializing = False
            
            # Clear the registry
            func_registry.FUNC_REGISTRY.clear()
            
            # Clear decoration tracking to allow re-decoration
            func_registry._decoration_applied.clear()
        
        # Step 3: Force re-initialization
        print("🔄 Force re-initializing function registry...")
        func_registry._auto_initialize_registry()
        
        # Step 4: Verify the new registry
        new_initialized = func_registry.is_registry_initialized()
        new_count = sum(len(funcs) for funcs in func_registry.FUNC_REGISTRY.values())
        
        print(f"\n📊 New registry status: {'✅ Initialized' if new_initialized else '❌ Failed to initialize'}")
        print(f"📊 New function count: {new_count}")
        
        if new_count > current_count:
            print(f"🎉 Registry expanded by {new_count - current_count} functions!")
        elif new_count == current_count:
            print("✅ Registry function count unchanged (expected if no new functions)")
        else:
            print(f"⚠️  Registry function count decreased by {current_count - new_count} functions")
        
        # Step 5: Test a specific function to verify dtype_conversion parameter
        print("\n🧪 Testing function signature updates...")
        try:
            from openhcs.processing.backends.processors.torch_processor import max_projection
            import inspect
            
            sig = inspect.signature(max_projection)
            has_slice_by_slice = 'slice_by_slice' in sig.parameters
            has_dtype_conversion = 'dtype_conversion' in sig.parameters
            
            print(f"   max_projection has slice_by_slice: {'✅' if has_slice_by_slice else '❌'}")
            print(f"   max_projection has dtype_conversion: {'✅' if has_dtype_conversion else '❌'}")
            
            if has_dtype_conversion:
                dtype_param = sig.parameters['dtype_conversion']
                print(f"   dtype_conversion type: {dtype_param.annotation}")
                print(f"   dtype_conversion default: {dtype_param.default}")
            
        except Exception as e:
            print(f"⚠️  Could not test function signature: {e}")
        
        print("\n✅ Function registry recache completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Restart the TUI to pick up the changes")
        print("   2. Check that functions now show dtype_conversion radio lists")
        print("   3. Verify that slice_by_slice parameters are working correctly")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during recache: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point."""
    print("OpenHCS Function Registry Recache Tool")
    print("=" * 50)
    
    success = recache_function_registry()
    
    if success:
        print("\n🎉 Recache completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Recache failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
