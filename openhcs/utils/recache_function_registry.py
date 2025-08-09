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
from datetime import datetime

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

        # Show cache status for all libraries
        print("\n📋 Cache status for all libraries:")

        # Scikit-image cache status
        try:
            from openhcs.processing.backends.analysis.cache_utils import get_cache_status
            skimage_status = get_cache_status("skimage")
            status_icon = "✅" if skimage_status['exists'] else "❌"
            print(f"  {status_icon} scikit-image: {skimage_status['function_count'] or 0} functions cached")
        except Exception as e:
            print(f"  ❌ scikit-image: Error checking cache status: {e}")

        # Pyclesperanto cache status
        try:
            pycle_status = get_cache_status("pyclesperanto")
            status_icon = "✅" if pycle_status['exists'] else "❌"
            print(f"  {status_icon} pyclesperanto: {pycle_status['function_count'] or 0} functions cached")
        except Exception as e:
            print(f"  ❌ pyclesperanto: Error checking cache status: {e}")

        # CuPy cache status
        try:
            cupy_status = get_cache_status("cupy")
            status_icon = "✅" if cupy_status['exists'] else "❌"
            print(f"  {status_icon} CuPy: {cupy_status['function_count'] or 0} functions cached")
        except Exception as e:
            print(f"  ❌ CuPy: Error checking cache status: {e}")

        # Step 0: Migrate legacy cache files to XDG locations
        print("\n🔄 Migrating legacy cache files to XDG locations...")
        from openhcs.core.xdg_paths import migrate_all_legacy_cache_files
        migrate_all_legacy_cache_files()

        # Step 1: Clear all function metadata caches
        print("\n🧹 Clearing all function metadata caches...")

        # Clear scikit-image cache
        scikit_registry.clear_function_metadata_cache()

        # Clear pyclesperanto cache
        try:
            from openhcs.processing.backends.analysis.pyclesperanto_registry import clear_pyclesperanto_cache
            clear_pyclesperanto_cache()
        except ImportError:
            print("  ⚠️  pyclesperanto not available - skipping cache clear")

        # Clear CuPy cache
        try:
            from openhcs.processing.backends.analysis.cupy_registry import clear_cupy_cache
            clear_cupy_cache()
        except ImportError:
            print("  ⚠️  CuPy not available - skipping cache clear")
        
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
        
        # Step 6: Show final cache status
        print("\n📋 Final cache status for all libraries:")

        try:
            from openhcs.processing.backends.analysis.cache_utils import get_cache_status

            # Check all library caches
            for library in ["skimage", "pyclesperanto", "cupy"]:
                try:
                    status = get_cache_status(library)
                    status_icon = "✅" if status['exists'] else "❌"
                    age_info = f" ({status['cache_age_days']:.1f} days old)" if status['cache_age_days'] else ""
                    print(f"  {status_icon} {library}: {status['function_count'] or 0} functions cached{age_info}")
                except Exception as e:
                    print(f"  ❌ {library}: Error checking cache status: {e}")

        except Exception as e:
            print(f"  ❌ Error checking final cache status: {e}")

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

def generate_audit_table():
    """Generate audit table after successful recache."""
    try:
        print("\n📊 Generating audit table...")

        # Import the audit functionality
        from audit_function_registry import FunctionRegistryAuditor

        # Create auditor and run audit
        auditor = FunctionRegistryAuditor()
        auditor.run_complete_audit()

        # Export to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"function_registry_audit_{timestamp}.csv"
        auditor.export_to_csv(csv_filename)

        print(f"📄 Audit table saved to: {csv_filename}")
        print(f"📊 Total functions audited: {len(auditor.audit_records)}")

        # Show summary by package
        package_counts = {}
        for record in auditor.audit_records:
            package = record.top_level_package
            package_counts[package] = package_counts.get(package, 0) + 1

        print("\n📋 Summary by package:")
        for package, count in sorted(package_counts.items()):
            print(f"   {package}: {count} functions")

        return True

    except Exception as e:
        print(f"💥 Failed to generate audit table: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point."""
    print("OpenHCS Function Registry Recache Tool")
    print("=" * 50)

    # Step 1: Recache the registry
    recache_success = recache_function_registry()

    if not recache_success:
        print("\n💥 Recache failed!")
        sys.exit(1)

    print("\n🎉 Recache completed successfully!")

    # Step 2: Generate audit table
    audit_success = generate_audit_table()

    if audit_success:
        print("\n✅ Complete success: Registry recached and audit table generated!")
        sys.exit(0)
    else:
        print("\n⚠️  Registry recached successfully, but audit table generation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
