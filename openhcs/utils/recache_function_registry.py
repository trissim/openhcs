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
    python recache_function_registry.py [--unified|--legacy]

    --unified    Use the new unified registry system (default)
    --legacy     Use the legacy registry system for comparison

The script will:
- Clear function metadata cache (unified or legacy)
- Reset registry initialization flags
- Clear decoration tracking
- Force complete re-initialization using selected system
- Verify the registry is working correctly
"""

import sys
import logging
from datetime import datetime

# Toggle between legacy and unified registry systems
USE_UNIFIED_REGISTRIES = True  # Set to False to use legacy registries

def _switch_to_unified_registries():
    """Temporarily switch func_registry to use unified registries."""
    import openhcs.processing.func_registry as func_registry

    # Monkey patch the _register_external_libraries function to use unified registries
    def _register_external_libraries_unified():
        """Register external libraries using unified registries."""
        logger = logging.getLogger(__name__)
        logger.info("Phase 2: Registering external library functions using unified registries...")

        try:
            from openhcs.processing.backends.analysis.pyclesperanto_registry_clean import PyclesperantoRegistry
            registry = PyclesperantoRegistry()
            registry.register_functions_direct()
            logger.info("Successfully registered pyclesperanto functions")
        except ImportError as e:
            logger.warning(f"Could not register pyclesperanto functions: {e}")
        except Exception as e:
            logger.error(f"Error registering pyclesperanto functions: {e}")

        try:
            from openhcs.processing.backends.analysis.scikit_image_registry_clean import SkimageRegistry
            registry = SkimageRegistry()
            registry.register_functions_direct()
            logger.info("Successfully registered scikit-image functions")
        except ImportError as e:
            logger.warning(f"Could not register scikit-image functions: {e}")
        except Exception as e:
            logger.error(f"Error registering scikit-image functions: {e}")

        try:
            from openhcs.processing.backends.lib_registry.cupy_registry import CupyRegistry
            registry = CupyRegistry()
            registry.register_functions_direct()
            logger.info("Successfully registered CuPy ndimage functions")
        except ImportError as e:
            logger.warning(f"Could not register CuPy functions: {e}")
        except Exception as e:
            logger.error(f"Error registering CuPy functions: {e}")

    # Replace the function
    func_registry._register_external_libraries = _register_external_libraries_unified
    print("  ✅ Switched to unified registry system")

def recache_function_registry():
    """Force a complete recache of the OpenHCS function registry."""
    
    print("🔄 Starting OpenHCS function registry recache...")
    
    try:
        # Import required modules
        import openhcs.processing.func_registry as func_registry

        # Only import legacy registry if needed
        if not USE_UNIFIED_REGISTRIES:
            import openhcs.processing.backends.analysis.scikit_image_registry as scikit_registry
        
        # Show current status
        current_initialized = func_registry.is_registry_initialized()
        current_count = sum(len(funcs) for funcs in func_registry.FUNC_REGISTRY.values()) if current_initialized else 0
        
        print(f"📊 Current registry status: {'✅ Initialized' if current_initialized else '❌ Not initialized'}")
        print(f"📊 Current function count: {current_count}")

        # Show cache status for available libraries only
        print("\n📋 Cache status for available libraries:")

        # Check which libraries are actually available
        available_libraries = []

        # Check scikit-image (always available as it's in base dependencies)
        available_libraries.append("skimage")

        # Check optional GPU libraries
        try:
            import pyclesperanto
            available_libraries.append("pyclesperanto")
        except ImportError:
            print("  ⚠️  pyclesperanto not installed - skipping")

        try:
            import cupy
            available_libraries.append("cupy")
        except ImportError:
            print("  ⚠️  CuPy not installed - skipping")

        # Show cache status for available libraries
        try:
            from openhcs.processing.backends.analysis.cache_utils import get_cache_status

            for library in available_libraries:
                try:
                    status = get_cache_status(library)
                    status_icon = "✅" if status['exists'] else "❌"
                    library_name = {
                        "skimage": "scikit-image",
                        "pyclesperanto": "pyclesperanto",
                        "cupy": "CuPy"
                    }.get(library, library)
                    print(f"  {status_icon} {library_name}: {status['function_count'] or 0} functions cached")
                except Exception as e:
                    print(f"  ❌ {library_name}: Error checking cache status: {e}")

        except Exception as e:
            print(f"  ❌ Error importing cache utilities: {e}")

        # Step 0: Migrate legacy cache files to XDG locations
        print("\n🔄 Migrating legacy cache files to XDG locations...")
        from openhcs.core.xdg_paths import migrate_all_legacy_cache_files
        migrate_all_legacy_cache_files()

        # Step 1: Clear function metadata caches for available libraries
        print(f"\n🧹 Clearing function metadata caches for available libraries...")
        print(f"   Using {'unified' if USE_UNIFIED_REGISTRIES else 'legacy'} registry system")

        if USE_UNIFIED_REGISTRIES:
            # Clear caches using unified registry system
            print("  🧹 Clearing unified registry caches...")
            try:
                from openhcs.processing.backends.analysis.cache_utils import clear_library_cache

                # Clear scikit-image cache (always available)
                print("    🧹 Clearing scikit-image cache...")
                clear_library_cache("skimage")

                # Clear pyclesperanto cache (if available)
                try:
                    import pyclesperanto
                    print("    🧹 Clearing pyclesperanto cache...")
                    clear_library_cache("pyclesperanto")
                except ImportError:
                    print("    ⚠️  pyclesperanto not installed - skipping cache clear")

                # Clear CuPy cache (if available)
                try:
                    import cupy
                    print("    🧹 Clearing CuPy cache...")
                    clear_library_cache("cupy")
                except ImportError:
                    print("    ⚠️  CuPy not installed - skipping cache clear")

            except ImportError as e:
                print(f"    ❌ Error importing unified cache utilities: {e}")
                print("    🔄 Falling back to legacy cache clearing...")
                USE_UNIFIED_REGISTRIES = False

        if not USE_UNIFIED_REGISTRIES:
            # Clear caches using legacy registry system
            print("  🧹 Clearing legacy registry caches...")

            # Clear scikit-image cache (always available)
            print("    🧹 Clearing scikit-image cache...")
            scikit_registry.clear_function_metadata_cache()

            # Clear pyclesperanto cache (if available)
            try:
                import pyclesperanto  # Check if available first
                from openhcs.processing.backends.analysis.pyclesperanto_registry import clear_pyclesperanto_cache
                print("    🧹 Clearing pyclesperanto cache...")
                clear_pyclesperanto_cache()
            except ImportError:
                print("    ⚠️  pyclesperanto not installed - skipping cache clear")

            # Clear CuPy cache (if available)
            try:
                import cupy  # Check if available first
                from openhcs.processing.backends.analysis.cupy_registry import clear_cupy_cache
                print("    🧹 Clearing CuPy cache...")
                clear_cupy_cache()
            except ImportError:
                print("    ⚠️  CuPy not installed - skipping cache clear")
        
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
        
        # Step 3: Force re-initialization with selected registry system
        print(f"🔄 Force re-initializing function registry using {'unified' if USE_UNIFIED_REGISTRIES else 'legacy'} system...")

        if USE_UNIFIED_REGISTRIES:
            # Temporarily switch func_registry to use unified registries
            _switch_to_unified_registries()

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
        
        # Step 5: Test function signatures (if torch is available)
        print("\n🧪 Testing function signature updates...")
        try:
            import torch  # Check if torch is available first
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

        except ImportError:
            print("   ⚠️  PyTorch not installed - skipping function signature test")
        except Exception as e:
            print(f"   ⚠️  Could not test function signature: {e}")
        
        # Step 6: Show final cache status for available libraries
        print("\n📋 Final cache status for available libraries:")

        try:
            from openhcs.processing.backends.analysis.cache_utils import get_cache_status

            # Check which libraries are available and their cache status
            libraries_to_check = []

            # scikit-image is always available
            libraries_to_check.append(("skimage", "scikit-image"))

            # Check optional libraries
            try:
                import pyclesperanto
                libraries_to_check.append(("pyclesperanto", "pyclesperanto"))
            except ImportError:
                pass

            try:
                import cupy
                libraries_to_check.append(("cupy", "CuPy"))
            except ImportError:
                pass

            # Check cache status for available libraries
            for library_key, library_name in libraries_to_check:
                try:
                    status = get_cache_status(library_key)
                    status_icon = "✅" if status['exists'] else "❌"
                    age_info = f" ({status['cache_age_days']:.1f} days old)" if status['cache_age_days'] else ""
                    print(f"  {status_icon} {library_name}: {status['function_count'] or 0} functions cached{age_info}")
                except Exception as e:
                    print(f"  ❌ {library_name}: Error checking cache status: {e}")

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
    import argparse

    parser = argparse.ArgumentParser(description="OpenHCS Function Registry Recache Tool")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy registry system instead of unified registries"
    )
    parser.add_argument(
        "--unified",
        action="store_true",
        help="Use unified registry system (default)"
    )
    args = parser.parse_args()

    # Update the global toggle based on command line arguments
    global USE_UNIFIED_REGISTRIES
    if args.legacy:
        USE_UNIFIED_REGISTRIES = False
    elif args.unified:
        USE_UNIFIED_REGISTRIES = True
    # Otherwise use the default value set at module level

    print("OpenHCS Function Registry Recache Tool")
    print("=" * 50)
    print(f"Registry System: {'Unified' if USE_UNIFIED_REGISTRIES else 'Legacy'}")
    print()

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
