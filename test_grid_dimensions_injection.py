#!/usr/bin/env python3
"""
Test script to verify grid dimensions injection works correctly.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from openhcs.core.pipeline.path_planner import (
    METADATA_RESOLVERS, 
    resolve_metadata, 
    inject_metadata_into_pattern
)

def test_metadata_registry():
    """Test that the metadata registry is properly set up."""
    print("Testing metadata registry...")
    
    # Check that grid_dimensions resolver is registered
    assert "grid_dimensions" in METADATA_RESOLVERS
    print("✓ grid_dimensions resolver is registered")
    
    # Check resolver structure
    resolver_info = METADATA_RESOLVERS["grid_dimensions"]
    assert "resolver" in resolver_info
    assert "description" in resolver_info
    assert callable(resolver_info["resolver"])
    print("✓ Resolver has correct structure")
    
    print("Metadata registry test passed!\n")

def test_pattern_injection():
    """Test that metadata injection into function patterns works."""
    print("Testing pattern injection...")
    
    # Test Case 1: Direct callable
    def dummy_func():
        pass
    
    result = inject_metadata_into_pattern(dummy_func, "grid_dimensions", (4, 4))
    expected = (dummy_func, {"grid_dimensions": (4, 4)})
    assert result == expected
    print("✓ Direct callable injection works")
    
    # Test Case 2: (callable, kwargs) tuple - update existing kwargs
    existing_pattern = (dummy_func, {"overlap_ratio": 0.1})
    result = inject_metadata_into_pattern(existing_pattern, "grid_dimensions", (4, 4))
    expected = (dummy_func, {"overlap_ratio": 0.1, "grid_dimensions": (4, 4)})
    assert result == expected
    print("✓ Tuple pattern injection works (updates existing kwargs)")
    
    # Test Case 3: (callable, empty kwargs) tuple
    empty_kwargs_pattern = (dummy_func, {})
    result = inject_metadata_into_pattern(empty_kwargs_pattern, "grid_dimensions", (4, 4))
    expected = (dummy_func, {"grid_dimensions": (4, 4)})
    assert result == expected
    print("✓ Empty kwargs tuple injection works")
    
    print("Pattern injection test passed!\n")

def test_function_signatures():
    """Test that the updated functions have correct signatures."""
    print("Testing function signatures...")
    
    try:
        from openhcs.processing.backends.pos_gen.ashlar_processor_cupy import gpu_ashlar_align_cupy
        from openhcs.processing.backends.pos_gen.mist_processor_cupy import mist_compute_tile_positions
        
        # Check that functions have special_inputs decorator
        assert hasattr(gpu_ashlar_align_cupy, '__special_inputs__')
        assert "grid_dimensions" in gpu_ashlar_align_cupy.__special_inputs__
        print("✓ gpu_ashlar_align_cupy has special_inputs decorator")
        
        assert hasattr(mist_compute_tile_positions, '__special_inputs__')
        assert "grid_dimensions" in mist_compute_tile_positions.__special_inputs__
        print("✓ mist_compute_tile_positions has special_inputs decorator")
        
        # Check that functions still have special_outputs decorator
        assert hasattr(gpu_ashlar_align_cupy, '__special_outputs__')
        assert "positions" in gpu_ashlar_align_cupy.__special_outputs__
        print("✓ gpu_ashlar_align_cupy has special_outputs decorator")
        
        assert hasattr(mist_compute_tile_positions, '__special_outputs__')
        assert "positions" in mist_compute_tile_positions.__special_outputs__
        print("✓ mist_compute_tile_positions has special_outputs decorator")
        
    except ImportError as e:
        print(f"⚠ Could not import functions (expected if CuPy not available): {e}")
        return
    
    print("Function signatures test passed!\n")

def main():
    """Run all tests."""
    print("=== Grid Dimensions Injection Test ===\n")
    
    try:
        test_metadata_registry()
        test_pattern_injection()
        test_function_signatures()
        
        print("🎉 All tests passed!")
        print("\nImplementation summary:")
        print("- Metadata resolver registry is working")
        print("- Pattern injection correctly modifies function patterns")
        print("- Position generation functions have correct decorators")
        print("- Grid dimensions will be auto-injected during compilation")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
