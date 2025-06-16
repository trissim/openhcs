#!/usr/bin/env python3
"""
Simple test to verify tophat function equivalence between NumPy and CuPy.
"""

import numpy as np
import cupy as cp
from openhcs.processing.backends.processors import numpy_processor, cupy_processor

def test_tophat_equivalence():
    """Test tophat function equivalence."""
    print("🔍 TESTING TOPHAT EQUIVALENCE: NUMPY vs CUPY")
    print("=" * 60)
    
    # Create test data
    np_data = np.random.randint(5000, 40000, (1, 40, 40), dtype=np.uint16)
    
    # Add some background structure that tophat should remove
    y, x = np.ogrid[0:40, 0:40]
    background = 8000 + (y * 100) + (x * 80)
    np_data[0] = np.clip(np_data[0] + background, 0, 65535).astype(np.uint16)
    
    # Add some bright spots that should be preserved
    np_data[0, 15:20, 15:20] = 50000
    np_data[0, 25:30, 25:30] = 55000
    
    cp_data = cp.asarray(np_data)
    
    print(f"Test data shape: {np_data.shape}")
    print(f"Test data range: {np_data.min()} to {np_data.max()}")
    
    # Test parameters
    params = {'selem_radius': 15, 'downsample_factor': 2}
    print(f"Test parameters: {params}")
    
    try:
        # Run NumPy implementation
        print("\nRunning NumPy tophat...")
        np_result = numpy_processor.tophat(np_data, **params)
        print(f"NumPy result: shape={np_result.shape}, range={np_result.min()}-{np_result.max()}")
        
        # Run CuPy implementation
        print("Running CuPy tophat...")
        cp_result = cupy_processor.tophat(cp_data, **params)
        cp_result_np = cp.asnumpy(cp_result)
        print(f"CuPy result: shape={cp_result_np.shape}, range={cp_result_np.min()}-{cp_result_np.max()}")
        
        # Calculate differences
        abs_diff = np.abs(np_result.astype(float) - cp_result_np.astype(float))
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        
        print(f"\nDifference analysis:")
        print(f"Max absolute difference: {max_diff:.1f}")
        print(f"Mean absolute difference: {mean_diff:.1f}")
        
        # Calculate relative differences
        max_val = max(np_result.max(), cp_result_np.max())
        if max_val > 0:
            rel_max_diff = max_diff / max_val
            rel_mean_diff = mean_diff / max_val
            print(f"Max relative difference: {rel_max_diff:.4f} ({rel_max_diff*100:.2f}%)")
            print(f"Mean relative difference: {rel_mean_diff:.4f} ({rel_mean_diff*100:.2f}%)")
        
        # Check correlation
        correlation = np.corrcoef(np_result.flatten(), cp_result_np.flatten())[0, 1]
        print(f"Correlation coefficient: {correlation:.6f}")
        
        # Tolerance checks
        close_strict = np.allclose(np_result, cp_result_np, rtol=1e-3, atol=10)
        close_moderate = np.allclose(np_result, cp_result_np, rtol=1e-2, atol=100)
        close_loose = np.allclose(np_result, cp_result_np, rtol=5e-2, atol=500)
        
        print(f"\nTolerance checks:")
        print(f"Close (strict: rtol=1e-3, atol=10): {close_strict}")
        print(f"Close (moderate: rtol=1e-2, atol=100): {close_moderate}")
        print(f"Close (loose: rtol=5e-2, atol=500): {close_loose}")
        
        # Assessment
        print(f"\nAssessment:")
        if max_diff < 100:
            print("✅ EXCELLENT: Very close results")
            return "excellent"
        elif max_diff < 1000:
            print("✅ GOOD: Reasonably close results")
            return "good"
        elif max_diff < 5000:
            print("⚠️ MODERATE: Some differences but acceptable")
            return "moderate"
        else:
            print("❌ POOR: Large differences detected")
            return "poor"
            
    except Exception as e:
        print(f"❌ Error during tophat test: {e}")
        import traceback
        traceback.print_exc()
        return "error"

if __name__ == "__main__":
    result = test_tophat_equivalence()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    if result == "excellent":
        print("✅ NumPy and CuPy tophat functions are very close")
    elif result == "good":
        print("✅ NumPy and CuPy tophat functions are reasonably equivalent")
    elif result == "moderate":
        print("⚠️ NumPy and CuPy tophat functions have some differences")
    elif result == "poor":
        print("❌ NumPy and CuPy tophat functions differ significantly")
    else:
        print("❌ Test failed with errors")
    
    print("cuCIM dependency removed to avoid Jitify hanging issues.")
    print("Using improved custom CuPy implementation.")
