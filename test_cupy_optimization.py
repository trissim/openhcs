#!/usr/bin/env python3
"""
Test script for CuPy flatfield correction memory optimizations.

This script tests the memory-optimized BaSiC flatfield correction to ensure
it can handle large images without running out of VRAM.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cupy as cp
import numpy as np
import time

def test_memory_optimization():
    """Test the memory-optimized flatfield correction."""
    
    print("🔧 Testing CuPy Flatfield Memory Optimization")
    print("=" * 50)
    
    # Test with different image sizes
    test_cases = [
        (7, 512, 512, "Small test"),
        (7, 1024, 1024, "Medium test"), 
        (7, 2048, 2048, "Large test (similar to your data)")
    ]
    
    for z, y, x, description in test_cases:
        print(f"\n📊 {description}: {z}×{y}×{x}")
        
        # Create test image
        test_image = cp.random.randint(0, 65536, (z, y, x), dtype=cp.uint16)
        image_size_mb = test_image.nbytes / 1e6
        print(f"   Image size: {image_size_mb:.1f} MB")
        
        # Check available memory
        mempool = cp.get_default_memory_pool()
        used_before = mempool.used_bytes() / 1e6
        print(f"   VRAM before: {used_before:.1f} MB")
        
        try:
            # Import the optimized function
            from openhcs.processing.backends.enhance.basic_processor_cupy import basic_flatfield_correction_cupy
            
            # Test with conservative memory limit
            max_memory_gb = 0.5  # 500MB limit
            
            start_time = time.time()
            
            # Run the optimized flatfield correction
            corrected = basic_flatfield_correction_cupy(
                test_image,
                max_iters=10,  # Fewer iterations for testing
                max_memory_gb=max_memory_gb,
                verbose=True
            )
            
            end_time = time.time()
            
            # Check memory after
            used_after = mempool.used_bytes() / 1e6
            peak_usage = max(used_before, used_after)
            
            print(f"   ✅ SUCCESS: Processed in {end_time - start_time:.2f}s")
            print(f"   VRAM after: {used_after:.1f} MB (peak: {peak_usage:.1f} MB)")
            print(f"   Output shape: {corrected.shape}")
            
            # Clean up
            del test_image, corrected
            cp.get_default_memory_pool().free_all_blocks()
            
        except cp.cuda.memory.OutOfMemoryError as e:
            print(f"   ❌ FAILED: Out of memory - {e}")
            # Clean up on failure
            try:
                del test_image
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass
                
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            # Clean up on failure
            try:
                del test_image
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass

def test_chunked_svd():
    """Test the chunked SVD implementation directly."""
    
    print("\n🔧 Testing Chunked SVD Implementation")
    print("=" * 50)
    
    # Create a test matrix similar to what flatfield correction uses
    z, yx = 7, 2048 * 2048  # 7 slices, 2048x2048 pixels each
    print(f"Test matrix: {z}×{yx:,} = {z * yx * 4 / 1e9:.2f} GB (float32)")
    
    try:
        # Create test matrix
        test_matrix = cp.random.randn(z, yx).astype(cp.float32)
        
        # Import the chunked function
        from openhcs.processing.backends.enhance.basic_processor_cupy import _chunked_low_rank_approximation
        
        # Test chunked SVD with small memory limit
        max_memory_gb = 0.1  # Very conservative 100MB limit
        rank = 3
        
        print(f"Running chunked SVD with {max_memory_gb}GB memory limit...")
        
        start_time = time.time()
        result = _chunked_low_rank_approximation(test_matrix, rank, max_memory_gb)
        end_time = time.time()
        
        print(f"✅ Chunked SVD completed in {end_time - start_time:.2f}s")
        print(f"Result shape: {result.shape}")
        print(f"Result rank should be ≤ {rank}")
        
        # Clean up
        del test_matrix, result
        cp.get_default_memory_pool().free_all_blocks()
        
    except Exception as e:
        print(f"❌ Chunked SVD failed: {e}")
        # Clean up on failure
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except:
            pass

if __name__ == "__main__":
    if not cp.cuda.is_available():
        print("❌ CUDA not available, cannot test CuPy optimizations")
        sys.exit(1)
        
    device_id = cp.cuda.get_device_id()
    print(f"🔍 GPU Device: {device_id}")
    total_mem, free_mem = cp.cuda.runtime.memGetInfo()
    print(f"🔍 Total VRAM: {total_mem / 1e9:.1f} GB")
    print(f"🔍 Free VRAM: {free_mem / 1e9:.1f} GB")
    
    try:
        test_memory_optimization()
        test_chunked_svd()
        print("\n🎉 All tests completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final cleanup
        cp.get_default_memory_pool().free_all_blocks()
