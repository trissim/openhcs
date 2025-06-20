#!/usr/bin/env python3

import numpy as np
import glob
from PIL import Image
from openhcs.processing.backends.assemblers.assemble_stack_cpu import assemble_stack_cpu
from openhcs.processing.backends.assemblers.assemble_stack_cupy import assemble_stack_cupy

def test_gpu_cpu_identical():
    """Test that GPU and CPU assemblers produce identical results"""
    
    # Load G05 tiles
    data_path = "/home/ts/code/projects/openhcs/tests/integration/tests_data/mfd-hips-cell-density-ctb-4x_Plate_12994_workspace/TimePoint_1"
    g05_files = sorted(glob.glob(f"{data_path}/G05_s*_w1_z001.TIF"))[:6]
    
    print(f"🔬 Loading {len(g05_files)} G05 tiles...")
    tiles = []
    for i, tile_file in enumerate(g05_files):
        img = Image.open(tile_file)
        tile_array = np.array(img, dtype=np.float32)
        tiles.append(tile_array)
    
    tiles_array = np.stack(tiles, axis=0)
    tile_h, tile_w = tiles_array.shape[1], tiles_array.shape[2]
    
    # Create 2×3 grid positions
    overlap = int(tile_w * 0.1)
    positions = np.array([
        [0, 0],
        [tile_w - overlap, 0],
        [2 * (tile_w - overlap), 0],
        [0, tile_h - overlap],
        [tile_w - overlap, tile_h - overlap],
        [2 * (tile_w - overlap), tile_h - overlap]
    ], dtype=np.float32)
    
    print(f"📊 Tiles shape: {tiles_array.shape}")
    print(f"📍 Positions shape: {positions.shape}")
    
    # Test CPU version with timing
    print(f"\n🔥 Testing CPU assembler...")
    import time
    start_time = time.time()
    result_cpu = assemble_stack_cpu(tiles_array, positions, blend_method="fixed", fixed_margin_ratio=0.1)
    cpu_time = time.time() - start_time

    print(f"✅ CPU result: shape={result_cpu.shape}, min={result_cpu.min()}, max={result_cpu.max()}, mean={result_cpu.mean():.1f}")
    print(f"⏱️  CPU time: {cpu_time:.3f} seconds")

    # Test GPU version with timing
    print(f"\n🔥 Testing OPTIMIZED GPU assembler...")
    try:
        import cupy as cp
        tiles_gpu = cp.asarray(tiles_array)
        positions_gpu = cp.asarray(positions)

        # Warm up GPU
        _ = cp.zeros((100, 100))
        cp.cuda.Stream.null.synchronize()

        start_time = time.time()
        result_gpu = assemble_stack_cupy(tiles_gpu, positions_gpu, blend_method="fixed", fixed_margin_ratio=0.1)
        cp.cuda.Stream.null.synchronize()  # Wait for GPU to finish
        gpu_time = time.time() - start_time

        result_gpu_cpu = cp.asnumpy(result_gpu)  # Convert back to numpy for comparison

        print(f"✅ GPU result: shape={result_gpu_cpu.shape}, min={result_gpu_cpu.min()}, max={result_gpu_cpu.max()}, mean={result_gpu_cpu.mean():.1f}")
        print(f"⏱️  GPU time: {gpu_time:.3f} seconds")
        print(f"🚀 Speedup: {cpu_time/gpu_time:.2f}x faster")
        
        # Compare results
        print(f"\n📊 COMPARISON:")
        print(f"  Shapes identical: {result_cpu.shape == result_gpu_cpu.shape}")
        print(f"  Results identical: {np.array_equal(result_cpu, result_gpu_cpu)}")
        
        if not np.array_equal(result_cpu, result_gpu_cpu):
            diff = np.abs(result_cpu.astype(np.float64) - result_gpu_cpu.astype(np.float64))
            print(f"  Max difference: {diff.max()}")
            print(f"  Mean difference: {diff.mean():.3f}")
            print(f"  Pixels with difference > 0: {(diff > 0).sum()}")
            print(f"  Pixels with difference > 1: {(diff > 1).sum()}")
        
        # Test dynamic mode too
        print(f"\n🔥 Testing DYNAMIC mode...")
        result_cpu_dyn = assemble_stack_cpu(tiles_array, positions, blend_method="dynamic", overlap_blend_fraction=1.0)
        result_gpu_dyn = assemble_stack_cupy(tiles_gpu, positions_gpu, blend_method="dynamic", overlap_blend_fraction=1.0)
        result_gpu_dyn_cpu = cp.asnumpy(result_gpu_dyn)
        
        print(f"  CPU dynamic: mean={result_cpu_dyn.mean():.1f}")
        print(f"  GPU dynamic: mean={result_gpu_dyn_cpu.mean():.1f}")
        print(f"  Dynamic results identical: {np.array_equal(result_cpu_dyn, result_gpu_dyn_cpu)}")
        print(f"  Fixed vs Dynamic identical: {np.array_equal(result_cpu, result_cpu_dyn)}")
        
    except ImportError:
        print("❌ CuPy not available, skipping GPU test")
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gpu_cpu_identical()
