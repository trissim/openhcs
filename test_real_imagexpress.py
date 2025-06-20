#!/usr/bin/env python3

import numpy as np
import glob
from PIL import Image
from openhcs.processing.backends.assemblers.assemble_stack_cpu import assemble_stack_cpu

def test_real_imagexpress_g05():
    """Test blending with real ImageXpress G05 data"""
    
    # Find G05 files
    data_path = "/home/ts/code/projects/openhcs/tests/integration/tests_data/mfd-hips-cell-density-ctb-4x_Plate_12994_workspace/TimePoint_1"
    g05_files = sorted(glob.glob(f"{data_path}/G05_s*_w1_z001.TIF"))
    
    print(f"🔬 Found {len(g05_files)} G05 tiles:")
    for f in g05_files:
        print(f"  📁 {f.split('/')[-1]}")
    
    if len(g05_files) < 2:
        print("❌ Need at least 2 tiles for blending test")
        return
    
    # Load all 6 tiles for 2×3 grid
    tiles_to_test = g05_files[:6]  # Use all 6 tiles
    tiles = []

    for tile_file in tiles_to_test:
        img = Image.open(tile_file)
        tile_array = np.array(img, dtype=np.float32)
        tiles.append(tile_array)
        print(f"📊 {tile_file.split('/')[-1]}: shape={tile_array.shape}, dtype={tile_array.dtype}")
        print(f"    Values: min={tile_array.min():.1f}, max={tile_array.max():.1f}, mean={tile_array.mean():.1f}")

    # Stack tiles
    tiles_array = np.stack(tiles, axis=0)
    print(f"\n🎯 Stacked tiles shape: {tiles_array.shape}")

    # Create positions for a 2×3 grid with some overlap
    # Assume each tile is roughly the same size
    tile_h, tile_w = tiles[0].shape
    overlap = int(tile_w * 0.1)  # 10% overlap

    # 2 rows × 3 columns layout:
    # [0] [1] [2]
    # [3] [4] [5]
    positions = np.array([
        # Top row
        [0, 0],                                    # s001: Top-left
        [tile_w - overlap, 0],                     # s002: Top-center
        [2 * (tile_w - overlap), 0],              # s003: Top-right
        # Bottom row
        [0, tile_h - overlap],                     # s004: Bottom-left
        [tile_w - overlap, tile_h - overlap],      # s005: Bottom-center
        [2 * (tile_w - overlap), tile_h - overlap] # s006: Bottom-right
    ], dtype=np.float32)
    
    print(f"📐 Tile size: {tile_h}×{tile_w}")
    print(f"🔗 Overlap: {overlap}px ({100*overlap/tile_w:.1f}%)")
    print(f"📍 Positions (2×3 grid):")
    for i, pos in enumerate(positions):
        print(f"  s{i+1:03d}: ({pos[0]:4.0f}, {pos[1]:4.0f})")
    
    # Test both approaches with NEW function signature
    print(f"\n🔥 Testing FIXED margin ratio 0.1 (using critical logic from old working version)...")
    result_fixed = assemble_stack_cpu(tiles_array, positions, blend_method="fixed", fixed_margin_ratio=0.1)

    print(f"\n✅ Fixed ratio assembly complete!")
    print(f"📊 Result shape: {result_fixed.shape}")
    print(f"📊 Result values: min={result_fixed.min()}, max={result_fixed.max()}, mean={result_fixed.mean():.1f}")

    # Check for black pixels
    black_pixels_fixed = (result_fixed[0] == 0).sum()
    total_pixels = result_fixed[0].size
    print(f"🖤 Black pixels: {black_pixels_fixed}/{total_pixels} ({100*black_pixels_fixed/total_pixels:.2f}%)")

    print(f"\n🔥 Testing DYNAMIC blending (using critical logic from old working version)...")
    result_dynamic = assemble_stack_cpu(tiles_array, positions, blend_method="dynamic", overlap_blend_fraction=1.0)

    print(f"\n✅ Dynamic assembly complete!")
    print(f"📊 Result shape: {result_dynamic.shape}")
    print(f"📊 Result values: min={result_dynamic.min()}, max={result_dynamic.max()}, mean={result_dynamic.mean():.1f}")

    # Check for black pixels
    black_pixels_dynamic = (result_dynamic[0] == 0).sum()
    print(f"🖤 Black pixels: {black_pixels_dynamic}/{total_pixels} ({100*black_pixels_dynamic/total_pixels:.2f}%)")

    # Compare results
    print(f"\n📊 COMPARISON:")
    print(f"  Fixed 0.1:  {black_pixels_fixed}/{total_pixels} ({100*black_pixels_fixed/total_pixels:.2f}%) black pixels")
    print(f"  Dynamic:    {black_pixels_dynamic}/{total_pixels} ({100*black_pixels_dynamic/total_pixels:.2f}%) black pixels")

    # Check if results are identical
    print(f"  Results identical: {np.array_equal(result_fixed, result_dynamic)}")

    return result_fixed, result_dynamic

if __name__ == "__main__":
    test_real_imagexpress_g05()
