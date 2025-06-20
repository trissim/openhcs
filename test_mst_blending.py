#!/usr/bin/env python3

import numpy as np
from openhcs.processing.backends.assemblers.assemble_stack_cpu import assemble_stack_cpu

# Create simple test data with overlapping tiles
def test_mst_blending():
    # Create 4 tiles in a 2x2 grid with overlaps
    tile_size = 100
    overlap = 20
    
    # Generate test tiles (100x100 each) with distinct values
    tiles = np.zeros((4, tile_size, tile_size), dtype=np.float32)
    tiles[0] = 0.25  # Tile 0: gray
    tiles[1] = 0.50  # Tile 1: medium gray
    tiles[2] = 0.75  # Tile 2: light gray
    tiles[3] = 1.00  # Tile 3: white

    print(f"🎨 Tile values: {[tiles[i].mean() for i in range(4)]}")
    
    # Position tiles with overlaps
    positions = np.array([
        [0, 0],                           # Top-left
        [tile_size - overlap, 0],         # Top-right (overlap with top-left)
        [0, tile_size - overlap],         # Bottom-left (overlap with top-left)
        [tile_size - overlap, tile_size - overlap]  # Bottom-right (overlaps with all)
    ], dtype=np.float32)
    
    print(f"🧪 Testing MST blending with {len(tiles)} tiles")
    print(f"📐 Tile size: {tile_size}x{tile_size}, Overlap: {overlap}px")
    print(f"📍 Positions: {positions}")
    
    # Test the new MST-based blending
    result = assemble_stack_cpu(tiles, positions, blend_method="custom_per_tile")
    
    print(f"✅ Assembly complete! Result shape: {result.shape}")

    # Test the blending in overlap regions
    print(f"\n🔍 Testing blending quality:")

    # Check overlap region between tiles 0 and 1 (horizontal overlap at x=80-100)
    overlap_region = result[0, 40:60, 85:95]  # Middle of overlap region
    print(f"📊 Horizontal overlap region stats:")
    print(f"   Min: {overlap_region.min():.3f}, Max: {overlap_region.max():.3f}")
    print(f"   Mean: {overlap_region.mean():.3f}, Std: {overlap_region.std():.3f}")

    # Check for black gaps (values near 0)
    black_pixels = (result[0] < 0.01).sum()
    total_pixels = result[0].size
    print(f"🖤 Black pixels: {black_pixels}/{total_pixels} ({100*black_pixels/total_pixels:.2f}%)")

    return result

if __name__ == "__main__":
    test_mst_blending()
