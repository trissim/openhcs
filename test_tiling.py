#!/usr/bin/env python3
"""Quick test script to verify tiling functionality works."""

import asyncio
from textual_window import window_manager, TilingLayout

async def test_tiling():
    """Test tiling functionality programmatically."""
    print("Testing tiling functionality...")
    
    # Test setting different layouts
    layouts = [
        TilingLayout.FLOATING,
        TilingLayout.HORIZONTAL_SPLIT,
        TilingLayout.VERTICAL_SPLIT,
        TilingLayout.GRID,
        TilingLayout.MASTER_DETAIL,
    ]
    
    for layout in layouts:
        window_manager.set_tiling_layout(layout)
        print(f"Set layout to: {layout.value}")
        await asyncio.sleep(0.1)  # Small delay
    
    print("Tiling test complete!")

if __name__ == "__main__":
    asyncio.run(test_tiling())
