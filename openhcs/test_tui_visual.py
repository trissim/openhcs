#!/usr/bin/env python3
"""Visual test for the hybrid TUI."""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_tui_visual():
    """Test the TUI visually for a few seconds."""
    from openhcs.tui_hybrid import HybridTUIApp
    
    print("Starting TUI visual test...")
    print("The TUI will run for 3 seconds, then exit automatically.")
    print("You should see:")
    print("- Top bar with Global Settings, Help, Exit, OpenHCS V1.0")
    print("- Two panes: Plate Manager (left) and Pipeline Editor (right)")
    print("- Buttons under each pane title")
    print("- Status bar at bottom")
    print("- Panes extending to the bottom")
    print()
    
    app = HybridTUIApp()
    
    # Create a task to exit after 3 seconds
    async def auto_exit():
        await asyncio.sleep(3)
        if hasattr(app, 'app_controller') and app.app_controller:
            if hasattr(app.app_controller, '_app') and app.app_controller._app:
                app.app_controller._app.exit()
    
    # Start auto-exit task
    exit_task = asyncio.create_task(auto_exit())
    
    try:
        # Run the TUI
        await app.run()
        print("\n✅ TUI test completed successfully!")
        print("If you saw the layout described above, the TUI is working correctly.")
        
    except Exception as e:
        print(f"\n❌ TUI test failed: {e}")
        return False
    finally:
        exit_task.cancel()
        
    return True

if __name__ == "__main__":
    success = asyncio.run(test_tui_visual())
    sys.exit(0 if success else 1)
