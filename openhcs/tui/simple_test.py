#!/usr/bin/env python3
"""Simple test to verify hybrid TUI is working."""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def test_hybrid_tui():
    """Test the hybrid TUI components."""
    print("🚀 Testing Hybrid TUI...")

    try:
        # Test import
        print("📦 Testing imports...")
        from openhcs.tui_hybrid import HybridTUIApp
        print("✅ Import successful")

        # Test app creation
        print("🏗️ Creating app...")
        app = HybridTUIApp()
        print("✅ App created")

        # Test app initialization
        print("⚙️ Initializing app...")
        await app.initialize()
        print("✅ App initialized")

        # Test application creation
        print("🖥️ Creating prompt_toolkit application...")
        prompt_app = app.app_controller.create_application()
        print("✅ Application created")

        # Test cleanup
        print("🧹 Cleaning up...")
        await app.cleanup()
        print("✅ Cleanup complete")

        print("🎉 All tests passed! Hybrid TUI is working correctly.")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_hybrid_tui())
    sys.exit(0 if success else 1)
