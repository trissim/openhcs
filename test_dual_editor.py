#!/usr/bin/env python3
"""
Test script to trigger the dual editor screen directly.
"""

import asyncio
import logging
from textual.app import App

# Setup logging
logging.basicConfig(level=logging.DEBUG)

from openhcs.textual_tui.screens.dual_editor import DualEditorScreen

class TestApp(App):
    """Simple test app to show the dual editor."""
    
    def on_mount(self):
        """Show the dual editor on startup."""
        def handle_result(result):
            print(f"Dialog result: {result}")
            self.exit()
        
        # Show the dual editor as a modal
        self.push_screen(DualEditorScreen(is_new=True), handle_result)

async def main():
    """Run the test app."""
    app = TestApp()
    await app.run_async()

if __name__ == "__main__":
    asyncio.run(main())
