#!/usr/bin/env python3
"""
Test script to verify file browser improvements:
1. Left-aligned buttons
2. Overwrite confirmation dialog
"""

import asyncio
from pathlib import Path
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from openhcs.constants.constants import Backend
from openhcs.io.filemanager import FileManager
from openhcs.textual_tui.screens.enhanced_file_browser import EnhancedFileBrowserScreen, BrowserMode
from openhcs.textual_tui.services.file_browser_service import SelectionMode


class TestFileBrowserApp(App):
    """Test app to verify file browser improvements."""
    
    def __init__(self):
        super().__init__()
        self.filemanager = FileManager()
    
    def compose(self) -> ComposeResult:
        """Compose the test app."""
        yield Static("File Browser Improvements Test", id="title")
        yield Static("Test left-aligned buttons and overwrite confirmation", id="instruction")
        
        yield Button("Test Save Dialog", id="test_save_btn")
        yield Button("Test Load Dialog", id="test_load_btn")
        yield Button("Exit", id="exit_btn")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "test_save_btn":
            self._test_save_dialog()
        elif event.button.id == "test_load_btn":
            self._test_load_dialog()
        elif event.button.id == "exit_btn":
            self.exit()
    
    def _test_save_dialog(self):
        """Test the save dialog with overwrite confirmation."""
        # Create a test file to trigger overwrite confirmation
        test_dir = Path.cwd() / "test_browser"
        test_dir.mkdir(exist_ok=True)
        test_file = test_dir / "test_file.txt"
        test_file.write_text("Test content for overwrite confirmation")
        
        def handle_save_result(result):
            if result:
                self.notify(f"Save result: {result}")
            else:
                self.notify("Save cancelled")
        
        browser = EnhancedFileBrowserScreen(
            file_manager=self.filemanager,
            initial_path=test_dir,
            backend=Backend.DISK,
            title="Test Save Dialog - Try saving 'test_file.txt'",
            mode=BrowserMode.SAVE,
            selection_mode=SelectionMode.FILES_ONLY,
            filter_extensions=['.txt'],
            default_filename="test_file.txt"  # This should trigger overwrite confirmation
        )
        self.push_screen(browser, handle_save_result)
    
    def _test_load_dialog(self):
        """Test the load dialog with left-aligned buttons."""
        def handle_load_result(result):
            if result:
                self.notify(f"Load result: {result}")
            else:
                self.notify("Load cancelled")
        
        browser = EnhancedFileBrowserScreen(
            file_manager=self.filemanager,
            initial_path=Path.cwd(),
            backend=Backend.DISK,
            title="Test Load Dialog - Check button alignment",
            mode=BrowserMode.LOAD,
            selection_mode=SelectionMode.FILES_ONLY
        )
        self.push_screen(browser, handle_load_result)


async def main():
    """Run the test app."""
    app = TestFileBrowserApp()
    await app.run_async()


if __name__ == "__main__":
    print("Testing file browser improvements...")
    print("Expected behavior:")
    print("1. Buttons should be left-aligned instead of centered")
    print("2. When saving 'test_file.txt', should show overwrite confirmation")
    print("3. Confirmation dialog should have Yes/No buttons")
    print("4. Clicking Yes should proceed with save, No should cancel")
    print()
    
    asyncio.run(main())
