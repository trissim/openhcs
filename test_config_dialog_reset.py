#!/usr/bin/env python3
"""
Test the config dialog reset fix to ensure reset buttons don't close the dialog.
"""

import asyncio
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from openhcs.textual_tui.screens.config_dialog import ConfigDialogScreen
from openhcs.core.config import GlobalPipelineConfig


class TestConfigDialogApp(App):
    """Test app to verify config dialog reset functionality."""
    
    def compose(self) -> ComposeResult:
        """Compose the test app."""
        yield Static("Config Dialog Reset Test", id="title")
        yield Static("Click 'Open Config' to test reset buttons", id="instruction")
        yield Button("Open Config", id="open_config")
        yield Static("", id="status")
        yield Button("Exit", id="exit_btn")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "exit_btn":
            self.exit()
        elif event.button.id == "open_config":
            self._open_config_dialog()
    
    def _open_config_dialog(self) -> None:
        """Open the config dialog."""
        def handle_result(result):
            if result is None:
                self.query_one("#status").update("Config dialog cancelled")
            else:
                self.query_one("#status").update("Config dialog saved")
        
        # Create config dialog (same as menu bar)
        config = GlobalPipelineConfig()
        dialog = ConfigDialogScreen(GlobalPipelineConfig, config)
        
        self.push_screen(dialog, handle_result)


async def main():
    """Run the test app."""
    app = TestConfigDialogApp()
    await app.run_async()


if __name__ == "__main__":
    print("Testing config dialog reset functionality...")
    print("1. Click 'Open Config' to open the dialog")
    print("2. Modify some values in the form")
    print("3. Click individual Reset buttons")
    print("4. The dialog should NOT close when clicking Reset")
    print("5. Only Save/Cancel should close the dialog")
    print()
    
    asyncio.run(main())
