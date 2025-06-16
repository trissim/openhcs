#!/usr/bin/env python3
"""
Test script to verify horizontal button layout in confirmation dialogs.
"""

import asyncio
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from openhcs.textual_tui.widgets.floating_window import ConfirmationWindow


class TestHorizontalButtonsApp(App):
    """Test app to verify horizontal button layout."""
    
    def compose(self) -> ComposeResult:
        """Compose the test app."""
        yield Static("Centered Horizontal Confirmation Buttons Test", id="title")
        yield Static("Click the button below to see centered horizontal Yes/No buttons", id="instruction")
        
        yield Button("Show Confirmation Dialog", id="show_dialog_btn")
        yield Button("Exit", id="exit_btn")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "show_dialog_btn":
            self._show_confirmation()
        elif event.button.id == "exit_btn":
            self.exit()
    
    def _show_confirmation(self):
        """Show confirmation dialog with horizontal buttons."""
        def handle_confirmation(result):
            if result:
                self.notify("You clicked Yes!")
            else:
                self.notify("You clicked No!")
        
        confirmation = ConfirmationWindow(
            title="Test Centered Horizontal Buttons",
            message="The Yes and No buttons should be side-by-side and centered.\nDo you see them horizontally aligned and centered in the dialog?"
        )
        self.push_screen(confirmation, handle_confirmation)


async def main():
    """Run the test app."""
    app = TestHorizontalButtonsApp()
    await app.run_async()


if __name__ == "__main__":
    print("Testing centered horizontal confirmation button layout...")
    print("Expected behavior:")
    print("- Confirmation dialog should show Yes and No buttons side-by-side")
    print("- Buttons should be centered within the dialog")
    print("- Buttons should be in a horizontal row, not stacked vertically")
    print("- Layout should be compact and visually appealing")
    print()
    
    asyncio.run(main())
