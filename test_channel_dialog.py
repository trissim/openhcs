#!/usr/bin/env python3
"""
Test the channel selection dialog properly by mounting it.
"""

from textual.app import App, ComposeResult
from textual.widgets import Button, Static
from textual.containers import Vertical

from openhcs.textual_tui.screens.channel_selection_dialog import ChannelSelectionDialog


class TestChannelDialogApp(App):
    """Test app for the channel selection dialog."""
    
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Channel Selection Dialog Test", id="title")
            yield Button("Show Dialog", id="show_dialog", variant="primary")
            yield Static("Result: None", id="result")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "show_dialog":
            self.show_channel_dialog()
    
    def show_channel_dialog(self) -> None:
        """Show the channel selection dialog."""
        def handle_result(selected_channels):
            result_widget = self.query_one("#result", Static)
            if selected_channels is not None:
                result_widget.update(f"Result: {selected_channels}")
            else:
                result_widget.update("Result: Cancelled")
        
        # Test with some sample channels
        available_channels = [1, 2, 3, 4, 5]
        selected_channels = [2, 4]  # Pre-select some channels
        
        dialog = ChannelSelectionDialog(
            available_channels=available_channels,
            selected_channels=selected_channels,
            callback=handle_result
        )
        
        self.push_screen(dialog)


if __name__ == "__main__":
    app = TestChannelDialogApp()
    app.run()
