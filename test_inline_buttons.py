#!/usr/bin/env python3
"""
Test the inline button functionality with a visual TUI.
"""

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer
from openhcs.textual_tui.widgets.button_list_widget import ButtonListWidget, ButtonConfig


class TestInlineButtonsApp(App):
    """Test app for inline buttons."""
    
    CSS = """
    Screen {
        align: center middle;
    }
    
    ButtonListWidget {
        width: 80%;
        height: 80%;
        border: solid $accent;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        # Create button configs
        configs = [
            ButtonConfig('Add Item', 'add_btn'),
            ButtonConfig('Remove Item', 'remove_btn'),
        ]
        
        # Create the widget
        self.widget = ButtonListWidget(
            button_configs=configs,
            list_id='test_list',
            container_id='test_container',
            on_button_pressed=self.on_button_pressed,
            on_selection_changed=self.on_selection_changed,
            on_item_moved=self.on_item_moved
        )
        yield self.widget
        yield Footer()
    
    def on_mount(self):
        """Set up test data."""
        test_items = [
            {'name': 'First Item', 'path': '/test/path1'},
            {'name': 'Second Item', 'path': '/test/path2'},
            {'name': 'Third Item', 'path': '/test/path3'},
            {'name': 'Fourth Item', 'path': '/test/path4'},
        ]
        self.widget.items = test_items
        self.title = "Inline Button Test - Click ↑↓ to reorder items"
    
    def on_button_pressed(self, button_id: str):
        """Handle button presses."""
        self.notify(f"Button pressed: {button_id}")
    
    def on_selection_changed(self, selected_values):
        """Handle selection changes."""
        self.notify(f"Selection: {selected_values}")
    
    def on_item_moved(self, from_index: int, to_index: int):
        """Handle item movement."""
        self.notify(f"Moved item from {from_index} to {to_index}")
        
        # Actually move the item in the data
        items = list(self.widget.items)
        if 0 <= from_index < len(items) and 0 <= to_index < len(items):
            item = items.pop(from_index)
            items.insert(to_index, item)
            self.widget.items = items


if __name__ == "__main__":
    app = TestInlineButtonsApp()
    app.run()
