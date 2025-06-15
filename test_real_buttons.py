#!/usr/bin/env python3
"""
Test actual Button widgets in a SelectionList-like interface.
"""

from textual.app import App, ComposeResult
from textual.widgets import Button, Static
from textual.containers import Horizontal, Vertical
from textual import on
from textual.message import Message

class ButtonListItem(Horizontal):
    """A single list item with actual Button widgets."""
    
    class SelectionChanged(Message):
        def __init__(self, index: int, selected: bool):
            super().__init__()
            self.index = index
            self.selected = selected
    
    class ItemMoved(Message):
        def __init__(self, from_index: int, to_index: int):
            super().__init__()
            self.from_index = from_index
            self.to_index = to_index
    
    def __init__(self, text: str, value: str, index: int, **kwargs):
        super().__init__(**kwargs)
        self.text = text
        self.value = value
        self.index = index
        self.selected = False
        self.styles.height = 1  # Keep items compact
    
    def compose(self) -> ComposeResult:
        # Actual Button widgets for up/down
        up_btn = Button('↑', id=f'up_{self.index}', compact=True)
        up_btn.styles.width = 2
        up_btn.styles.min_width = 2
        yield up_btn

        down_btn = Button('↓', id=f'down_{self.index}', compact=True)
        down_btn.styles.width = 2
        down_btn.styles.min_width = 2
        yield down_btn

        # Selection checkbox (clickable)
        self.checkbox = Button('☐', id=f'check_{self.index}', compact=True)
        self.checkbox.styles.width = 2
        self.checkbox.styles.min_width = 2
        yield self.checkbox

        # Text content (make sure it's visible)
        self.text_widget = Static(f" {self.text}", expand=True)
        self.text_widget.styles.padding = (0, 1)
        yield self.text_widget
    
    def set_selected(self, selected: bool):
        """Update selection state."""
        self.selected = selected
        if hasattr(self, 'checkbox'):
            self.checkbox.label = '☑' if selected else '☐'
        if hasattr(self, 'text_widget'):
            # Highlight selected items
            if selected:
                self.text_widget.styles.background = 'blue'
                self.text_widget.styles.color = 'white'
            else:
                self.text_widget.styles.background = 'transparent'
                self.text_widget.styles.color = 'auto'
    
    @on(Button.Pressed)
    def handle_button(self, event):
        """Handle button presses."""
        if event.button.id.startswith('up_'):
            if self.index > 0:
                self.post_message(self.ItemMoved(self.index, self.index - 1))
        elif event.button.id.startswith('down_'):
            self.post_message(self.ItemMoved(self.index, self.index + 1))
        elif event.button.id.startswith('check_'):
            # Handle checkbox button press
            self.set_selected(not self.selected)
            self.post_message(self.SelectionChanged(self.index, self.selected))

    @on(Static.Clicked)
    def handle_text_click(self, event):
        """Handle clicking on text to toggle selection."""
        if event.static == self.text_widget:
            self.set_selected(not self.selected)
            self.post_message(self.SelectionChanged(self.index, self.selected))

class RealButtonSelectionList(Vertical):
    """A SelectionList-like widget with actual Button widgets."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.items = []
    
    def add_options(self, options):
        """Add options like SelectionList.add_options([('text', 'value'), ...])"""
        for i, (text, value) in enumerate(options):
            item = ButtonListItem(text=text, value=value, index=i)
            self.items.append(item)
            self.mount(item)
    
    def clear_options(self):
        """Clear all options."""
        for item in self.items:
            item.remove()
        self.items.clear()
    
    def get_selected(self):
        """Get list of selected values."""
        return [item.value for item in self.items if item.selected]

class TestApp(App):
    """Test app for real button widgets."""
    
    CSS = """
    Screen {
        align: center middle;
    }
    
    RealButtonSelectionList {
        width: 80%;
        height: 80%;
        border: solid white;
        padding: 1;
    }
    
    ButtonListItem {
        margin-bottom: 0;
    }
    """
    
    def compose(self) -> ComposeResult:
        self.list_widget = RealButtonSelectionList()
        yield self.list_widget
    
    def on_mount(self):
        # Add test items
        self.list_widget.add_options([
            ('First Item - Click me to select', 'item1'),
            ('Second Item - Use ↑↓ to reorder', 'item2'),
            ('Third Item - Real buttons!', 'item3'),
            ('Fourth Item - Test selection', 'item4')
        ])

        # Select first item by default to show selection works
        if self.list_widget.items:
            self.list_widget.items[0].set_selected(True)

        self.title = 'Real Button Test - Click buttons, checkboxes, and text'
    
    def on_button_list_item_selection_changed(self, event):
        """Handle selection changes."""
        selected = self.list_widget.get_selected()
        self.notify(f'Selected: {selected}')
    
    def on_button_list_item_item_moved(self, event):
        """Handle item movement."""
        self.notify(f'Move: {event.from_index} -> {event.to_index}')

if __name__ == '__main__':
    app = TestApp()
    app.run()
