#!/usr/bin/env python3
"""
Test script to verify reset button UI functionality.
"""

import asyncio
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static, Input

from openhcs.textual_tui.widgets.config_form import ConfigFormWidget
from openhcs.textual_tui.services.config_reflection_service import FieldIntrospector
from openhcs.core.config import GlobalPipelineConfig


class TestResetUIApp(App):
    """Test app to verify reset UI functionality."""
    
    def __init__(self):
        super().__init__()
        self.config = GlobalPipelineConfig()
        
        # Analyze the config
        field_specs = FieldIntrospector().analyze_dataclass(GlobalPipelineConfig, self.config)
        
        # Create form widget
        self.config_form = ConfigFormWidget(field_specs)
    
    def compose(self) -> ComposeResult:
        """Compose the test app."""
        yield Static("Reset Button UI Test", id="title")
        yield Static("1. Modify the num_workers field below", id="instruction1")
        yield Static("2. Click the Reset button next to it", id="instruction2")
        yield Static("3. The value should reset to 16", id="instruction3")
        yield self.config_form
        yield Button("Test Reset Programmatically", id="test_btn")
        yield Static("", id="status")
        yield Button("Exit", id="exit_btn")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "exit_btn":
            self.exit()
        elif event.button.id == "test_btn":
            self._test_reset_programmatically()
    
    def _test_reset_programmatically(self) -> None:
        """Test reset functionality programmatically."""
        try:
            # Find the num_workers input widget
            widget_id = "config_num_workers"
            
            # Try to find the widget
            try:
                widget = self.query_one(f"#{widget_id}")
                self.query_one("#status").update(f"Found widget: {widget.__class__.__name__}")
            except Exception as e:
                # Try alternative search
                widgets = self.query(Input)
                found_widgets = []
                for w in widgets:
                    if w.id and 'num_workers' in w.id:
                        found_widgets.append(w.id)
                
                self.query_one("#status").update(f"Widget not found. Available Input widgets: {found_widgets}")
                return
            
            # Get current value
            current_value = widget.value
            self.query_one("#status").update(f"Current value: {current_value}")
            
            # Modify the value
            widget.value = "999"
            self.query_one("#status").update(f"Modified to: {widget.value}")
            
            # Test the reset functionality
            self.config_form._reset_field('num_workers')
            
            # Check if it reset
            new_value = widget.value
            self.query_one("#status").update(f"After reset: {new_value} (should be 16)")
            
        except Exception as e:
            self.query_one("#status").update(f"Error: {e}")


async def main():
    """Run the test app."""
    app = TestResetUIApp()
    await app.run_async()


if __name__ == "__main__":
    print("Testing reset UI functionality...")
    asyncio.run(main())
