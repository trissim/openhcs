#!/usr/bin/env python3
"""
Test script to verify reset button functionality in OpenHCS TUI forms.
"""

import asyncio
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from openhcs.textual_tui.widgets.config_form import ConfigFormWidget
from openhcs.textual_tui.services.config_reflection_service import FieldIntrospector
from openhcs.core.config import GlobalPipelineConfig


class TestResetApp(App):
    """Test app to verify reset functionality."""
    
    def __init__(self):
        super().__init__()
        self.config = GlobalPipelineConfig()
        
        # Analyze the config
        field_specs = FieldIntrospector().analyze_dataclass(GlobalPipelineConfig, self.config)
        
        # Create form widget
        self.config_form = ConfigFormWidget(field_specs)
    
    def compose(self) -> ComposeResult:
        """Compose the test app."""
        yield Static("Reset Button Test - Modify values then click Reset buttons", id="title")
        yield self.config_form
        yield Button("Exit", id="exit_btn")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "exit_btn":
            self.exit()
        else:
            # Let the config form handle reset buttons
            pass


async def main():
    """Run the test app."""
    app = TestResetApp()
    await app.run_async()


if __name__ == "__main__":
    print("Testing reset functionality...")
    print("1. The app will show a config form")
    print("2. Modify some values in the form")
    print("3. Click individual Reset buttons to test if they work")
    print("4. Values should reset to their default values")
    print("5. Press Exit when done")
    print()
    
    asyncio.run(main())
