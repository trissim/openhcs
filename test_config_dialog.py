#!/usr/bin/env python3

"""Test script to check config dialog sizing."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from dataclasses import dataclass
from textual.app import App, ComposeResult
from textual.widgets import Button

from openhcs.textual_tui.screens.config_dialog import ConfigDialogScreen

@dataclass
class TestConfig:
    """Simple test config."""
    name: str = "test"
    value: int = 42
    enabled: bool = True

class TestApp(App):
    """Test app to show config dialog."""
    
    def compose(self) -> ComposeResult:
        yield Button("Open Config", id="open_config")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "open_config":
            config = TestConfig()
            dialog = ConfigDialogScreen(TestConfig, config)
            self.push_screen(dialog, self.handle_config_result)
    
    def handle_config_result(self, result):
        if result:
            print(f"Config saved: {result}")
        else:
            print("Config cancelled")

if __name__ == "__main__":
    app = TestApp()
    app.run()
