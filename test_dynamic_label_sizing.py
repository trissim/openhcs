#!/usr/bin/env python3
"""
Test script to verify dynamic label sizing in parameter forms.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from openhcs.textual_tui.widgets.config_form import ConfigFormWidget
from openhcs.textual_tui.services.config_reflection_service import FieldIntrospector


class TestEnum(Enum):
    SHORT = "short"
    VERY_LONG_OPTION_NAME = "very_long_option_name"


@dataclass
class TestConfig:
    # Short parameter names
    x: int = 1
    y: float = 2.0
    
    # Medium parameter names
    num_workers: int = 16
    memory_limit: float = 8.0
    
    # Long parameter names
    very_long_parameter_name_for_testing: str = "default"
    extremely_long_parameter_name_that_should_resize_dynamically: bool = False
    
    # Enum with varying option lengths
    test_enum: TestEnum = TestEnum.SHORT


class TestDynamicLabelApp(App):
    """Test app to verify dynamic label sizing."""
    
    def compose(self) -> ComposeResult:
        """Compose the test app."""
        yield Static("Dynamic Label Sizing Test", id="title")
        yield Static("Labels should size to their content, not fixed width", id="instruction")
        
        # Create test config
        config = TestConfig()
        field_specs = FieldIntrospector().analyze_dataclass(TestConfig, config)
        
        # Create config form widget
        config_form = ConfigFormWidget(field_specs)
        yield config_form
        
        yield Button("Exit", id="exit_btn")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "exit_btn":
            self.exit()


async def main():
    """Run the test app."""
    app = TestDynamicLabelApp()
    await app.run_async()


if __name__ == "__main__":
    print("Testing dynamic label sizing...")
    print("Expected behavior:")
    print("- Short labels (x:, y:) should take minimal space")
    print("- Medium labels (num_workers:, memory_limit:) should take moderate space")
    print("- Long labels should take more space as needed")
    print("- Input fields should adapt and fill remaining space")
    print("- Reset buttons should remain compact on the right")
    print()
    
    asyncio.run(main())
