"""Start Menu Button for OpenHCS TUI - integrates with WindowBar system."""

import logging
from typing import Any

from textual import events, work
from textual.app import ComposeResult
from textual.containers import Container
from textual.geometry import Offset
from textual.screen import ModalScreen

# Import the button base from textual-window
from textual_window.button_bases import NoSelectStatic, ButtonStatic

logger = logging.getLogger(__name__)


class StartMenuButton(NoSelectStatic):
    """
    Start menu button that integrates with WindowBar.

    Uses the same pattern as WindowBarButton (content-sized, not space-filling).
    """

    # Use the same CSS classes as WindowBarButton for consistent styling
    DEFAULT_CSS = """
    StartMenuButton {
        height: 1;
        width: auto;  /* Content-sized like WindowBarButton, not 1fr like WindowBarAllButton */
        padding: 0 1;
        &:hover { background: $panel-lighten-1; }
        &.pressed { background: $primary; color: $text; }
    }
    """
    
    def __init__(self, window_bar, **kwargs: Any):
        super().__init__(content="☰ Start", **kwargs)
        self.window_bar = window_bar
        self.click_started_on: bool = False

    def on_mouse_down(self, event: events.MouseDown) -> None:
        """Handle mouse down - add pressed styling."""
        self.add_class("pressed")
        self.click_started_on = True

    async def on_mouse_up(self, event: events.MouseUp) -> None:
        """Handle mouse up - show dropdown menu."""
        self.remove_class("pressed")
        if self.click_started_on:
            self.show_popup(event)
            self.click_started_on = False

    def on_leave(self, event: events.Leave) -> None:
        """Handle mouse leave - remove pressed styling."""
        self.remove_class("pressed")
        self.click_started_on = False

    @work
    async def show_popup(self, event: events.MouseUp) -> None:
        """Show the start menu dropdown using WindowBar's menu system."""
        absolute_offset = self.screen.get_offset(self)
        await self.app.push_screen_wait(
            StartMenuDropdown(
                menu_offset=absolute_offset,
                dock=self.window_bar.dock,
            )
        )


class StartMenuDropdown(ModalScreen[None]):
    """
    Start menu dropdown - follows the same pattern as WindowBarMenu.
    """

    CSS = """
    StartMenuDropdown {
        background: $background 0%;
        align: left top;
    }
    #start_menu_container {
        background: $surface;
        width: 8; height: 4;
        border-left: wide $panel;
        border-right: wide $panel;
        /* Remove problematic borders that get cut off */
        & > ButtonStatic {
            &:hover { background: $panel-lighten-2; }
            &.pressed { background: $primary; }
        }
    }
    """

    def __init__(self, menu_offset: Offset, dock: str) -> None:
        super().__init__()
        self.menu_offset = menu_offset
        self.dock = dock

    def compose(self) -> ComposeResult:
        """Compose the start menu dropdown."""
        with Container(id="start_menu_container"):
            yield ButtonStatic("Main", id="main")
            yield ButtonStatic("Config", id="config")
            yield ButtonStatic("Help", id="help")
            yield ButtonStatic("Quit", id="quit")

    def on_mount(self) -> None:
        """Position the dropdown menu based on dock position."""
        menu = self.query_one("#start_menu_container")
        
        if self.dock == "top":
            # Bar is at top, dropdown should appear below
            y_offset = self.menu_offset.y + 1
        elif self.dock == "bottom":
            # Bar is at bottom, dropdown should appear above
            y_offset = self.menu_offset.y - 4  # 4 is height of our menu
        else:
            raise ValueError("Dock must be either 'top' or 'bottom'")

        menu.offset = Offset(self.menu_offset.x, y_offset)
        menu.add_class(self.dock)

    def on_mouse_up(self) -> None:
        """Close dropdown when clicking outside."""
        self.dismiss(None)

    async def on_button_static_pressed(self, event: ButtonStatic.Pressed) -> None:
        """Handle button presses in the dropdown."""
        button_id = event.button.id

        if button_id == "main":
            await self._handle_main()
        elif button_id == "config":
            await self._handle_config()
        elif button_id == "help":
            await self._handle_help()
        elif button_id == "quit":
            await self._handle_quit()

        # Close the dropdown
        self.dismiss(None)

    async def _handle_main(self) -> None:
        """Handle main button press - open the shared PipelinePlateWindow with both components."""
        from openhcs.textual_tui.windows import PipelinePlateWindow
        from textual.css.query import NoMatches

        # Try to find existing window - if it doesn't exist, create new one
        try:
            window = self.app.query_one(PipelinePlateWindow)
            # Window exists, show both components and open it
            window.show_both()
            window.open_state = True
        except NoMatches:
            # Expected case: window doesn't exist yet, create new one
            window = PipelinePlateWindow(self.app.filemanager, self.app.global_config)
            await self.app.mount(window)
            window.show_both()
            window.open_state = True

    async def _handle_config(self) -> None:
        """Handle config button press."""
        from openhcs.textual_tui.windows import ConfigWindow
        from openhcs.core.config import GlobalPipelineConfig
        from textual.css.query import NoMatches

        def handle_config_save(new_config):
            self.app.global_config = new_config
            logger.info("Configuration updated from start menu")

        # Try to find existing config window - if it doesn't exist, create new one
        try:
            window = self.app.query_one(ConfigWindow)
            # Window exists, just open it
            window.open_state = True
        except NoMatches:
            # Expected case: window doesn't exist yet, create new one
            window = ConfigWindow(
                GlobalPipelineConfig,
                self.app.global_config,
                on_save_callback=handle_config_save
            )
            await self.app.mount(window)
            window.open_state = True

    async def _handle_help(self) -> None:
        """Handle help button press."""
        from openhcs.textual_tui.windows import HelpWindow
        from textual.css.query import NoMatches

        # Try to find existing help window - if it doesn't exist, create new one
        try:
            window = self.app.query_one(HelpWindow)
            # Window exists, just open it
            window.open_state = True
        except NoMatches:
            # Expected case: window doesn't exist yet, create new one
            window = HelpWindow()
            await self.app.mount(window)
            window.open_state = True

    async def _handle_quit(self) -> None:
        """Handle quit button press."""
        self.app.action_quit()


