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
        width: auto;
        min-width: 8;
        height: auto;
        border-left: wide $panel;
        border-right: wide $panel;
        /* Remove problematic borders that get cut off */
        & > ButtonStatic {
            width: 100%;
            min-width: 8;
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
            yield ButtonStatic("Monitor", id="toggle_monitor")
            yield ButtonStatic("Term", id="term")
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
            menu_height = len(list(menu.children))
            y_offset = self.menu_offset.y - menu_height
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
        elif button_id == "term":
            await self._handle_term()
        elif button_id == "help":
            await self._handle_help()
        elif button_id == "toggle_monitor":
            await self._handle_toggle_monitor()
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
            # new_config is already GlobalPipelineConfig (concrete dataclass)
            global_config = new_config

            # Apply config changes to app
            self.app.global_config = global_config

            # Update thread-local storage for MaterializationPathConfig defaults
            from openhcs.core.config import set_current_pipeline_config
            set_current_pipeline_config(global_config)

            # Propagate config changes to all existing orchestrators and plate manager
            self._propagate_global_config_to_orchestrators(global_config)

            # Save config to cache for future sessions
            self._save_config_to_cache(global_config)

            logger.info("Configuration updated and applied from start menu")

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
                on_save_callback=handle_config_save,
                is_global_config_editing=True
            )
            await self.app.mount(window)
            window.open_state = True

    async def _handle_term(self) -> None:
        """Handle term button press - open terminal window."""
        from openhcs.textual_tui.windows.terminal_window import TerminalWindow
        from textual.css.query import NoMatches

        # Try to find existing terminal window - if it doesn't exist, create new one
        try:
            window = self.app.query_one(TerminalWindow)
            # Window exists, just open it
            window.open_state = True
        except NoMatches:
            # Expected case: window doesn't exist yet, create new one
            window = TerminalWindow()
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

    async def _handle_toggle_monitor(self) -> None:
        """Handle toggle monitor button press."""
        try:
            # Find the system monitor widget
            main_content = self.app.query_one("MainContent")
            system_monitor = main_content.query_one("SystemMonitorTextual")

            # Toggle monitoring
            system_monitor.toggle_monitoring()

            # Update button text
            toggle_btn = self.query_one("#toggle_monitor", ButtonStatic)
            if system_monitor.is_monitoring:
                toggle_btn.content = "Monitor"
            else:
                toggle_btn.content = "Monitor"

        except Exception as e:
            logger.error(f"Failed to toggle monitoring: {e}")

    async def _handle_quit(self) -> None:
        """Handle quit button press."""
        self.app.action_quit()

    def _propagate_global_config_to_orchestrators(self, new_config) -> None:
        """Propagate global config changes to all existing orchestrators and plate manager."""
        try:
            # Find the plate manager widget
            main_content = self.app.query_one("MainContent")
            plate_manager = main_content.query_one("PlateManagerWidget")

            # CRITICAL: Update plate manager's global config reference
            # This ensures future orchestrators and subprocesses use the latest config
            plate_manager.global_config = new_config
            logger.info("Updated plate manager global config reference")

            # Also update pipeline editor if it exists (though it should use app.global_config)
            try:
                pipeline_editor = main_content.query_one("PipelineEditorWidget")
                # Pipeline editor is designed to use self.app.global_config, but let's be safe
                logger.debug("Pipeline editor will automatically use updated app.global_config")
            except Exception:
                # Pipeline editor might not exist or be mounted
                pass

            # Update all orchestrators that don't have plate-specific configs
            updated_count = 0
            for plate_path, orchestrator in plate_manager.orchestrators.items():
                # Only update if this plate doesn't have a plate-specific config override
                if plate_path not in plate_manager.plate_configs:
                    # Use the async method to apply the new config
                    import asyncio
                    asyncio.create_task(orchestrator.apply_new_global_config(new_config))
                    updated_count += 1

            if updated_count > 0:
                logger.info(f"Applied global config changes to {updated_count} orchestrators")
            else:
                logger.info("No orchestrators updated (all have plate-specific configs)")

        except Exception as e:
            logger.error(f"Failed to propagate global config to orchestrators: {e}")
            # Don't fail the config update if propagation fails
            pass

    def _save_config_to_cache(self, config) -> None:
        """Save config to cache asynchronously."""
        async def _async_save():
            from openhcs.textual_tui.services.config_cache_adapter import save_global_config_to_cache
            try:
                success = await save_global_config_to_cache(config)
                if success:
                    logger.info("Global config saved to cache for future sessions")
                else:
                    logger.warning("Failed to save global config to cache")
            except Exception as e:
                logger.error(f"Error saving global config to cache: {e}")

        # Schedule the async save operation
        import asyncio
        asyncio.create_task(_async_save())


