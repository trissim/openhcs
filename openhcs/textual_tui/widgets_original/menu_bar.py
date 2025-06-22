"""
MenuBar Widget for OpenHCS Textual TUI

Top menu bar with Global Config, Help, title, and Quit buttons.
Matches the layout from the current prompt-toolkit TUI.
"""

import logging
from typing import Optional, Any

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Button, Static
from textual.widget import Widget

from openhcs.core.config import GlobalPipelineConfig

logger = logging.getLogger(__name__)


class MenuBar(Widget):
    """
    Top menu bar widget.
    
    Layout: [Global Config] [Help] ——————— OpenHCS v1.0 ——————— [Quit]
    """
    
    # Reactive state
    app_title = reactive("OpenHCS v1.0")
    
    def __init__(self, global_config: GlobalPipelineConfig):
        """
        Initialize the menu bar.

        Args:
            global_config: Global configuration for the application (for initial setup only)
        """
        super().__init__()
        # Note: We don't store global_config as it can become stale
        # Always use self.app.global_config to get the current config
        logger.debug("MenuBar initialized")
    
    def compose(self) -> ComposeResult:
        """Compose the menu bar layout."""
        with Horizontal():
            # Left side buttons
            yield Button("Config", id="global_config_btn", compact=True)
            yield Button("Help", id="help_btn", compact=True)

            # Center title (expandable and centered)
            yield Static(f"{self.app_title}", expand=True, id="app_title")

            # Right side button
            yield Button("Quit", id="quit_btn", compact=True)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses in the menu bar."""
        button_id = event.button.id
        
        if button_id == "global_config_btn":
            self.action_show_global_config()
        elif button_id == "help_btn":
            self.action_show_help()
        elif button_id == "quit_btn":
            self.action_quit()
    
    def action_show_global_config(self) -> None:
        """Show global configuration dialog."""

        def handle_result(result: Any) -> None:
            if result:  # User saved config changes
                # Apply config changes to app
                self.app.global_config = result

                # Propagate config changes to all existing orchestrators
                self._propagate_global_config_to_orchestrators(result)

                # Save config to cache for future sessions
                self._save_config_to_cache(result)

                logger.info("Configuration updated and applied to all plates")
            else:
                logger.info("Configuration cancelled")

        # LAZY IMPORT to avoid circular import (evidence: pipeline_editor.py lines 166, 212)
        from openhcs.textual_tui.screens.config_dialog import ConfigDialogScreen
        from openhcs.core.config import GlobalPipelineConfig

        # Get current config from app (evidence: app.py line 145)
        current_config = self.app.global_config

        # Launch modal
        self.app.push_screen(
            ConfigDialogScreen(GlobalPipelineConfig, current_config),
            handle_result
        )

    def _propagate_global_config_to_orchestrators(self, new_config: GlobalPipelineConfig) -> None:
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
                logger.info("Pipeline editor will automatically use updated app.global_config")
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

    def _save_config_to_cache(self, config: GlobalPipelineConfig) -> None:
        """Save config to cache asynchronously."""
        async def _async_save():
            from openhcs.textual_tui.services.global_config_cache import save_global_config_to_cache
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
    
    def action_show_help(self) -> None:
        """Show help dialog."""

        # LAZY IMPORT to avoid circular import
        from openhcs.textual_tui.screens.help_dialog import HelpDialogScreen

        # Launch modal (no result callback needed)
        self.app.push_screen(HelpDialogScreen())
    
    def action_quit(self) -> None:
        """Quit the application."""
        self.app.action_quit()
    
    def watch_app_title(self, title: str) -> None:
        """Update the title display when app_title changes."""
        try:
            title_widget = self.query_one("#app_title")
            title_widget.update(title)
        except Exception:
            # Widget might not be mounted yet
            pass
