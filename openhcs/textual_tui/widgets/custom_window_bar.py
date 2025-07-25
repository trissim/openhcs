"""
Custom WindowBar for OpenHCS that removes left button bar and adds separators.

Extends textual-window WindowBar to customize button layout and appearance.
"""

import logging
from typing import Any

from textual import work
from textual.app import ComposeResult
from textual.widgets import Static
from textual_window import WindowBar
from textual_window.windowbar import WindowBarAllButton, WindowBarButton
from textual_window.window import Window

logger = logging.getLogger(__name__)


# ButtonSeparator removed - not using separators anymore


class CustomWindowBar(WindowBar):
    """
    Custom WindowBar that removes the left button bar and adds separators between buttons.

    Layout: [Window1] | [Window2] | [Window3] [Right All Button]
    """

    def __init__(self, **kwargs):
        """Initialize with logging."""
        super().__init__(**kwargs)

        # Verify our methods are being used
    
    DEFAULT_CSS = """
    CustomWindowBar {
        align: center bottom;
        background: $panel;
    }
    WindowBarButton {
        height: 1; width: auto;
        padding: 0 1;
        border-left: solid $panel-lighten-2;
        &:hover { background: $panel-lighten-1; }
        &.pressed { background: $primary; color: $text; }
        &.right_pressed { background: $accent-darken-3; color: $text; }
    }
    WindowBarAllButton {
        height: 1; width: 1fr;  /* Keep 1fr to fill remaining space */
        padding: 0 1;
        border-left: solid $panel-lighten-2;
        &:hover { background: $boost; }
        &.pressed { background: $panel-lighten-1; }
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the window bar with only the right button (no left button)."""
        # Only yield the right button - no left button
        yield WindowBarAllButton(window_bar=self, id="windowbar_button_right")

    def on_mount(self) -> None:
        """Log children after mounting to see what actually exists."""
        all_children = [f"{child.__class__.__name__}(id={getattr(child, 'id', 'no-id')})" for child in self.children]

        # Check if both buttons exist
        try:
            left_button = self.query_one("#windowbar_button_left")
        except Exception as e:
            logger.error(f"🔘 WINDOWBAR MOUNT: Left button missing: {e}")

        try:
            right_button = self.query_one("#windowbar_button_right")
        except Exception as e:
            logger.error(f"🔘 WINDOWBAR MOUNT: Right button missing: {e}")



    @work(group="windowbar")
    async def add_window_button(self, window: Window) -> None:
        """
        Add a window button with separator.

        Override the parent method to add separators between buttons.
        """
        try:

            # Check if button already exists
            try:
                existing_button = self.query_one(f"#{window.id}_button")
                return
            except Exception:
                pass  # Button doesn't exist, continue with creation

            display_name = (window.icon + " " + window.name) if window.icon else window.name
            logger.debug(f"🔘 BUTTON CREATE: Display name = '{display_name}'")

            # Check if right button exists
            try:
                right_button = self.query_one("#windowbar_button_right")
                logger.debug(f"🔘 BUTTON CREATE: Right button found: {right_button}")
            except Exception as e:
                logger.error(f"🔘 BUTTON CREATE: Right button missing! {e}")
                raise

            # Add the window button directly (no separators)
            logger.debug(f"🔘 BUTTON CREATE: Creating WindowBarButton for {window.id}")
            try:
                button = WindowBarButton(
                    content=display_name,
                    window=window,
                    window_bar=self,
                    id=f"{window.id}_button",
                )
                logger.debug(f"🔘 BUTTON CREATE: WindowBarButton created: {button}")

                await self.mount(
                    button,
                    before=self.query_one("#windowbar_button_right"),
                )
                logger.debug(f"🔘 BUTTON CREATE: Button mounted for {window.id}")

                # Verify button was actually added
                try:
                    verify_button = self.query_one(f"#{window.id}_button")
                except Exception as e:
                    logger.error(f"🔘 BUTTON CREATE: Button not found after mount! {window.id} - {e}")
                    raise

            except Exception as e:
                logger.error(f"🔘 BUTTON CREATE: Button mount failed for {window.id} - {e}")
                raise

        except Exception as e:
            logger.error(f"🔘 BUTTON CREATE FAILED: {window.id} - {type(e).__name__}: {e}")
            import traceback
            logger.error(f"🔘 BUTTON CREATE TRACEBACK: {traceback.format_exc()}")
            raise  # Re-raise to expose the actual error

    @work(group="windowbar")
    async def remove_window_button(self, window: Window) -> None:
        """
        Remove a window button.

        Simplified version without separators.
        """
        # Remove the window button
        try:
            self.query_one(f"#{window.id}_button").remove()
        except Exception as e:
            logger.warning(f"🔘 BUTTON REMOVE FAILED: {window.id} - {e}")





    def update_window_button_state(self, window: Window, state: bool) -> None:
        """
        Override to add comprehensive logging for button state updates.

        This is called by the WindowManager when a window is minimized or opened.
        """

        try:
            # Log current WindowBar state
            all_children = [child.id for child in self.children if hasattr(child, 'id')]
            button_children = [
                child.id for child in self.children
                if isinstance(child, WindowBarButton)
            ]
            logger.debug(f"🔘 BUTTON UPDATE: All children: {all_children}")
            logger.debug(f"🔘 BUTTON UPDATE: Button children: {button_children}")

            # Try to find the button
            button_id = f"#{window.id}_button"
            logger.debug(f"🔘 BUTTON UPDATE: Looking for button: {button_id}")

            button = self.query_one(button_id, WindowBarButton)
            logger.debug(f"🔘 BUTTON UPDATE: Found button: {button}")

            # Update the button state
            if state:
                button.window_state = True
            else:
                button.window_state = False

        except Exception as e:
            # Button doesn't exist yet - this might be normal during window creation
            # But let's log it to see what's happening
            logger.warning(f"🔘 BUTTON UPDATE FAILED: {window.id} - {type(e).__name__}: {e}")
            logger.debug(f"🔘 BUTTON UPDATE: Current children: {[child.id for child in self.children if hasattr(child, 'id')]}")
            # The button should be added later via add_window_button

    def __getattribute__(self, name):
        """Override to log when manager accesses our methods."""
        return super().__getattribute__(name)
