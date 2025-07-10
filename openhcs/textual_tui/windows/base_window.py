"""Base window class for OpenHCS TUI windows."""

from textual_window import Window
from textual.app import ComposeResult
from textual.widgets import Button
from textual.containers import Container


class BaseOpenHCSWindow(Window):
    """
    Base class for all OpenHCS windows with common functionality.

    Features:
    - Instant window open/close (no fade animations)
    - Automatic window position/size caching
    """
    
    def __init__(self, window_id: str, title: str, mode: str = "temporary", **kwargs):
        """
        Initialize base OpenHCS window.

        Args:
            window_id: Unique window identifier
            title: Window title
            mode: "temporary" or "permanent"

        Features:
            - Instant window operations (no animations)
            - Automatic position/size caching
        """
        super().__init__(
            id=window_id,
            name=title,
            mode=mode,
            allow_resize=True,
            animated=False,  # Disable fade effects for instant window open/close
            **kwargs
        )
    
    def close_window(self):
        """Close this window (handles both temporary and permanent modes)."""
        # Save window position and size to cache before closing
        import logging
        logger = logging.getLogger(__name__)

        try:
            # Get window identifier
            window_id = getattr(self, 'id', None) or self.__class__.__name__

            # Get window position and size
            if hasattr(self, 'offset') and hasattr(self, 'size'):
                position = self.offset
                size = self.size

                # Save to cache
                from openhcs.textual_tui.services.window_cache import get_window_cache
                cache = get_window_cache()
                cache.save_window_state(window_id, position, size)

                logger.info(f"💾 WINDOW CLOSING: {window_id} - Position: ({position.x},{position.y}), Size: {size.width}x{size.height}")
            else:
                logger.info(f"💾 WINDOW CLOSING: {window_id} - Position/size unavailable")
        except Exception as e:
            window_id = getattr(self, 'id', None) or self.__class__.__name__
            logger.warning(f"💾 WINDOW CLOSING: {window_id} - Cache save failed: {e}")

        # Use the correct textual-window API method
        super().close_window()  # Call the textual-window Window.close_window() method

    # Position caching removed - it was interfering with window button creation
    # Only keeping animated=False for instant window open/close
