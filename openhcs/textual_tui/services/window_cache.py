"""
Window position and size caching service for OpenHCS TUI.

Provides persistent storage of window positions and sizes across application sessions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from textual.geometry import Offset, Size

logger = logging.getLogger(__name__)


class WindowCache:
    """Service for caching window positions and sizes."""
    
    def __init__(self):
        """Initialize the window cache service."""
        # Use the same directory as other OpenHCS data
        self.cache_dir = Path.home() / ".local" / "share" / "openhcs"
        self.cache_file = self.cache_dir / "window_cache.json"
        self._cache_data: Dict[str, Dict[str, Dict[str, int]]] = {}
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cache data from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    self._cache_data = json.load(f)
                logger.debug(f"Loaded window cache from {self.cache_file}")
            else:
                self._cache_data = {}
                logger.debug("No existing window cache found, starting fresh")
        except Exception as e:
            logger.warning(f"Failed to load window cache: {e}")
            self._cache_data = {}
    
    def _save_cache(self) -> None:
        """Save cache data to disk."""
        try:
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache_data, f, indent=2)
            logger.debug(f"Saved window cache to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save window cache: {e}")
    
    def save_window_state(self, window_id: str, position: Offset, size: Size) -> None:
        """
        Save window position and size to cache.
        
        Args:
            window_id: Unique window identifier (window button name)
            position: Window position (Offset with x, y)
            size: Window size (Size with width, height)
        """
        try:
            self._cache_data[window_id] = {
                "position": {"x": position.x, "y": position.y},
                "size": {"width": size.width, "height": size.height}
            }
            self._save_cache()
            logger.debug(f"Cached window state for '{window_id}': pos=({position.x},{position.y}), size=({size.width}x{size.height})")
        except Exception as e:
            logger.error(f"Failed to save window state for '{window_id}': {e}")
    
    def load_window_state(self, window_id: str) -> Optional[Tuple[Offset, Size]]:
        """
        Load window position and size from cache.
        
        Args:
            window_id: Unique window identifier (window button name)
            
        Returns:
            Tuple of (position, size) if cached, None if not found
        """
        try:
            if window_id in self._cache_data:
                data = self._cache_data[window_id]
                position = Offset(data["position"]["x"], data["position"]["y"])
                size = Size(data["size"]["width"], data["size"]["height"])
                logger.debug(f"Loaded cached window state for '{window_id}': pos=({position.x},{position.y}), size=({size.width}x{size.height})")
                return position, size
            else:
                logger.debug(f"No cached state found for window '{window_id}'")
                return None
        except Exception as e:
            logger.error(f"Failed to load window state for '{window_id}': {e}")
            return None
    
    def clear_window_cache(self, window_id: Optional[str] = None) -> None:
        """
        Clear cached window state.
        
        Args:
            window_id: Specific window to clear, or None to clear all
        """
        try:
            if window_id:
                if window_id in self._cache_data:
                    del self._cache_data[window_id]
                    self._save_cache()
                    logger.debug(f"Cleared cache for window '{window_id}'")
            else:
                self._cache_data.clear()
                self._save_cache()
                logger.debug("Cleared all window cache data")
        except Exception as e:
            logger.error(f"Failed to clear window cache: {e}")


# Global cache instance
_window_cache: Optional[WindowCache] = None


def get_window_cache() -> WindowCache:
    """Get the global window cache instance."""
    global _window_cache
    if _window_cache is None:
        _window_cache = WindowCache()
    return _window_cache
