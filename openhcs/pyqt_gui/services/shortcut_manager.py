"""
Declarative Shortcut Manager for PyQt GUI.

Applies keyboard shortcuts from ShortcutConfig to the application.
All shortcuts are global (application-wide) and defined in config.py.
"""

import logging
from typing import Dict, Callable, Optional

from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtGui import QShortcut, QKeySequence
from PyQt6.QtCore import Qt

from openhcs.pyqt_gui.config import get_shortcut_config

logger = logging.getLogger(__name__)


class ShortcutManager:
    """
    Manages global keyboard shortcuts for the PyQt GUI application.
    
    Usage:
        # In MainWindow.__init__:
        self.shortcut_manager = ShortcutManager(self)
        self.shortcut_manager.register_action("show_plate_manager", self.show_plate_manager)
        self.shortcut_manager.register_action("time_travel_back", ObjectStateRegistry.time_travel_back)
        self.shortcut_manager.apply_shortcuts()
    """
    
    def __init__(self, main_window: QMainWindow):
        self.main_window = main_window
        self.config = get_shortcut_config()
        self._actions: Dict[str, Callable] = {}
        self._shortcuts: Dict[str, QShortcut] = {}

    def register_action(self, action_name: str, callback: Callable) -> None:
        """Register a callback for an action name."""
        self._actions[action_name] = callback
    
    def register_actions(self, actions: Dict[str, Callable]) -> None:
        """Register multiple actions at once."""
        self._actions.update(actions)
    
    def apply_shortcuts(self) -> None:
        """
        Apply all shortcuts from config to registered actions.
        
        Creates QShortcut objects for each shortcut in ShortcutConfig
        that has a registered action callback.
        """
        for shortcut_name, shortcut in self.config.shortcut_items():
            action_name = shortcut.action
            
            if action_name not in self._actions:
                logger.debug(
                    "Shortcut %r action %r not registered, skipping",
                    shortcut_name,
                    action_name,
                )
                continue
            
            # Create global shortcut
            qshortcut = QShortcut(QKeySequence(shortcut.key), self.main_window)
            qshortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            qshortcut.activated.connect(self._actions[action_name])
            
            self._shortcuts[shortcut_name] = qshortcut
            logger.debug(f"Applied shortcut: {shortcut.key} -> {action_name}")
        
        logger.info(f"Applied {len(self._shortcuts)} keyboard shortcuts")
    
    def get_shortcut_help(self) -> str:
        """Generate help text for all shortcuts."""
        lines = ["Keyboard Shortcuts:", ""]
        for _, shortcut in self.config.shortcut_items():
            lines.append(f"  {shortcut.key:20} {shortcut.description}")
        return "\n".join(lines)
