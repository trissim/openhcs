"""
QPalette Manager for OpenHCS PyQt6 GUI

Manages QPalette integration with PyQt6ColorScheme for system-wide theming.
Provides utilities for applying color schemes to Qt's palette system and
managing theme switching across the entire application.
"""

import logging
from typing import Optional
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtWidgets import QApplication
from openhcs.pyqt_gui.shared.color_scheme import PyQt6ColorScheme

logger = logging.getLogger(__name__)


class PaletteManager:
    """
    Manages QPalette integration with PyQt6ColorScheme.
    
    Provides methods to apply color schemes to Qt's palette system,
    enabling system-wide theming and consistent color application.
    """
    
    def __init__(self, color_scheme: PyQt6ColorScheme):
        """
        Initialize the palette manager with a color scheme.
        
        Args:
            color_scheme: PyQt6ColorScheme instance to use for palette generation
        """
        self.color_scheme = color_scheme
        self._original_palette = None
    
    def update_color_scheme(self, color_scheme: PyQt6ColorScheme):
        """
        Update the color scheme used for palette generation.
        
        Args:
            color_scheme: New PyQt6ColorScheme instance
        """
        self.color_scheme = color_scheme
    
    def create_palette(self) -> QPalette:
        """
        Create a QPalette from the current color scheme.
        
        Returns:
            QPalette: Configured palette with color scheme colors
        """
        palette = QPalette()
        cs = self.color_scheme
        
        # Window colors
        palette.setColor(QPalette.ColorRole.Window, cs.to_qcolor(cs.window_bg))
        palette.setColor(QPalette.ColorRole.WindowText, cs.to_qcolor(cs.text_primary))
        
        # Base colors (input fields, etc.)
        palette.setColor(QPalette.ColorRole.Base, cs.to_qcolor(cs.input_bg))
        palette.setColor(QPalette.ColorRole.AlternateBase, cs.to_qcolor(cs.panel_bg))
        palette.setColor(QPalette.ColorRole.Text, cs.to_qcolor(cs.input_text))
        
        # Button colors
        palette.setColor(QPalette.ColorRole.Button, cs.to_qcolor(cs.button_normal_bg))
        palette.setColor(QPalette.ColorRole.ButtonText, cs.to_qcolor(cs.button_text))
        
        # Selection colors
        palette.setColor(QPalette.ColorRole.Highlight, cs.to_qcolor(cs.selection_bg))
        palette.setColor(QPalette.ColorRole.HighlightedText, cs.to_qcolor(cs.selection_text))
        
        # Disabled colors
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, 
                        cs.to_qcolor(cs.text_disabled))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, 
                        cs.to_qcolor(cs.text_disabled))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, 
                        cs.to_qcolor(cs.button_disabled_text))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Button, 
                        cs.to_qcolor(cs.button_disabled_bg))
        
        # Additional color roles
        palette.setColor(QPalette.ColorRole.ToolTipBase, cs.to_qcolor(cs.panel_bg))
        palette.setColor(QPalette.ColorRole.ToolTipText, cs.to_qcolor(cs.text_primary))
        
        # Border/frame colors
        palette.setColor(QPalette.ColorRole.Mid, cs.to_qcolor(cs.border_color))
        palette.setColor(QPalette.ColorRole.Dark, cs.to_qcolor(cs.separator_color))
        palette.setColor(QPalette.ColorRole.Light, cs.to_qcolor(cs.border_light))
        
        return palette
    
    def apply_palette_to_application(self, app: Optional[QApplication] = None):
        """
        Apply the color scheme palette to the entire application.
        
        Args:
            app: QApplication instance (uses QApplication.instance() if None)
        """
        if app is None:
            app = QApplication.instance()
        
        if app is None:
            logger.warning("No QApplication instance found, cannot apply palette")
            return
        
        # Store original palette for restoration
        if self._original_palette is None:
            self._original_palette = app.palette()
        
        # Apply new palette
        new_palette = self.create_palette()
        app.setPalette(new_palette)
        
        logger.debug("Applied color scheme palette to application")
    
    def restore_original_palette(self, app: Optional[QApplication] = None):
        """
        Restore the original application palette.
        
        Args:
            app: QApplication instance (uses QApplication.instance() if None)
        """
        if app is None:
            app = QApplication.instance()
        
        if app is None or self._original_palette is None:
            logger.warning("Cannot restore original palette")
            return
        
        app.setPalette(self._original_palette)
        logger.debug("Restored original application palette")
    
    def get_palette_info(self) -> dict:
        """
        Get information about the current palette configuration.
        
        Returns:
            dict: Dictionary with palette color information
        """
        palette = self.create_palette()
        cs = self.color_scheme
        
        return {
            "window_bg": cs.to_hex(cs.window_bg),
            "window_text": cs.to_hex(cs.text_primary),
            "base_bg": cs.to_hex(cs.input_bg),
            "base_text": cs.to_hex(cs.input_text),
            "button_bg": cs.to_hex(cs.button_normal_bg),
            "button_text": cs.to_hex(cs.button_text),
            "selection_bg": cs.to_hex(cs.selection_bg),
            "selection_text": cs.to_hex(cs.selection_text),
            "disabled_text": cs.to_hex(cs.text_disabled),
        }


class ThemeManager:
    """
    High-level theme management for the entire application.
    
    Coordinates color scheme, style sheet generation, and palette management
    to provide seamless theme switching capabilities.
    """
    
    def __init__(self, initial_color_scheme: Optional[PyQt6ColorScheme] = None):
        """
        Initialize the theme manager.
        
        Args:
            initial_color_scheme: Initial color scheme (defaults to dark theme)
        """
        self.color_scheme = initial_color_scheme or PyQt6ColorScheme()
        self.palette_manager = PaletteManager(self.color_scheme)
        
        # Import here to avoid circular imports
        from openhcs.pyqt_gui.shared.style_generator import StyleSheetGenerator
        self.style_generator = StyleSheetGenerator(self.color_scheme)
        
        self._theme_change_callbacks = []
    
    def switch_to_dark_theme(self):
        """Switch to dark theme variant."""
        self.apply_color_scheme(PyQt6ColorScheme.create_dark_theme())
    
    def switch_to_light_theme(self):
        """Switch to light theme variant."""
        self.apply_color_scheme(PyQt6ColorScheme.create_light_theme())
    
    def apply_color_scheme(self, color_scheme: PyQt6ColorScheme):
        """
        Apply a new color scheme to the entire application.
        
        Args:
            color_scheme: New PyQt6ColorScheme to apply
        """
        self.color_scheme = color_scheme
        self.palette_manager.update_color_scheme(color_scheme)
        self.style_generator.update_color_scheme(color_scheme)
        
        # Apply to application
        self.palette_manager.apply_palette_to_application()
        
        # Notify callbacks
        for callback in self._theme_change_callbacks:
            try:
                callback(color_scheme)
            except Exception as e:
                logger.warning(f"Theme change callback failed: {e}")
        
        logger.info("Applied new color scheme to application")
    
    def register_theme_change_callback(self, callback):
        """
        Register a callback to be called when theme changes.
        
        Args:
            callback: Function to call with new color scheme
        """
        self._theme_change_callbacks.append(callback)
    
    def unregister_theme_change_callback(self, callback):
        """
        Unregister a theme change callback.
        
        Args:
            callback: Function to remove from callbacks
        """
        if callback in self._theme_change_callbacks:
            self._theme_change_callbacks.remove(callback)
    
    def get_current_style_sheet(self) -> str:
        """
        Get the current complete application style sheet.
        
        Returns:
            str: Complete QStyleSheet for current theme
        """
        return self.style_generator.generate_complete_application_style()
    
    def load_theme_from_config(self, config_path: str) -> bool:
        """
        Load and apply theme from configuration file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            color_scheme = PyQt6ColorScheme.load_color_scheme_from_config(config_path)
            self.apply_color_scheme(color_scheme)
            return True
        except Exception as e:
            logger.error(f"Failed to load theme from {config_path}: {e}")
            return False
    
    def save_current_theme(self, config_path: str) -> bool:
        """
        Save current theme to configuration file.
        
        Args:
            config_path: Path to save JSON configuration file
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.color_scheme.save_to_json(config_path)
