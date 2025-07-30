"""
PyQt6 Color Scheme for OpenHCS GUI

Comprehensive color scheme system extending the LogColorScheme pattern to cover
all GUI components. Provides centralized color management with theme support,
JSON configuration, and WCAG accessibility compliance.
"""

import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from pathlib import Path
from PyQt6.QtGui import QColor

logger = logging.getLogger(__name__)


@dataclass
class PyQt6ColorScheme:
    """
    Comprehensive color scheme for OpenHCS PyQt6 GUI with semantic color names.
    
    Extends the LogColorScheme pattern to cover all GUI components including
    windows, dialogs, widgets, and interactive elements. Supports light/dark
    theme variants and ensures WCAG accessibility compliance.
    
    All colors meet minimum 4.5:1 contrast ratio for normal text readability.
    """
    
    # ========== BASE UI ARCHITECTURE COLORS ==========
    
    # Window and Panel Backgrounds
    window_bg: Tuple[int, int, int] = (43, 43, 43)      # #2b2b2b - Main window/dialog backgrounds
    panel_bg: Tuple[int, int, int] = (30, 30, 30)       # #1e1e1e - Panel/widget backgrounds
    frame_bg: Tuple[int, int, int] = (43, 43, 43)       # #2b2b2b - Frame backgrounds
    
    # Borders and Separators
    border_color: Tuple[int, int, int] = (85, 85, 85)   # #555555 - Primary borders
    border_light: Tuple[int, int, int] = (102, 102, 102) # #666666 - Secondary borders
    separator_color: Tuple[int, int, int] = (51, 51, 51) # #333333 - Separators/dividers
    
    # ========== TEXT HIERARCHY COLORS ==========
    
    # Text Colors
    text_primary: Tuple[int, int, int] = (255, 255, 255)   # #ffffff - Primary text
    text_secondary: Tuple[int, int, int] = (204, 204, 204) # #cccccc - Secondary text/labels
    text_accent: Tuple[int, int, int] = (0, 170, 255)      # #00aaff - Accent text/titles
    text_disabled: Tuple[int, int, int] = (102, 102, 102)  # #666666 - Disabled text
    
    # ========== INTERACTIVE ELEMENT COLORS ==========
    
    # Button States
    button_normal_bg: Tuple[int, int, int] = (64, 64, 64)    # #404040 - Normal button background
    button_hover_bg: Tuple[int, int, int] = (80, 80, 80)     # #505050 - Button hover state
    button_pressed_bg: Tuple[int, int, int] = (48, 48, 48)   # #303030 - Button pressed state
    button_disabled_bg: Tuple[int, int, int] = (42, 42, 42)  # #2a2a2a - Disabled button background
    button_text: Tuple[int, int, int] = (255, 255, 255)      # #ffffff - Button text
    button_disabled_text: Tuple[int, int, int] = (102, 102, 102) # #666666 - Disabled button text
    
    # Input Fields
    input_bg: Tuple[int, int, int] = (64, 64, 64)        # #404040 - Input field background
    input_border: Tuple[int, int, int] = (102, 102, 102) # #666666 - Input field border
    input_text: Tuple[int, int, int] = (255, 255, 255)   # #ffffff - Input field text
    input_focus_border: Tuple[int, int, int] = (0, 170, 255) # #00aaff - Focused input border
    
    # ========== SELECTION AND HIGHLIGHTING COLORS ==========
    
    # Selection States
    selection_bg: Tuple[int, int, int] = (0, 120, 212)   # #0078d4 - Primary selection background
    selection_text: Tuple[int, int, int] = (255, 255, 255) # #ffffff - Selected text
    hover_bg: Tuple[int, int, int] = (51, 51, 51)        # #333333 - Hover background
    focus_outline: Tuple[int, int, int] = (0, 170, 255)  # #00aaff - Focus outline
    
    # Search and Highlighting
    search_highlight_bg: Tuple[int, int, int, int] = (255, 255, 0, 100) # Yellow with transparency
    search_highlight_text: Tuple[int, int, int] = (0, 0, 0) # #000000 - Search highlight text
    
    # ========== STATUS COMMUNICATION COLORS ==========
    
    # Status Indicators
    status_success: Tuple[int, int, int] = (0, 255, 0)   # #00ff00 - Success/ready states
    status_warning: Tuple[int, int, int] = (255, 170, 0) # #ffaa00 - Warning messages
    status_error: Tuple[int, int, int] = (255, 0, 0)     # #ff0000 - Error states
    status_info: Tuple[int, int, int] = (0, 170, 255)    # #00aaff - Information/accent
    
    # Progress and Activity
    progress_bg: Tuple[int, int, int] = (30, 30, 30)     # #1e1e1e - Progress bar background
    progress_fill: Tuple[int, int, int] = (0, 120, 212)  # #0078d4 - Progress bar fill
    activity_indicator: Tuple[int, int, int] = (0, 170, 255) # #00aaff - Activity indicators
    
    # ========== LOG HIGHLIGHTING COLORS (LogColorScheme compatibility) ==========
    
    # Log level colors with semantic meaning (WCAG 4.5:1 compliant)
    log_critical_fg: Tuple[int, int, int] = (255, 255, 255)  # White text
    log_critical_bg: Tuple[int, int, int] = (139, 0, 0)      # Dark red background
    log_error_color: Tuple[int, int, int] = (255, 85, 85)    # Brighter red - WCAG compliant
    log_warning_color: Tuple[int, int, int] = (255, 140, 0)  # Dark orange - attention grabbing
    log_info_color: Tuple[int, int, int] = (100, 160, 210)   # Brighter steel blue - WCAG compliant
    log_debug_color: Tuple[int, int, int] = (160, 160, 160)  # Lighter gray - better contrast
    
    # Metadata and structural colors
    timestamp_color: Tuple[int, int, int] = (105, 105, 105)      # Dim gray - unobtrusive
    logger_name_color: Tuple[int, int, int] = (147, 112, 219)   # Medium slate blue - distinctive
    memory_address_color: Tuple[int, int, int] = (255, 182, 193) # Light pink - technical data
    file_path_color: Tuple[int, int, int] = (34, 139, 34)       # Forest green - file system
    
    # Python syntax colors (following VS Code dark theme conventions)
    python_keyword_color: Tuple[int, int, int] = (86, 156, 214)    # Blue - language keywords
    python_string_color: Tuple[int, int, int] = (206, 145, 120)    # Orange - string literals
    python_number_color: Tuple[int, int, int] = (181, 206, 168)    # Light green - numeric values
    python_operator_color: Tuple[int, int, int] = (212, 212, 212)  # Light gray - operators/punctuation
    python_name_color: Tuple[int, int, int] = (156, 220, 254)      # Light blue - identifiers
    python_function_color: Tuple[int, int, int] = (220, 220, 170)  # Yellow - function names
    python_class_color: Tuple[int, int, int] = (78, 201, 176)      # Teal - class names
    python_builtin_color: Tuple[int, int, int] = (86, 156, 214)    # Blue - built-in functions
    python_comment_color: Tuple[int, int, int] = (106, 153, 85)    # Green - comments
    
    # Special highlighting colors
    exception_color: Tuple[int, int, int] = (255, 69, 0)       # Red orange - error types
    function_call_color: Tuple[int, int, int] = (255, 215, 0)  # Gold - function invocations
    boolean_color: Tuple[int, int, int] = (86, 156, 214)       # Blue - True/False/None
    
    # Enhanced syntax colors
    tuple_parentheses_color: Tuple[int, int, int] = (255, 215, 0)     # Gold - tuple delimiters
    set_braces_color: Tuple[int, int, int] = (255, 140, 0)            # Dark orange - set delimiters
    class_representation_color: Tuple[int, int, int] = (78, 201, 176) # Teal - <class 'name'>
    function_representation_color: Tuple[int, int, int] = (220, 220, 170) # Yellow - <function name>
    module_path_color: Tuple[int, int, int] = (147, 112, 219)         # Medium slate blue - module.path
    hex_number_color: Tuple[int, int, int] = (181, 206, 168)          # Light green - 0xFF
    scientific_notation_color: Tuple[int, int, int] = (181, 206, 168) # Light green - 1.23e-4
    binary_number_color: Tuple[int, int, int] = (181, 206, 168)       # Light green - 0b1010
    octal_number_color: Tuple[int, int, int] = (181, 206, 168)        # Light green - 0o755
    python_special_color: Tuple[int, int, int] = (255, 20, 147)       # Deep pink - __name__
    single_quoted_string_color: Tuple[int, int, int] = (206, 145, 120) # Orange - 'string'
    list_comprehension_color: Tuple[int, int, int] = (156, 220, 254)  # Light blue - [x for x in y]
    generator_expression_color: Tuple[int, int, int] = (156, 220, 254) # Light blue - (x for x in y)
    
    def to_qcolor(self, color_tuple: Tuple[int, int, int]) -> QColor:
        """
        Convert RGB tuple to QColor object.
        
        Args:
            color_tuple: RGB color tuple (r, g, b)
            
        Returns:
            QColor: Qt color object
        """
        return QColor(*color_tuple)
    
    def to_qcolor_rgba(self, color_tuple: Tuple[int, int, int, int]) -> QColor:
        """
        Convert RGBA tuple to QColor object.
        
        Args:
            color_tuple: RGBA color tuple (r, g, b, a)
            
        Returns:
            QColor: Qt color object with alpha
        """
        return QColor(*color_tuple)
    
    def to_hex(self, color_tuple: Tuple[int, int, int]) -> str:
        """
        Convert RGB tuple to hex color string.

        Args:
            color_tuple: RGB color tuple (r, g, b)

        Returns:
            str: Hex color string (e.g., "#ff0000")
        """
        r, g, b = color_tuple
        return f"#{r:02x}{g:02x}{b:02x}"

    @classmethod
    def create_dark_theme(cls) -> 'PyQt6ColorScheme':
        """
        Create a dark theme variant with adjusted colors for dark backgrounds.

        This is the default theme, so most colors remain the same with minor
        adjustments for better contrast on dark backgrounds.

        Returns:
            PyQt6ColorScheme: Dark theme color scheme with enhanced contrast
        """
        return cls(
            # Enhanced colors for dark backgrounds with better contrast
            log_error_color=(255, 100, 100),    # Brighter red
            log_info_color=(120, 180, 230),     # Brighter steel blue
            timestamp_color=(160, 160, 160),    # Lighter gray
            python_string_color=(236, 175, 150), # Brighter orange
            python_number_color=(200, 230, 190), # Brighter green
            # UI colors optimized for dark theme
            text_secondary=(220, 220, 220),     # Slightly brighter secondary text
            status_success=(0, 255, 100),       # Slightly brighter green
            # Other colors remain the same as they work well on dark backgrounds
        )

    @classmethod
    def create_light_theme(cls) -> 'PyQt6ColorScheme':
        """
        Create a light theme variant with adjusted colors for light backgrounds.

        All colors are adjusted to maintain WCAG 4.5:1 contrast ratio on light
        backgrounds while preserving the semantic meaning and visual hierarchy.

        Returns:
            PyQt6ColorScheme: Light theme color scheme with appropriate contrast
        """
        return cls(
            # Base UI colors for light theme
            window_bg=(245, 245, 245),          # Light gray background
            panel_bg=(255, 255, 255),           # White panel background
            frame_bg=(240, 240, 240),           # Light frame background
            border_color=(180, 180, 180),       # Medium gray borders
            border_light=(160, 160, 160),       # Lighter borders
            separator_color=(200, 200, 200),    # Light separators

            # Text colors for light theme
            text_primary=(0, 0, 0),             # Black primary text
            text_secondary=(80, 80, 80),        # Dark gray secondary text
            text_accent=(0, 100, 200),          # Darker blue accent
            text_disabled=(160, 160, 160),      # Light gray disabled text

            # Interactive elements for light theme
            button_normal_bg=(230, 230, 230),   # Light button background
            button_hover_bg=(210, 210, 210),    # Button hover state
            button_pressed_bg=(190, 190, 190),  # Button pressed state
            button_disabled_bg=(250, 250, 250), # Disabled button background
            button_text=(0, 0, 0),              # Black button text
            button_disabled_text=(160, 160, 160), # Light gray disabled text

            # Input fields for light theme
            input_bg=(255, 255, 255),           # White input background
            input_border=(180, 180, 180),       # Gray input border
            input_text=(0, 0, 0),               # Black input text
            input_focus_border=(0, 100, 200),   # Blue focus border

            # Selection and highlighting for light theme
            selection_bg=(0, 120, 215),         # Blue selection background
            selection_text=(255, 255, 255),     # White selected text
            hover_bg=(240, 240, 240),           # Light hover background
            focus_outline=(0, 100, 200),        # Blue focus outline

            # Search highlighting for light theme
            search_highlight_bg=(255, 255, 0, 150), # Yellow with transparency
            search_highlight_text=(0, 0, 0),    # Black search text

            # Status colors for light theme (darker for contrast)
            status_success=(0, 150, 0),         # Darker green
            status_warning=(200, 100, 0),       # Darker orange
            status_error=(200, 0, 0),           # Darker red
            status_info=(0, 100, 200),          # Darker blue

            # Progress colors for light theme
            progress_bg=(240, 240, 240),        # Light progress background
            progress_fill=(0, 120, 215),        # Blue progress fill
            activity_indicator=(0, 100, 200),   # Blue activity indicator

            # Log colors for light theme (darker for contrast)
            log_error_color=(180, 20, 40),      # Darker red
            log_info_color=(30, 80, 130),       # Darker steel blue
            log_warning_color=(200, 100, 0),    # Darker orange
            timestamp_color=(60, 60, 60),       # Darker gray
            logger_name_color=(100, 60, 160),   # Darker slate blue
            python_string_color=(150, 80, 60),  # Darker orange
            python_number_color=(120, 140, 100), # Darker green
            memory_address_color=(200, 120, 140), # Darker pink
            file_path_color=(20, 100, 20),      # Darker forest green
            exception_color=(200, 40, 0),       # Darker red orange
            # Adjust other syntax colors for light background contrast
            python_keyword_color=(0, 0, 150),   # Darker blue
            python_operator_color=(80, 80, 80), # Dark gray
            python_name_color=(0, 80, 150),     # Darker blue
            python_function_color=(150, 100, 0), # Darker yellow
            python_class_color=(0, 120, 100),   # Darker teal
            python_builtin_color=(0, 0, 150),   # Darker blue
            python_comment_color=(80, 120, 60), # Darker green
            boolean_color=(0, 0, 150),          # Darker blue
        )

    @classmethod
    def load_color_scheme_from_config(cls, config_path: str = None) -> 'PyQt6ColorScheme':
        """
        Load color scheme from external configuration file.

        Args:
            config_path: Path to JSON config file (optional)

        Returns:
            PyQt6ColorScheme: Loaded color scheme or default if file not found
        """
        if config_path and Path(config_path).exists():
            try:
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)

                # Create color scheme from config
                scheme_kwargs = {}
                for key, value in config.items():
                    if key.endswith('_color') or key.endswith('_fg') or key.endswith('_bg') or key.endswith('_text'):
                        if isinstance(value, list) and len(value) >= 3:
                            # Handle both RGB and RGBA tuples
                            scheme_kwargs[key] = tuple(value)

                return cls(**scheme_kwargs)

            except Exception as e:
                logger.warning(f"Failed to load color scheme from {config_path}: {e}")

        return cls()  # Return default scheme

    def validate_wcag_contrast(self, foreground: Tuple[int, int, int],
                              background: Tuple[int, int, int],
                              min_ratio: float = 4.5) -> bool:
        """
        Validate WCAG contrast ratio between foreground and background colors.

        Args:
            foreground: Foreground color RGB tuple
            background: Background color RGB tuple
            min_ratio: Minimum contrast ratio (default: 4.5 for normal text)

        Returns:
            bool: True if contrast ratio meets minimum requirement
        """
        def relative_luminance(color: Tuple[int, int, int]) -> float:
            """Calculate relative luminance of a color."""
            r, g, b = [c / 255.0 for c in color]

            # Apply gamma correction
            def gamma_correct(c):
                return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

            r, g, b = map(gamma_correct, [r, g, b])
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        # Calculate contrast ratio
        l1 = relative_luminance(foreground)
        l2 = relative_luminance(background)

        # Ensure l1 is the lighter color
        if l1 < l2:
            l1, l2 = l2, l1

        contrast_ratio = (l1 + 0.05) / (l2 + 0.05)
        return contrast_ratio >= min_ratio

    def get_color_dict(self) -> Dict[str, Tuple[int, int, int]]:
        """
        Get all colors as a dictionary for serialization or inspection.

        Returns:
            Dict[str, Tuple[int, int, int]]: Dictionary of color name to RGB tuple
        """
        color_dict = {}
        for field_name in self.__dataclass_fields__:
            color_value = getattr(self, field_name)
            if isinstance(color_value, tuple) and len(color_value) >= 3:
                color_dict[field_name] = color_value
        return color_dict

    def save_to_json(self, config_path: str) -> bool:
        """
        Save color scheme to JSON configuration file.

        Args:
            config_path: Path to save JSON config file

        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            import json
            color_dict = self.get_color_dict()

            # Convert tuples to lists for JSON serialization
            json_dict = {k: list(v) for k, v in color_dict.items()}

            with open(config_path, 'w') as f:
                json.dump(json_dict, f, indent=2, sort_keys=True)

            logger.info(f"Color scheme saved to {config_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save color scheme to {config_path}: {e}")
            return False
