"""
Configuration Validator for OpenHCS PyQt6 Color Schemes

Validates color scheme JSON configurations for WCAG compliance, proper format,
and semantic correctness. Provides hot-reload capability and error reporting.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import fields
from openhcs.pyqt_gui.shared.color_scheme import PyQt6ColorScheme

logger = logging.getLogger(__name__)


class ColorSchemeValidator:
    """
    Validates PyQt6 color scheme configurations.
    
    Provides comprehensive validation including format checking, WCAG compliance,
    and semantic correctness validation for color scheme JSON files.
    """
    
    def __init__(self):
        """Initialize the validator."""
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate_config_file(self, config_path: str) -> bool:
        """
        Validate a color scheme configuration file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        self.validation_errors.clear()
        self.validation_warnings.clear()
        
        try:
            # Check file exists
            if not Path(config_path).exists():
                self.validation_errors.append(f"Configuration file not found: {config_path}")
                return False
            
            # Load and parse JSON
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Validate structure
            if not self._validate_structure(config):
                return False
            
            # Validate colors
            if not self._validate_colors(config):
                return False
            
            # Validate WCAG compliance
            if not self._validate_wcag_compliance(config):
                return False
            
            # Validate theme variants
            if not self._validate_theme_variants(config):
                return False
            
            logger.info(f"Color scheme configuration validated successfully: {config_path}")
            return True
            
        except json.JSONDecodeError as e:
            self.validation_errors.append(f"Invalid JSON format: {e}")
            return False
        except Exception as e:
            self.validation_errors.append(f"Validation error: {e}")
            return False
    
    def _validate_structure(self, config: Dict[str, Any]) -> bool:
        """
        Validate the basic structure of the configuration.
        
        Args:
            config: Parsed JSON configuration
            
        Returns:
            bool: True if structure is valid
        """
        required_sections = [
            "base_ui_colors",
            "text_colors", 
            "interactive_elements",
            "selection_and_highlighting",
            "status_colors",
            "log_highlighting",
            "python_syntax"
        ]
        
        for section in required_sections:
            if section not in config:
                self.validation_errors.append(f"Missing required section: {section}")
                return False
        
        # Validate metadata
        if "_schema_version" not in config:
            self.validation_warnings.append("Missing schema version")
        
        return True
    
    def _validate_colors(self, config: Dict[str, Any]) -> bool:
        """
        Validate color format and values.
        
        Args:
            config: Parsed JSON configuration
            
        Returns:
            bool: True if colors are valid
        """
        valid = True
        
        def validate_color_section(section_name: str, section_data: Dict[str, Any]):
            nonlocal valid
            for color_name, color_value in section_data.items():
                if color_name.startswith("_"):
                    continue  # Skip metadata
                
                if not self._validate_color_value(color_name, color_value):
                    valid = False
        
        # Validate all color sections
        for section_name, section_data in config.items():
            if isinstance(section_data, dict) and not section_name.startswith("_"):
                if section_name not in ["theme_variants", "accessibility", "metadata"]:
                    validate_color_section(section_name, section_data)
        
        return valid
    
    def _validate_color_value(self, color_name: str, color_value: Any) -> bool:
        """
        Validate a single color value.
        
        Args:
            color_name: Name of the color
            color_value: Color value to validate
            
        Returns:
            bool: True if color value is valid
        """
        if not isinstance(color_value, list):
            self.validation_errors.append(f"Color {color_name} must be a list, got {type(color_value)}")
            return False
        
        # Check RGB or RGBA format
        if len(color_value) not in [3, 4]:
            self.validation_errors.append(f"Color {color_name} must have 3 (RGB) or 4 (RGBA) values")
            return False
        
        # Validate color component values
        for i, component in enumerate(color_value):
            if not isinstance(component, int):
                self.validation_errors.append(f"Color {color_name} component {i} must be integer")
                return False
            
            max_value = 255 if i < 3 else 255  # RGB: 0-255, Alpha: 0-255
            if not (0 <= component <= max_value):
                self.validation_errors.append(f"Color {color_name} component {i} out of range (0-{max_value})")
                return False
        
        return True
    
    def _validate_wcag_compliance(self, config: Dict[str, Any]) -> bool:
        """
        Validate WCAG contrast ratio compliance.
        
        Args:
            config: Parsed JSON configuration
            
        Returns:
            bool: True if WCAG compliance is satisfied
        """
        # Create temporary color scheme for validation
        try:
            color_scheme = self._create_color_scheme_from_config(config)
            
            # Define critical color combinations to validate
            critical_combinations = [
                (color_scheme.text_primary, color_scheme.window_bg, "Primary text on window"),
                (color_scheme.text_secondary, color_scheme.window_bg, "Secondary text on window"),
                (color_scheme.button_text, color_scheme.button_normal_bg, "Button text on button"),
                (color_scheme.input_text, color_scheme.input_bg, "Input text on input field"),
                (color_scheme.selection_text, color_scheme.selection_bg, "Selected text on selection"),
            ]
            
            min_ratio = 4.5  # WCAG AA standard
            all_valid = True
            
            for fg, bg, description in critical_combinations:
                if not color_scheme.validate_wcag_contrast(fg, bg, min_ratio):
                    contrast_ratio = self._calculate_contrast_ratio(fg, bg)
                    self.validation_errors.append(
                        f"WCAG compliance failure: {description} "
                        f"(contrast ratio: {contrast_ratio:.2f}, required: {min_ratio})"
                    )
                    all_valid = False
            
            return all_valid
            
        except Exception as e:
            self.validation_errors.append(f"WCAG validation error: {e}")
            return False
    
    def _validate_theme_variants(self, config: Dict[str, Any]) -> bool:
        """
        Validate theme variant configurations.
        
        Args:
            config: Parsed JSON configuration
            
        Returns:
            bool: True if theme variants are valid
        """
        if "theme_variants" not in config:
            return True  # Optional section
        
        variants = config["theme_variants"]
        
        for variant_name, variant_config in variants.items():
            if variant_name.startswith("_"):
                continue  # Skip metadata
            
            # Validate variant colors
            for color_name, color_value in variant_config.items():
                if color_name.startswith("_"):
                    continue  # Skip metadata
                
                if not self._validate_color_value(f"{variant_name}.{color_name}", color_value):
                    return False
        
        return True
    
    def _create_color_scheme_from_config(self, config: Dict[str, Any]) -> PyQt6ColorScheme:
        """
        Create a PyQt6ColorScheme from configuration for validation.
        
        Args:
            config: Parsed JSON configuration
            
        Returns:
            PyQt6ColorScheme: Color scheme instance
        """
        # Flatten all color sections into a single dict
        all_colors = {}
        
        for section_name, section_data in config.items():
            if isinstance(section_data, dict) and not section_name.startswith("_"):
                if section_name not in ["theme_variants", "accessibility", "metadata"]:
                    for color_name, color_value in section_data.items():
                        if not color_name.startswith("_"):
                            all_colors[color_name] = tuple(color_value)
        
        # Create color scheme with available colors
        valid_fields = {f.name for f in fields(PyQt6ColorScheme)}
        scheme_kwargs = {k: v for k, v in all_colors.items() if k in valid_fields}
        
        return PyQt6ColorScheme(**scheme_kwargs)
    
    def _calculate_contrast_ratio(self, fg: Tuple[int, int, int], bg: Tuple[int, int, int]) -> float:
        """
        Calculate contrast ratio between two colors.
        
        Args:
            fg: Foreground color RGB tuple
            bg: Background color RGB tuple
            
        Returns:
            float: Contrast ratio
        """
        def relative_luminance(color: Tuple[int, int, int]) -> float:
            r, g, b = [c / 255.0 for c in color]
            
            def gamma_correct(c):
                return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
            
            r, g, b = map(gamma_correct, [r, g, b])
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        l1 = relative_luminance(fg)
        l2 = relative_luminance(bg)
        
        if l1 < l2:
            l1, l2 = l2, l1
        
        return (l1 + 0.05) / (l2 + 0.05)
    
    def get_validation_report(self) -> Dict[str, List[str]]:
        """
        Get the validation report with errors and warnings.
        
        Returns:
            Dict[str, List[str]]: Dictionary with 'errors' and 'warnings' lists
        """
        return {
            "errors": self.validation_errors.copy(),
            "warnings": self.validation_warnings.copy()
        }
    
    def print_validation_report(self):
        """Print the validation report to console."""
        if self.validation_errors:
            print("❌ Validation Errors:")
            for error in self.validation_errors:
                print(f"  • {error}")
        
        if self.validation_warnings:
            print("⚠️  Validation Warnings:")
            for warning in self.validation_warnings:
                print(f"  • {warning}")
        
        if not self.validation_errors and not self.validation_warnings:
            print("✅ Validation passed with no errors or warnings")
