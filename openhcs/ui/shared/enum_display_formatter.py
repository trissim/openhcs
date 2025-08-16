"""
Centralized enum display text formatting for consistent UI presentation.

This module provides a single source of truth for how enum values should be
displayed across all UI frameworks (PyQt6, Textual TUI, etc.) to eliminate
code duplication and ensure consistency.
"""

from enum import Enum
from typing import Any, Optional


class EnumDisplayFormatter:
    """
    Centralized formatter for enum display text across all UI frameworks.
    
    This class provides consistent enum formatting methods that should be used
    by all UI components instead of implementing their own enum formatting logic.
    
    Design Principles:
    - Single source of truth for enum display formatting
    - Framework-agnostic (works with PyQt6, Textual TUI, etc.)
    - Consistent behavior across the entire application
    - Clear separation between display text and internal values
    """
    
    @staticmethod
    def get_display_text(enum_value: Enum) -> str:
        """
        Get the standard display text for an enum value.
        
        This is the primary method that should be used for displaying enum values
        in UI components. It returns the enum name in uppercase format for
        consistent presentation across all frameworks.
        
        Args:
            enum_value: The enum value to format for display
            
        Returns:
            Formatted display text (enum name in uppercase)
            
        Example:
            >>> from openhcs.constants.constants import Microscope
            >>> EnumDisplayFormatter.get_display_text(Microscope.IMAGEXPRESS)
            'IMAGEXPRESS'
            >>> EnumDisplayFormatter.get_display_text(Microscope.AUTO)
            'AUTO'
        """
        if not isinstance(enum_value, Enum):
            raise TypeError(f"Expected Enum instance, got {type(enum_value)}")
        
        return enum_value.name.upper()
    
    @staticmethod
    def get_placeholder_text(enum_value: Enum, prefix: str = "Pipeline default: ") -> str:
        """
        Get formatted placeholder text for enum values.
        
        This method is specifically for creating placeholder text in form fields
        and other UI components that need to show default values.
        
        Args:
            enum_value: The enum value to format
            prefix: The prefix to use for placeholder text
            
        Returns:
            Formatted placeholder text
            
        Example:
            >>> from openhcs.constants.constants import Microscope
            >>> EnumDisplayFormatter.get_placeholder_text(Microscope.AUTO)
            'Pipeline default: AUTO'
        """
        if not isinstance(enum_value, Enum):
            raise TypeError(f"Expected Enum instance, got {type(enum_value)}")
        
        display_text = EnumDisplayFormatter.get_display_text(enum_value)
        return f"{prefix}{display_text}"
    
    @staticmethod
    def get_code_representation(enum_value: Enum) -> str:
        """
        Get the code representation of an enum value.
        
        This method is for generating code or debug output where you need
        the full qualified enum name (ClassName.ENUM_NAME format).
        
        Args:
            enum_value: The enum value to format
            
        Returns:
            Code representation of the enum
            
        Example:
            >>> from openhcs.constants.constants import Microscope
            >>> EnumDisplayFormatter.get_code_representation(Microscope.AUTO)
            'Microscope.AUTO'
        """
        if not isinstance(enum_value, Enum):
            raise TypeError(f"Expected Enum instance, got {type(enum_value)}")
        
        return f"{enum_value.__class__.__name__}.{enum_value.name}"
    
    @staticmethod
    def find_enum_by_display_text(enum_class: type, display_text: str) -> Optional[Enum]:
        """
        Find an enum value by its display text.
        
        This method allows reverse lookup of enum values from their display text,
        useful for parsing user input or widget values back to enum instances.
        
        Args:
            enum_class: The enum class to search in
            display_text: The display text to match (case-insensitive)
            
        Returns:
            The matching enum value, or None if not found
            
        Example:
            >>> from openhcs.constants.constants import Microscope
            >>> EnumDisplayFormatter.find_enum_by_display_text(Microscope, "IMAGEXPRESS")
            <Microscope.IMAGEXPRESS: 'ImageXpress'>
            >>> EnumDisplayFormatter.find_enum_by_display_text(Microscope, "imagexpress")
            <Microscope.IMAGEXPRESS: 'ImageXpress'>
        """
        if not issubclass(enum_class, Enum):
            raise TypeError(f"Expected Enum class, got {type(enum_class)}")
        
        normalized_text = display_text.upper()
        
        for enum_value in enum_class:
            if enum_value.name.upper() == normalized_text:
                return enum_value
        
        return None
    
    @staticmethod
    def get_all_display_texts(enum_class: type) -> list[str]:
        """
        Get all display texts for an enum class.
        
        This method returns a list of all possible display texts for an enum,
        useful for populating UI components like combo boxes or radio button sets.
        
        Args:
            enum_class: The enum class to get display texts for
            
        Returns:
            List of all display texts for the enum
            
        Example:
            >>> from openhcs.constants.constants import Microscope
            >>> EnumDisplayFormatter.get_all_display_texts(Microscope)
            ['AUTO', 'OPENHCS', 'IMAGEXPRESS']
        """
        if not issubclass(enum_class, Enum):
            raise TypeError(f"Expected Enum class, got {type(enum_class)}")
        
        return [EnumDisplayFormatter.get_display_text(enum_value) for enum_value in enum_class]


# Convenience functions for backward compatibility and ease of use
def get_enum_display_text(enum_value: Enum) -> str:
    """Convenience function for getting enum display text."""
    return EnumDisplayFormatter.get_display_text(enum_value)


def get_enum_placeholder_text(enum_value: Enum, prefix: str = "Pipeline default: ") -> str:
    """Convenience function for getting enum placeholder text."""
    return EnumDisplayFormatter.get_placeholder_text(enum_value, prefix)
