"""
Pattern Key Selector - UI component for Dict key selection and management.

This component handles key selection and management for Dict-based function patterns
with proper None key semantics and clean UI patterns.
"""

from typing import Dict, List, Tuple, Any, Optional, Callable
import logging

from prompt_toolkit.layout.containers import HSplit, VSplit
from prompt_toolkit.widgets import Button, Label, RadioList as Dropdown

from openhcs.tui.utils.unified_task_manager import get_task_manager

logger = logging.getLogger(__name__)


class PatternKeySelector:
    """
    UI component for Dict key selection and management.
    
    Handles key selection, addition, removal with proper None key semantics
    and follows established TUI patterns.
    """
    
    def __init__(self, pattern: Dict, current_key: Any, is_dict: bool,
                 on_key_change: Optional[Callable] = None,
                 on_add_key: Optional[Callable] = None,
                 on_remove_key: Optional[Callable] = None,
                 on_convert_to_dict: Optional[Callable] = None):
        """
        Initialize the pattern key selector.
        
        Args:
            pattern: Current pattern (Dict or List)
            current_key: Currently selected key
            is_dict: Whether pattern is in dict mode
            on_key_change: Callback for key selection changes
            on_add_key: Callback for adding new keys
            on_remove_key: Callback for removing keys
            on_convert_to_dict: Callback for converting list to dict
        """
        self.pattern = pattern
        self.current_key = current_key
        self.is_dict = is_dict
        self.on_key_change = on_key_change
        self.on_add_key = on_add_key
        self.on_remove_key = on_remove_key
        self.on_convert_to_dict = on_convert_to_dict
        
        self._container = self._build_key_selector()
    
    @property
    def container(self):
        """Return the main container for the key selector."""
        return self._container
    
    def _build_key_selector(self) -> HSplit:
        """
        Build key selector UI based on pattern type.
        
        Returns:
            HSplit container with key selector components
        """
        if self.is_dict:
            return self._build_dict_key_selector()
        else:
            return self._build_list_converter()
    
    def _build_dict_key_selector(self) -> HSplit:
        """
        Build key selector for Dict patterns.
        
        Returns:
            HSplit with dropdown and management buttons
        """
        # Create key dropdown
        key_dropdown = self._create_key_dropdown()
        
        # Create key management buttons
        add_key_button = Button(
            "Add Key",
            handler=lambda: get_task_manager().fire_and_forget(
                self._handle_add_key(), "add_key"
            ),
            width=10
        )
        
        remove_key_button = Button(
            "Remove Key",
            handler=lambda: get_task_manager().fire_and_forget(
                self._handle_remove_key(), "remove_key"
            ),
            width=12
        )
        
        key_management_buttons = VSplit([add_key_button, remove_key_button], padding=1)

        # RadioList can be used directly in VSplit - no wrapping needed

        return HSplit([
            VSplit([
                Label("Pattern Keys: "),
                key_dropdown,
                key_management_buttons
            ])
        ])
    
    def _build_list_converter(self) -> HSplit:
        """
        Build empty container for List patterns (Apply per Component removed).

        Returns:
            Empty HSplit container
        """
        # Apply per Component button removed as it means nothing
        return HSplit([])
    
    def _create_key_dropdown(self) -> Dropdown:
        """
        Create dropdown for key selection with None key handling.
        
        CRITICAL: Displays 'Unnamed' for None key while preserving None in data model.
        
        Returns:
            Dropdown widget for key selection
        """
        display_keys = self._create_key_dropdown_options(self.pattern)
        
        key_dropdown = Dropdown(
            options=display_keys,
            default=self.current_key
        )
        
        # Set handler for key selection
        def on_key_change(key):
            get_task_manager().fire_and_forget(
                self._handle_key_change(key), f"switch_key_{key}"
            )
        
        key_dropdown.handler = on_key_change
        return key_dropdown
    
    def _create_key_dropdown_options(self, pattern: Dict) -> List[Tuple]:
        """
        Create dropdown options for experimental component identifiers.

        Args:
            pattern: Dict pattern to extract keys from

        Returns:
            List of (value, display_text) tuples for dropdown
        """
        display_keys = []
        for k in pattern.keys():
            # Show actual experimental component identifiers
            display_keys.append((k, f"Component: {k}"))

        # Handle empty dict case - add placeholder option
        if not display_keys:
            display_keys.append((None, "No components yet - click Add Key"))

        return display_keys
    
    async def _handle_key_change(self, new_key: Any):
        """Handle key selection change."""
        if self.on_key_change:
            await self.on_key_change(new_key)
    
    async def _handle_add_key(self):
        """Handle add key button click."""
        if self.on_add_key:
            await self.on_add_key()
    
    async def _handle_remove_key(self):
        """Handle remove key button click."""
        if self.on_remove_key:
            await self.on_remove_key()
    
    async def _handle_convert_to_dict(self):
        """Handle convert to dict button click."""
        if self.on_convert_to_dict:
            await self.on_convert_to_dict()
    
    def update_pattern(self, pattern: Dict, current_key: Any, is_dict: bool):
        """
        Update the key selector with new pattern data.
        
        Args:
            pattern: New pattern data
            current_key: New current key
            is_dict: Whether pattern is in dict mode
        """
        self.pattern = pattern
        self.current_key = current_key
        self.is_dict = is_dict
        
        # Rebuild the container
        self._container = self._build_key_selector()
    
    def _should_convert_to_list(self, pattern: Dict) -> bool:
        """
        Check if dict pattern should be converted back to list.

        Args:
            pattern: Dict pattern to check

        Returns:
            True if should convert to list, False otherwise
        """
        return len(pattern) == 0  # Convert to list if dict is empty
    
    @staticmethod
    def create_key_selector(pattern: Dict, current_key: Any, is_dict: bool,
                          on_key_change: Optional[Callable] = None,
                          on_add_key: Optional[Callable] = None,
                          on_remove_key: Optional[Callable] = None,
                          on_convert_to_dict: Optional[Callable] = None) -> 'PatternKeySelector':
        """
        Factory method for creating key selector instances.
        
        Args:
            pattern: Pattern data
            current_key: Current selected key
            is_dict: Whether pattern is in dict mode
            on_key_change: Key change callback
            on_add_key: Add key callback
            on_remove_key: Remove key callback
            on_convert_to_dict: Convert to dict callback
            
        Returns:
            PatternKeySelector instance
        """
        return PatternKeySelector(
            pattern=pattern,
            current_key=current_key,
            is_dict=is_dict,
            on_key_change=on_key_change,
            on_add_key=on_add_key,
            on_remove_key=on_remove_key,
            on_convert_to_dict=on_convert_to_dict
        )
