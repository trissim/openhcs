"""
Layout calculation utilities for OpenHCS TUI.

Provides content-based width calculation to eliminate magic numbers
from form layouts and ensure proper alignment.
"""
from typing import List


class FormLayoutCalculator:
    """Calculates optimal layout dimensions based on actual content."""
    
    # Semantic constants - no magic numbers
    LABEL_SUFFIX_WIDTH = 2      # For ": " after label
    BUTTON_PADDING = 2          # Space around button text  
    MIN_LABEL_WIDTH = 8         # Minimum readable label width
    MAX_LABEL_WIDTH = 40        # Prevent excessive width
    
    @staticmethod
    def calculate_label_width(field_labels: List[str]) -> int:
        """
        Calculate optimal label width based on actual content.
        
        Args:
            field_labels: List of field label strings
            
        Returns:
            Optimal width for consistent label alignment
        """
        if not field_labels:
            return FormLayoutCalculator.MIN_LABEL_WIDTH
            
        try:
            # Find longest label and add suffix space
            max_length = max(len(label) for label in field_labels)
            optimal_width = max_length + FormLayoutCalculator.LABEL_SUFFIX_WIDTH
            
            # Apply reasonable bounds
            return max(
                FormLayoutCalculator.MIN_LABEL_WIDTH,
                min(optimal_width, FormLayoutCalculator.MAX_LABEL_WIDTH)
            )
        except (ValueError, TypeError):
            # Fallback for edge cases
            return FormLayoutCalculator.MIN_LABEL_WIDTH
    
    @staticmethod
    def calculate_button_width(button_text: str) -> int:
        """
        Calculate button width based on text content.
        
        Args:
            button_text: Text that will appear on the button
            
        Returns:
            Width needed for button including padding
        """
        if not button_text:
            return FormLayoutCalculator.BUTTON_PADDING
            
        return len(button_text) + FormLayoutCalculator.BUTTON_PADDING
