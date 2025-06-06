"""
Custom Button component with [ ] symbols instead of < >.

This module provides a Button class that overrides the default prompt_toolkit Button
to use [ ] symbols instead of < > for a cleaner appearance.
"""

from prompt_toolkit.widgets import Button as PromptToolkitButton
from typing import Callable, Optional


class Button(PromptToolkitButton):
    """
    Custom Button with [ ] symbols instead of < >.
    
    This class inherits from prompt_toolkit.widgets.Button but changes
    the default left_symbol and right_symbol to use square brackets.
    """
    
    def __init__(
        self,
        text: str,
        handler: Optional[Callable[[], None]] = None,
        width: int = 12,
        left_symbol: str = '[',
        right_symbol: str = ']'
    ):
        """
        Initialize custom button with [ ] symbols.
        
        Args:
            text: Button text
            handler: Click handler function
            width: Button width
            left_symbol: Left symbol (default: '[')
            right_symbol: Right symbol (default: ']')
        """
        super().__init__(
            text=text,
            handler=handler,
            width=width,
            left_symbol=left_symbol,
            right_symbol=right_symbol
        )
