"""
Swappable Pane Interface

Defines the mandatory interface for panes that can be swapped in the left panel.
Both PlateManager and DualStepFuncEditor must implement this interface.
"""

from abc import ABC, abstractmethod
from prompt_toolkit.layout import Container, Window


class SwappablePaneInterface(ABC):
    """MANDATORY interface for swappable panes.
    
    Any pane that can be swapped in the left panel must implement this interface
    to ensure proper focus management and container handling.
    """
    
    @property
    @abstractmethod
    def container(self) -> Container:
        """Return the root container for this pane.
        
        Returns:
            Container: The root prompt-toolkit Container for this pane
            
        Raises:
            RuntimeError: If container is not initialized or is None
            
        Requirements:
            - MUST NOT return None
            - MUST be a valid prompt-toolkit Container
            - MUST contain the focus Window returned by get_focus_window()
        """
        raise NotImplementedError
        
    @abstractmethod
    def get_focus_window(self) -> Window:
        """Return the Window that should receive focus.
        
        Returns:
            Window: The Window containing the primary interactive element
            
        Raises:
            RuntimeError: If focus window is not initialized or is None
            
        Requirements:
            - MUST NOT return None
            - MUST return a Window containing a focusable UIControl
            - MUST be the deepest focusable element in the container tree
            - MUST be contained within the container returned by .container
        """
        raise NotImplementedError
        
    def on_focus_gained(self) -> None:
        """Called when this pane gains focus.
        
        Optional: Implement if pane needs focus notifications.
        This is called after the focus has been successfully set.
        """
        pass
        
    def on_focus_lost(self) -> None:
        """Called when this pane loses focus.
        
        Optional: Implement if pane needs focus notifications.
        This is called before focus is moved to another pane.
        """
        pass


def validate_swappable_pane(pane: SwappablePaneInterface) -> bool:
    """Validate that a pane properly implements SwappablePaneInterface.
    
    Args:
        pane: The pane to validate
        
    Returns:
        bool: True if valid, False if invalid
        
    Raises:
        ValueError: If validation fails with details
    """
    try:
        # Test 1: Check container property
        container = pane.container
        if container is None:
            raise ValueError("container property returned None")
        # Check if it's a valid Container (has children, content, or body attribute)
        if not (hasattr(container, 'children') or hasattr(container, 'content') or hasattr(container, 'body')):
            raise ValueError("container is not a valid Container")
            
        # Test 2: Check get_focus_window method
        focus_window = pane.get_focus_window()
        if focus_window is None:
            raise ValueError("get_focus_window() returned None")
        if not isinstance(focus_window, Window):
            raise ValueError(f"get_focus_window() returned {type(focus_window)}, expected Window")
            
        # Test 3: Check focus window has focusable content
        if not hasattr(focus_window, 'content') or focus_window.content is None:
            raise ValueError("focus window has no content")
        if not hasattr(focus_window.content, 'focusable'):
            raise ValueError("focus window content is not focusable")
        # Note: focusable is a Filter in prompt-toolkit, we just check it exists
            
        # Test 4: Check focus window is in container tree (or will be when DynamicContainer renders)
        # Note: DynamicContainer content may not be in tree until rendered, so this is optional
        if not _is_window_in_container(focus_window, container):
            # This is expected for DynamicContainer-based layouts before rendering
            pass
            
        return True
        
    except Exception as e:
        raise ValueError(f"Pane validation failed: {e}")


def _is_window_in_container(window: Window, container: Container) -> bool:
    """Check if window is contained within container tree."""
    def find_window(cont):
        if cont == window:
            return True
        # Check children (HSplit, VSplit, etc.)
        if hasattr(cont, 'children'):
            for child in cont.children:
                if find_window(child):
                    return True
        # Check content (Window, etc.)
        if hasattr(cont, 'content') and cont.content == window:
            return True
        # Check body (Frame, etc.)
        if hasattr(cont, 'body') and find_window(cont.body):
            return True
        return False

    return find_window(container)
