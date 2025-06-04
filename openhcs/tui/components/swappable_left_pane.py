"""
Swappable Left Pane Manager

Manages container swapping for the left pane with bulletproof focus management.
Allows switching between PlateManager and DualStepFuncEditor while maintaining
proper focus for mouse handlers.
"""

from typing import Dict, Optional
from prompt_toolkit.application import get_app
from prompt_toolkit.layout import Container, DynamicContainer

from openhcs.tui.interfaces.swappable_pane import SwappablePaneInterface, validate_swappable_pane


class SwappableLeftPane:
    """Manages swapping between different panes in the left panel.
    
    Provides bulletproof container swapping with proper focus management
    to ensure mouse handlers work correctly after pane changes.
    """
    
    def __init__(self):
        """Initialize the swappable left pane manager."""
        self.current_left_pane: str = "plate_manager"
        self.panes: Dict[str, SwappablePaneInterface] = {}
        self._container: Optional[DynamicContainer] = None
        
    def register_pane(self, name: str, pane: SwappablePaneInterface) -> None:
        """Register a pane that can be swapped to.
        
        Args:
            name: Unique name for the pane ("plate_manager", "dual_editor", etc.)
            pane: The pane object implementing SwappablePaneInterface
            
        Raises:
            ValueError: If pane doesn't implement interface correctly
        """
        # VALIDATION: Ensure pane implements interface correctly
        validate_swappable_pane(pane)
        
        self.panes[name] = pane
        
    def get_container(self) -> DynamicContainer:
        """Get the DynamicContainer for the left pane.
        
        Returns:
            DynamicContainer: Container that shows the current pane
        """
        if self._container is None:
            self._container = DynamicContainer(self._get_current_pane_container)
        return self._container
        
    def _get_current_pane_container(self) -> Container:
        """Get the container for the currently active pane."""
        if self.current_left_pane not in self.panes:
            # Fallback to first available pane
            if self.panes:
                self.current_left_pane = next(iter(self.panes.keys()))
            else:
                # No panes registered - return empty container
                from prompt_toolkit.widgets import Label
                from prompt_toolkit.layout import HSplit
                return HSplit([Label("No panes registered")])
                
        current_pane = self.panes[self.current_left_pane]
        return current_pane.container
        
    def swap_left_pane(self, new_pane: str) -> bool:
        """Swap left pane with BULLETPROOF error handling.
        
        Args:
            new_pane: Name of the pane to swap to
            
        Returns:
            bool: True if swap successful, False if failed
        """
        try:
            # VALIDATION 1: Input validation
            if not isinstance(new_pane, str):
                raise ValueError(f"new_pane must be string, got {type(new_pane)}")
            if new_pane not in self.panes:
                raise ValueError(f"Invalid pane: {new_pane}. Available: {list(self.panes.keys())}")
            
            # VALIDATION 2: Current state check
            if self.current_left_pane == new_pane:
                return True  # Already showing requested pane
                
            # VALIDATION 3: App state check
            app = get_app()
            if not app or not hasattr(app, 'layout'):
                raise RuntimeError("App not running or no layout")
                
            # VALIDATION 4: Pane objects exist and are initialized
            pane_obj = self.panes[new_pane]
            
            # VALIDATION 5: Pane has required methods
            if not hasattr(pane_obj, 'container'):
                raise RuntimeError(f"{new_pane} missing container property")
            if not hasattr(pane_obj, 'get_focus_window'):
                raise RuntimeError(f"{new_pane} missing get_focus_window method")
                
            # VALIDATION 6: Get container and focus target
            new_container = pane_obj.container
            if not new_container:
                raise RuntimeError(f"{new_pane} container is None")
                
            focus_target = pane_obj.get_focus_window()
            if not focus_target:
                raise RuntimeError(f"{new_pane} focus target is None")
                
            # NOTIFICATION: Call focus lost on current pane
            if self.current_left_pane in self.panes:
                current_pane_obj = self.panes[self.current_left_pane]
                if hasattr(current_pane_obj, 'on_focus_lost'):
                    current_pane_obj.on_focus_lost()
                    
            # ATOMIC SWAP OPERATION
            # Step 1: Update current pane (this triggers DynamicContainer rebuild)
            self.current_left_pane = new_pane
            
            # Step 2: Focus immediately (with validation)
            focus_success = self._focus_target_bulletproof(focus_target)
            if not focus_success:
                # Swap succeeded but focus failed - log warning but continue
                print(f"Warning: Pane swap succeeded but focus failed for {new_pane}")
                
            # Step 3: Invalidate to trigger UI update
            app.invalidate()
            
            # NOTIFICATION: Call focus gained on new pane
            if hasattr(pane_obj, 'on_focus_gained'):
                pane_obj.on_focus_gained()
            
            return True
            
        except Exception as e:
            print(f"Pane swap failed: {e}")
            return False

    def _focus_target_bulletproof(self, target) -> bool:
        """Focus target with ALL edge cases handled."""
        try:
            app = get_app()
            if not app or not hasattr(app, 'layout'):
                return False
                
            # Check if target is in layout tree
            if not self._is_window_in_layout(target, app.layout):
                return False
                
            # Attempt focus
            app.layout.focus(target)
            return True
            
        except Exception as e:
            print(f"Focus failed: {e}")
            return False

    def _is_window_in_layout(self, window, layout):
        """Check if window is actually in the layout tree."""
        try:
            # Traverse layout tree to find window
            def find_window(container):
                if container == window:
                    return True
                if hasattr(container, 'children'):
                    for child in container.children:
                        if find_window(child):
                            return True
                if hasattr(container, 'content') and container.content == window:
                    return True
                return False
            
            return find_window(layout.container)
        except:
            return False
            
    def get_current_pane_name(self) -> str:
        """Get the name of the currently active pane."""
        return self.current_left_pane
        
    def get_current_pane(self) -> Optional[SwappablePaneInterface]:
        """Get the currently active pane object."""
        return self.panes.get(self.current_left_pane)
