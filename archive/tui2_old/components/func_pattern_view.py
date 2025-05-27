"""
Function Pattern View Component for OpenHCS TUI.

This module defines the FuncPatternView class, which wraps and manages
the UI for editing the function pattern of a step.
"""
import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from prompt_toolkit.layout import DynamicContainer, Container
from prompt_toolkit.widgets import Label, Frame

# Assuming the original FunctionPatternEditor is refactored or can be used as is.
# If FunctionPatternEditor itself needs significant changes to work without direct core dependencies,
# that would be a separate refactoring task. For now, we'll assume it can be wrapped.
try:
    from openhcs.tui.function_pattern_editor import FunctionPatternEditor
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Original FunctionPatternEditor not found. FuncPatternView will use a placeholder.")
    # Define a placeholder if the original is not yet available or refactored
    class FunctionPatternEditor: # type: ignore
        def __init__(self, pattern_data: Any, on_change: Callable[[Any], None], func_registry: Optional[Any] = None):
            self.pattern_data = pattern_data
            self.on_change = on_change
            self.container = Frame(Label(f"Placeholder for FunctionPatternEditor. Data: {pattern_data}"))
        def get_pattern(self) -> Any: return self.pattern_data
        def __pt_container__(self) -> Container: return self.container


if TYPE_CHECKING:
    from openhcs.tui.interfaces import CoreStepData # For type hinting
    # If FunctionPatternEditor takes a func_registry like object from core:
    # from openhcs.processing.func_registry import FUNC_REGISTRY 

logger = logging.getLogger(__name__)

class FuncPatternView:
    """
    Wraps the FunctionPatternEditor and manages its state for the DualEditorController.
    """
    def __init__(self,
                 on_pattern_change: Callable[[Any], None], # Callback: new_pattern_data
                 # The func_registry_provider would be a callable that the controller provides,
                 # potentially fetching available functions/patterns via an adapter if needed.
                 # For now, can be Optional or mocked if not immediately available.
                 func_registry_provider: Optional[Callable[[], Any]] = None
                ):
        self.current_pattern_data: Any = None # The 'func' part of CoreStepData
        self.on_pattern_change_callback = on_pattern_change
        self.func_registry_provider = func_registry_provider
        
        self.editor_instance: Optional[FunctionPatternEditor] = None
        self._view_container: Container = Frame(Label("No function pattern loaded.")) # Initial placeholder
        
        # The main container for this view will be a DynamicContainer
        self.container = DynamicContainer(lambda: self._view_container)

    async def set_pattern_data(self, pattern_data: Any, step_name: Optional[str] = "Current Step"):
        """
        Sets the function pattern data to be edited and rebuilds/updates the editor.
        'pattern_data' is typically the 'func' attribute from a CoreStepData object.
        """
        self.current_pattern_data = pattern_data
        
        # Get func_registry if provider exists
        func_registry_instance = None
        if self.func_registry_provider:
            try:
                func_registry_instance = self.func_registry_provider()
            except Exception as e:
                logger.error(f"FuncPatternView: Error getting func_registry: {e}")

        try:
            # FunctionPatternEditor might need to be adapted if its constructor changes
            # or if it expects specific types not available directly from pattern_data.
            self.editor_instance = FunctionPatternEditor(
                pattern_data=self.current_pattern_data,
                on_change=self._handle_internal_editor_change, # Internal handler
                func_registry=func_registry_instance
            )
            # The title of the frame should ideally be dynamic, e.g., showing step name
            self._view_container = Frame(
                self.editor_instance.container, 
                title=f"Function Pattern: {step_name or ''}"
            )
            logger.info(f"FuncPatternView: FunctionPatternEditor instance created for pattern: {pattern_data}")
        except Exception as e:
            logger.error(f"FuncPatternView: Failed to instantiate FunctionPatternEditor: {e}", exc_info=True)
            self._view_container = Frame(Label(f"Error loading function editor: {e}"))
        
        if get_app().is_running:
            get_app().invalidate()

    def _handle_internal_editor_change(self, new_pattern_from_editor: Any):
        """
        Handles the 'on_change' event from the wrapped FunctionPatternEditor.
        It then calls the callback provided by the DualEditorController.
        """
        self.current_pattern_data = new_pattern_from_editor # Update internal state
        if self.on_pattern_change_callback:
            self.on_pattern_change_callback(self.current_pattern_data)

    def get_current_pattern_data(self) -> Any:
        """Returns the current state of the pattern data from the editor."""
        if self.editor_instance:
            return self.editor_instance.get_pattern()
        return self.current_pattern_data

    def __pt_container__(self) -> Container:
        return self.container
