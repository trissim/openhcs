"""
Visual Programming Dialog Service for OpenHCS TUI.

Handles creation and management of visual programming dialogs using DualEditorPane.
Separates dialog concerns from pipeline management.
"""
import asyncio
import copy
import logging
from typing import Any, List, Optional

from prompt_toolkit.application import get_app
from prompt_toolkit.widgets import Dialog, Button
from prompt_toolkit.layout import HSplit

from openhcs.constants.constants import Backend
from openhcs.core.pipeline import Pipeline
from openhcs.core.steps.function_step import FunctionStep
from openhcs.tui.editors.dual_editor_pane import DualEditorPane
# Global error handling will catch all exceptions automatically

logger = logging.getLogger(__name__)


class VisualProgrammingDialogService:
    """
    Service for managing visual programming dialogs.
    
    Handles dialog creation, lifecycle, and integration with DualEditorPane.
    Keeps dialog concerns separate from pipeline management.
    """
    
    def __init__(self, state: Any, context: Any):
        """
        Initialize the visual programming dialog service.
        
        Args:
            state: TUI state object
            context: Processing context
        """
        self.state = state
        self.context = context
        
        # Dialog state management
        self.current_dialog = None
        self.current_dialog_future = None
    
    async def show_add_step_dialog(self, target_pipelines: List[Pipeline]) -> Optional[FunctionStep]:
        """
        Show visual programming dialog for adding a new step.
        
        Args:
            target_pipelines: List of pipelines to add the step to
            
        Returns:
            Created FunctionStep if successful, None if cancelled
        """
        # Create empty FunctionStep for new step creation
        empty_step = FunctionStep(
            name="New Step",
            description="",
            func=None,
            variable_components=[],
            group_by=""
        )

        # Create DualEditorPane for visual programming
        dual_editor = DualEditorPane(
            state=self.state,
            func_step=empty_step
        )

        # Create and show dialog
        result = await self._show_dialog(
            title="Visual Programming - Add Step",
            dual_editor=dual_editor,
            ok_handler=lambda: self._handle_add_step_ok(dual_editor)
        )

        return result
    
    async def show_edit_step_dialog(self, target_step: FunctionStep) -> Optional[FunctionStep]:
        """
        Show visual programming dialog for editing an existing step.
        
        Args:
            target_step: The step to edit
            
        Returns:
            Edited FunctionStep if successful, None if cancelled
        """
        # Create DualEditorPane for visual programming with existing step
        dual_editor = DualEditorPane(
            state=self.state,
            func_step=target_step
        )

        # Create and show dialog
        result = await self._show_dialog(
            title="Visual Programming - Edit Step",
            dual_editor=dual_editor,
            ok_handler=lambda: self._handle_edit_step_ok(dual_editor)
        )

        return result
    
    async def _show_dialog(self, title: str, dual_editor: DualEditorPane, ok_handler) -> Optional[FunctionStep]:
        """
        Show a visual programming dialog with the given DualEditorPane.
        
        Args:
            title: Dialog title
            dual_editor: The DualEditorPane instance
            ok_handler: Handler for OK button
            
        Returns:
            Result from OK handler or None if cancelled
        """
        # Create dialog result future
        self.current_dialog_future = asyncio.Future()
        
        # Create dialog
        dialog = Dialog(
            title=title,
            body=dual_editor.container,
            buttons=[
                Button("OK", handler=lambda: self._handle_ok(ok_handler)),
                Button("Cancel", handler=self._handle_cancel)
            ],
            width=120,
            height=40,
            modal=True
        )
        
        # Store dialog reference
        self.current_dialog = dialog
        
        # Show dialog using state.show_dialog
        await self.state.show_dialog(dialog, result_future=self.current_dialog_future)
        
        # Wait for dialog completion
        result = await self.current_dialog_future
        
        # Cleanup
        self.current_dialog = None
        self.current_dialog_future = None
        
        return result
    
    def _handle_ok(self, ok_handler):
        """Handle OK button click."""
        try:
            result = ok_handler()
            if self.current_dialog_future and not self.current_dialog_future.done():
                self.current_dialog_future.set_result(result)
        except Exception as e:
            logger.error(f"Error in OK handler: {e}")
            if self.current_dialog_future and not self.current_dialog_future.done():
                self.current_dialog_future.set_result(None)
    
    def _handle_cancel(self):
        """Handle Cancel button click."""
        if self.current_dialog_future and not self.current_dialog_future.done():
            self.current_dialog_future.set_result(None)
    
    def _handle_add_step_ok(self, dual_editor: DualEditorPane) -> Optional[FunctionStep]:
        """Handle OK button for add step dialog."""
        created_step = dual_editor.get_created_step()
        return created_step
    
    def _handle_edit_step_ok(self, dual_editor: DualEditorPane) -> Optional[FunctionStep]:
        """Handle OK button for edit step dialog."""
        edited_step = dual_editor.get_created_step()
        return edited_step
