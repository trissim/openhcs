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
# DualEditorPane injected via constructor to break circular dependency
# Global error handling will catch all exceptions automatically

logger = logging.getLogger(__name__)


class VisualProgrammingDialogService:
    """
    Service for managing visual programming dialogs.

    Handles dialog creation, lifecycle, and integration with DualEditorPane.
    Keeps dialog concerns separate from pipeline management.
    Uses dependency injection to avoid circular imports.
    """

    def __init__(self, state: Any, context: Any, dual_editor_pane_class: type):
        """
        Initialize the visual programming dialog service.

        Args:
            state: TUI state object
            context: Processing context
            dual_editor_pane_class: DualEditorPane class injected to break circular dependency
        """
        self.state = state
        self.context = context
        self.dual_editor_pane_class = dual_editor_pane_class

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
        # Create empty FunctionStep for new step creation - BACKEND API COMPLIANT
        empty_step = FunctionStep(
            func=None,  # Required positional parameter
            name="New Step",
            # Let variable_components use FunctionStep's default [VariableComponents.SITE]
            group_by=""
        )

        # Create DualEditorPane for visual programming with dialog callbacks
        dual_editor = self.dual_editor_pane_class(
            state=self.state,
            func_step=empty_step,
            on_save=lambda step: self._handle_save_and_close(step),
            on_cancel=lambda: self._handle_cancel_and_close()
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
        # Create DualEditorPane for visual programming with existing step and dialog callbacks
        dual_editor = self.dual_editor_pane_class(
            state=self.state,
            func_step=target_step,
            on_save=lambda step: self._handle_save_and_close(step),
            on_cancel=lambda: self._handle_cancel_and_close()
        )

        # Create and show dialog
        result = await self._show_dialog(
            title="Visual Programming - Edit Step",
            dual_editor=dual_editor,
            ok_handler=lambda: self._handle_edit_step_ok(dual_editor)
        )

        return result
    
    async def _show_dialog(self, title: str, dual_editor: Any, ok_handler) -> Optional[FunctionStep]:
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
        
        # Create dialog WITHOUT buttons - DualEditorPane handles its own save/cancel
        # Set minimum width to accommodate Function Pattern Editor button row
        from prompt_toolkit.layout.dimension import Dimension

        # Calculate minimum width for Function Pattern Editor buttons:
        # "Function Pattern Editor" (title) + "Add Function" + "Load .func" + "Save .func As" + "Edit in Vim"
        # ≈ 23 + 15 + 12 + 15 + 12 + padding ≈ 85 characters
        min_dialog_width = 85

        dialog = Dialog(
            title=title,
            body=dual_editor.container,
            buttons=[],  # No buttons - DualEditorPane has its own Save/Cancel
            width=Dimension(min=min_dialog_width),  # Ensure minimum width for button row
            modal=True
        )
        
        # Store dialog reference
        self.current_dialog = dialog

        # Show dialog (async)
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

    def _handle_save_and_close(self, step: FunctionStep):
        """Handle save from DualEditorPane and close dialog."""
        if self.current_dialog_future and not self.current_dialog_future.done():
            self.current_dialog_future.set_result(step)

    def _handle_cancel_and_close(self):
        """Handle cancel from DualEditorPane and close dialog."""
        if self.current_dialog_future and not self.current_dialog_future.done():
            self.current_dialog_future.set_result(None)
    
    def _handle_add_step_ok(self, dual_editor: Any) -> Optional[FunctionStep]:
        """Handle OK button for add step dialog."""
        created_step = dual_editor.get_created_step()
        return created_step

    def _handle_edit_step_ok(self, dual_editor: Any) -> Optional[FunctionStep]:
        """Handle OK button for edit step dialog."""
        edited_step = dual_editor.get_created_step()
        return edited_step
