"""
Global Settings Editor Dialog for OpenHCS TUI.

This module implements a dialog for viewing and editing the
GlobalPipelineConfig.
"""
import asyncio
import logging
from typing import Any, Optional

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import HSplit, VSplit
from prompt_toolkit.widgets import Button, Dialog, Label, TextArea, RadioList, Checkbox

from openhcs.core.config import GlobalPipelineConfig, VFSConfig, PathPlanningConfig # Assuming these are dataclasses
from openhcs.constants.constants import Backend, Microscope # For dropdowns

# Assuming TUIState is accessible, e.g., from a shared module or passed in
# from ..tui_architecture import TUIState # Avoid circular import

logger = logging.getLogger(__name__)

class GlobalSettingsEditorDialog:
    """
    A dialog for editing global OpenHCS settings (GlobalPipelineConfig).
    """
    def __init__(self, state: Any, initial_config: GlobalPipelineConfig):
        """
        Initialize the GlobalSettingsEditorDialog.

        Args:
            state: The TUIState instance.
            initial_config: The initial GlobalPipelineConfig to edit.
        """
        self.state = state
        self.original_config = initial_config
        # Create a working copy to modify
        self.editing_config = initial_config.model_copy(deep=True)

        self.dialog: Optional[Dialog] = None
        self._build_dialog()

    def _build_dialog(self):
        """Constructs the dialog UI elements."""
        # UI elements will be created here based on GlobalPipelineConfig fields
        # Example: num_workers
        self.num_workers_input = TextArea(
            text=str(self.editing_config.num_workers),
            multiline=False,
            height=1,
            prompt="Number of Workers: ",
            # TODO: Add on_text_changed to enable save button
        )
        
        # Example: VFS default_storage_backend (using RadioList as Dropdown)
        vfs_backend_options = [(b.value, b.name) for b in Backend] # label, value
        self.vfs_backend_selector = RadioList(
            values=[(b.value, b.name) for b in Backend], # value, label
            current_value=self.editing_config.vfs.default_storage_backend.value
            # TODO: Add on_value_changed
        )

        # Example: Microscope default_microscope_type
        microscope_options = [(m.value, m.name) for m in Microscope]
        self.microscope_selector = RadioList(
            values=[(m.value, m.name) for m in Microscope], # value, label
            current_value=self.editing_config.microscope.default_microscope_type.value
            # TODO: Add on_value_changed
        )

        # ... more fields for PathPlanningConfig, etc.

        body_content = [
            Label("Global Pipeline Settings:"),
            HSplit([Label("Num Workers:", width=25), self.num_workers_input]),
            Label("VFS Settings:"),
            HSplit([Label("  Default Storage Backend:", width=25), self.vfs_backend_selector]),
            Label("Microscope Settings:"),
            HSplit([Label("  Default Microscope Type:", width=25), self.microscope_selector]),
            # TODO: Add other config fields here
            Label("Path Planning Settings: (TODO)"),
            Label("Advanced VFS Settings: (TODO)"),
        ]

        ok_button = Button("Save", handler=self._save_settings)
        cancel_button = Button("Cancel", handler=self._cancel)
        
        self.dialog = Dialog(
            title="Global Settings",
            body=HSplit(body_content, padding=1),
            buttons=[ok_button, cancel_button],
            width=80,
            modal=True
        )

    async def _save_settings(self):
        logger.info("GlobalSettingsEditorDialog: Saving settings.")
        # TODO: Update self.editing_config from UI elements
        try:
            self.editing_config.num_workers = int(self.num_workers_input.text)
            self.editing_config.vfs.default_storage_backend = Backend(self.vfs_backend_selector.current_value)
            self.editing_config.microscope.default_microscope_type = Microscope(self.microscope_selector.current_value)
            # ... update other fields
        except ValueError as e:
            logger.error(f"Error parsing settings: {e}")
            # TODO: Show error in dialog
            await self.state.notify('error', {'message': f"Invalid setting value: {e}", 'source': self.__class__.__name__})
            return

        self.original_config = self.editing_config.model_copy(deep=True)
        await self.state.notify('global_config_changed', self.original_config)
        
        if self.dialog and hasattr(self.dialog, '__ohcs_future'):
             future = getattr(self.dialog, '__ohcs_future', None)
             if future and not future.done():
                future.set_result(self.original_config) # Return the saved config
        # else: # Dialog might be directly managed by MenuBar, no future needed
        #    get_app().layout.focus_last() # Or some other way to close/hide

    async def _cancel(self):
        logger.info("GlobalSettingsEditorDialog: Cancelled.")
        if self.dialog and hasattr(self.dialog, '__ohcs_future'):
            future = getattr(self.dialog, '__ohcs_future', None)
            if future and not future.done():
                future.set_result(None) # Indicate cancellation
        # else:
        #    get_app().layout.focus_last()

    async def show(self) -> Optional[GlobalPipelineConfig]:
        """Shows the dialog and returns the config if saved, else None."""
        if not self.dialog:
            self._build_dialog() # Should have been called in __init__

        app = get_app()
        previous_focus = app.layout.current_window if hasattr(app.layout, 'current_window') else None
        
        float_ = Float(self.dialog)
        setattr(self.dialog, '__ohcs_float', float_) # For closing
        
        future = asyncio.Future()
        setattr(self.dialog, '__ohcs_future', future)

        app.layout.container.floats.append(float_)
        app.layout.focus(self.dialog)
        
        try:
            return await future
        finally:
            if float_ in app.layout.container.floats:
                app.layout.container.floats.remove(float_)
            if previous_focus and hasattr(app.layout, 'walk'):
                try:
                    if previous_focus in app.layout.walk(): # Check if focus target still exists
                        app.layout.focus(previous_focus)
                except Exception: # Broad except as focus restoration can be tricky
                    pass # Silently ignore if previous focus is no longer valid

# Example usage (would be called from MenuBar typically)
# async def open_global_settings_dialog(state: TUIState, current_config: GlobalPipelineConfig):
#     dialog = GlobalSettingsEditorDialog(state, current_config)
#     updated_config = await dialog.show()
#     if updated_config:
#         logger.info("Global settings updated.")
#     else:
#         logger.info("Global settings dialog cancelled.")