"""
Dual Editor Controller for OpenHCS TUI.

This module defines the DualEditorController, which manages the
StepSettingsEditorView and FuncPatternView for editing a single step's
configuration and function pattern.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Callable
import copy # For deep copying step data

from prompt_toolkit.layout import HSplit, VSplit, Window, Dimension, FormattedTextControl, Container
from prompt_toolkit.widgets import Button, Frame, Label, Box

from ..components.list_item_base import FramedButton # Assuming FramedButton is in components.list_item_base

if TYPE_CHECKING:
    from openhcs.tui.tui_architecture import TUIState
    from openhcs.tui.interfaces import CoreApplicationAdapterInterface, CoreOrchestratorAdapterInterface, CoreStepData, ParamSchema
    from ..components.step_settings_editor import StepSettingsEditorView
    from ..components.func_pattern_view import FuncPatternView
    from openhcs.tui.async_manager import AsyncUIManager # Import AsyncUIManager
    # from ..commands import UpdateStepCommand

logger = logging.getLogger(__name__)

class DualEditorController:
    """
    Manages StepSettingsEditorView and FuncPatternView for editing a step.
    Replaces the old DualStepFuncEditorPane.
    """
    def __init__(self,
                 ui_state: 'TUIState',
                 app_adapter: 'CoreApplicationAdapterInterface',
                 async_ui_manager: 'AsyncUIManager', # Added AsyncUIManager
                 step_data: 'CoreStepData', 
                 step_settings_editor_view_class: type['StepSettingsEditorView'],
                 func_pattern_view_class: type['FuncPatternView']
                ):
        self.ui_state = ui_state
        self.app_adapter = app_adapter
        self.async_ui_manager = async_ui_manager # Store AsyncUIManager
        
        self.original_step_data: 'CoreStepData' = copy.deepcopy(step_data)
        self.editing_step_data: 'CoreStepData' = copy.deepcopy(step_data)

        self.step_settings_view = step_settings_editor_view_class(
            on_parameter_change=self._handle_parameter_change,
            get_current_step_schema=self._get_current_step_schema_for_view
        )
        self.func_pattern_view = func_pattern_view_class(
            on_pattern_change=self._handle_pattern_change,
            func_registry_provider=self._get_func_registry_for_view # Example provider
        )

        # Buttons for Save and Cancel
        self.save_button = FramedButton("Save", handler=self._save_changes, width=8)
        self.cancel_button = FramedButton("Cancel", handler=self._cancel_editing, width=10)
        
        self._build_layout() # Sets up self.container

        # Initialize views with current data
        # This needs to be done after views are created and layout is ready for them.
        # The AppController will call initialize_controller after instantiation.
        self.param_schema: Optional[Dict[str, 'ParamSchema']] = None # To store fetched schema


    async def initialize_controller(self):
        """Asynchronously fetches schema and initializes child views with data."""
        logger.info(f"DualEditorController: Initializing for step {self.original_step_data.get('id')}")
        func_id = self.original_step_data.get('function_identifier')
        if func_id and hasattr(self.app_adapter, 'get_function_schema'):
            try:
                self.param_schema = await self.app_adapter.get_function_schema(func_id)
                logger.info(f"Schema fetched for func_id '{func_id}': {self.param_schema is not None}")
            except Exception as e:
                logger.error(f"Failed to fetch schema for func_id '{func_id}': {e}")
                self.param_schema = {"error": {"type": "str", "label": "Schema Error", "default": f"Could not load schema: {e}"}} # Provide error schema
        else:
            logger.warning(f"No function_identifier or get_function_schema method for step {self.original_step_data.get('id')}")
            # Fallback: try to infer schema from existing parameters if no explicit schema can be fetched
            if self.original_step_data.get('parameters'):
                self.param_schema = {
                    name: {'type': type(val).__name__.lower(), 'label': name, 'default': val} # Infer type from value
                    for name, val in self.original_step_data['parameters'].items()
                }
                logger.info(f"Inferred schema from parameters: {self.param_schema}")
            else:
                self.param_schema = {"info": {"type": "str", "label": "Info", "default": "No parameters defined or schema available."}}


        # Now that schema is potentially available (or error state is set), initialize views
        await self.step_settings_view.set_step_data(self.editing_step_data) 
        await self.func_pattern_view.set_pattern_data(
            self.editing_step_data.get('func_pattern', self.editing_step_data.get('func')), 
            step_name=self.editing_step_data.get('name')
        )
        logger.info("DualEditorController: Views initialized with data and schema.")


    def _build_layout(self):
        """Constructs the UI container for this controller."""
        buttons_toolbar = VSplit([
            self.save_button,
            Window(width=1, char=' '),
            self.cancel_button,
            Window(width=Dimension(weight=1)), # Spacer
            Label(lambda: f"Editing: {self.original_step_data.get('name', 'Unnamed Step')}", style="class:editor-title")
        ], height=1, padding=1)

        # Layout: Settings on left, Function Pattern on right, buttons at bottom
        # Or, stack them if preferred. For now, side-by-side:
        # HSplit might be better if editors are tall
        main_editor_area = HSplit([ # VSplit for side-by-side, HSplit for stacked
            Frame(self.step_settings_view.container, title="Parameters"),
            Frame(self.func_pattern_view.container, title="Function/Pattern") 
        ], padding=1)

        self.container = HSplit([
            buttons_toolbar,
            Window(height=1, char='-'), # Separator
            main_editor_area
        ])
        # If this controller is to be a full "pane" replacement:
        # self.container = Frame( HSplit([...]), title="Step Editor" )


    # --- Callbacks for Views ---
    def _handle_parameter_change(self, param_name: str, new_value: Any):
        """Called by StepSettingsEditorView when a parameter's value changes."""
        logger.debug(f"DualEditorController: Parameter '{param_name}' changed to '{new_value}'")
        if 'parameters' not in self.editing_step_data:
            self.editing_step_data['parameters'] = {}
        self.editing_step_data['parameters'][param_name] = new_value
        # Potentially mark as dirty, or live update if TUIState supports partial step updates.

    def _handle_pattern_change(self, new_pattern_data: Any):
        """Called by FuncPatternView when the function pattern changes."""
        logger.debug(f"DualEditorController: Function pattern changed to: {new_pattern_data}")
        # 'func' or 'func_pattern' key depends on CoreStepData definition
        self.editing_step_data['func_pattern'] = new_pattern_data # Or 'func'
        # Potentially mark as dirty.

    def _get_current_step_schema_for_view(self) -> Optional[Dict[str, 'ParamSchema']]:
        """
        Provides the parameter schema to the StepSettingsEditorView.
        This might involve fetching it via app_adapter if not part of CoreStepData.
        """
        # For now, assume schema might be part of CoreStepData or fetched by controller init
        # This now returns the pre-fetched schema.
        if self.param_schema:
            return self.param_schema
        
        # Fallback if schema somehow wasn't fetched during init (should not happen if init is always called)
        logger.warning("DualEditorController: _get_current_step_schema_for_view called before schema was fetched.")
        if self.editing_step_data and self.editing_step_data.get('parameters'):
             return {
                 name: {'type': type(val).__name__.lower(), 'label': name, 'default': val}
                 for name, val in self.editing_step_data['parameters'].items()
             }
        return {"error": {"type": "str", "label": "Schema Error", "default": "Schema not available."}}


    def _get_func_registry_for_view(self) -> Optional[Any]:
        """Provides func_registry-like data to FuncPatternView if needed."""
        # If FuncPatternView (or underlying FunctionPatternEditor) needs this,
        # it would be fetched via self.app_adapter.
        # For now, returning None, assuming FuncPatternView can handle it or is simplified.
        # Example: return self.app_adapter.get_available_functions_for_step_type(...)
        logger.debug("DualEditorController: Func registry provider called (returning None for now).")
        return None

    # --- Actions ---
    async def _save_changes(self):
        logger.info(f"DualEditorController: Saving changes for step '{self.original_step_data.get('id')}'")
        # Consolidate changes from views if not already live-updated
        # self.editing_step_data['parameters'] = self.step_settings_view.get_all_values() # If view holds state
        # self.editing_step_data['func_pattern'] = self.func_pattern_view.get_current_pattern_data()

        from ..commands import UpdateStepCommand # Local import
        
        # The command will need the plate_id to get the correct plate_adapter
        plate_id = self.ui_state.active_plate_id 
        if not plate_id:
            logger.error("DualEditorController: Cannot save step, no active plate ID in TUIState.")
            await self.ui_state.notify("error", {"message": "Cannot save step: No active plate selected."})
            return

        cmd = UpdateStepCommand(plate_id=plate_id, step_data=self.editing_step_data)
        
        plate_adapter = await self.app_adapter.get_orchestrator_adapter(plate_id)
        if not plate_adapter:
            logger.error(f"DualEditorController: Could not get plate adapter for plate {plate_id}.")
            await self.ui_state.notify("error", {"message": f"Cannot save step: Plate adapter unavailable for {plate_id}."})
            return
            
        try:
            await cmd.execute(self.app_adapter, plate_adapter, self.ui_state)
            # On successful save, notify TUIState to close this editor view.
            # AppController listens to this and calls ui_state.set_active_editor("PLATE_MANAGER")
            await self.ui_state.notify('step_editor_closed', {'step_id': self.original_step_data.get('id'), 'saved': True})
        except Exception as e:
            logger.error(f"Error executing UpdateStepCommand: {e}", exc_info=True)
            await self.ui_state.notify("error", {"message": f"Failed to save step: {e}"})


    async def _cancel_editing(self):
        logger.info(f"DualEditorController: Cancelling edit for step '{self.original_step_data.get('id')}'")
        # Notify TUIState to close this editor view.
        # AppController listens to this and calls ui_state.set_active_editor("PLATE_MANAGER")
        await self.ui_state.notify('step_editor_closed', {'step_id': self.original_step_data.get('id'), 'saved': False})

    def get_container(self) -> Container:
        """Returns the root UI container for this controller."""
        return self.container

    async def shutdown(self):
        """Cleanup if needed."""
        logger.info(f"DualEditorController for step '{self.original_step_data.get('id')}' shutting down.")
        # No specific resources to clean for now, but good practice.