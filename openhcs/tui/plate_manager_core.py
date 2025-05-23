"""
Plate Manager Pane for OpenHCS TUI - Core Implementation.

This module implements the left pane of the OpenHCS TUI, which displays
a list of plates (filesystem directories) and allows the user to select,
add, and manage them.

🔒 Clause 295: Component Boundaries
Maintain clear boundaries between components with explicit interfaces.

🔒 Clause 306: Backend Positional Parameters
All backend parameters must be passed positionally, not as keywords.

🔒 Clause 310: Function Backend Propagation
Any function accepting filemanager must also accept backend.

🔒 Clause 315: TUI_FILEMANAGER_INJECTION
All TUI managers must receive context.filemanager and pass backend positionally.

🔒 Clause 317: TUI_STATUS_THREADSAFETY
All updates to state.operation_status must be serialized through a lock.

🔒 Clause 319: TUI_NO_VIRTUALPATH_EXPOSURE
TUI must work only with plain strings, not VirtualPath objects.
"""
import asyncio
import logging
import os
import shutil  # For terminal size fallback
import signal  # For signal handling in register_with_app
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

from prompt_toolkit.application import get_app
from prompt_toolkit.filters import Condition, has_focus
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Container, HSplit, VSplit, DynamicContainer, Dimension
from prompt_toolkit.widgets import Box, Button, Frame, Label # Removed TextArea
from openhcs.tui.status_bar import STATUS_ICONS
from openhcs.tui.dialogs.plate_dialog_manager import PlateDialogManager
from .components import InteractiveListItem # Import the new component
from openhcs.tui.services.plate_validation import PlateValidationService

from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.io.base import storage_registry
from openhcs.constants.constants import Backend
from openhcs.io.filemanager import FileManager

logger = logging.getLogger(__name__)


# Define interfaces for component communication
class PlateEventHandler(Protocol):
    """Protocol defining the interface for plate event handling."""
    async def on_plate_added(self, plate: Dict[str, Any]) -> None: ...
    async def on_plate_removed(self, plate: Dict[str, Any]) -> None: ...
    async def on_plate_selected(self, plate: Dict[str, Any]) -> None: ...
    async def on_plate_status_changed(self, plate_id: str, status: str) -> None: ...


class PlateManagerPane:
    """
    Left pane for managing plates in the OpenHCS TUI.
    """
    def __init__(self, state, context: ProcessingContext, storage_registry: Any):
        self.state = state
        self.context = context
        self.registry = storage_registry
        # Use the filemanager from the context instead of creating a new one
        self.filemanager = context.filemanager if hasattr(context, 'filemanager') else None

        self.plates: List[Dict[str, Any]] = []
        self.selected_index = 0
        self.is_loading = False
        self.plates_lock = asyncio.Lock()

        self.io_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="plate-io-")

        # Initialize dialog manager and validation service
        self._initialize_services()

        # Register for filemanager_available event
        self.state.add_observer('filemanager_available', self._on_filemanager_available)

        # Initialize UI components immediately
        try:
            app = get_app()
            app.create_background_task(self._initialize_ui())
        except Exception as e:
            logger.warning(f"PlateManagerPane: Error initializing UI: {e}")
            # Will be initialized later when app is available

        self._ui_initialized = False
        self.plate_items_container_widget: Optional[HSplit] = None
        self._dynamic_plate_list_wrapper: Optional[DynamicContainer] = None
        self._container: Optional[Frame] = None

        # Buttons
        self.add_button: Optional[Button] = None
        self.remove_button: Optional[Button] = None
        self.edit_button: Optional[Button] = None
        self.init_button: Optional[Button] = None
        self.compile_button: Optional[Button] = None
        self.run_button: Optional[Button] = None
        self.kb: Optional[KeyBindings] = None


    def _initialize_services(self):
        """Initialize dialog manager and validation service."""
        # Initialize with current filemanager if available
        if self.filemanager is not None:
            self.dialog_manager = PlateDialogManager(
                on_add_dialog_result=self._handle_add_dialog_result,
                on_remove_dialog_result=self._handle_remove_dialog_result,
                on_error=self._handle_error,
                file_manager=self.filemanager,
                default_backend=Backend.DISK
            )

            self.validation_service = PlateValidationService(
                context=self.context,
                on_validation_result=self._handle_validation_result,
                on_error=self._handle_error,
                storage_registry=self.registry,
                io_executor=self.io_executor
            )
            logger.info("PlateManagerPane: Services initialized with filemanager.")
        else:
            self.dialog_manager = None
            self.validation_service = None
            logger.warning("PlateManagerPane: Services not initialized - filemanager not available.")

    async def _on_filemanager_available(self, data: Dict[str, Any]):
        """Handle filemanager_available event."""
        if 'filemanager' not in data:
            logger.warning("PlateManagerPane: filemanager_available event received but no filemanager in data.")
            return

        self.filemanager = data['filemanager']
        logger.info("PlateManagerPane: Received filemanager from event.")

        # Initialize services with the new filemanager
        self._initialize_services()

        # Initialize UI if not already done
        if not self._ui_initialized:
            try:
                app = get_app()
                app.create_background_task(self._initialize_ui())
            except Exception as e:
                logger.warning(f"PlateManagerPane: Error initializing UI after filemanager available: {e}")

    async def _initialize_ui(self):
        if self._ui_initialized:
            return

        app = get_app()
        if not hasattr(app, 'is_running') or not app.is_running:
            logger.warning("PlateManagerPane._initialize_ui called before app is fully running. Deferring.")
            app.call_later(0.1, lambda: get_app().create_background_task(self._initialize_ui()))
            return

        # Ensure self.context is available (should be set in __init__)
        if not hasattr(self, 'context') or self.context is None:
            logger.error("PlateManagerPane: ProcessingContext not available for command execution.")
            # Handle error appropriately, maybe disable buttons or show an error message
            # For now, buttons might fail if context is None.

        # Register observers with the TUIState
        self.register_with_app()

        # Import commands here to avoid circular imports
        from openhcs.tui.commands import (
            ShowAddPlateDialogCommand, DeleteSelectedPlatesCommand,
            ShowEditPlateConfigDialogCommand, InitializePlatesCommand,
            CompilePlatesCommand, RunPlatesCommand
        )

        self.add_button = Button("Add", handler=lambda: get_app().create_background_task(
            ShowAddPlateDialogCommand().execute(self.state, self.context, plate_dialog_manager=self.dialog_manager)
        ))
        self.remove_button = Button("Del", handler=lambda: get_app().create_background_task(
            DeleteSelectedPlatesCommand().execute(self.state, self.context, selected_plates_data=self._get_selected_plate_data_for_action())
        ))
        self.edit_button = Button("Edit", handler=lambda: get_app().create_background_task(
            ShowEditPlateConfigDialogCommand().execute(self.state, self.context) # Relies on state.active_orchestrator
        ))
        self.init_button = Button("Init", handler=lambda: get_app().create_background_task(
            InitializePlatesCommand().execute(self.state, self.context, orchestrators_to_init=self._get_selected_orchestrators_for_action())
        ))
        self.compile_button = Button("Compile", handler=lambda: get_app().create_background_task(
            CompilePlatesCommand().execute(self.state, self.context, orchestrators_to_compile=self._get_selected_orchestrators_for_action())
        ))
        self.run_button = Button("Run", handler=lambda: get_app().create_background_task(
            RunPlatesCommand().execute(self.state, self.context, orchestrators_to_run=self._get_selected_orchestrators_for_action())
        ))

        # Initialize with a placeholder until plates are loaded
        self.plates = [{
            'id': 'loading',
            'path': 'N/A',
            'status': 'info',
            'name': 'Loading plates...',
            'error_details': 'Please wait while plates are being loaded'
        }]

        self.plate_items_container_widget = await self._build_plate_items_container()
        self.kb = self._create_key_bindings()

        self.get_current_plate_list_container = lambda: self.plate_items_container_widget or HSplit([Label("Loading plates...")])
        self._dynamic_plate_list_wrapper = DynamicContainer(self.get_current_plate_list_container)
        self._container = Frame(self._dynamic_plate_list_wrapper, title="Plates")

        self._ui_initialized = True

        # Trigger a refresh of plates after UI is initialized
        app.create_background_task(self._refresh_plates())

    def _get_selected_plate_data_for_action(self) -> Optional[List[Dict[str, Any]]]:
        """Helper to get data of the currently selected plate(s) for commands."""
        # This logic needs to be robust based on how multi-selection is handled.
        # For now, assume single selection via self.selected_index or a list if multi-select is implemented.
        if self.plates and 0 <= self.selected_index < len(self.plates):
            return [self.plates[self.selected_index]] # Return as a list for consistency
        # Could also check self.state.selected_plates if that's a list from multi-select UI
        return None

    def _get_selected_orchestrators_for_action(self) -> List["PipelineOrchestrator"]:
        """Helper to get orchestrator instances of selected plate(s)."""
        # This method should ideally use a more robust way to get selected plates,
        # e.g. if multi-selection is supported by InteractiveListItem or a similar mechanism.
        # For now, it relies on _get_selected_plate_data_for_action which is single-select.
        selected_data = self._get_selected_plate_data_for_action()
        orchestrators: List["PipelineOrchestrator"] = []
        if selected_data:
            for plate_dict in selected_data:
                orchestrator = plate_dict.get('orchestrator')
                if orchestrator: # Add type check for PipelineOrchestrator if possible
                    orchestrators.append(orchestrator)

        # Fallback to active_orchestrator if no specific selection for action is found
        # and the command implies action on the single active one.
        if not orchestrators and hasattr(self.state, 'active_orchestrator') and self.state.active_orchestrator:
            orchestrators.append(self.state.active_orchestrator)

        return orchestrators

    @property
    def container(self) -> Container:
        if not self._container:
            # This case should ideally not be hit if _initialize_ui is called correctly.
            # Fallback to a simple label if not initialized.
            return Frame(Label("Plate Manager not initialized."))
        return self._container

    def get_buttons_container(self) -> Container:
        # Ensure buttons are initialized before creating the container
        if not all([self.add_button, self.remove_button, self.edit_button, self.init_button, self.compile_button, self.run_button]):
             # This can happen if get_buttons_container is called before _initialize_ui completes
             return HSplit([Label("Buttons not ready.")])

        return HSplit([
            VSplit([ # VSplit for horizontal arrangement of buttons
                Box(self.add_button, padding_left=1, padding_right=1),
                Box(self.remove_button, padding_right=1),
                Box(self.edit_button, padding_right=1),
                Box(self.init_button, padding_right=1),
                Box(self.compile_button, padding_right=1),
                Box(self.run_button, padding_right=1),
            ])
        ])

    def register_with_app(self):
        """Registers observers with the TUIState."""
        self.state.add_observer('filemanager_available', self._on_filemanager_available)
        self.state.add_observer('refresh_plates', lambda data=None: get_app().create_background_task(self._refresh_plates(data)))
        self.state.add_observer('plate_status_changed', lambda data: get_app().create_background_task(self._update_plate_status(data)))
        self.state.add_observer('ui_request_show_add_plate_dialog', lambda data=None: get_app().create_background_task(self._handle_request_show_add_plate_dialog(data)))
        self.state.add_observer('add_predefined_plate', lambda data=None: get_app().create_background_task(self._handle_add_predefined_plate(data)))
        self.state.add_observer('delete_plates_requested', self._handle_delete_plates_request) # Added observer
        # Note: Shutdown hook is handled by the main TUI application ensuring self.shutdown() is called.

    async def _handle_request_show_add_plate_dialog(self, data=None):
        logger.info("PlateManagerPane: Received ui_request_show_add_plate_dialog event.")
        await self._show_add_plate_dialog()

    async def _handle_add_predefined_plate(self, data: Optional[Dict[str, Any]] = None):
        if not data or 'path' not in data or 'backend' not in data:
            logger.error("PlateManagerPane: Received 'add_predefined_plate' event with missing data.")
            await self._handle_error("Invalid data for predefined plate.", f"Received: {data}")
            return
        path = data['path']
        backend = data['backend']
        logger.info(f"PlateManagerPane: Received 'add_predefined_plate' event for path='{path}', backend='{backend}'.")
        try:
            await self.validation_service.validate_plate(path, backend) # Validation service will add it via _handle_validation_result
            logger.info(f"PlateManagerPane: Validation initiated for predefined plate '{path}'.")
        except Exception as e:
            logger.error(f"PlateManagerPane: Error initiating validation for predefined plate '{path}': {e}", exc_info=True)
            await self._handle_error(f"Error adding test plate {Path(path).name}", str(e))

    def _on_filemanager_available(self, data):
        """Handle filemanager becoming available."""
        if 'filemanager' in data:
            self.filemanager = data['filemanager']
            # Update validation service with filemanager
            self.validation_service.filemanager = self.filemanager
            # Initialize UI asynchronously now that filemanager is ready
            get_app().create_background_task(self._initialize_and_refresh())


    async def _initialize_and_refresh(self):
        """Initialize UI and refresh plates asynchronously."""
        app = get_app()
        while not hasattr(app, 'is_running') or not app.is_running: # Wait for app to be fully running
            await asyncio.sleep(0.1)

        if not self._ui_initialized: # Ensure UI is initialized only once
            await self._initialize_ui()
            # register_with_app() is now called from _initialize_ui
            logger.info("PlateManagerPane: UI initialized successfully.")
        else:
            logger.info("PlateManagerPane: UI already initialized.")

        # Force a refresh of the plates
        await self._refresh_plates() # Initial refresh
        logger.info("PlateManagerPane: Initial plate refresh completed.")


    async def _update_selection(self):
        """Rebuilds the plate items container to reflect current selection and data."""
        if not self._ui_initialized:
            logger.debug("PlateManagerPane: _update_selection called before UI initialized.")
            return

        self.plate_items_container_widget = await self._build_plate_items_container()

        # Ensure TUIState is updated with the current selection status
        current_selection_valid = self.plates and 0 <= self.selected_index < len(self.plates)
        if current_selection_valid:
            await self._select_plate(self.selected_index)
        elif not self.plates: # If list is empty, ensure no selection is active in TUIState
             await self.state.notify('plate_selected', None)
        # If selection is invalid but list is not empty, _select_plate will handle notifying None if index is bad.

        get_app().invalidate()

    async def _build_plate_items_container(self) -> HSplit:
        """Builds the HSplit container holding individual InteractiveListItem widgets for plates."""
        item_widgets = []
        async with self.plates_lock:
            if self.is_loading and not self.plates:
                 item_widgets.append(Label("Loading plates..."))
            elif not self.plates:
                item_widgets.append(Label("No plates. Click 'Add'."))
            else:
                for i, plate_data in enumerate(self.plates):
                    is_selected = (i == self.selected_index)
                    can_move_up = i > 0
                    can_move_down = i < len(self.plates) - 1
                    item_widget = InteractiveListItem(
                        item_data=plate_data, item_index=i, is_selected=is_selected,
                        display_text_func=self._get_plate_display_text,
                        on_select=self._handle_plate_item_select,
                        on_move_up=self._handle_plate_item_move_up,
                        on_move_down=self._handle_plate_item_move_down,
                        can_move_up=can_move_up, can_move_down=can_move_down
                    )
                    item_widgets.append(item_widget)
        # Ensure HSplit always has children, even if it's just a placeholder label
        return HSplit(item_widgets if item_widgets else [Label(" ")], width=Dimension(weight=1), height=Dimension(weight=1))

    def _get_plate_display_text(self, plate_data: Dict[str, Any], is_selected: bool) -> str:
        """Generates the display text for a single plate item."""
        plate_status = plate_data.get('status', 'unknown')

        # Updated status symbol logic based on V4 plan and common error states
        if plate_status == 'not_initialized': status_symbol = "?"
        elif plate_status == 'initialized': status_symbol = "-"
        elif plate_status == 'compiled_ok': status_symbol = "✓"
        elif plate_status and ('error' in plate_status or plate_status in ['error_init', 'error_compile', 'error_run']):
            status_symbol = "!"
        else: status_symbol = " " # Remove the circle, use space instead

        name = plate_data.get('name', 'Unknown Plate')

        path_str = "[No Path]"
        # Ensure self.filemanager is checked before use, as it's set asynchronously
        fm = getattr(self, 'filemanager', None)
        if 'path' in plate_data and 'backend' in plate_data and fm:
            try:
                # Assuming fm.get_path might not exist or backend is not directly used here
                # If path is already an OS path for disk backend:
                raw_path = plate_data['path']
                # If it's a VFS path object, get its string representation
                if hasattr(raw_path, 'os_path'): # Check if it's a VirtualPath like object
                    path_str = str(raw_path.os_path)
                else:
                    path_str = str(raw_path)

                if len(path_str) > 40:
                    path_str = "(...)" + path_str[-(40-5):]
            except Exception as e:
                logger.debug(f"Error formatting path for display: {plate_data.get('path')}, {e}")
                path_str = plate_data.get('path', '[Path Error]')
        elif 'path' in plate_data: # Fallback if backend or fm not fully set up yet
             path_str = str(plate_data['path'])
             if len(path_str) > 40: path_str = "(...)" + path_str[-(40-5):]

        # Format the display text to match the desired layout
        # The ^/v symbols for reordering are handled by InteractiveListItem
        return f"{status_symbol} {name} | {path_str}"

    # _format_plate_list is obsolete.

    # --- New callback handlers for InteractiveListItem ---
    async def _handle_plate_item_select(self, index: int):
        """Handles selection of a plate item from the list via click."""
        if 0 <= index < len(self.plates): # Ensure index is valid
            self.selected_index = index
            await self._update_selection() # This will rebuild list and call _select_plate

    async def _handle_plate_item_move_up(self, index: int):
        """Handles 'move up' button click for a plate item."""
        if 0 <= index < len(self.plates): # Ensure index is valid
            self.selected_index = index # Set current selection to the item being moved
            await self._move_plate_up()

    async def _handle_plate_item_move_down(self, index: int):
        """Handles 'move down' button click for a plate item."""
        if 0 <= index < len(self.plates): # Ensure index is valid
            self.selected_index = index # Set current selection to the item being moved
            await self._move_plate_down()

    async def _update_selection_and_notify_order(self):
        """Helper to update selection UI and notify about plate order changes."""
        await self._update_selection()
        async with self.plates_lock: # Ensure thread-safe access to self.plates for notification
            await self.state.notify('plate_order_changed', {'plates': list(self.plates)}) # Send a copy
    # --- End of new callback handlers ---

    def _create_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        # Ensure self._container is used for focus check, which is the Frame
        list_container_focused = has_focus(self._container) if self._container else Condition(lambda: False)

        @kb.add('up', filter=list_container_focused)
        def _(event):
            if self.plates: get_app().create_background_task(self._move_selection(-1))
        @kb.add('down', filter=list_container_focused)
        def _(event):
            if self.plates: get_app().create_background_task(self._move_selection(1))

        vim_mode_condition = Condition(lambda: getattr(self.state, "vim_mode", False))
        @kb.add('k', filter=list_container_focused & vim_mode_condition)
        def _(event):
            if self.plates: get_app().create_background_task(self._move_selection(-1))
        @kb.add('j', filter=list_container_focused & vim_mode_condition)
        def _(event):
            if self.plates: get_app().create_background_task(self._move_selection(1))

        @kb.add('s-up', filter=list_container_focused)
        def _(event): get_app().create_background_task(self._move_plate_up())
        @kb.add('s-down', filter=list_container_focused)
        def _(event): get_app().create_background_task(self._move_plate_down())

        @kb.add('enter', filter=list_container_focused)
        def _(event):
            if self.plates and 0 <= self.selected_index < len(self.plates):
                # Enter on a plate confirms selection and potentially opens an edit/detail view in future
                get_app().create_background_task(self._select_plate(self.selected_index))
        return kb

    async def _move_plate_up(self):
        logger.info("PlateManagerPane: Move plate up.")
        async with self.plates_lock:
            if not self.plates or self.selected_index == 0: return
            idx = self.selected_index
            plate_to_move = self.plates.pop(idx)
            self.plates.insert(idx - 1, plate_to_move)
            self.selected_index = idx - 1 # Update selected index to follow the moved item
            await self._update_selection_and_notify_order()

    async def _move_plate_down(self):
        logger.info("PlateManagerPane: Move plate down.")
        async with self.plates_lock:
            if not self.plates or self.selected_index >= len(self.plates) - 1: return
            idx = self.selected_index
            plate_to_move = self.plates.pop(idx)
            self.plates.insert(idx + 1, plate_to_move)
            self.selected_index = idx + 1 # Update selected index to follow the moved item
            await self._update_selection_and_notify_order()

    # Old handler methods _show_add_plate_dialog, _show_remove_plate_dialog,
    # and the missing _on_edit_plate_clicked, _on_init_plate_clicked,
    # _on_compile_plate_clicked, _on_run_plate_clicked are now effectively
    # replaced by the Command executions in _initialize_ui.
    # Their direct invocation is removed.
    # The logic for what these actions do is now encapsulated within their respective Command classes.

    # _handle_add_dialog_result is still needed if ShowAddPlateDialogCommand (or dialog manager)
    # calls back to PlateManagerPane after dialog completion.
    # Or, ShowAddPlateDialogCommand handles the result itself.
    # For now, assuming commands might notify TUIState, and PlateManagerPane reacts.
    # Let's keep _handle_add_dialog_result as it's used by dialog_manager.
    # The ShowAddPlateDialogCommand will trigger dialog_manager.show_add_plate_dialog,
    # which then calls this handler.
    async def _handle_add_dialog_result(self, result: Dict[str, Any]):
        if not result or 'paths' not in result or not result['paths']: # Check for None, 'paths' key, and empty list
            logger.error("PlateManagerPane: '_handle_add_dialog_result' received no 'paths' or empty list.")
            # Notify error or simply log and return if cancellation is handled by dialog manager
            # await self._handle_error("Failed to add plate(s): No paths provided.", f"Result: {result}")
            return # Assuming cancellation or empty selection is handled, so just return

        paths_to_process = result['paths'] # Expecting a list of path strings
        # Ensure it's a list, though PlateDialogManager should now always send a list
        if not isinstance(paths_to_process, list):
            logger.warning(f"PlateManagerPane: _handle_add_dialog_result expected a list of paths, got {type(paths_to_process)}. Wrapping.")
            paths_to_process = [str(paths_to_process)]

        if not self.state.global_config:
            logger.error("PlateManagerPane: Global config not available.")
            await self._handle_error("Cannot add plate(s): Global configuration is missing.")
            return

        added_plate_details = []
        for path_str in paths_to_process:
            try:
                orchestrator = PipelineOrchestrator(plate_path=path_str, config=self.state.global_config, storage_registry=self.registry)
                plate_tui_id = str(Path(path_str).name) + f"_{orchestrator.config.vfs.default_storage_backend}"
                new_plate_entry = {'id': plate_tui_id, 'name': Path(path_str).name, 'path': path_str,
                                   'status': 'not_initialized', 'orchestrator': orchestrator,
                                   'backend': orchestrator.config.vfs.default_storage_backend}
                async with self.plates_lock:
                    if any(p['id'] == new_plate_entry['id'] for p in self.plates):
                        logger.warning(f"Plate with TUI ID '{new_plate_entry['id']}' already exists. Skipping.")
                        continue
                    self.plates.append(new_plate_entry)
                added_plate_details.append({'plate_id': new_plate_entry['id'], 'path': new_plate_entry['path']})
                logger.info(f"Added new plate '{new_plate_entry['name']}' (ID: {new_plate_entry['id']}). Status: not_initialized.")
            except Exception as e:
                logger.error(f"Error creating orchestrator for path '{path_str}': {e}", exc_info=True)
                await self._handle_error(f"Failed to add plate for path: {path_str}", str(e))

        if added_plate_details:
            async with self.plates_lock:
                if self.plates:
                    self.selected_index = len(self.plates) - 1
                await self._update_selection() # This will rebuild UI and call _select_plate
            # Notify for each added plate (orchestrator instance)
            for plate_data in self.plates: # Iterate through current plates
                for detail in added_plate_details:
                    if plate_data['id'] == detail['plate_id']:
                         await self.state.notify('plate_orchestrator_added', {
                            'plate_id': plate_data['id'],
                            'orchestrator': plate_data['orchestrator'], # Send the orchestrator instance
                            'path': plate_data['path']
                        })
                         break # Found the added plate, move to next detail
            logger.info(f"Processed and notified for {len(added_plate_details)} new plate(s).")
        else:
            logger.info("No new plates were added from dialog result.")

    async def _handle_delete_plates_request(self, data: Dict[str, Any]):
        """Handles the 'delete_plates_requested' event from DeleteSelectedPlatesCommand."""
        plates_to_delete_data = data.get('plates_to_delete', [])
        if not plates_to_delete_data:
            logger.info("PlateManagerPane: Received delete_plates_requested but no plates specified.")
            return

        ids_to_remove = {plate_data.get('id') for plate_data in plates_to_delete_data if plate_data.get('id')}
        if not ids_to_remove:
            logger.warning("PlateManagerPane: No valid plate IDs found in delete_plates_requested.")
            return

        # Remove plates with matching IDs
        async with self.plates_lock:
            original_length = len(self.plates)
            self.plates = [p for p in self.plates if p.get('id') not in ids_to_remove]
            num_removed = original_length - len(self.plates)

            # Update selected index if needed
            if self.plates:
                self.selected_index = min(self.selected_index, len(self.plates) - 1)
            else:
                self.selected_index = 0

            # Update UI
            await self._update_selection()

            # Notify about removed plates
            for plate_id in ids_to_remove:
                await self.state.notify('plate_removed', {'id': plate_id})

            logger.info(f"Removed {num_removed} plates from PlateManagerPane.")

    async def _handle_error(self, message: str, details: str = ""):
        """Handle errors by notifying the TUIState."""
        logger.error(f"PlateManagerPane error: {message} - {details}")
        await self.state.notify('error', {
            'source': 'PlateManagerPane',
            'message': message,
            'details': details
        })

    async def _handle_validation_result(self, data: Dict[str, Any]):
        """Handle plate validation result."""
        if not data:
            logger.error("PlateManagerPane: _handle_validation_result called with no data.")
            return

        is_valid = data.get('is_valid', False)
        path = data.get('path')
        backend = data.get('backend', Backend.DISK)

        if not is_valid:
            error_details = data.get('error_details', "Unknown validation error")
            await self._handle_error(f"Failed to validate plate: {path}", error_details)
            return

        # Plate is valid, add it to the list
        try:
            plate_name = Path(path).name
            plate_id = f"{plate_name}_{backend}"

            # Check if plate already exists
            async with self.plates_lock:
                if any(p['id'] == plate_id for p in self.plates):
                    logger.warning(f"Plate with ID '{plate_id}' already exists. Skipping.")
                    return

                # Create a new plate entry
                new_plate_entry = {
                    'id': plate_id,
                    'name': plate_name,
                    'path': path,
                    'status': 'not_initialized',
                    'backend': backend
                }

                # Add the plate to the list
                self.plates.append(new_plate_entry)
                self.selected_index = len(self.plates) - 1

                # Update the UI
                await self._update_selection()

                # Notify that a plate was added
                await self.state.notify('plate_added', new_plate_entry)

        except Exception as e:
            logger.error(f"Error adding validated plate '{path}': {e}", exc_info=True)
            await self._handle_error(f"Error adding plate: {path}", str(e))

    async def _handle_add_dialog_result(self, result: Dict[str, Any]):
        """Handle the result of the add plate dialog."""
        if not result or not result.get('paths'):
            logger.info("PlateManagerPane: Add plate dialog cancelled or no paths provided.")
            return

        paths = result.get('paths', [])
        backend = result.get('backend', Backend.DISK)

        if not paths:
            logger.info("PlateManagerPane: No paths provided in add plate dialog result.")
            return

        # Validate each path
        for path in paths:
            # Use the validation service to validate the plate
            if self.validation_service:
                await self.validation_service.validate_plate(path, backend)
            else:
                logger.error("PlateManagerPane: Validation service not initialized.")
                await self._handle_error("Cannot validate plate", "Validation service not initialized")

        logger.info(f"PlateManagerPane: Submitted {len(paths)} paths for validation.")

    async def _handle_remove_dialog_result(self, result: Dict[str, Any]):
        """Handle the result of the remove plate dialog."""
        if not result or 'plate_id' not in result:
            logger.info("PlateManagerPane: Remove plate dialog cancelled or no plate ID provided.")
            return

        plate_id_to_remove = result['plate_id']

        # Find the plate to remove
        async with self.plates_lock:
            plate_to_remove = None
            for i, plate in enumerate(self.plates):
                if plate.get('id') == plate_id_to_remove:
                    plate_to_remove = plate
                    break

            if not plate_to_remove:
                logger.warning(f"PlateManagerPane: Plate with ID '{plate_id_to_remove}' not found for removal.")
                return

            # Remove the plate
            self.plates.remove(plate_to_remove)

            # Update selected index if needed
            if self.plates:
                self.selected_index = min(self.selected_index, len(self.plates) - 1)
            else:
                self.selected_index = 0

            # Update UI
            await self._update_selection()

            # Notify that a plate was removed
            await self.state.notify('plate_removed', {'id': plate_id_to_remove})

            logger.info(f"Removed plate with ID '{plate_id_to_remove}' from PlateManagerPane.")
            return

        removed_plate_details = []
        async with self.plates_lock:
            original_length = len(self.plates)

            # Store details of plates being removed before modifying self.plates
            for plate_id_to_remove in ids_to_remove:
                for plate_in_list in self.plates:
                    if plate_in_list.get('id') == plate_id_to_remove:
                        removed_plate_details.append(dict(plate_in_list)) # Store a copy
                        break

            self.plates = [p for p in self.plates if p.get('id') not in ids_to_remove]
            num_removed = original_length - len(self.plates)

            if self.selected_index >= len(self.plates):
                self.selected_index = max(0, len(self.plates) - 1)

        if num_removed > 0:
            logger.info(f"PlateManagerPane: Removed {num_removed} plate(s) with IDs: {ids_to_remove}.")
            await self._update_selection() # This refreshes UI and notifies 'plate_selected'

            # Notify TUIState for each actually removed plate so TUILauncher can clean up orchestrators
            for removed_plate_detail in removed_plate_details:
                 await self.state.notify('plate_removed', removed_plate_detail) # Send full detail for TUILauncher
        else:
            logger.warning(f"PlateManagerPane: No plates found matching IDs {ids_to_remove} for deletion.")


    async def _handle_remove_dialog_result(self, plate_to_remove: Dict[str, Any]):
        # This method is called by PlateDialogManager after user confirms removal of a single plate.
        # The DeleteSelectedPlatesCommand might show a single confirmation for multiple plates,
        # then notify 'delete_plates_requested'. This _handle_remove_dialog_result might become
        # less used if multi-delete confirmation is handled centrally by the command.
        # For now, keeping its logic as it's tied to the dialog manager's current single-item remove flow.
        # If DeleteSelectedPlatesCommand directly calls this for each, it's fine.
        # However, the current DeleteSelectedPlatesCommand notifies 'delete_plates_requested'
        # which is now handled by _handle_delete_plates_request.
        # This method might be simplified or removed if PlateDialogManager's remove dialog
        # is only ever triggered by a command that handles multi-selection confirmation.
        # For safety, let's assume it might still be called for single deletions.
        logger.info(f"PlateManagerPane: _handle_remove_dialog_result for plate ID: {plate_to_remove.get('id')}")
        await self._handle_delete_plates_request({'plates_to_delete': [plate_to_remove]})


    async def _handle_validation_result(self, validated_plate: Dict[str, Any]):
        """Handles validation result, typically adding or updating a plate."""
        async with self.plates_lock:
            existing_plate_index = -1
            for i, p in enumerate(self.plates):
                if p['id'] == validated_plate['id']:
                    existing_plate_index = i
                    break

            if existing_plate_index != -1:
                self.plates[existing_plate_index].update(validated_plate)
            else:
                self.plates.append(validated_plate)
                if self.plates:
                    self.selected_index = len(self.plates) - 1
            await self._update_selection() # Rebuilds UI and calls _select_plate

        # Notify TUIState about the added/updated plate if it's ready
        if validated_plate.get('status') == 'ready': # Check status from validated_plate
            self.state.notify('plate_added', validated_plate) # Send the full plate data

    async def _handle_error(self, message: str, details: str = None):
        if not hasattr(self.state, 'error_logs'):
            from collections import deque
            self.state.error_logs = deque(maxlen=200)
        if details: self.state.error_logs.append(f"PlateManagerPane Error: {message} - Details: {details}")
        else: self.state.error_logs.append(f"PlateManagerPane Error: {message}")
        self.state.notify('error', {'source': 'PlateManagerPane', 'message': message, 'details': details})

    async def _on_edit_plate_clicked(self):
        logger.info("PlateManagerPane: Edit plate clicked.")
        if not self.plates or not (0 <= self.selected_index < len(self.plates)):
            await self._handle_error("No plate selected.", "Select a plate to edit its configuration.")
            return
        plate_entry = self.plates[self.selected_index]
        orchestrator = plate_entry.get('orchestrator')
        if not orchestrator:
             await self._handle_error(f"Orchestrator not found for selected plate '{plate_entry['name']}'. Cannot edit.")
             return
        logger.info(f"Attempting to edit config for plate '{plate_entry['name']}'.")
        # Actual edit dialog logic would go here.
        await self._handle_error(f"Edit Plate Configuration for '{plate_entry['name']}' not yet implemented.", "This feature (PlateConfigEditorDialog) is part of a future phase.")

    async def _on_init_plate_clicked(self):
        logger.info("Init plate clicked (Not Implemented).")
        await self._handle_error("Initialize Plate functionality not yet implemented.", "This feature is pending.")
    async def _on_compile_plate_clicked(self):
        logger.info("Compile plate clicked (Not Implemented).")
        await self._handle_error("Compile Plate functionality not yet implemented.", "This feature is pending.")
    async def _on_run_plate_clicked(self):
        logger.info("Run plate clicked (Not Implemented).")
        await self._handle_error("Run Plate functionality not yet implemented.", "This feature is pending.")

    async def _move_selection(self, delta: int) -> None:
        """Move selection up or down (typically for keyboard navigation)."""
        async with self.plates_lock:
            if self.plates:
                new_index = max(0, min(len(self.plates) - 1, self.selected_index + delta))
                if new_index != self.selected_index:
                    self.selected_index = new_index
                    await self._update_selection() # This will rebuild UI and call _select_plate

    def _ensure_selection_visible(self) -> None:
        """Placeholder. Relies on PTK default scroll-to-focus for Frame/DynamicContainer."""
        pass

    async def _select_plate(self, index: int) -> None:
        """Updates TUIState with the currently selected plate's information."""
        plate_to_select: Optional[Dict[str, Any]] = None
        is_valid_selection = False
        async with self.plates_lock: # Lock for accessing self.plates and self.selected_index
            if self.plates and 0 <= index < len(self.plates):
                self.selected_index = index # Ensure selected_index is up-to-date
                plate_to_select = self.plates[index]
                is_valid_selection = True
            # Note: _update_selection() is NOT called here to prevent recursion,
            # as _update_selection() itself calls _select_plate().
            # The UI refresh is handled by the caller of _select_plate (e.g., _update_selection).

        if is_valid_selection and plate_to_select:
            await self.state.notify('plate_selected', {
                'id': plate_to_select.get('id'),
                'backend': plate_to_select.get('backend'),
                'path': plate_to_select.get('path'),
                'orchestrator': plate_to_select.get('orchestrator') # Pass orchestrator on selection
            })
        else:
            await self.state.notify('plate_selected', None) # No valid selection


    async def _update_plate_status(self, data):
        """Handles 'plate_status_changed' events from TUIState."""
        plate_id = data.get('plate_id')
        new_status = data.get('status')
        error_message = data.get('error_message') # Optional error message
        if not plate_id or not new_status: return
        # Update local state and UI, but don't re-notify TUIState (notify_state=False)
        await self._update_plate_status_locally_and_notify(plate_id, new_status, error_message, notify_state=False)

    async def _update_plate_status_locally_and_notify(self, plate_id: str, new_status: str, message: Optional[str] = None, notify_state: bool = True):
        """Helper to update plate status in self.plates, refresh UI, and optionally notify TUIState."""
        updated = False
        async with self.plates_lock:
            for plate in self.plates:
                if plate.get('id') == plate_id:
                    plate['status'] = new_status
                    if message and new_status == 'error': plate['error_message'] = message
                    else: plate.pop('error_message', None) # Clear error if status is not error
                    updated = True
                    break
            if updated:
                await self._update_selection() # Rebuild UI to reflect status change

        if notify_state and updated: # If this method is called directly to also notify TUIState
            await self.state.notify('plate_status_changed', {
                'plate_id': plate_id,
                'status': new_status,
                'message': message
            })

    async def _refresh_plates(self, _=None):
        """Refresh the plate list, re-fetching data from the filesystem."""
        logger.info("Refreshing plates...")
        async with self.plates_lock:
            self.is_loading = True
            # Update UI to show loading state
            await self._update_selection()

        try:
            # Get the common output directory from the context
            common_output_directory = getattr(self.context, 'common_output_directory', None)
            if not common_output_directory:
                logger.error("PlateManagerPane: Cannot refresh plates - no common_output_directory in context.")
                # Create a placeholder entry to show the error
                async with self.plates_lock:
                    self.plates = [{
                        'id': 'error',
                        'path': 'N/A',
                        'status': 'error',
                        'name': 'Output directory not set',
                        'error_details': 'No common_output_directory in context'
                    }]
                return

            # Ensure we have a filemanager
            if not hasattr(self, 'filemanager') or self.filemanager is None:
                logger.error("PlateManagerPane: Cannot refresh plates - no filemanager available.")
                # Create a placeholder entry to show the error
                async with self.plates_lock:
                    self.plates = [{
                        'id': 'error',
                        'path': 'N/A',
                        'status': 'error',
                        'name': 'FileManager not available',
                        'error_details': 'No filemanager available'
                    }]
                return

            # Ensure the directory exists
            if not self.filemanager.exists(common_output_directory, backend='disk'):
                logger.info(f"PlateManagerPane: Creating common output directory: {common_output_directory}")
                self.filemanager.make_dir(common_output_directory, backend='disk')

            # List all plate directories in the common output directory
            paths = self.filemanager.list_dir(common_output_directory, backend='disk')

            # Filter to only include directories
            plate_paths = []
            for path in paths:
                if self.filemanager.is_dir(path, backend='disk'):
                    plate_paths.append(path)

            # Update the plates list with the new data
            async with self.plates_lock:
                # If no plates found, create a placeholder entry
                if not plate_paths:
                    logger.info("PlateManagerPane: No plates found. Creating placeholder.")
                    self.plates = [{
                        'id': 'no_plates',
                        'path': common_output_directory,
                        'status': 'info',
                        'name': 'No plates found',
                        'error_details': 'Use [Add] to create a new plate'
                    }]
                else:
                    # Update the plates list with the new data
                    self.plates = []
                    for path in plate_paths:
                        # Extract plate ID from path
                        plate_id = Path(path).name
                        # Create a plate detail entry
                        plate_detail = {
                            'id': plate_id,
                            'path': str(path),
                            'status': 'not_initialized',  # Default status
                            'name': plate_id,  # Use ID as name by default
                            'orchestrator': None  # Will be set when initialized
                        }
                        self.plates.append(plate_detail)

                logger.info(f"PlateManagerPane: Refreshed {len(self.plates)} plates.")
        except Exception as e:
            logger.error(f"Error during plate refresh: {e}", exc_info=True)
            await self._handle_error("Failed to refresh plates.", str(e))
            # Create a placeholder entry to show the error
            async with self.plates_lock:
                self.plates = [{
                    'id': 'error',
                    'path': 'N/A',
                    'status': 'error',
                    'name': f'Error: {str(e)}',
                    'error_details': str(e)
                }]
        finally:
            async with self.plates_lock:
                self.is_loading = False
            await self._update_selection() # Rebuild with current (possibly unchanged) self.plates data

    async def shutdown(self):
        """Performs cleanup of resources like the ThreadPoolExecutor."""
        logger.info("PlateManagerPane: Shutting down...")
        if hasattr(self, 'validation_service') and self.validation_service is not None:
            await self.validation_service.close()
            self.validation_service = None
        if hasattr(self, 'io_executor') and self.io_executor is not None:
            # Ensure executor shutdown is handled correctly in async context
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.io_executor.shutdown, True) # wait=True
            self.io_executor = None
        logger.info("PlateManagerPane: Shutdown complete.")
