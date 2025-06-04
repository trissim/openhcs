"""
Clean Plate Manager Pane using unified list management architecture.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path


from prompt_toolkit.layout import Container, Window
from prompt_toolkit.application import get_app

from openhcs.io.filemanager import FileManager
from openhcs.tui.components import ListManagerPane, ListConfig, ButtonConfig
from openhcs.tui.utils.dialog_helpers import prompt_for_multi_folder_dialog
from openhcs.tui.interfaces.swappable_pane import SwappablePaneInterface
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.orchestrator.gpu_scheduler import setup_global_gpu_registry
from openhcs.core.config import GlobalPipelineConfig
from openhcs.tui.components.config_editor import ConfigEditor

logger = logging.getLogger(__name__)





class PlateManagerPane(SwappablePaneInterface):
    """Clean plate manager using unified list management architecture."""

    def __init__(self, state, filemanager: FileManager, global_config: GlobalPipelineConfig):
        """Initialize with clean architecture."""
        self.state = state
        self.filemanager = filemanager
        self.global_config = global_config

        # Business state
        self.orchestrators: Dict[str, Any] = {}
        self.pipelines: Dict[str, List] = {}
        self.plate_configs: Dict[str, GlobalPipelineConfig] = {}  # Store plate-specific config overrides

        # Create list manager configuration (without enabled_func initially)
        config = ListConfig(
            title="Plate Manager",
            button_configs=[
                ButtonConfig("Add", self._handle_add_plates, width=len("Add") + 2),
                ButtonConfig("Del", self._handle_delete_plates, width=len("Del") + 2),
                ButtonConfig("Edit", self._handle_edit_plate, width=len("Edit") + 2),
                ButtonConfig("Init", self._handle_initialize_plates, width=len("Init") + 2),
                ButtonConfig("Compile", self._handle_compile_plates, width=len("Compile") + 2),
                ButtonConfig("Run", self._handle_run_plates, width=len("Run") + 2),
            ],
            display_func=self._get_display_text,
            empty_message="Click 'Add' to add plates.\n\nStatus: ? = added, - = initialized, o = compiled, ! = running, X = error",
            allow_multi_select=True  # Enable checkbox clicking
        )

        self.list_manager = ListManagerPane(config, self.filemanager.registry)

        # Now set enabled functions after list_manager exists
        config.button_configs[1].enabled_func = lambda: len(self.list_manager.model.items) > 0  # Del
        config.button_configs[2].enabled_func = lambda: self.list_manager.get_selected_item() is not None  # Edit
        self.list_manager._on_model_changed = self._on_selection_changed

        logger.info("PlateManagerPane: Initialized")
    
    @property
    def container(self) -> Container:
        """Return the PlateManager Frame container."""
        if not hasattr(self, 'list_manager') or not self.list_manager:
            raise RuntimeError("PlateManager list_manager not initialized")
        return self.list_manager.container

    def get_focus_window(self):
        """Return the ListView's FormattedTextControl (like FileManagerBrowser)."""
        if not hasattr(self, 'list_manager') or not self.list_manager:
            raise RuntimeError("PlateManager list_manager not initialized")
        if not hasattr(self.list_manager, 'view') or not self.list_manager.view:
            raise RuntimeError("PlateManager ListView not initialized")
        if not hasattr(self.list_manager.view, 'list_control'):
            raise RuntimeError("ListView list_control not initialized")
        if not self.list_manager.view.list_control:
            raise RuntimeError("ListView list_control is None")
        return self.list_manager.view.list_control

    def _get_display_text(self, plate_data: Dict[str, Any], is_selected: bool) -> str:
        """Generate display text for a plate."""
        status = self._get_status_symbol(plate_data.get('status', '?'))
        name = plate_data.get('name', 'Unknown Plate')
        path = plate_data.get('path', 'Unknown Path')
        return f"{status} {name} | {path}"

    def _get_status_symbol(self, status: str) -> str:
        """Get status symbol for plate."""
        symbols = {
            '?': '?',  # Added but not initialized
            '-': '-',  # Initialized but not compiled
            'o': 'o',  # Compiled and ready
            '!': '!',  # Running
            'X': 'X'   # Error
        }
        return symbols.get(status, '?')

    def _on_selection_changed(self):
        """Handle selection changes."""
        # CRITICAL: Call the original method to trigger UI invalidation
        from prompt_toolkit.application import get_app
        get_app().invalidate()

        # Then handle our business logic
        selected_item = self.list_manager.get_selected_item()
        if selected_item:
            self.state.set_selected_plate(selected_item)

    # Selection state logic (BULLETPROOF)
    def get_selection_state(self) -> Tuple[List[Dict], str]:
        """Get current selection state with bulletproof validation.

        Returns:
            Tuple[List[Dict], str]: (selected_items, selection_mode)
            - selected_items: List of item dictionaries to operate on
            - selection_mode: "empty" | "all" | "cursor" | "checkbox"
        """
        # VALIDATION 1: Check if list exists and has items
        if not hasattr(self, 'list_manager') or not self.list_manager:
            raise RuntimeError("list_manager not initialized")
        if not hasattr(self.list_manager, 'model') or not self.list_manager.model:
            raise RuntimeError("list_manager.model not initialized")

        all_items = self.list_manager.model.get_all_items()
        if not all_items:
            return [], "empty"

        # VALIDATION 2: Check if any checkboxes are currently checked (highest priority)
        checked_items = self.list_manager.model.get_checked_items()
        if checked_items:
            # Validate checked items are still in the list
            valid_checked = [item for item in checked_items if item in all_items]
            if not valid_checked:
                # Checked items were removed - clear checks and fall through to cursor/all logic
                self.list_manager.model.clear_all_checks()
            else:
                # At least one checkbox is currently checked - use checkbox mode
                return valid_checked, "checkbox"

        # VALIDATION 3: Check for cursor selection
        highlighted_item = self.list_manager.get_selected_item()
        if highlighted_item and highlighted_item in all_items:
            return [highlighted_item], "cursor"

        # VALIDATION 4: No specific selection - default to all items
        return all_items, "all"

    def get_operation_description(self, selected_items: List[Dict], selection_mode: str, operation: str) -> str:
        """Generate human-readable description of what will be operated on."""
        count = len(selected_items)
        if selection_mode == "empty":
            return f"No plates available for {operation}"
        elif selection_mode == "all":
            return f"{operation.title()} ALL {count} plates"
        elif selection_mode == "cursor":
            item_name = selected_items[0].get('name', 'Unknown')
            return f"{operation.title()} highlighted plate: {item_name}"
        elif selection_mode == "checkbox":
            if count == 1:
                item_name = selected_items[0].get('name', 'Unknown')
                return f"{operation.title()} checked plate: {item_name}"
            else:
                return f"{operation.title()} {count} checked plates"
        else:
            raise ValueError(f"Unknown selection_mode: {selection_mode}")

    # Action handlers
    async def _handle_add_plates(self):
        """Handle Add Plates button."""
        logger.info("PlateManager: _handle_add_plates called!")
        await self._add_plates()

    async def _add_plates(self):
        """Add plates via multi-folder selection and create orchestrators."""
        folder_paths = await prompt_for_multi_folder_dialog(
            title="Add Plates",
            prompt_message="Select plate folders:",
            app_state=self.state,
            filemanager=self.filemanager
        )

        if not folder_paths:
            return

        new_plates = []
        for folder_path in folder_paths:
            path_obj = Path(folder_path)
            plate_path = str(path_obj)

            # Use plate-specific config if available, otherwise global config
            plate_config = self.plate_configs.get(plate_path, self.global_config)

            # Create orchestrator (construction only, no initialization)
            # This will raise an exception if there are issues - let global handler catch it
            orchestrator = PipelineOrchestrator(
                plate_path=plate_path,
                global_config=plate_config,
                storage_registry=self.filemanager.registry
            )

            # Store orchestrator
            self.orchestrators[plate_path] = orchestrator

            # Add plate entry with '?' status (created but not initialized)
            new_plates.append({
                'name': path_obj.name,
                'path': plate_path,
                'status': '?'
            })

            logger.info(f"Created orchestrator for plate: {path_obj.name}")

        # Add to existing plates
        current_plates = self.list_manager.model.items
        self.list_manager.load_items(current_plates + new_plates)
        logger.info(f"Added {len(new_plates)} plates")

    async def _edit_single_plate_config(self, selected_plate: Dict[str, Any]):
        """Edit configuration for a single plate."""
        if not selected_plate:
            logger.warning("No plate provided for editing")
            return

        plate_path = selected_plate['path']

        # Get current plate-specific config or use global config as base
        current_plate_config = self.plate_configs.get(plate_path)
        if current_plate_config is None:
            # Use global config as starting point
            current_plate_config = self.global_config

        # Create config editor for plate-specific overrides
        config_editor = ConfigEditor(
            config_class=GlobalPipelineConfig,
            current_config=current_plate_config,
            backend=getattr(self.state, 'backend', 'disk'),
            scope="plate",
            base_config=self.global_config,  # Show what's being overridden
            on_config_change=self._on_plate_config_change,
            on_reset_field=self._on_plate_config_reset_field,
            on_reset_all=self._on_plate_config_reset_all
        )

        # Build the UI container
        config_container = config_editor.build_ui()

        def save_and_close():
            # Get updated config from editor
            updated_config = config_editor.get_current_config()

            # Store plate-specific config
            self.plate_configs[plate_path] = updated_config

            # Update orchestrator if it exists
            if plate_path in self.orchestrators:
                orchestrator = self.orchestrators[plate_path]
                get_app().create_background_task(orchestrator.apply_new_global_config(updated_config))

            logger.info(f"Plate config updated for {selected_plate['name']}: {updated_config}")
            self._hide_dialog()

        def cancel_and_close():
            self._hide_dialog()

        # Import dialog components
        from prompt_toolkit.widgets import Dialog
        from prompt_toolkit.layout.containers import HSplit, VSplit
        from prompt_toolkit.widgets import Button
        from prompt_toolkit.layout.containers import Window

        def dialog_button(text, handler):
            return Button(text, handler=handler, width=len(text) + 2)

        # Create dialog with config editor
        settings_dialog = Dialog(
            title=f"Plate Settings - {selected_plate['name']}",
            body=HSplit([
                config_container,
                VSplit([
                    dialog_button("Save", handler=save_and_close),
                    Window(width=2, char=' '),  # Spacer
                    dialog_button("Cancel", handler=cancel_and_close)
                ], height=1)
            ]),
            buttons=[],
            width=80,
            modal=True
        )

        self._show_dialog(settings_dialog)

    def _show_dialog(self, dialog):
        """Show dialog using the same pattern as menu_bar."""
        from prompt_toolkit.application import get_app
        from prompt_toolkit.layout.containers import Float

        layout = get_app().layout
        if hasattr(layout, 'container') and hasattr(layout.container, 'floats'):
            float_dialog = Float(content=dialog)
            layout.container.floats.append(float_dialog)
            get_app().invalidate()

    def _hide_dialog(self):
        """Hide dialog using the same pattern as menu_bar."""
        from prompt_toolkit.application import get_app

        layout = get_app().layout
        if hasattr(layout, 'container') and hasattr(layout.container, 'floats'):
            if layout.container.floats:
                layout.container.floats.pop()
                get_app().invalidate()

    async def _on_plate_config_change(self, field_name: str, new_value: Any, scope: str):
        """Handle plate config field changes."""
        logger.info(f"Plate config field changed: {field_name} = {new_value}")

    async def _on_plate_config_reset_field(self, field_name: str, scope: str):
        """Handle plate config field reset."""
        logger.info(f"Plate config field reset: {field_name}")

    async def _on_plate_config_reset_all(self, scope: str):
        """Handle plate config reset all."""
        logger.info("Plate config reset all")

    async def _handle_delete_plates(self):
        """Handle Delete Plates button based on selection state."""
        selected_items, selection_mode = self.get_selection_state()

        if selection_mode == "empty":
            logger.warning("No plates available for deletion")
            return

        # Generate description and confirm deletion
        description = self.get_operation_description(selected_items, selection_mode, "delete")
        # TODO: Implement confirmation dialog
        logger.info(f"Would delete: {description}")

        # Clean up orchestrators and remove items from model
        for item in selected_items:
            plate_path = item['path']

            # Remove orchestrator if it exists
            if plate_path in self.orchestrators:
                del self.orchestrators[plate_path]
                logger.info(f"Cleaned up orchestrator for: {item['name']}")

            # Remove plate-specific config if it exists
            if plate_path in self.plate_configs:
                del self.plate_configs[plate_path]
                logger.info(f"Cleaned up config for: {item['name']}")

            # Remove from UI model
            self.list_manager.model.remove_item_by_data(item)

        # Clear checkboxes if in checkbox mode
        if selection_mode == "checkbox":
            self.list_manager.model.clear_all_checks()

        logger.info(f"Deleted {len(selected_items)} plates")

    async def _handle_edit_plate(self):
        """Handle Edit Plate button based on selection state."""
        selected_items, selection_mode = self.get_selection_state()

        if selection_mode == "empty":
            logger.warning("No plates available for editing")
            return

        if selection_mode == "checkbox" and len(selected_items) > 1:
            # Multi-edit mode: apply config to all checked plates
            logger.info(f"Multi-edit mode for {len(selected_items)} plates")
            # TODO: Implement multi-plate config editor
            await self._edit_single_plate_config(selected_items[0])  # For now, edit first one
        else:
            # Single edit mode: edit first/highlighted plate only
            await self._edit_single_plate_config(selected_items[0])

    async def _handle_initialize_plates(self):
        """Handle Initialize Plates button based on selection state."""
        selected_items, selection_mode = self.get_selection_state()

        if selection_mode == "empty":
            logger.warning("No plates available for initialization")
            return

        # Validate all selected plates are created but not initialized
        invalid_plates = [item for item in selected_items if item.get('status') not in ['?', '-']]
        if invalid_plates:
            names = [item['name'] for item in invalid_plates]
            raise ValueError(f"Cannot initialize plates with invalid status: {', '.join(names)}")

        # Generate description and log operation
        description = self.get_operation_description(selected_items, selection_mode, "initialize")
        logger.info(f"Initializing: {description}")

        await self._initialize_selected_plates(selected_items)

    async def _handle_compile_plates(self):
        """Handle Compile Plates button based on selection state."""
        selected_items, selection_mode = self.get_selection_state()

        if selection_mode == "empty":
            logger.warning("No plates available for compilation")
            return

        # Validate all selected plates are initialized
        uninitialized = [item for item in selected_items if item.get('status') == '?']
        if uninitialized:
            names = [item['name'] for item in uninitialized]
            raise ValueError(f"Cannot compile uninitialized plates: {', '.join(names)}")

        # Generate description and log operation
        description = self.get_operation_description(selected_items, selection_mode, "compile")
        logger.info(f"Compiling: {description}")

        await self._compile_selected_plates(selected_items)

    async def _handle_run_plates(self):
        """Handle Run Plates button based on selection state."""
        selected_items, selection_mode = self.get_selection_state()

        if selection_mode == "empty":
            logger.warning("No plates available for execution")
            return

        # Validate all selected plates are compiled
        uncompiled = [item for item in selected_items if item.get('status') != 'o']
        if uncompiled:
            names = [item['name'] for item in uncompiled]
            raise ValueError(f"Cannot run uncompiled plates: {', '.join(names)}")

        # Generate description and log operation
        description = self.get_operation_description(selected_items, selection_mode, "run")
        logger.info(f"Running: {description}")

        await self._run_selected_plates(selected_items)

    async def _initialize_selected_plates(self, selected_items: List[Dict[str, Any]]):
        """Initialize existing orchestrators for selected plates."""
        setup_global_gpu_registry()

        for plate_data in selected_items:
            plate_path = plate_data['path']

            # Get existing orchestrator (must have been created by Add button)
            orchestrator = self.orchestrators.get(plate_path)
            if not orchestrator:
                raise ValueError(f"No orchestrator found for {plate_path}. Use Add button first.")

            # Check if already initialized
            if orchestrator.is_initialized():
                logger.info(f"Orchestrator for {plate_data['name']} already initialized")
                plate_data['status'] = '-'  # Ensure status is correct
                continue

            # Run heavy orchestrator initialization in executor to avoid blocking UI
            # Use safe wrapper that ensures exceptions are caught by global handler
            await safe_run_in_executor(None, orchestrator.initialize)

            # Update status to initialized
            plate_data['status'] = '-'  # Yellow: initialized but not compiled
            logger.info(f"Initialized orchestrator for plate: {plate_data['name']}")

        # Refresh UI
        self.list_manager.model.notify_observers()
        logger.info(f"Initialized {len(selected_items)} plates")

    async def _compile_selected_plates(self, selected_items: List[Dict[str, Any]]):
        """Compile pipelines for selected plates."""
        for plate_data in selected_items:
            plate_path = plate_data['path']

            # Get orchestrator (must be initialized)
            orchestrator = self.orchestrators.get(plate_path)
            if not orchestrator:
                raise ValueError(f"No orchestrator found for {plate_path}")

            # Get current pipeline definition from state
            pipeline_definition = self._get_current_pipeline_definition()
            if not pipeline_definition:
                raise ValueError("No pipeline defined")

            # Run heavy compilation operations in executor to avoid blocking UI
            # Use safe wrapper that ensures exceptions are caught by global handler
            wells = await safe_run_in_executor(None, orchestrator.get_wells)
            compiled_contexts = await safe_run_in_executor(
                None,
                lambda: orchestrator.compile_pipelines(
                    pipeline_definition=pipeline_definition,
                    well_filter=wells
                )
            )

            # Store compiled contexts in item for execution
            plate_data['compiled_contexts'] = compiled_contexts
            plate_data['pipeline_definition'] = pipeline_definition
            plate_data['status'] = 'o'  # Green: compiled and ready

        # Refresh UI
        self.list_manager.model.notify_observers()
        logger.info(f"Compiled {len(selected_items)} plates")

    async def _run_selected_plates(self, selected_items: List[Dict[str, Any]]):
        """Execute compiled plates."""
        for plate_data in selected_items:
            plate_path = plate_data['path']

            # Get orchestrator and compiled contexts
            orchestrator = self.orchestrators.get(plate_path)
            if not orchestrator:
                raise ValueError(f"No orchestrator found for {plate_path}")

            compiled_contexts = plate_data.get('compiled_contexts')
            if not compiled_contexts:
                raise ValueError("No compiled contexts found")

            # Get pipeline definition
            pipeline_definition = plate_data.get('pipeline_definition')
            if not pipeline_definition:
                raise ValueError("No pipeline definition found")

            # Set status to running
            plate_data['status'] = '!'  # Red: running
            self.list_manager.model.notify_observers()

            # Run heavy execution operations in executor to avoid blocking UI
            # Use safe wrapper that ensures exceptions are caught by global handler
            results = await safe_run_in_executor(
                None,
                lambda: orchestrator.execute_compiled_plate(
                    pipeline_definition=pipeline_definition,
                    compiled_contexts=compiled_contexts
                )
            )

            # Store results and update status
            plate_data['execution_results'] = results
            if results and all(r.get('status') != 'error' for r in results.values()):
                plate_data['status'] = 'o'  # Green: completed successfully
            else:
                plate_data['status'] = 'X'  # X: execution error
                plate_data['error'] = "Execution failed - check logs"

        # Refresh UI
        self.list_manager.model.notify_observers()
        logger.info(f"Executed {len(selected_items)} plates")

    # Helper methods
    def _get_selected_or_all_plates(self) -> List[Dict[str, Any]]:
        """Get selected plate or all plates."""
        selected = self.list_manager.get_selected_item()
        return [selected] if selected else self.list_manager.model.items

    def _get_current_pipeline_definition(self) -> List:
        """Get current pipeline definition from active orchestrator."""
        return self.state.active_orchestrator.pipeline_definition

    # Compatibility methods
    async def shutdown(self):
        """Clean up resources."""
        logger.info("PlateManagerPane: Shutdown complete")
    
    def handle_key(self, key_event):
        """Handle keyboard input."""
        pass
