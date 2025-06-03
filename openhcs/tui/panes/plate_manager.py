"""
Clean Plate Manager Pane using unified list management architecture.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

from prompt_toolkit.layout import Container
from prompt_toolkit.application import get_app

from openhcs.io.filemanager import FileManager
from openhcs.tui.components import ListManagerPane, ListConfig, ButtonConfig
from openhcs.tui.utils.dialog_helpers import prompt_for_multi_folder_dialog
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.orchestrator.gpu_scheduler import setup_global_gpu_registry
from openhcs.core.config import GlobalPipelineConfig
from openhcs.tui.components.config_editor import ConfigEditor

logger = logging.getLogger(__name__)


class PlateManagerPane:
    """Clean plate manager using unified list management architecture."""

    def __init__(self, state, filemanager: FileManager):
        """Initialize with clean architecture."""
        self.state = state
        self.filemanager = filemanager

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
            empty_message="Click 'Add' to add plates. Status: ? = added, - = initialized, o = compiled, ! = running"
        )

        self.list_manager = ListManagerPane(config, self.filemanager.registry)

        # Now set enabled functions after list_manager exists
        config.button_configs[1].enabled_func = lambda: len(self.list_manager.model.items) > 0  # Del
        config.button_configs[2].enabled_func = lambda: self.list_manager.get_selected_item() is not None  # Edit
        self.list_manager._on_model_changed = self._on_selection_changed

        logger.info("PlateManagerPane: Initialized")
    
    @property
    def container(self) -> Container:
        """Get the UI container."""
        return self.list_manager.container

    def _get_display_text(self, plate_data: Dict[str, Any], is_selected: bool) -> str:
        """Generate display text for a plate."""
        status = self._get_status_symbol(plate_data.get('status', '?'))
        name = plate_data.get('name', 'Unknown Plate')
        path = plate_data.get('path', 'Unknown Path')
        return f"{status} {name} | {path}"

    def _get_status_symbol(self, status: str) -> str:
        """Get status symbol for plate."""
        symbols = {'?': '?', '-': '-', 'o': 'o', '!': '!'}
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

    # Action handlers
    async def _handle_add_plates(self):
        """Handle Add Plates button."""
        logger.info("PlateManager: _handle_add_plates called!")
        await self._add_plates()

    async def _add_plates(self):
        """Add plates via multi-folder selection."""
        try:
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
                new_plates.append({
                    'name': path_obj.name,
                    'path': str(path_obj),
                    'status': '?'
                })

            # Add to existing plates
            current_plates = self.list_manager.model.items
            self.list_manager.load_items(current_plates + new_plates)
            logger.info(f"Added {len(new_plates)} plates")

        except Exception as e:
            logger.error(f"Error adding plates: {e}", exc_info=True)

    async def _edit_plate_config(self):
        """Edit configuration for selected plate."""
        try:
            selected_plate = self.list_manager.get_selected_item()
            if not selected_plate:
                logger.warning("No plate selected for editing")
                return

            plate_path = selected_plate['path']

            # Get current plate-specific config or use global config as base
            current_plate_config = self.plate_configs.get(plate_path)
            if current_plate_config is None:
                # Use global config as starting point
                current_plate_config = self.state.global_config

            # Create config editor for plate-specific overrides
            config_editor = ConfigEditor(
                config_class=GlobalPipelineConfig,
                current_config=current_plate_config,
                backend=getattr(self.state, 'backend', 'disk'),
                scope="plate",
                base_config=self.state.global_config,  # Show what's being overridden
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

        except Exception as e:
            logger.error(f"Error editing plate config: {e}", exc_info=True)

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
        """Handle Delete Plates button."""
        selected_index = self.list_manager.model.selected_index
        if self.list_manager.model.remove_item(selected_index):
            logger.info("Removed selected plate")

    async def _handle_edit_plate(self):
        """Handle Edit Plate button."""
        await self._edit_plate_config()

    async def _handle_initialize_plates(self):
        """Handle Initialize Plates button."""
        await self._initialize_plates()

    async def _handle_compile_plates(self):
        """Handle Compile Plates button."""
        await self._compile_plates()

    async def _handle_run_plates(self):
        """Handle Run Plates button."""
        await self._run_plates()

    async def _initialize_plates(self):
        """Initialize selected plates."""
        try:
            plates = self._get_selected_or_all_plates()
            setup_global_gpu_registry()

            for plate_data in plates:
                try:
                    plate_path = plate_data['path']

                    # Use plate-specific config if available, otherwise global config
                    plate_config = self.plate_configs.get(plate_path, self.state.global_config)

                    # Run heavy orchestrator operations in executor to avoid blocking UI
                    loop = asyncio.get_event_loop()
                    orchestrator = await loop.run_in_executor(
                        None,
                        lambda: PipelineOrchestrator(plate_path, global_config=plate_config).initialize()
                    )

                    self.orchestrators[plate_path] = orchestrator
                    plate_data['status'] = '-'

                except Exception as e:
                    logger.error(f"Error initializing {plate_data['name']}: {e}")
                    plate_data['status'] = '!'

            self.list_manager.load_items(self.list_manager.model.items)
            logger.info(f"Initialized {len(plates)} plates")

        except Exception as e:
            logger.error(f"Error in initialize: {e}", exc_info=True)

    async def _compile_plates(self):
        """Compile initialized plates."""
        try:
            plates = [p for p in self.list_manager.model.items if p.get('status') in ['-', 'o']]

            for plate_data in plates:
                try:
                    plate_path = plate_data['path']
                    orchestrator = self.orchestrators[plate_path]
                    pipeline_definition = self._get_current_pipeline_definition()

                    # Run heavy compilation operations in executor to avoid blocking UI
                    loop = asyncio.get_event_loop()

                    wells = await loop.run_in_executor(None, orchestrator.get_wells)
                    compiled_contexts = await loop.run_in_executor(
                        None,
                        lambda: orchestrator.compile_pipelines(
                            pipeline_definition=pipeline_definition,
                            well_filter=wells
                        )
                    )

                    self.pipelines[plate_path] = {
                        'pipeline_definition': pipeline_definition,
                        'compiled_contexts': compiled_contexts
                    }
                    plate_data['status'] = 'o'

                except Exception as e:
                    logger.error(f"Error compiling {plate_data['name']}: {e}")
                    plate_data['status'] = '!'

            self.list_manager.load_items(self.list_manager.model.items)
            logger.info(f"Compiled {len(plates)} plates")

        except Exception as e:
            logger.error(f"Error in compile: {e}", exc_info=True)

    async def _run_plates(self):
        """Execute compiled plates."""
        try:
            plates = [p for p in self.list_manager.model.items if p.get('status') == 'o']

            for plate_data in plates:
                try:
                    plate_path = plate_data['path']
                    orchestrator = self.orchestrators[plate_path]
                    pipeline_data = self.pipelines[plate_path]

                    plate_data['status'] = '!'  # Running
                    self.list_manager.load_items(self.list_manager.model.items)

                    # Run heavy execution operations in executor to avoid blocking UI
                    loop = asyncio.get_event_loop()
                    results = await loop.run_in_executor(
                        None,
                        lambda: orchestrator.execute_compiled_plate(
                            pipeline_definition=pipeline_data['pipeline_definition'],
                            compiled_contexts=pipeline_data['compiled_contexts']
                        )
                    )

                    if results and all(r.get('status') != 'error' for r in results.values()):
                        plate_data['status'] = 'o'
                    else:
                        plate_data['status'] = '!'

                except Exception as e:
                    logger.error(f"Error executing {plate_data['name']}: {e}")
                    plate_data['status'] = '!'

            self.list_manager.load_items(self.list_manager.model.items)
            logger.info(f"Executed {len(plates)} plates")

        except Exception as e:
            logger.error(f"Error in run: {e}", exc_info=True)

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
