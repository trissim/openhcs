"""
PlateManagerWidget for OpenHCS Textual TUI

Plate management widget with complete button set and reactive state management.
Matches the functionality from the current prompt-toolkit TUI.
"""

import asyncio
import copy
import dataclasses
import inspect
import json
import logging
import numpy as np
import os
import pickle
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple

from openhcs.core.config import PipelineConfig

from PIL import Image
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer
from textual.reactive import reactive
from textual.widgets import Button, Static, SelectionList
from textual.widget import Widget
from textual.css.query import NoMatches
from .button_list_widget import ButtonListWidget, ButtonConfig
from textual import work, on

from openhcs.core.config import GlobalPipelineConfig, VFSConfig, MaterializationBackend
from openhcs.core.pipeline import Pipeline
from openhcs.io.filemanager import FileManager
from openhcs.core.memory.gpu_cleanup import cleanup_all_gpu_frameworks
from openhcs.io.base import storage_registry
from openhcs.io.memory import MemoryStorageBackend
from openhcs.io.zarr import ZarrStorageBackend
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.constants.constants import GroupBy, Backend, VariableComponents, OrchestratorState
from openhcs.textual_tui.services.file_browser_service import SelectionMode
from openhcs.textual_tui.services.window_service import WindowService
from openhcs.core.path_cache import get_cached_browser_path, PathCacheKey, get_path_cache
from openhcs.textual_tui.widgets.shared.signature_analyzer import SignatureAnalyzer

logger = logging.getLogger(__name__)

# Note: Using subprocess approach instead of multiprocessing to avoid TUI FD conflicts

def get_orchestrator_status_symbol(orchestrator: PipelineOrchestrator) -> str:
    """Get UI symbol for orchestrator state - simple mapping without over-engineering."""
    if orchestrator is None:
        return "?"  # No orchestrator (newly added plate)

    state_to_symbol = {
        OrchestratorState.CREATED: "?",         # Created but not initialized
        OrchestratorState.READY: "-",           # Initialized, ready for compilation
        OrchestratorState.COMPILED: "o",        # Compiled, ready for execution
        OrchestratorState.EXECUTING: "!",       # Execution in progress
        OrchestratorState.COMPLETED: "C",       # Execution completed successfully
        OrchestratorState.INIT_FAILED: "I",     # Initialization failed
        OrchestratorState.COMPILE_FAILED: "P",  # Compilation failed (P for Pipeline)
        OrchestratorState.EXEC_FAILED: "X",     # Execution failed
    }

    return state_to_symbol.get(orchestrator.state, "?")

def get_current_log_file_path() -> str:
    """Get the current log file path from the logging system."""
    try:
        # Get the root logger and find the FileHandler
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                return handler.baseFilename

        # Fallback: try to get from openhcs logger
        openhcs_logger = logging.getLogger("openhcs")
        for handler in openhcs_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                return handler.baseFilename

        # Last resort: create a default path
        log_dir = Path.home() / ".local" / "share" / "openhcs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return str(log_dir / f"openhcs_subprocess_{int(time.time())}.log")

    except Exception as e:
        logger.warning(f"Could not determine log file path: {e}")
        return "/tmp/openhcs_subprocess.log"







class PlateManagerWidget(ButtonListWidget):
    """
    Plate management widget using Textual reactive state.
    """

    # Semantic reactive property (like PipelineEditor's pipeline_steps)
    selected_plate = reactive("")
    orchestrators = reactive({})
    plate_configs = reactive({})
    orchestrator_state_version = reactive(0)  # Increment to trigger UI refresh
    
    def __init__(self, filemanager: FileManager, global_config: GlobalPipelineConfig):
        button_configs = [
            ButtonConfig("Add", "add_plate"),
            ButtonConfig("Del", "del_plate", disabled=True),
            ButtonConfig("Edit", "edit_config", disabled=True),  # Unified edit button for config editing
            ButtonConfig("Init", "init_plate", disabled=True),
            ButtonConfig("Compile", "compile_plate", disabled=True),
            ButtonConfig("Run", "run_plate", disabled=True),
            ButtonConfig("Code", "code_plate", disabled=True),  # Generate Python code
            ButtonConfig("Save", "save_python_script", disabled=True),  # Save Python script
            # ButtonConfig("Export", "export_ome_zarr", disabled=True),  # Export to OME-ZARR - HIDDEN FROM UI
        ]
        super().__init__(
            button_configs=button_configs,
            list_id="plate_content",
            container_id="plate_list",
            on_button_pressed=self._handle_button_press,
            on_selection_changed=self._handle_selection_change,
            on_item_moved=self._handle_item_moved
        )
        self.filemanager = filemanager
        self.global_config = global_config
        self.plate_compiled_data = {}
        self.on_plate_selected: Optional[Callable[[str], None]] = None
        self.pipeline_editor: Optional['PipelineEditorWidget'] = None

        # Initialize window service to avoid circular imports
        self.window_service = None  # Will be set in on_mount

        # --- Subprocess Architecture ---
        self.current_process: Optional[subprocess.Popen] = None
        self.log_file_path: Optional[str] = None  # Single source of truth
        self.log_file_position: int = 0  # Track position in log file for incremental reading
        # Async monitoring using Textual's interval system
        self.monitoring_interval = None
        self.monitoring_active = False
        # ---
        
        logger.debug("PlateManagerWidget initialized")





    def on_unmount(self) -> None:
        logger.debug("Unmounting PlateManagerWidget, ensuring worker process is terminated.")
        # Schedule async stop execution since on_unmount is sync
        import asyncio
        if self.current_process and self.current_process.poll() is None:
            # Create a task to stop execution asynchronously
            asyncio.create_task(self.action_stop_execution())
        self._stop_monitoring()

    def format_item_for_display(self, plate: Dict) -> Tuple[str, str]:
        # Get status from orchestrator instead of magic string
        plate_path = plate.get('path', '')
        orchestrator = self.orchestrators.get(plate_path)
        status_symbol = get_orchestrator_status_symbol(orchestrator)

        status_symbols = {
            "?": "➕",   # Created (not initialized)
            "-": "✅",   # Ready (initialized)
            "o": "⚡",   # Compiled
            "!": "🔄",   # Executing
            "C": "🏁",   # Completed
            "I": "🚫",   # Init failed
            "P": "💥",   # Compile failed (Pipeline)
            "X": "❌"    # Execution failed
        }
        status_icon = status_symbols.get(status_symbol, "❓")
        plate_name = plate.get('name', 'Unknown')
        display_text = f"{status_icon} {plate_name} - {plate_path}"
        return display_text, plate_path

    async def _handle_button_press(self, button_id: str) -> None:
        action_map = {
            "add_plate": self.action_add_plate,
            "del_plate": self.action_delete_plate,
            "edit_config": self.action_edit_config,  # Unified edit button
            "init_plate": self.action_init_plate,
            "compile_plate": self.action_compile_plate,
            "code_plate": self.action_code_plate,  # Generate Python code
            "save_python_script": self.action_save_python_script,  # Save Python script
            # "export_ome_zarr": self.action_export_ome_zarr,  # HIDDEN
        }
        if button_id in action_map:
            action = action_map[button_id]
            if inspect.iscoroutinefunction(action):
                await action()
            else:
                action()
        elif button_id == "run_plate":
            if self._is_any_plate_running():
                await self.action_stop_execution()
            else:
                await self.action_run_plate()

    def _handle_selection_change(self, selected_values: List[str]) -> None:
        logger.debug(f"Checkmarks changed: {len(selected_values)} items selected")

    def _handle_item_moved(self, from_index: int, to_index: int) -> None:
        current_plates = list(self.items)
        plate = current_plates.pop(from_index)
        current_plates.insert(to_index, plate)
        self.items = current_plates
        plate_name = plate['name']
        direction = "up" if to_index < from_index else "down"
        self.app.current_status = f"Moved plate '{plate_name}' {direction}"

    def on_mount(self) -> None:
        # Initialize window service
        self.window_service = WindowService(self.app)

        self.call_later(self._delayed_update_display)
        self.call_later(self._update_button_states)
    
    def watch_items(self, items: List[Dict]) -> None:
        """Automatically update UI when items changes (follows ButtonListWidget pattern)."""
        # DEBUG: Log when items list changes to track the source of the reset
        stack_trace = ''.join(traceback.format_stack()[-3:-1])  # Get last 2 stack frames
        logger.debug(f"🔍 ITEMS CHANGED: {len(items)} plates. Call stack:\n{stack_trace}")

        # CRITICAL: Call parent's watch_items to update the SelectionList
        super().watch_items(items)

        logger.debug(f"Plates updated: {len(items)} plates")
        self._update_button_states()
    
    def watch_highlighted_item(self, plate_path: str) -> None:
        self.selected_plate = plate_path
        logger.debug(f"Highlighted plate: {plate_path}")

    def watch_selected_plate(self, plate_path: str) -> None:
        self._update_button_states()
        if self.on_plate_selected and plate_path:
            self.on_plate_selected(plate_path)
        logger.debug(f"Selected plate: {plate_path}")

    def watch_orchestrator_state_version(self, version: int) -> None:
        """Automatically refresh UI when orchestrator states change."""
        # Only update UI if widget is properly mounted
        if not self.is_mounted:
            return

        # Force SelectionList to update by calling _update_selection_list
        # This re-calls format_item_for_display() for all items
        self._update_selection_list()

        # CRITICAL: Update main button states when orchestrator states change
        self._update_button_states()

        # Also notify PipelineEditor if connected
        if self.pipeline_editor:
            logger.debug(f"PlateManager: Notifying PipelineEditor of orchestrator state change (version {version})")
            self.pipeline_editor._update_button_states()

    def get_selection_state(self) -> tuple[List[Dict], str]:
        # Check if widget is properly mounted first
        if not self.is_mounted:
            logger.debug("get_selection_state called on unmounted widget")
            return [], "empty"

        try:
            selection_list = self.query_one(f"#{self.list_id}")
            multi_selected_values = selection_list.selected
            if multi_selected_values:
                selected_items = [p for p in self.items if p.get('path') in multi_selected_values]
                return selected_items, "checkbox"
            elif self.selected_plate:
                selected_items = [p for p in self.items if p.get('path') == self.selected_plate]
                return selected_items, "cursor"
            else:
                return [], "empty"
        except Exception as e:
            # DOM CORRUPTION DETECTED - This is a critical error
            stack_trace = ''.join(traceback.format_stack()[-3:-1])
            logger.error(f"🚨 DOM CORRUPTION: Failed to get selection state: {e}")
            logger.error(f"🚨 DOM CORRUPTION: Call stack:\n{stack_trace}")
            logger.error(f"🚨 DOM CORRUPTION: Widget mounted: {self.is_mounted}")
            logger.error(f"🚨 DOM CORRUPTION: Looking for: #{self.list_id}")
            logger.error(f"🚨 DOM CORRUPTION: Plates count: {len(self.items)}")

            # Try to diagnose what widgets actually exist
            try:
                all_widgets = list(self.query("*"))
                widget_ids = [w.id for w in all_widgets if w.id]
                logger.error(f"🚨 DOM CORRUPTION: Available widget IDs: {widget_ids}")
            except Exception as diag_e:
                logger.error(f"🚨 DOM CORRUPTION: Could not diagnose widgets: {diag_e}")

            if self.selected_plate:
                selected_items = [p for p in self.items if p.get('path') == self.selected_plate]
                return selected_items, "cursor"
            return [], "empty"

    def get_operation_description(self, selected_items: List[Dict], selection_mode: str, operation: str) -> str:
        count = len(selected_items)
        if count == 0: return f"No items for {operation}"
        if count == 1: return f"{operation.title()} item: {selected_items[0].get('name', 'Unknown')}"
        return f"{operation.title()} {count} items"

    def _delayed_update_display(self) -> None:
        """Trigger UI update - no longer needed since reactive system handles this automatically."""
        # The reactive system now handles updates automatically via watch_plates()
        # This method is kept for compatibility but does nothing
        pass

    def _trigger_ui_refresh(self) -> None:
        """Force UI refresh when orchestrator state changes without items list changing."""
        # Increment reactive counter to trigger automatic UI refresh
        self.orchestrator_state_version += 1

    def _update_button_states(self) -> None:
        try:
            # Check if widget is mounted and buttons exist
            if not self.is_mounted:
                return

            has_selection = bool(self.selected_plate)
            is_running = self._is_any_plate_running()

            # Check if there are any selected items (for delete button)
            selected_items, _ = self.get_selection_state()
            has_selected_items = bool(selected_items)

            can_run = has_selection and any(p['path'] in self.plate_compiled_data for p in self.items if p.get('path') == self.selected_plate)

            # Try to get run button - if it doesn't exist, widget is not fully mounted
            try:
                run_button = self.query_one("#run_plate")
                if is_running:
                    run_button.label = "Stop"
                    run_button.disabled = False
                else:
                    run_button.label = "Run"
                    run_button.disabled = not can_run
            except:
                # Buttons not mounted yet, skip update
                return

            self.query_one("#add_plate").disabled = is_running
            self.query_one("#del_plate").disabled = not self.items or not has_selected_items or is_running

            # Edit button (config editing) enabled when 1+ orchestrators selected and initialized
            selected_items, _ = self.get_selection_state()
            edit_enabled = (
                len(selected_items) > 0 and
                all(self._is_orchestrator_initialized(item['path']) for item in selected_items) and
                not is_running
            )
            self.query_one("#edit_config").disabled = not edit_enabled

            # Init button - enabled when plates are selected, can be initialized, and not running
            init_enabled = (
                len(selected_items) > 0 and
                any(self._can_orchestrator_be_initialized(item['path']) for item in selected_items) and
                not is_running
            )
            self.query_one("#init_plate").disabled = not init_enabled

            # Compile button - enabled when plates are selected, initialized, and not running
            selected_items, _ = self.get_selection_state()
            compile_enabled = (
                len(selected_items) > 0 and
                all(self._is_orchestrator_initialized(item['path']) for item in selected_items) and
                not is_running
            )
            self.query_one("#compile_plate").disabled = not compile_enabled

            # Code button - enabled when plates are selected, initialized, and not running
            code_enabled = (
                len(selected_items) > 0 and
                all(self._is_orchestrator_initialized(item['path']) for item in selected_items) and
                not is_running
            )
            self.query_one("#code_plate").disabled = not code_enabled

            # Save Python script button - enabled when plates are selected, initialized, and not running
            save_enabled = (
                len(selected_items) > 0 and
                all(self._is_orchestrator_initialized(item['path']) for item in selected_items) and
                not is_running
            )
            self.query_one("#save_python_script").disabled = not save_enabled

            # Export button - enabled when plate is initialized and has workspace (HIDDEN FROM UI)
            # export_enabled = (
            #     has_selection and
            #     self.selected_plate in self.orchestrators and
            #     not is_running
            # )
            # try:
            #     self.query_one("#export_ome_zarr").disabled = not export_enabled
            # except:
            #     pass  # Button is hidden from UI

            # Debug button removed - no longer needed

        except Exception as e:
            # Only log if it's not a mounting/unmounting issue
            if self.is_mounted:
                logger.debug(f"Button state update skipped (widget not ready): {e}")
            # Don't log errors during unmounting

    def _is_any_plate_running(self) -> bool:
        return self.current_process is not None and self.current_process.poll() is None

    def _has_pipelines(self, plates: List[Dict]) -> bool:
        """Check if all plates have pipeline definitions."""
        if not self.pipeline_editor:
            return False

        for plate in plates:
            pipeline = self.pipeline_editor.get_pipeline_for_plate(plate['path'])
            if not pipeline:
                return False
        return True

    def get_plate_status(self, plate_path: str) -> str:
        """Get status for specific plate - now uses orchestrator state."""
        orchestrator = self.orchestrators.get(plate_path)
        return get_orchestrator_status_symbol(orchestrator)

    def _is_orchestrator_initialized(self, plate_path: str) -> bool:
        """Check if orchestrator exists and is in an initialized state."""
        orchestrator = self.orchestrators.get(plate_path)
        if orchestrator is None:
            return False
        return orchestrator.state in [OrchestratorState.READY, OrchestratorState.COMPILED,
                                     OrchestratorState.COMPLETED, OrchestratorState.COMPILE_FAILED,
                                     OrchestratorState.EXEC_FAILED]

    def _can_orchestrator_be_initialized(self, plate_path: str) -> bool:
        """Check if orchestrator can be initialized (doesn't exist or is in a re-initializable state)."""
        orchestrator = self.orchestrators.get(plate_path)
        if orchestrator is None:
            return True  # No orchestrator exists, can be initialized
        return orchestrator.state in [OrchestratorState.CREATED, OrchestratorState.INIT_FAILED]

    def _notify_pipeline_editor_status_change(self, plate_path: str, new_status: str) -> None:
        """Notify pipeline editor when plate status changes (enables Add button immediately)."""
        if self.pipeline_editor and self.pipeline_editor.current_plate == plate_path:
            # Update pipeline editor's status and trigger button state update
            self.pipeline_editor.current_plate_status = new_status

    def _get_current_pipeline_definition(self, plate_path: str = None) -> List:
        """Get current pipeline definition from PipelineEditor (now returns FunctionStep objects directly)."""
        if not self.pipeline_editor:
            logger.warning("No pipeline editor reference - using empty pipeline")
            return []

        # Get pipeline for specific plate or current plate
        target_plate = plate_path or self.pipeline_editor.current_plate
        if not target_plate:
            logger.warning("No plate specified - using empty pipeline")
            return []

        # Get pipeline from editor (now returns List[FunctionStep] directly)
        pipeline_steps = self.pipeline_editor.get_pipeline_for_plate(target_plate)

        # No conversion needed - pipeline_steps are already FunctionStep objects with memory type decorators
        return pipeline_steps

    def get_operation_description(self, selected_items: List[Dict], selection_mode: str, operation: str) -> str:
        """Generate human-readable description of what will be operated on."""
        count = len(selected_items)
        if selection_mode == "empty":
            return f"No items available for {operation}"
        elif selection_mode == "all":
            return f"{operation.title()} ALL {count} items"
        elif selection_mode == "checkbox":
            if count == 1:
                item_name = selected_items[0].get('name', 'Unknown')
                return f"{operation.title()} selected item: {item_name}"
            else:
                return f"{operation.title()} {count} selected items"
        else:
            return f"{operation.title()} {count} items"

    def _reset_execution_state(self, status_message: str, force_fail_executing: bool = True):
        if self.current_process:
            if self.current_process.poll() is None:  # Still running
                logger.warning("Forcefully terminating subprocess during reset.")
                self.current_process.terminate()
                try:
                    self.current_process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.current_process.kill()  # Force kill if terminate fails
            self.current_process = None

        # Clear log file reference (no temp files - log file is single source of truth)
        self.log_file_path = None
        self.log_file_position = 0

        # Stop async monitoring
        self._stop_monitoring()

        # Only reset executing orchestrators to failed if this is a forced termination
        # Natural completion should preserve the states set by the completion handler
        if force_fail_executing:
            for plate_path, orchestrator in self.orchestrators.items():
                if orchestrator.state == OrchestratorState.EXECUTING:
                    orchestrator._state = OrchestratorState.EXEC_FAILED

        # Trigger UI refresh after state changes - this is essential for button states
        self._trigger_ui_refresh()

        # Update button states - but only if widget is properly mounted
        try:
            if self.is_mounted and hasattr(self, 'query_one'):
                self._update_button_states()
        except Exception as e:
            logger.error(f"Failed to update button states during reset: {e}")

        self.app.current_status = status_message

    async def action_run_plate(self) -> None:
        # Clear logs from singleton toolong window before starting new run
        try:
            from openhcs.textual_tui.windows.toolong_window import clear_toolong_logs
            logger.info("Clearing logs from singleton toolong window before new run")
            clear_toolong_logs(self.app)
            logger.info("Toolong logs cleared")
        except Exception as e:
            logger.error(f"Failed to clear toolong logs: {e}")
            import traceback
            logger.error(traceback.format_exc())

        selected_items, _ = self.get_selection_state()
        if not selected_items:
            self.app.show_error("No plates selected to run.")
            return

        ready_items = [item for item in selected_items if item.get('path') in self.plate_compiled_data]
        if not ready_items:
            self.app.show_error("Selected plates are not compiled. Please compile first.")
            return

        try:
            # Use subprocess approach like integration tests
            logger.debug("🔥 Using subprocess approach for clean isolation")

            plate_paths_to_run = [item['path'] for item in ready_items]

            # Pass definition pipeline steps - subprocess will make fresh copy and compile
            pipeline_data = {}
            for plate_path in plate_paths_to_run:
                definition_pipeline = self._get_current_pipeline_definition(plate_path)
                pipeline_data[plate_path] = definition_pipeline

            logger.info(f"🔥 Starting subprocess for {len(plate_paths_to_run)} plates")

            # Create data file for subprocess (only file needed besides log)
            data_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl')

            # Generate unique ID for this subprocess
            import time
            subprocess_timestamp = int(time.time())
            plate_names = [Path(path).name for path in plate_paths_to_run]
            unique_id = f"plates_{'_'.join(plate_names[:2])}_{subprocess_timestamp}"  # Limit to first 2 plates for filename length

            # Build subprocess log name from TUI log base using log utilities
            from openhcs.core.log_utils import get_current_log_file_path
            try:
                tui_log_path = get_current_log_file_path()
                if tui_log_path.endswith('.log'):
                    tui_base = tui_log_path[:-4]  # Remove .log extension
                else:
                    tui_base = tui_log_path
                log_file_base = f"{tui_base}_subprocess_{subprocess_timestamp}"
            except RuntimeError:
                # Fallback if no main log found
                log_dir = Path.home() / ".local" / "share" / "openhcs" / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file_base = str(log_dir / f"tui_subprocess_{subprocess_timestamp}")

            # Pickle data for subprocess
            subprocess_data = {
                'plate_paths': plate_paths_to_run,
                'pipeline_data': pipeline_data,
                'global_config': self.app.global_config  # Pickle config object directly
            }

            # Resolve all lazy configurations to concrete values before pickling
            from openhcs.core.lazy_config import resolve_lazy_configurations_for_serialization
            resolved_subprocess_data = resolve_lazy_configurations_for_serialization(subprocess_data)

            # Wrap pickle operation in executor to avoid blocking UI
            def _write_pickle_data():
                import dill as pickle
                with open(data_file.name, 'wb') as f:
                    pickle.dump(resolved_subprocess_data, f)
                data_file.close()

            await asyncio.get_event_loop().run_in_executor(None, _write_pickle_data)

            logger.debug(f"🔥 Created data file: {data_file.name}")

            # Create subprocess (like integration tests)
            subprocess_script = Path(__file__).parent.parent / "subprocess_runner.py"

            # Generate actual log file path that subprocess will create
            actual_log_file_path = f"{log_file_base}_{unique_id}.log"
            logger.debug(f"🔥 Log file base: {log_file_base}")
            logger.debug(f"🔥 Unique ID: {unique_id}")
            logger.debug(f"🔥 Actual log file: {actual_log_file_path}")

            # Store log file path for monitoring (subprocess logger writes to this)
            self.log_file_path = actual_log_file_path
            self.log_file_position = self._get_current_log_position()  # Start from current end

            logger.debug(f"🔥 Subprocess command: {sys.executable} {subprocess_script} {data_file.name} {log_file_base} {unique_id}")
            logger.debug(f"🔥 Subprocess logger will write to: {self.log_file_path}")
            logger.debug(f"🔥 Subprocess stdout will be silenced (logger handles output)")

            # SIMPLE SUBPROCESS: Let subprocess log to its own file (single source of truth)
            # Wrap subprocess creation in executor to avoid blocking UI
            def _create_subprocess():
                return subprocess.Popen([
                    sys.executable, str(subprocess_script),
                    data_file.name, log_file_base, unique_id  # Only data file and log - no temp files
                ],
                stdout=subprocess.DEVNULL,  # Subprocess logs to its own file
                stderr=subprocess.DEVNULL,  # Subprocess logs to its own file
                text=True,  # Text mode for easier handling
                )

            self.current_process = await asyncio.get_event_loop().run_in_executor(None, _create_subprocess)

            logger.info(f"🔥 Subprocess started with PID: {self.current_process.pid}")

            # Subprocess logs to its own dedicated file - no output monitoring needed

            # Update orchestrator states to show running state
            for plate in ready_items:
                plate_path = plate['path']
                if plate_path in self.orchestrators:
                    self.orchestrators[plate_path]._state = OrchestratorState.EXECUTING

            # Trigger UI refresh after state changes
            self._trigger_ui_refresh()

            self.app.current_status = f"Running {len(ready_items)} plate(s) in subprocess..."
            self._update_button_states()
            
            # Start reactive log monitoring
            self._start_log_monitoring()

            # Start async monitoring
            await self._start_monitoring()

        except Exception as e:
            logger.critical(f"Failed to start subprocess: {e}", exc_info=True)
            self.app.show_error("Failed to start subprocess", str(e))
            self._reset_execution_state("Subprocess failed to start")

    def _start_log_monitoring(self) -> None:
        """Start reactive log monitoring for subprocess logs."""
        if not self.log_file_path:
            logger.warning("Cannot start log monitoring: no log file path")
            return

        try:
            # Extract base path from log file path (remove .log extension)
            log_path = Path(self.log_file_path)
            base_log_path = str(log_path.with_suffix(''))

            # Notify status bar to start log monitoring
            if hasattr(self.app, 'status_bar') and self.app.status_bar:
                self.app.status_bar.start_log_monitoring(base_log_path)
                logger.debug(f"Started reactive log monitoring for: {base_log_path}")
            else:
                logger.warning("Status bar not available for log monitoring")

        except Exception as e:
            logger.error(f"Failed to start log monitoring: {e}")

    def _stop_log_monitoring(self) -> None:
        """Stop reactive log monitoring."""
        try:
            # Notify status bar to stop log monitoring
            if hasattr(self.app, 'status_bar') and self.app.status_bar:
                self.app.status_bar.stop_log_monitoring()
                logger.debug("Stopped reactive log monitoring")
        except Exception as e:
            logger.error(f"Failed to stop log monitoring: {e}")

    def _get_current_log_position(self) -> int:
        """Get current position in log file."""
        if not self.log_file_path or not Path(self.log_file_path).exists():
            return 0
        try:
            return Path(self.log_file_path).stat().st_size
        except Exception:
            return 0


            
    def _stop_file_watcher(self) -> None:
        """Stop file system watcher without blocking."""
        if not self.file_observer:
            return
            
        try:
            # Just stop and abandon - don't wait for anything
            self.file_observer.stop()
        except Exception:
            pass  # Ignore errors
        finally:
            # Always clear references immediately
            self.file_observer = None
            self.file_watcher = None



    async def _start_monitoring(self) -> None:
        """Start async monitoring using Textual's interval system."""
        # Stop any existing monitoring
        self._stop_monitoring()

        if self.monitoring_active:
            return

        self.monitoring_active = True
        # Use Textual's set_interval for periodic async monitoring
        self.monitoring_interval = self.set_interval(
            10.0,  # Check every 10 seconds
            self._check_process_status_async,
            pause=False
        )
        logger.debug("Started async process monitoring")

    def _stop_monitoring(self) -> None:
        """Stop async monitoring."""
        if self.monitoring_interval:
            self.monitoring_interval.stop()
            self.monitoring_interval = None
        self.monitoring_active = False

        # Also stop log monitoring
        self._stop_log_monitoring()

        logger.debug("Stopped async process monitoring")

    async def _check_process_status_async(self) -> None:
        """Async process status check - replaces worker thread."""
        if not self.monitoring_active:
            return

        try:
            # Simple direct access - no threading, no locks needed
            if not self._is_any_plate_running():
                logger.debug("🔥 MONITOR: Subprocess finished")

                # Stop monitoring first
                self._stop_monitoring()

                # Handle completion directly - no call_from_thread needed
                await self._handle_process_completion()

        except Exception as e:
            logger.debug(f"Error in async process monitoring: {e}")
            # Continue monitoring on error

    async def _handle_process_completion(self) -> None:
        """Handle subprocess completion - read from log file (single source of truth)."""
        # Determine success/failure from log file content (single source of truth)
        success = False

        if self.log_file_path and Path(self.log_file_path).exists():
            try:
                # Read log file directly to check for success markers
                with open(self.log_file_path, 'r') as f:
                    log_content = f.read()
                    # Look for success markers in the log
                    has_execution_success = "🔥 SUBPROCESS: EXECUTION SUCCESS:" in log_content
                    has_all_completed = "All plates completed successfully" in log_content
                    if has_execution_success and has_all_completed:
                        success = True

            except Exception as e:
                logger.error(f"Error reading subprocess log file: {e}")
                success = False

        # Clean up the subprocess
        logger.info("🔥 MONITOR: Starting process cleanup...")
        if self.current_process:
            try:
                self.current_process.wait()  # Clean up the zombie process
                logger.info("🔥 MONITOR: Process cleanup completed")
            except Exception as e:
                logger.warning(f"🔥 MONITOR: Error during process cleanup: {e}")

        # Update orchestrator states based on log file analysis (single source of truth)
        if success:
            # Success - update orchestrators to completed
            for plate_path, orchestrator in self.orchestrators.items():
                if orchestrator.state == OrchestratorState.EXECUTING:
                    orchestrator._state = OrchestratorState.COMPLETED

            # Reset execution state (this will trigger UI refresh internally)
            self._reset_execution_state("Execution completed successfully.", force_fail_executing=False)
        else:
            # Failure - update orchestrators to failed
            for plate_path, orchestrator in self.orchestrators.items():
                if orchestrator.state == OrchestratorState.EXECUTING:
                    orchestrator._state = OrchestratorState.EXEC_FAILED

            # Reset execution state (this will trigger UI refresh internally)
            self._reset_execution_state("Execution failed.", force_fail_executing=False)

        self._stop_monitoring()  # Stop monitoring since process is done

    async def _read_log_file_incremental(self) -> None:
        """Read new content from the log file since last read."""
        if not self.log_file_path:
            self.app.current_status = "🔥 LOG READER: No log file"
            return

        try:
            # Wrap all file I/O operations in executor to avoid blocking UI
            def _read_log_content():
                if not Path(self.log_file_path).exists():
                    return None, self.log_file_position

                with open(self.log_file_path, 'r') as f:
                    # Seek to where we left off
                    f.seek(self.log_file_position)
                    new_content = f.read()
                    # Update position for next read
                    new_position = f.tell()

                return new_content, new_position

            new_content, new_position = await asyncio.get_event_loop().run_in_executor(None, _read_log_content)
            self.log_file_position = new_position

            if new_content is None:
                self.app.current_status = "🔥 LOG READER: No log file"
                return

            if new_content and new_content.strip():
                # Get the last non-empty line from new content
                lines = new_content.strip().split('\n')
                non_empty_lines = [line.strip() for line in lines if line.strip()]

                if non_empty_lines:
                    # Show the last line, truncated if too long
                    last_line = non_empty_lines[-1]
                    if len(last_line) > 100:
                        last_line = last_line[:97] + "..."

                    self.app.current_status = last_line
                else:
                    self.app.current_status = "🔥 LOG READER: No lines found"
            else:
                self.app.current_status = "🔥 LOG READER: No new content"

        except Exception as e:
            self.app.current_status = f"🔥 LOG READER ERROR: {e}"



    async def action_stop_execution(self) -> None:
        logger.info("🛑 Stop button pressed. Terminating subprocess.")
        self.app.current_status = "Terminating execution..."

        # Stop async monitoring first
        self._stop_monitoring()

        if self.current_process and self.current_process.poll() is None:  # Still running
            try:
                # Kill the entire process group, not just the parent process
                # The subprocess creates its own process group, so we need to kill that group
                logger.info(f"🛑 Killing process group for PID {self.current_process.pid}...")

                # Get the process group ID (should be same as PID since subprocess calls os.setpgrp())
                process_group_id = self.current_process.pid

                # Kill entire process group (negative PID kills process group)
                os.killpg(process_group_id, signal.SIGTERM)

                # Give processes time to exit gracefully
                await asyncio.sleep(1)

                # Force kill if still alive
                try:
                    os.killpg(process_group_id, signal.SIGKILL)
                    logger.info(f"🛑 Force killed process group {process_group_id}")
                except ProcessLookupError:
                    logger.info(f"🛑 Process group {process_group_id} already terminated")

            except Exception as e:
                logger.warning(f"🛑 Error killing process group: {e}, falling back to single process kill")
                # Fallback to killing just the main process
                self.current_process.kill()

        self._reset_execution_state("Execution terminated by user.")



    async def action_add_plate(self) -> None:
        """Handle Add Plate button."""
        await self._open_plate_directory_browser()

    async def action_export_ome_zarr(self) -> None:
        """Export selected plate to OME-ZARR format."""
        if not self.selected_plate:
            self.app.show_error("No Selection", "Please select a plate to export.")
            return

        # Get the orchestrator for the selected plate
        orchestrator = self.orchestrators.get(self.selected_plate)
        if not orchestrator:
            self.app.show_error("Not Initialized", "Please initialize the plate before exporting.")
            return

        # Open file browser for export location
        def handle_export_result(selected_paths):
            if selected_paths:
                export_path = Path(selected_paths[0]) if isinstance(selected_paths, list) else Path(selected_paths)
                self._start_ome_zarr_export(orchestrator, export_path)

        await self.window_service.open_file_browser(
            file_manager=self.filemanager,
            initial_path=get_cached_browser_path(PathCacheKey.GENERAL),
            backend=Backend.DISK,
            title="Select OME-ZARR Export Directory",
            mode="save",
            selection_mode=SelectionMode.DIRECTORIES_ONLY,
            cache_key=PathCacheKey.GENERAL,
            on_result_callback=handle_export_result,
            caller_id="plate_manager_export"
        )

    def _start_ome_zarr_export(self, orchestrator, export_path: Path):
        """Start OME-ZARR export process."""
        async def run_export():
            try:
                self.app.current_status = f"Exporting to OME-ZARR: {export_path}"

                # Create export-specific config with ZARR materialization
                export_config = orchestrator.global_config
                export_vfs_config = VFSConfig(
                    intermediate_backend=export_config.vfs.intermediate_backend,
                    materialization_backend=MaterializationBackend.ZARR
                )

                # Update orchestrator config for export
                export_global_config = dataclasses.replace(export_config, vfs=export_vfs_config)

                # Create zarr backend with OME-ZARR enabled
                zarr_backend = ZarrStorageBackend(ome_zarr_metadata=True)

                # Copy processed data from current workspace/plate to OME-ZARR format
                # For OpenHCS format, workspace_path is None, so use input_dir (plate path)
                source_path = orchestrator.workspace_path or orchestrator.input_dir
                if source_path and source_path.exists():
                    # Find processed images in workspace/plate
                    processed_images = list(source_path.rglob("*.tif"))

                    if processed_images:
                        # Group by well for batch operations
                        wells_data = defaultdict(list)

                        for img_path in processed_images:
                            # Extract well from filename
                            well_match = None
                            # Try ImageXpress pattern: A01_s001_w1_z001.tif
                            match = re.search(r'([A-Z]\d{2})_', img_path.name)
                            if match:
                                well_id = match.group(1)
                                wells_data[well_id].append(img_path)

                        # Export each well to OME-ZARR
                        export_store_path = export_path / "plate.zarr"

                        for well_id, well_images in wells_data.items():
                            # Load images
                            images = []
                            for img_path in well_images:
                                img = Image.open(img_path)
                                images.append(np.array(img))

                            # Create output paths for OME-ZARR structure
                            output_paths = [export_store_path / f"{well_id}_{i:03d}.tif"
                                          for i in range(len(images))]

                            # Save to OME-ZARR format
                            zarr_backend.save_batch(images, output_paths, chunk_name=well_id)

                        self.app.current_status = f"✅ OME-ZARR export completed: {export_store_path}"
                    else:
                        self.app.show_error("No Data", "No processed images found in workspace.")
                else:
                    self.app.show_error("No Workspace", "Plate workspace not found. Run pipeline first.")

            except Exception as e:
                logger.error(f"OME-ZARR export failed: {e}", exc_info=True)
                self.app.show_error("Export Failed", f"OME-ZARR export failed: {str(e)}")

        # Run export in background
        asyncio.create_task(run_export())

    # Debug functionality removed - no longer needed

    async def _open_plate_directory_browser(self):
        """Open textual-window file browser for plate directory selection."""
        # Get cached path for better UX - remembers last used directory
        path_cache = get_path_cache()
        initial_path = path_cache.get_initial_path(PathCacheKey.PLATE_IMPORT, Path.home())

        # Open textual-window file browser for directory selection
        await self.window_service.open_file_browser(
            file_manager=self.filemanager,
            initial_path=initial_path,
            backend=Backend.DISK,
            title="Select Plate Directory",
            mode="load",
            selection_mode=SelectionMode.DIRECTORIES_ONLY,
            cache_key=PathCacheKey.PLATE_IMPORT,
            on_result_callback=self._add_plate_callback,
            caller_id="plate_manager",
            enable_multi_selection=True
        )

    def _add_plate_callback(self, selected_paths) -> None:
        """Handle directory selection from file browser."""
        logger.debug(f"_add_plate_callback called with: {selected_paths} (type: {type(selected_paths)})")

        if selected_paths is None or selected_paths is False:
            self.app.current_status = "Plate selection cancelled"
            return

        # Handle both single path and list of paths
        if not isinstance(selected_paths, list):
            selected_paths = [selected_paths]

        added_plates = []
        current_plates = list(self.items)

        for selected_path in selected_paths:
            # Ensure selected_path is a Path object
            if isinstance(selected_path, str):
                selected_path = Path(selected_path)
            elif not isinstance(selected_path, Path):
                selected_path = Path(str(selected_path))

            # Check if plate already exists
            if any(plate['path'] == str(selected_path) for plate in current_plates):
                continue

            # Add the plate to the list
            plate_name = selected_path.name
            plate_path = str(selected_path)
            plate_entry = {
                'name': plate_name,
                'path': plate_path,
                # No status field - state comes from orchestrator
            }

            current_plates.append(plate_entry)
            added_plates.append(plate_name)

        # Cache the parent directory for next time (save user navigation time)
        if selected_paths:
            # Use parent of first selected path as the cached directory
            first_path = selected_paths[0] if isinstance(selected_paths[0], Path) else Path(selected_paths[0])
            parent_dir = first_path.parent
            get_path_cache().set_cached_path(PathCacheKey.PLATE_IMPORT, parent_dir)

        # Update items list using reactive property (triggers automatic UI update)
        self.items = current_plates

        if added_plates:
            if len(added_plates) == 1:
                self.app.current_status = f"Added plate: {added_plates[0]}"
            else:
                self.app.current_status = f"Added {len(added_plates)} plates: {', '.join(added_plates)}"
        else:
            self.app.current_status = "No new plates added (duplicates skipped)"

    def action_delete_plate(self) -> None:
        selected_items, _ = self.get_selection_state()
        if not selected_items:
            self.app.show_error("No plate selected to delete.")
            return
        
        paths_to_delete = {p['path'] for p in selected_items}
        self.items = [p for p in self.items if p['path'] not in paths_to_delete]

        # Clean up orchestrators for deleted plates
        for path in paths_to_delete:
            if path in self.orchestrators:
                del self.orchestrators[path]

        if self.selected_plate in paths_to_delete:
            self.selected_plate = ""

        self.app.current_status = f"Deleted {len(paths_to_delete)} plate(s)"



    async def action_edit_config(self) -> None:
        """
        Handle Edit button - create per-orchestrator PipelineConfig instances.

        This enables per-orchestrator configuration without affecting global configuration.
        Shows resolved defaults from GlobalPipelineConfig with "Pipeline default: {value}" placeholders.
        """
        selected_items, selection_mode = self.get_selection_state()

        if selection_mode == "empty":
            self.app.current_status = "No orchestrators selected for configuration"
            return

        selected_orchestrators = [
            self.orchestrators[item['path']] for item in selected_items
            if item['path'] in self.orchestrators
        ]

        if not selected_orchestrators:
            self.app.current_status = "No initialized orchestrators selected"
            return

        # Load existing config or create new one for editing
        representative_orchestrator = selected_orchestrators[0]

        # Use orchestrator's existing config if it exists, otherwise use global config as source
        source_config = representative_orchestrator.pipeline_config or self.global_config

        from openhcs.core.config import create_pipeline_config_for_editing
        current_plate_config = create_pipeline_config_for_editing(source_config)

        def handle_config_save(new_config: PipelineConfig) -> None:
            """Apply per-orchestrator configuration without global side effects."""
            for orchestrator in selected_orchestrators:
                # Direct synchronous call - no async needed
                orchestrator.apply_pipeline_config(new_config)
            count = len(selected_orchestrators)
            self.app.current_status = f"Per-orchestrator configuration applied to {count} orchestrator(s)"

        # Open configuration window using PipelineConfig (not GlobalPipelineConfig)
        await self.window_service.open_config_window(
            PipelineConfig,
            current_plate_config,
            on_save_callback=handle_config_save
        )

    async def action_edit_global_config(self) -> None:
        """
        Handle global configuration editing - affects all orchestrators.

        This maintains the existing global configuration workflow but uses lazy loading.
        """
        from openhcs.core.config import get_default_global_config
        from openhcs.core.lazy_config import create_pipeline_config_for_editing, PipelineConfig

        # Get current global config from app or use default
        current_global_config = self.app.global_config or get_default_global_config()

        # Create lazy PipelineConfig for editing with proper thread-local context
        current_lazy_config = create_pipeline_config_for_editing(current_global_config, preserve_values=True)

        def handle_global_config_save(new_config: PipelineConfig) -> None:
            """Apply global configuration to all orchestrators."""
            # Convert lazy PipelineConfig back to GlobalPipelineConfig
            global_config = new_config.to_base_config()

            self.app.global_config = global_config  # Update app-level config

            # Update thread-local storage for MaterializationPathConfig defaults
            from openhcs.core.config import set_current_global_config, GlobalPipelineConfig
            set_current_global_config(GlobalPipelineConfig, global_config)

            for orchestrator in self.orchestrators.values():
                asyncio.create_task(orchestrator.apply_new_global_config(global_config))
            self.app.current_status = "Global configuration applied to all orchestrators"

        # PipelineConfig already imported from openhcs.core.config
        await self.window_service.open_config_window(
            PipelineConfig,
            current_lazy_config,
            on_save_callback=handle_global_config_save
        )



    def _analyze_orchestrator_configs(self, orchestrators: List['PipelineOrchestrator']) -> Dict[str, Dict[str, Any]]:
        """Analyze configs across multiple orchestrators to detect same/different values.

        Args:
            orchestrators: List of PipelineOrchestrator instances

        Returns:
            Dict mapping field names to analysis results:
            - {"type": "same", "value": actual_value, "default": default_value}
            - {"type": "different", "values": [val1, val2, ...], "default": default_value}
        """
        if not orchestrators:
            return {}

        # Get parameter info for defaults
        param_info = SignatureAnalyzer.analyze(GlobalPipelineConfig)

        config_analysis = {}

        # Analyze each field in GlobalPipelineConfig
        for field in dataclasses.fields(GlobalPipelineConfig):
            field_name = field.name

            # Get values from all orchestrators
            values = []
            for orch in orchestrators:
                try:
                    value = getattr(orch.global_config, field_name)
                    values.append(value)
                except AttributeError:
                    # Field doesn't exist in this config, skip
                    continue

            if not values:
                continue

            # Get default value from parameter info
            param_details = param_info.get(field_name)
            default_value = param_details.default_value if param_details else None

            # Check if all values are the same
            if all(self._values_equal(v, values[0]) for v in values):
                config_analysis[field_name] = {
                    "type": "same",
                    "value": values[0],
                    "default": default_value
                }
            else:
                config_analysis[field_name] = {
                    "type": "different",
                    "values": values,
                    "default": default_value
                }

        return config_analysis

    def _values_equal(self, val1: Any, val2: Any) -> bool:
        """Check if two values are equal, handling dataclasses and complex types."""
        # Handle dataclass comparison
        if dataclasses.is_dataclass(val1) and dataclasses.is_dataclass(val2):
            return dataclasses.asdict(val1) == dataclasses.asdict(val2)

        # Handle regular comparison
        return val1 == val2

    def action_init_plate(self) -> None:
        """Handle Init Plate button - initialize selected plates."""
        # Get current selection state
        selected_items, selection_mode = self.get_selection_state()

        if selection_mode == "empty":
            logger.warning("No plates available for initialization")
            return

        # Validate all selected plates can be initialized (allow ALL failed plates to be re-initialized)
        invalid_plates = []
        for item in selected_items:
            plate_path = item['path']
            orchestrator = self.orchestrators.get(plate_path)
            # Only block plates that are currently executing - all other states can be re-initialized
            if orchestrator is not None and orchestrator.state == OrchestratorState.EXECUTING:
                invalid_plates.append(item)

        if invalid_plates:
            names = [item['name'] for item in invalid_plates]
            logger.warning(f"Cannot initialize plates that are currently executing: {', '.join(names)}")
            return

        # Start async initialization
        self._start_async_init(selected_items, selection_mode)

    def _start_async_init(self, selected_items: List[Dict], selection_mode: str) -> None:
        """Start async initialization of selected plates."""
        # Generate operation description
        desc = self.get_operation_description(selected_items, selection_mode, "initialize")
        logger.info(f"Initializing: {desc}")

        # Start background worker
        self._init_plates_worker(selected_items)

    @work(exclusive=True)
    async def _init_plates_worker(self, selected_items: List[Dict]) -> None:
        """Background worker for plate initialization."""
        for plate_data in selected_items:
            plate_path = plate_data['path']

            # Find the actual plate in self.items (not the copy from get_selection_state)
            actual_plate = None
            for plate in self.items:
                if plate['path'] == plate_path:
                    actual_plate = plate
                    break

            if not actual_plate:
                logger.error(f"Plate not found in plates list: {plate_path}")
                continue

            try:
                # Run heavy initialization in executor to avoid blocking UI
                def init_orchestrator():
                    return PipelineOrchestrator(
                        plate_path=plate_path,
                        global_config=self.global_config,
                        storage_registry=self.filemanager.registry
                    ).initialize()

                orchestrator = await asyncio.get_event_loop().run_in_executor(None, init_orchestrator)

                # Store orchestrator for later use (channel selection, etc.)
                self.orchestrators[plate_path] = orchestrator
                # Orchestrator state is already set to READY by initialize() method
                logger.info(f"Plate {actual_plate['name']} initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize plate {plate_path}: {e}", exc_info=True)
                # Create a failed orchestrator to track the error state
                failed_orchestrator = PipelineOrchestrator(
                    plate_path=plate_path,
                    global_config=self.global_config,
                    storage_registry=self.filemanager.registry
                )
                failed_orchestrator._state = OrchestratorState.INIT_FAILED
                self.orchestrators[plate_path] = failed_orchestrator
                actual_plate['error'] = str(e)

            # Trigger UI refresh after orchestrator state changes
            self._trigger_ui_refresh()
            # Update button states immediately (reactive system handles UI updates automatically)
            self._update_button_states()
            # Notify pipeline editor of status change
            status_symbol = get_orchestrator_status_symbol(self.orchestrators.get(actual_plate['path']))
            self._notify_pipeline_editor_status_change(actual_plate['path'], status_symbol)
            logger.debug(f"Updated plate {actual_plate['name']} status")

        # Final UI update (reactive system handles this automatically when self.items is modified)
        self._update_button_states()

        # Update status
        success_count = len([p for p in selected_items if self.orchestrators.get(p['path']) and self.orchestrators[p['path']].state == OrchestratorState.READY])
        error_count = len([p for p in selected_items if self.orchestrators.get(p['path']) and self.orchestrators[p['path']].state == OrchestratorState.INIT_FAILED])

        if error_count == 0:
            logger.info(f"Successfully initialized {success_count} plates")
        else:
            logger.warning(f"Initialized {success_count} plates, {error_count} errors")

    def action_compile_plate(self) -> None:
        """Handle Compile Plate button - compile pipelines for selected plates."""
        # Get current selection state
        selected_items, selection_mode = self.get_selection_state()

        if selection_mode == "empty":
            logger.warning("No plates available for compilation")
            return

        # Validate all selected plates are ready for compilation (allow failed plates to be re-compiled)
        not_ready = []
        for item in selected_items:
            plate_path = item['path']
            orchestrator = self.orchestrators.get(plate_path)
            # Allow READY, COMPILE_FAILED, EXEC_FAILED, COMPILED, and COMPLETED states to be compiled/recompiled
            if orchestrator is None or orchestrator.state not in [OrchestratorState.READY, OrchestratorState.COMPILE_FAILED, OrchestratorState.EXEC_FAILED, OrchestratorState.COMPILED, OrchestratorState.COMPLETED]:
                not_ready.append(item)

        if not_ready:
            names = [item['name'] for item in not_ready]
            # More accurate error message based on actual state
            if any(self.orchestrators.get(item['path']) is None for item in not_ready):
                logger.warning(f"Cannot compile plates that haven't been initialized: {', '.join(names)}")
            elif any(self.orchestrators.get(item['path']).state == OrchestratorState.EXECUTING for item in not_ready):
                logger.warning(f"Cannot compile plates that are currently executing: {', '.join(names)}")
            else:
                logger.warning(f"Cannot compile plates in current state: {', '.join(names)}")
            return

        # Validate all selected plates have pipelines
        no_pipeline = []
        for item in selected_items:
            pipeline = self._get_current_pipeline_definition(item['path'])
            if not pipeline:
                no_pipeline.append(item)

        if no_pipeline:
            names = [item['name'] for item in no_pipeline]
            self.app.current_status = f"Cannot compile plates without pipelines: {', '.join(names)}"
            return

        # Start async compilation
        self._start_async_compile(selected_items, selection_mode)

    def _start_async_compile(self, selected_items: List[Dict], selection_mode: str) -> None:
        """Start async compilation of selected plates."""
        # Generate operation description
        desc = self.get_operation_description(selected_items, selection_mode, "compile")
        logger.info(f"Compiling: {desc}")

        # Start background worker
        self._compile_plates_worker(selected_items)

    @work(exclusive=True)
    async def _compile_plates_worker(self, selected_items: List[Dict]) -> None:
        """Background worker for plate compilation."""
        for plate_data in selected_items:
            plate_path = plate_data['path']

            # Find the actual plate in self.items (not the copy from get_selection_state)
            actual_plate = None
            for plate in self.items:
                if plate['path'] == plate_path:
                    actual_plate = plate
                    break

            if not actual_plate:
                logger.error(f"Plate not found in plates list: {plate_path}")
                continue

            # Get definition pipeline and make fresh copy
            definition_pipeline = self._get_current_pipeline_definition(plate_path)
            if not definition_pipeline:
                logger.warning(f"No pipeline defined for {actual_plate['name']}, using empty pipeline")
                definition_pipeline = []

            try:
                # Get or create orchestrator for compilation (run in executor to avoid blocking)
                def get_or_create_orchestrator():
                    if plate_path in self.orchestrators:
                        orchestrator = self.orchestrators[plate_path]
                        if not orchestrator.is_initialized():
                            orchestrator.initialize()
                        return orchestrator
                    else:
                        return PipelineOrchestrator(
                            plate_path=plate_path,
                            global_config=self.global_config,
                            storage_registry=self.filemanager.registry
                        ).initialize()

                orchestrator = await asyncio.get_event_loop().run_in_executor(None, get_or_create_orchestrator)
                self.orchestrators[plate_path] = orchestrator

                # Make fresh copy for compilation
                execution_pipeline = copy.deepcopy(definition_pipeline)

                # Fix step IDs after deep copy to match new object IDs
                for step in execution_pipeline:
                    step.step_id = str(id(step))
                    # Ensure variable_components is never None - use FunctionStep default
                    if step.variable_components is None:
                        logger.warning(f"🔥 Step '{step.name}' has None variable_components, setting FunctionStep default")
                        step.variable_components = [VariableComponents.SITE]
                    # Also ensure it's not an empty list
                    elif not step.variable_components:
                        logger.warning(f"🔥 Step '{step.name}' has empty variable_components, setting FunctionStep default")
                        step.variable_components = [VariableComponents.SITE]

                # Get wells and compile (async - run in executor to avoid blocking UI)
                # Wrap in Pipeline object like test_main.py does
                pipeline_obj = Pipeline(steps=execution_pipeline)

                # Run heavy operations in executor to avoid blocking UI
                wells = await asyncio.get_event_loop().run_in_executor(None, lambda: orchestrator.get_component_keys(GroupBy.WELL))
                compiled_contexts = await asyncio.get_event_loop().run_in_executor(
                    None, orchestrator.compile_pipelines, pipeline_obj.steps, wells
                )

                # Store state simply - no reactive property issues
                step_ids_in_pipeline = [id(step) for step in execution_pipeline]
                # Get step IDs from contexts (ProcessingContext objects)
                first_well_key = list(compiled_contexts.keys())[0] if compiled_contexts else None
                step_ids_in_contexts = list(compiled_contexts[first_well_key].step_plans.keys()) if first_well_key and hasattr(compiled_contexts[first_well_key], 'step_plans') else []
                logger.info(f"🔥 Storing compiled data for {plate_path}: pipeline={type(execution_pipeline)}, contexts={type(compiled_contexts)}")
                logger.info(f"🔥 Step IDs in pipeline: {step_ids_in_pipeline}")
                logger.info(f"🔥 Step IDs in contexts: {step_ids_in_contexts}")
                self.plate_compiled_data[plate_path] = (execution_pipeline, compiled_contexts)
                logger.info(f"🔥 Stored! Available compiled plates: {list(self.plate_compiled_data.keys())}")

                # Orchestrator state is already set to COMPILED by compile_pipelines() method
                logger.info(f"🔥 Successfully compiled {plate_path}")

            except Exception as e:
                logger.error(f"🔥 COMPILATION ERROR: Pipeline compilation failed for {plate_path}: {e}", exc_info=True)
                # Orchestrator state is already set to FAILED by compile_pipelines() method
                actual_plate['error'] = str(e)
                # Don't store anything in plate_compiled_data on failure

            # Trigger UI refresh after orchestrator state changes
            self._trigger_ui_refresh()
            # Update button states immediately (reactive system handles UI updates automatically)
            self._update_button_states()
            # Notify pipeline editor of status change
            status_symbol = get_orchestrator_status_symbol(self.orchestrators.get(actual_plate['path']))
            self._notify_pipeline_editor_status_change(actual_plate['path'], status_symbol)

        # Final UI update (reactive system handles this automatically when self.items is modified)
        self._update_button_states()

        # Update status
        success_count = len([p for p in selected_items if self.orchestrators.get(p['path']) and self.orchestrators[p['path']].state == OrchestratorState.COMPILED])
        error_count = len([p for p in selected_items if self.orchestrators.get(p['path']) and self.orchestrators[p['path']].state == OrchestratorState.COMPILE_FAILED])

        if error_count == 0:
            logger.info(f"Successfully compiled {success_count} plates")
        else:
            logger.warning(f"Compiled {success_count} plates, {error_count} errors")

    async def action_code_plate(self) -> None:
        """Generate Python code for selected plates and their pipelines."""
        logger.debug("Code button pressed - generating Python code for plates")

        selected_items, _ = self.get_selection_state()
        if not selected_items:
            self.app.current_status = "No plates selected for code generation"
            return

        try:
            # Get pipeline data for selected plates
            plate_paths = [item['path'] for item in selected_items]
            pipeline_data = {}

            # Collect pipeline steps for each plate
            for plate_path in plate_paths:
                if hasattr(self, 'pipeline_editor') and self.pipeline_editor:
                    # Get pipeline steps from pipeline editor if available
                    if plate_path in self.pipeline_editor.plate_pipelines:
                        pipeline_data[plate_path] = self.pipeline_editor.plate_pipelines[plate_path]
                    else:
                        pipeline_data[plate_path] = []
                else:
                    pipeline_data[plate_path] = []

            # Use existing pickle_to_python logic to generate complete script
            from openhcs.textual_tui.services.terminal_launcher import TerminalLauncher

            # Create data structure like pickle_to_python expects
            data = {
                'plate_paths': plate_paths,
                'pipeline_data': pipeline_data,
                'global_config': self.app.global_config
            }

            # Extract variables from data dict
            plate_paths = data['plate_paths']
            pipeline_data = data['pipeline_data']

            # Generate just the orchestrator configuration (no execution wrapper)
            from openhcs.debug.pickle_to_python import generate_complete_orchestrator_code

            python_code = generate_complete_orchestrator_code(
                plate_paths=plate_paths,
                pipeline_data=pipeline_data,
                global_config=self.app.global_config,
                clean_mode=False
            )

            # Create callback to handle edited code
            def handle_edited_code(edited_code: str):
                logger.debug("Orchestrator code edited, processing changes...")
                try:
                    # Execute the code (it has all necessary imports)
                    namespace = {}
                    exec(edited_code, namespace)

                    # Update pipeline data if present (composition: orchestrator contains pipelines)
                    if 'pipeline_data' in namespace:
                        new_pipeline_data = namespace['pipeline_data']
                        # Update pipeline editor using reactive system (like pipeline code button does)
                        if hasattr(self, 'pipeline_editor') and self.pipeline_editor:
                            # Update plate pipelines storage
                            current_pipelines = dict(self.pipeline_editor.plate_pipelines)
                            current_pipelines.update(new_pipeline_data)
                            self.pipeline_editor.plate_pipelines = current_pipelines

                            # If current plate is in the edited data, update the current view too
                            current_plate = self.pipeline_editor.current_plate
                            if current_plate and current_plate in new_pipeline_data:
                                self.pipeline_editor.pipeline_steps = new_pipeline_data[current_plate]

                        self.app.current_status = f"Pipeline data updated for {len(new_pipeline_data)} plates"

                    # Update global config if present
                    elif 'global_config' in namespace:
                        new_global_config = namespace['global_config']
                        import asyncio
                        for plate_path in plate_paths:
                            if plate_path in self.orchestrators:
                                orchestrator = self.orchestrators[plate_path]
                                asyncio.create_task(orchestrator.apply_new_global_config(new_global_config))
                        self.app.current_status = f"Global config updated for {len(plate_paths)} plates"

                    # Update orchestrators list if present
                    elif 'orchestrators' in namespace:
                        new_orchestrators = namespace['orchestrators']
                        self.app.current_status = f"Orchestrator list updated with {len(new_orchestrators)} orchestrators"

                    else:
                        self.app.show_error("Parse Error", "No valid assignments found in edited code")

                except SyntaxError as e:
                    self.app.show_error("Syntax Error", f"Invalid Python syntax: {e}")
                except Exception as e:
                    logger.error(f"Failed to parse edited orchestrator code: {e}")
                    self.app.show_error("Edit Error", f"Failed to parse orchestrator code: {str(e)}")

            # Launch terminal editor
            launcher = TerminalLauncher(self.app)
            await launcher.launch_editor_for_file(
                file_content=python_code,
                file_extension='.py',
                on_save_callback=handle_edited_code
            )

        except Exception as e:
            logger.error(f"Failed to generate plate code: {e}")
            self.app.current_status = f"Failed to generate code: {e}"

    async def action_save_python_script(self) -> None:
        """Save Python script for selected plates (like special_io_pipeline.py)."""
        logger.debug("Save button pressed - saving Python script for plates")

        selected_items, _ = self.get_selection_state()
        if not selected_items:
            self.app.current_status = "No plates selected for script generation"
            return

        try:
            # Get pipeline data for selected plates
            plate_paths = [item['path'] for item in selected_items]
            pipeline_data = {}

            # Collect pipeline steps for each plate
            for plate_path in plate_paths:
                if hasattr(self, 'pipeline_editor') and self.pipeline_editor:
                    # Get pipeline steps from pipeline editor if available
                    if plate_path in self.pipeline_editor.plate_pipelines:
                        pipeline_data[plate_path] = self.pipeline_editor.plate_pipelines[plate_path]
                    else:
                        pipeline_data[plate_path] = []
                else:
                    pipeline_data[plate_path] = []

            # Create data structure like pickle_to_python expects
            data = {
                'plate_paths': plate_paths,
                'pipeline_data': pipeline_data,
                'global_config': self.app.global_config
            }

            # Generate complete executable Python script using pickle_to_python logic
            python_code = self._generate_executable_script(data)

            # Launch file browser to save the script
            from openhcs.textual_tui.windows.file_browser_window import open_file_browser_window, BrowserMode
            from openhcs.textual_tui.services.file_browser_service import SelectionMode
            from openhcs.core.path_cache import get_cached_browser_path, PathCacheKey
            from openhcs.constants.constants import Backend

            def handle_save_result(result):
                if result:
                    # Handle both single Path and list of Paths
                    save_path = None
                    if isinstance(result, Path):
                        save_path = result
                    elif isinstance(result, list) and len(result) > 0:
                        save_path = result[0]  # Take first path

                    if save_path:
                        try:
                            # Write the Python script to the selected file
                            with open(save_path, 'w') as f:
                                f.write(python_code)

                            logger.info(f"Python script saved to: {save_path}")
                            self.app.current_status = f"Python script saved to: {save_path}"
                        except Exception as e:
                            logger.error(f"Failed to save Python script: {e}")
                            self.app.current_status = f"Failed to save script: {e}"

            # Generate default filename based on first plate
            first_plate_name = Path(plate_paths[0]).name if plate_paths else "pipeline"
            default_filename = f"{first_plate_name}_pipeline.py"

            await open_file_browser_window(
                app=self.app,
                file_manager=self.app.filemanager,
                initial_path=get_cached_browser_path(PathCacheKey.PIPELINE_FILES),
                backend=Backend.DISK,
                title="Save Python Pipeline Script",
                mode=BrowserMode.SAVE,
                selection_mode=SelectionMode.FILES_ONLY,
                filter_extensions=['.py'],
                default_filename=default_filename,
                cache_key=PathCacheKey.PIPELINE_FILES,
                on_result_callback=handle_save_result,
                caller_id="plate_manager_save_script"
            )

        except Exception as e:
            logger.error(f"Failed to save Python script: {e}")
            self.app.current_status = f"Failed to save script: {e}"

    def _generate_executable_script(self, data: Dict) -> str:
        """Generate fully executable Python script by creating a temp pickle and using existing convert_pickle_to_python."""
        import tempfile
        import dill as pickle
        from openhcs.debug.pickle_to_python import convert_pickle_to_python

        # Create temporary pickle file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as temp_pickle:
            pickle.dump(data, temp_pickle)
            temp_pickle_path = temp_pickle.name

        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_output:
            temp_output_path = temp_output.name

        try:
            # Use existing convert_pickle_to_python function
            convert_pickle_to_python(temp_pickle_path, temp_output_path)

            # Read the generated script
            with open(temp_output_path, 'r') as f:
                script_content = f.read()

            return script_content

        finally:
            # Clean up temp files
            import os
            try:
                os.unlink(temp_pickle_path)
                os.unlink(temp_output_path)
            except:
                pass


