"""
PlateManagerWidget for OpenHCS Textual TUI

Plate management widget with complete button set and reactive state management.
Matches the functionality from the current prompt-toolkit TUI.
"""

import asyncio
import logging
import subprocess
import sys
import tempfile
import pickle
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple

from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer
from textual.reactive import reactive
from textual.widgets import Button, Static, SelectionList
from textual.widget import Widget
from .button_list_widget import ButtonListWidget, ButtonConfig
from textual import work, on

from openhcs.core.config import GlobalPipelineConfig
from openhcs.io.filemanager import FileManager
from openhcs.core.memory.gpu_cleanup import cleanup_all_gpu_frameworks
from openhcs.io.base import storage_registry
from openhcs.io.memory import MemoryStorageBackend
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.constants.constants import GroupBy

logger = logging.getLogger(__name__)

# Note: Using subprocess approach instead of multiprocessing to avoid TUI FD conflicts

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
        from pathlib import Path
        log_dir = Path.home() / ".local" / "share" / "openhcs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return str(log_dir / f"openhcs_subprocess_{int(time.time())}.log")

    except Exception as e:
        logger.warning(f"Could not determine log file path: {e}")
        return "/tmp/openhcs_subprocess.log"





# REMOVED: LogFileWatcher class - using StatusBar's timer-based monitoring instead

class PlateManagerWidget(ButtonListWidget):
    """
    Plate management widget using Textual reactive state.
    """

    # Semantic reactive property (like PipelineEditor's pipeline_steps)
    plates = reactive([])
    selected_plate = reactive("")
    orchestrators = reactive({})
    plate_configs = reactive({})
    
    def __init__(self, filemanager: FileManager, global_config: GlobalPipelineConfig):
        button_configs = [
            ButtonConfig("Add", "add_plate"),
            ButtonConfig("Del", "del_plate", disabled=True),
            ButtonConfig("Edit", "edit_plate", disabled=True),
            ButtonConfig("Init", "init_plate", disabled=True),
            ButtonConfig("Compile", "compile_plate", disabled=True),
            ButtonConfig("Run", "run_plate", disabled=True),
            ButtonConfig("Debug", "save_debug_pickle", disabled=True),  # Enabled for debugging
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

        # --- Subprocess Architecture ---
        self.current_process: Optional[subprocess.Popen] = None
        self.status_file_path: Optional[str] = None
        self.result_file_path: Optional[str] = None
        self.data_file_path: Optional[str] = None
        self.log_file_path: Optional[str] = None
        self.log_file_position: int = 0  # Track position in log file for incremental reading
        # File watcher for real-time log monitoring
        self.file_observer: Optional[Observer] = None
        self.file_watcher: Optional[LogFileWatcher] = None
        # Keep timer for process status checking (reduced frequency)
        self.monitoring_timer = self.set_interval(10.0, self._check_process_status, pause=True)  # Check process every 10 seconds
        # ---
        
        logger.debug("PlateManagerWidget initialized")

    @property
    def plates(self):
        """Alias for items to maintain backward compatibility."""
        return self.items

    @plates.setter
    def plates(self, value):
        """Alias for items to maintain backward compatibility."""
        self.items = value

    def on_unmount(self) -> None:
        logger.info("Unmounting PlateManagerWidget, ensuring worker process is terminated.")
        self.action_stop_execution()
        # DISABLED: self._stop_file_watcher()

    def format_item_for_display(self, plate: Dict) -> Tuple[str, str]:
        status_symbols = {"?": "➕", "-": "✅", "o": "⚡", "!": "🔄", "X": "❌", "F": "🔥", "C": "🏁"}
        status_icon = status_symbols.get(plate.get("status", "?"), "❓")
        plate_name = plate.get('name', 'Unknown')
        plate_path = plate.get('path', '')
        display_text = f"{status_icon} {plate_name} - {plate_path}"
        return display_text, plate_path

    async def _handle_button_press(self, button_id: str) -> None:
        action_map = {
            "add_plate": self.action_add_plate,
            "del_plate": self.action_delete_plate,
            "edit_plate": self.action_edit_plate,
            "init_plate": self.action_init_plate,
            "compile_plate": self.action_compile_plate,
            "save_debug_pickle": self.action_save_debug_pickle,
        }
        if button_id in action_map:
            import inspect
            action = action_map[button_id]
            if inspect.iscoroutinefunction(action):
                await action()
            else:
                action()
        elif button_id == "run_plate":
            if self._is_any_plate_running():
                self.action_stop_execution()
            else:
                await self.action_run_plate()

    def _handle_selection_change(self, selected_values: List[str]) -> None:
        logger.debug(f"Checkmarks changed: {len(selected_values)} items selected")

    def _handle_item_moved(self, from_index: int, to_index: int) -> None:
        current_plates = list(self.plates)
        plate = current_plates.pop(from_index)
        current_plates.insert(to_index, plate)
        self.plates = current_plates
        plate_name = plate['name']
        direction = "up" if to_index < from_index else "down"
        self.app.current_status = f"Moved plate '{plate_name}' {direction}"

    def on_mount(self) -> None:
        self.call_later(self._delayed_update_display)
        self.set_timer(0.1, self._delayed_update_display)
        self.call_later(self._update_button_states)
    
    def watch_plates(self, plates: List[Dict]) -> None:
        """Automatically update UI when plates changes (follows PipelineEditor pattern)."""
        # DEBUG: Log when plates list changes to track the source of the reset
        import traceback
        stack_trace = ''.join(traceback.format_stack()[-3:-1])  # Get last 2 stack frames
        logger.info(f"🔍 PLATES CHANGED: {len(plates)} plates. Call stack:\n{stack_trace}")

        # Sync with ButtonListWidget's items property to trigger its reactive system
        self.items = list(plates)

        logger.debug(f"Plates updated: {len(plates)} plates")
        self._update_button_states()
    
    def watch_highlighted_item(self, plate_path: str) -> None:
        self.selected_plate = plate_path
        logger.debug(f"Highlighted plate: {plate_path}")

    def watch_selected_plate(self, plate_path: str) -> None:
        self._update_button_states()
        if self.on_plate_selected and plate_path:
            self.on_plate_selected(plate_path)
        logger.debug(f"Selected plate: {plate_path}")

    def get_selection_state(self) -> tuple[List[Dict], str]:
        # Check if widget is properly mounted first
        if not self.is_mounted:
            logger.debug("get_selection_state called on unmounted widget")
            return [], "empty"

        try:
            selection_list = self.query_one(f"#{self.list_id}")
            multi_selected_values = selection_list.selected
            if multi_selected_values:
                selected_items = [p for p in self.plates if p.get('path') in multi_selected_values]
                return selected_items, "checkbox"
            elif self.selected_plate:
                selected_items = [p for p in self.plates if p.get('path') == self.selected_plate]
                return selected_items, "cursor"
            else:
                return [], "empty"
        except Exception as e:
            # DOM CORRUPTION DETECTED - This is a critical error
            import traceback
            stack_trace = ''.join(traceback.format_stack()[-3:-1])
            logger.error(f"🚨 DOM CORRUPTION: Failed to get selection state: {e}")
            logger.error(f"🚨 DOM CORRUPTION: Call stack:\n{stack_trace}")
            logger.error(f"🚨 DOM CORRUPTION: Widget mounted: {self.is_mounted}")
            logger.error(f"🚨 DOM CORRUPTION: Looking for: #{self.list_id}")
            logger.error(f"🚨 DOM CORRUPTION: Plates count: {len(self.plates)}")

            # Try to diagnose what widgets actually exist
            try:
                all_widgets = list(self.query("*"))
                widget_ids = [w.id for w in all_widgets if w.id]
                logger.error(f"🚨 DOM CORRUPTION: Available widget IDs: {widget_ids}")
            except Exception as diag_e:
                logger.error(f"🚨 DOM CORRUPTION: Could not diagnose widgets: {diag_e}")

            if self.selected_plate:
                selected_items = [p for p in self.plates if p.get('path') == self.selected_plate]
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

            can_run = has_selection and any(p['path'] in self.plate_compiled_data for p in self.plates if p.get('path') == self.selected_plate)

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
            self.query_one("#del_plate").disabled = not self.plates or not has_selected_items or is_running
            self.query_one("#edit_plate").disabled = not has_selection or is_running
            self.query_one("#init_plate").disabled = not has_selection or is_running
            self.query_one("#compile_plate").disabled = not has_selection or is_running

            # Debug button - enabled when subprocess data is available
            has_debug_data = hasattr(self, '_last_subprocess_data')
            self.query_one("#save_debug_pickle").disabled = not has_debug_data

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
        """Get status for specific plate."""
        for plate in self.plates:
            if plate.get('path') == plate_path:
                return plate.get('status', '?')
        return '?'  # Default to added status

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

    def _reset_execution_state(self, status_message: str):
        logger.info(f"🧹 Resetting execution state. Reason: {status_message}")
        
        if self.current_process:
            if self.current_process.poll() is None:  # Still running
                logger.warning("Forcefully terminating subprocess during reset.")
                self.current_process.terminate()
                try:
                    self.current_process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.current_process.kill()  # Force kill if terminate fails
            self.current_process = None

        # DISABLED: Stop file watcher before cleanup
        # self._stop_file_watcher()

        # Clear file references and cleanup temp files (don't delete log file)
        for file_path in [self.status_file_path, self.result_file_path, self.data_file_path]:
            if file_path:
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.debug(f"Could not cleanup temp file {file_path}: {e}")

        self.status_file_path = None
        self.result_file_path = None
        self.data_file_path = None
        self.log_file_path = None
        self.log_file_position = 0
        
        if self.monitoring_timer:
            self.monitoring_timer.pause()
        
        current_plates = list(self.plates)
        for plate in current_plates:
            if plate.get('status') == '!':
                plate['status'] = 'F'
        self.plates = current_plates

        self._update_button_states()
        self.app.current_status = status_message
        logger.info("🧹 Execution state reset and UI updated.")

    async def action_run_plate(self) -> None:
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
            logger.info("🔥 Using subprocess approach for clean isolation")

            plate_paths_to_run = [item['path'] for item in ready_items]

            # Pass definition pipeline steps - subprocess will make fresh copy and compile
            pipeline_data = {}
            for plate_path in plate_paths_to_run:
                definition_pipeline = self._get_current_pipeline_definition(plate_path)
                pipeline_data[plate_path] = definition_pipeline

            logger.info(f"🔥 Starting subprocess for {len(plate_paths_to_run)} plates")

            # Create temporary files for communication
            status_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
            result_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
            data_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl')

            # Close files so subprocess can write to them
            status_file.close()
            result_file.close()

            # Store file paths for monitoring and cleanup
            self.status_file_path = status_file.name
            self.result_file_path = result_file.name
            self.data_file_path = data_file.name

            # Get current log file path
            log_file_path = get_current_log_file_path()

            # Pickle data for subprocess
            subprocess_data = {
                'plate_paths': plate_paths_to_run,
                'pipeline_data': pipeline_data,
                'global_config_dict': self.global_config.__dict__
            }

            # Wrap pickle operation in executor to avoid blocking UI
            import asyncio
            def _write_pickle_data():
                with open(data_file.name, 'wb') as f:
                    pickle.dump(subprocess_data, f)
                data_file.close()

            await asyncio.get_event_loop().run_in_executor(None, _write_pickle_data)

            logger.info(f"🔥 Created data file: {data_file.name}")
            logger.info(f"🔥 Status file: {status_file.name}")
            logger.info(f"🔥 Result file: {result_file.name}")
            logger.info(f"🔥 Log file: {log_file_path}")

            # DEBUGGING: Store subprocess data for manual debugging
            self._last_subprocess_data = subprocess_data
            self._last_data_file_path = data_file.name
            self._last_status_file_path = status_file.name
            self._last_result_file_path = result_file.name
            self._last_log_file_path = log_file_path

            # Create subprocess (like integration tests)
            subprocess_script = Path(__file__).parent.parent / "subprocess_runner.py"

            # Store log file path for monitoring (subprocess logger writes to this)
            self.log_file_path = log_file_path
            self.log_file_position = self._get_current_log_position()  # Start from current end

            logger.info(f"🔥 Subprocess command: {sys.executable} {subprocess_script} {data_file.name} {status_file.name} {result_file.name} {log_file_path}")
            logger.info(f"🔥 Subprocess logger will write to: {self.log_file_path}")
            logger.info(f"🔥 Subprocess stdout will be silenced (logger handles output)")

            # SILENT SUBPROCESS: Let subprocess logger handle output to avoid duplication
            # Wrap subprocess creation in executor to avoid blocking UI
            def _create_subprocess():
                return subprocess.Popen([
                    sys.executable, str(subprocess_script),
                    data_file.name, status_file.name, result_file.name, log_file_path
                ],
                stdout=subprocess.DEVNULL,  # Silence stdout to avoid duplication with logger
                stderr=subprocess.DEVNULL,  # Silence stderr to avoid duplication with logger
                text=True,  # Text mode for easier handling
                )

            self.current_process = await asyncio.get_event_loop().run_in_executor(None, _create_subprocess)

            logger.info(f"🔥 Subprocess started with PID: {self.current_process.pid}")

            # NUCLEAR FIX: Don't start any output monitoring that could interfere
            # Let the subprocess run completely independently

            # Update UI to show running state
            for plate in ready_items:
                plate['status'] = '!'
            self.plates = list(self.plates)

            self.app.current_status = f"Running {len(ready_items)} plate(s) in subprocess..."
            self._update_button_states()
            
            # DISABLED: Start log file watcher for real-time monitoring
            # self._start_log_file_watcher()
            
            if self.monitoring_timer:
                self.monitoring_timer.resume()

        except Exception as e:
            logger.critical(f"Failed to start subprocess: {e}", exc_info=True)
            self.app.show_error("Failed to start subprocess", str(e))
            self._reset_execution_state("Subprocess failed to start")

    def _get_current_log_position(self) -> int:
        """Get current position in log file."""
        if not self.log_file_path or not Path(self.log_file_path).exists():
            return 0
        try:
            return Path(self.log_file_path).stat().st_size
        except Exception:
            return 0

    def _start_log_file_watcher(self) -> None:
        """Start file system watcher for log file changes."""
        if not self.log_file_path:
            return
            
        try:
            self._stop_file_watcher()  # Stop any existing watcher
            
            log_path = Path(self.log_file_path)
            watch_directory = str(log_path.parent)
            
            self.file_watcher = LogFileWatcher(self, self.log_file_path)
            self.file_observer = Observer()
            self.file_observer.daemon = True  # Make daemon thread to prevent hanging
            self.file_observer.schedule(self.file_watcher, watch_directory, recursive=False)
            self.file_observer.start()
            
            logger.info(f"🔥 Started log file watcher for: {self.log_file_path}")
            
        except Exception as e:
            logger.error(f"🔥 Failed to start log file watcher: {e}")
            
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

    def _check_process_status(self) -> None:
        """Check if subprocess is still running (called by timer)."""
        if not self._is_any_plate_running():
            logger.info("🔥 MONITOR: Subprocess finished - investigating...")
            if self.monitoring_timer:
                self.monitoring_timer.pause()

            # Get subprocess exit code and detailed info
            if self.current_process:
                exit_code = self.current_process.poll()
                logger.info(f"🔥 MONITOR: Subprocess exit code: {exit_code}")

                # Classify the exit
                if exit_code == 0:
                    logger.info("🔥 MONITOR: Subprocess completed successfully")
                elif exit_code is None:
                    logger.error("🔥 MONITOR: Subprocess still running but poll() returned None - this is weird!")
                elif exit_code < 0:
                    logger.error(f"🔥 MONITOR: Subprocess killed by signal {-exit_code}")
                else:
                    logger.error(f"🔥 MONITOR: Subprocess failed with exit code {exit_code}")

                # Read any remaining log entries
                self._read_log_file_incremental()

            # Check results from files before cleaning up
            if self.status_file_path:
                # Debug: Check if temp files exist and have content
                logger.info(f"🔥 MONITOR: Checking temp files:")
                logger.info(f"🔥 MONITOR: Status file exists: {Path(self.status_file_path).exists()}")
                logger.info(f"🔥 MONITOR: Result file exists: {Path(self.result_file_path).exists()}")
                logger.info(f"🔥 MONITOR: Data file exists: {Path(self.data_file_path).exists()}")
                logger.info(f"🔥 MONITOR: Log file exists: {Path(self.log_file_path).exists() if self.log_file_path else False}")

                if Path(self.status_file_path).exists():
                    try:
                        with open(self.status_file_path, 'r') as f:
                            content = f.read()
                        logger.info(f"🔥 MONITOR: Status file content: '{content}'")
                        if not content.strip():
                            logger.warning("🔥 MONITOR: Status file is empty - subprocess may have crashed before writing anything")
                    except Exception as e:
                        logger.error(f"🔥 MONITOR: Could not read status file: {e}")

                self._process_execution_results()

            if self.current_process:
                # FIXED: Don't wait with timeout - process is already finished!
                # The poll() check above already confirmed it's done
                try:
                    self.current_process.wait()  # NO TIMEOUT! Just clean up the zombie process
                except Exception as e:
                    logger.warning(f"🔥 MONITOR: Error during process cleanup: {e}")

            self._reset_execution_state("Execution finished.")

    def _read_log_file_incremental(self) -> None:
        """Read new content from the log file since last read."""
        if not self.log_file_path or not Path(self.log_file_path).exists():
            self.app.current_status = "🔥 LOG READER: No log file"
            return
            
        try:
            with open(self.log_file_path, 'r') as f:
                # Seek to where we left off
                f.seek(self.log_file_position)
                new_content = f.read()
                # Update position for next read
                self.log_file_position = f.tell()
                
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

    def _process_execution_results(self) -> None:
        """Process results from the subprocess files."""
        if not self.status_file_path:
            return

        try:
            # Collect all status updates from file
            status_updates = {}
            result_updates = {}

            # Read status file (one JSON object per line)
            if Path(self.status_file_path).exists():
                try:
                    with open(self.status_file_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                data = json.loads(line)
                                plate_path = data['plate_path']
                                status = data['status']
                                status_updates[plate_path] = status
                except Exception as e:
                    logger.debug(f"Could not read status file: {e}")

            # Read result file (one JSON object per line)
            if Path(self.result_file_path).exists():
                try:
                    with open(self.result_file_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                data = json.loads(line)
                                plate_path = data['plate_path']
                                result = data['result']
                                result_updates[plate_path] = result
                except Exception as e:
                    logger.debug(f"Could not read result file: {e}")

            # Update plate statuses based on results
            current_plates = list(self.plates)
            for plate in current_plates:
                plate_path = plate['path']

                if plate_path in status_updates:
                    status = status_updates[plate_path]
                    if status == "COMPLETED":
                        plate['status'] = 'C'  # Completed
                        logger.info(f"🔥 Plate {plate['name']} completed successfully")
                    elif status.startswith("ERROR:"):
                        plate['status'] = 'F'  # Failed
                        plate['error'] = status[6:]  # Remove "ERROR:" prefix
                        logger.error(f"🔥 Plate {plate['name']} failed: {plate['error']}")

                        # Log full error details if available
                        if plate_path in result_updates:
                            logger.error(f"🔥 Full error for {plate['name']}:\n{result_updates[plate_path]}")

                    # Log results if available
                    if plate_path in result_updates and status == "COMPLETED":
                        logger.info(f"🔥 Results for {plate['name']}: {result_updates[plate_path]}")

            # Update the plates list
            self.plates = current_plates

        except Exception as e:
            logger.error(f"Error processing execution results: {e}", exc_info=True)

    def action_stop_execution(self) -> None:
        logger.info("🛑 Stop button pressed. Terminating subprocess.")
        self.app.current_status = "Terminating execution..."

        if self.current_process and self.current_process.poll() is None:  # Still running
            try:
                import os
                import signal
                
                # Kill the entire process group, not just the parent process
                # The subprocess creates its own process group, so we need to kill that group
                logger.info(f"🛑 Killing process group for PID {self.current_process.pid}...")
                
                # Get the process group ID (should be same as PID since subprocess calls os.setpgrp())
                process_group_id = self.current_process.pid
                
                # Kill entire process group (negative PID kills process group)
                os.killpg(process_group_id, signal.SIGTERM)
                
                # Give processes time to exit gracefully
                import time
                time.sleep(1)
                
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
        self._update_button_states()

    # REMOVED: _start_subprocess_output_monitoring
    # This was interfering with subprocess execution by trying to read stdout
    # Now subprocess runs completely independently

    async def action_add_plate(self) -> None:
        """Handle Add Plate button."""
        await self._open_plate_directory_browser()

    async def action_save_debug_pickle(self) -> None:
        """Save the last subprocess pickle file for manual debugging."""
        if not hasattr(self, '_last_subprocess_data'):
            self.app.show_error("No Debug Data", "No subprocess data available. Run execution first.")
            return

        from openhcs.textual_tui.screens.enhanced_file_browser import EnhancedFileBrowserScreen, BrowserMode, SelectionMode
        from openhcs.constants.constants import Backend
        from openhcs.textual_tui.utils.path_cache import get_cached_browser_path, PathCacheKey
        import pickle
        from datetime import datetime

        # Generate default filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"debug_subprocess_data_{timestamp}.pkl"

        def handle_save_result(selected_path):
            """Handle the file save result."""
            if selected_path is None:
                self.app.current_status = "Debug pickle save cancelled"
                return

            try:
                # Save the subprocess data to the selected file
                with open(selected_path, 'wb') as f:
                    pickle.dump(self._last_subprocess_data, f)

                # Also create companion files with the paths for easy manual testing
                base_path = selected_path.with_suffix('')

                # Save executable shell script
                shell_script = base_path.with_suffix('.sh')
                subprocess_script = Path(__file__).parent.parent / "subprocess_runner.py"

                with open(shell_script, 'w') as f:
                    f.write(f"#!/bin/bash\n")
                    f.write(f"# OpenHCS Subprocess Debug Script\n")
                    f.write(f"# Generated: {datetime.now()}\n")
                    f.write(f"# Plates: {self._last_subprocess_data['plate_paths']}\n\n")

                    f.write(f"echo \"🔥 Starting OpenHCS subprocess debugging...\"\n")
                    f.write(f"echo \"🔥 Pickle file: {selected_path.name}\"\n")
                    f.write(f"echo \"🔥 Press Ctrl+C to stop\"\n")
                    f.write(f"echo \"\"\n\n")

                    # Change to the directory containing the pickle file
                    f.write(f"cd \"{selected_path.parent}\"\n\n")

                    # Run the subprocess with the exact filenames
                    f.write(f"python \"{subprocess_script}\" \\\n")
                    f.write(f"    \"{selected_path.name}\" \\\n")
                    f.write(f"    \"debug_status.json\" \\\n")
                    f.write(f"    \"debug_result.json\" \\\n")
                    f.write(f"    \"debug.log\"\n\n")

                    f.write(f"echo \"\"\n")
                    f.write(f"echo \"🔥 Subprocess finished. Check the files:\"\n")
                    f.write(f"echo \"  - debug_status.json (progress/death markers)\"\n")
                    f.write(f"echo \"  - debug_result.json (final results)\"\n")
                    f.write(f"echo \"  - debug.log (detailed logs)\"\n")

                # Make shell script executable
                import stat
                shell_script.chmod(shell_script.stat().st_mode | stat.S_IEXEC)

                # Save command file for reference
                command_file = base_path.with_suffix('.cmd')
                command = f"python {subprocess_script} {selected_path.name} debug_status.json debug_result.json debug.log"

                with open(command_file, 'w') as f:
                    f.write(f"# Manual subprocess debugging command\n")
                    f.write(f"# Run this command to execute the subprocess manually:\n\n")
                    f.write(f"cd \"{selected_path.parent}\"\n")
                    f.write(f"{command}\n\n")
                    f.write(f"# Original files from TUI execution:\n")
                    f.write(f"# Data file: {self._last_data_file_path}\n")
                    f.write(f"# Status file: {self._last_status_file_path}\n")
                    f.write(f"# Result file: {self._last_result_file_path}\n")
                    f.write(f"# Log file: {self._last_log_file_path}\n")

                # Save info file
                info_file = base_path.with_suffix('.info')
                with open(info_file, 'w') as f:
                    f.write(f"Debug Subprocess Data\n")
                    f.write(f"====================\n\n")
                    f.write(f"Generated: {datetime.now()}\n")
                    f.write(f"Plates: {len(self._last_subprocess_data['plate_paths'])}\n")
                    f.write(f"Plate paths: {self._last_subprocess_data['plate_paths']}\n\n")
                    f.write(f"Pipeline data keys: {list(self._last_subprocess_data['pipeline_data'].keys())}\n\n")
                    f.write(f"Global config: {self._last_subprocess_data['global_config_dict']}\n\n")
                    f.write(f"To debug manually:\n")
                    f.write(f"1. Run: ./{shell_script.name} (executable shell script)\n")
                    f.write(f"2. Or run: {command}\n")
                    f.write(f"3. Check debug_status.json for progress/death markers\n")
                    f.write(f"4. Check debug_result.json for results\n")
                    f.write(f"5. Check debug.log for detailed logs\n")

                self.app.current_status = f"Debug files saved: {selected_path.name}, {shell_script.name} (executable), {command_file.name}, {info_file.name}"
                logger.info(f"Debug subprocess data saved to {selected_path}")

            except Exception as e:
                error_msg = f"Failed to save debug pickle: {e}"
                logger.error(error_msg, exc_info=True)
                self.app.show_error("Save Failed", error_msg)

        # Open textual-window file browser for saving
        from openhcs.textual_tui.windows import open_file_browser_window, BrowserMode
        from openhcs.textual_tui.services.file_browser_service import SelectionMode

        await open_file_browser_window(
            app=self.app,
            file_manager=self.filemanager,
            initial_path=get_cached_browser_path(PathCacheKey.DEBUG_FILES),
            backend=Backend.DISK,
            title="Save Debug Subprocess Data",
            mode=BrowserMode.SAVE,
            selection_mode=SelectionMode.FILES_ONLY,
            filter_extensions=['.pkl'],
            default_filename=default_filename,
            cache_key=PathCacheKey.DEBUG_FILES,
            on_result_callback=handle_save_result,
            caller_id="plate_manager"
        )

    async def _open_plate_directory_browser(self):
        """Open textual-window file browser for plate directory selection."""
        from openhcs.textual_tui.windows import open_file_browser_window, BrowserMode
        from openhcs.textual_tui.services.file_browser_service import SelectionMode
        from openhcs.constants.constants import Backend
        from openhcs.textual_tui.utils.path_cache import get_path_cache, PathCacheKey
        from pathlib import Path

        # Get cached path for better UX - remembers last used directory
        path_cache = get_path_cache()
        initial_path = path_cache.get_initial_path(PathCacheKey.PLATE_IMPORT, Path.home())

        # Open textual-window file browser for directory selection
        await open_file_browser_window(
            app=self.app,
            file_manager=self.filemanager,
            initial_path=initial_path,
            backend=Backend.DISK,
            title="Select Plate Directory",
            mode=BrowserMode.LOAD,
            selection_mode=SelectionMode.DIRECTORIES_ONLY,
            cache_key=PathCacheKey.PLATE_IMPORT,
            on_result_callback=self._add_plate_callback,
            caller_id="plate_manager"
        )

    def _add_plate_callback(self, selected_paths) -> None:
        """Handle directory selection from file browser."""
        from pathlib import Path

        logger.debug(f"_add_plate_callback called with: {selected_paths} (type: {type(selected_paths)})")

        if selected_paths is None or selected_paths is False:
            self.app.current_status = "Plate selection cancelled"
            return

        # Handle both single path and list of paths
        if not isinstance(selected_paths, list):
            selected_paths = [selected_paths]

        added_plates = []
        current_plates = list(self.plates)

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
                'status': '?'  # Added but not initialized
            }

            current_plates.append(plate_entry)
            added_plates.append(plate_name)

        # Cache the parent directory for next time (save user navigation time)
        if selected_paths:
            from openhcs.textual_tui.utils.path_cache import get_path_cache, PathCacheKey
            # Use parent of first selected path as the cached directory
            first_path = selected_paths[0] if isinstance(selected_paths[0], Path) else Path(selected_paths[0])
            parent_dir = first_path.parent
            get_path_cache().set_cached_path(PathCacheKey.PLATE_IMPORT, parent_dir)

        # Update plates list using reactive property (triggers automatic UI update)
        self.plates = current_plates

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
        self.plates = [p for p in self.plates if p['path'] not in paths_to_delete]

        # Clean up orchestrators for deleted plates
        for path in paths_to_delete:
            if path in self.orchestrators:
                del self.orchestrators[path]

        if self.selected_plate in paths_to_delete:
            self.selected_plate = ""

        self.app.current_status = f"Deleted {len(paths_to_delete)} plate(s)"

    def action_edit_plate(self) -> None:
        if self.selected_plate:
            self.app.current_status = f"Editing plate: {self.selected_plate}"
            # Launch dual editor for plate editing
            from openhcs.textual_tui.screens.dual_editor import DualEditorScreen

            editor = DualEditorScreen(plate_path=self.selected_plate)
            self.app.push_screen(editor)

    def action_init_plate(self) -> None:
        """Handle Init Plate button - initialize selected plates."""
        # Get current selection state
        selected_items, selection_mode = self.get_selection_state()

        if selection_mode == "empty":
            logger.warning("No plates available for initialization")
            return

        # Validate all selected plates can be initialized (allow failed plates to be re-initialized)
        invalid_plates = [item for item in selected_items if item.get('status') not in ['?', '-', 'F']]
        if invalid_plates:
            names = [item['name'] for item in invalid_plates]
            logger.warning(f"Cannot initialize plates with invalid status: {', '.join(names)}")
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
        import asyncio

        for plate_data in selected_items:
            plate_path = plate_data['path']

            # Find the actual plate in self.plates (not the copy from get_selection_state)
            actual_plate = None
            for plate in self.plates:
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

                actual_plate['status'] = '-'  # Initialized
                logger.info(f"Set plate {actual_plate['name']} status to '-' (initialized)")

            except Exception as e:
                logger.error(f"Failed to initialize plate {plate_path}: {e}", exc_info=True)
                actual_plate['status'] = 'F'  # Failed
                actual_plate['error'] = str(e)

            # Update button states immediately (reactive system handles UI updates automatically)
            self._update_button_states()
            # Notify pipeline editor of status change
            self._notify_pipeline_editor_status_change(actual_plate['path'], actual_plate['status'])
            logger.info(f"Updated plate {actual_plate['name']} status")

        # Final UI update (reactive system handles this automatically when self.plates is modified)
        self._update_button_states()

        # Update status
        success_count = len([p for p in selected_items if p.get('status') == '-'])
        error_count = len([p for p in selected_items if p.get('status') == 'X'])

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

        # Validate all selected plates are initialized (allow failed plates to be re-compiled)
        uninitialized = [item for item in selected_items if item.get('status') not in ['-', 'F']]
        if uninitialized:
            names = [item['name'] for item in uninitialized]
            logger.warning(f"Cannot compile uninitialized plates: {', '.join(names)}")
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
        import asyncio

        for plate_data in selected_items:
            plate_path = plate_data['path']

            # Find the actual plate in self.plates (not the copy from get_selection_state)
            actual_plate = None
            for plate in self.plates:
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
                import copy
                from openhcs.constants.constants import VariableComponents
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
                from openhcs.core.pipeline import Pipeline
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

                # Update plate status ONLY on successful compilation
                actual_plate['status'] = 'o'  # Compiled
                logger.info(f"🔥 Successfully compiled {plate_path}")

            except Exception as e:
                logger.error(f"🔥 COMPILATION ERROR: Pipeline compilation failed for {plate_path}: {e}", exc_info=True)
                actual_plate['status'] = 'F'  # Failed
                actual_plate['error'] = str(e)
                # Don't store anything in plate_compiled_data on failure

            # Update button states immediately (reactive system handles UI updates automatically)
            self._update_button_states()
            # Notify pipeline editor of status change
            self._notify_pipeline_editor_status_change(actual_plate['path'], actual_plate['status'])

        # Final UI update (reactive system handles this automatically when self.plates is modified)
        self._update_button_states()

        # Update status
        success_count = len([p for p in selected_items if p.get('status') == 'o'])
        error_count = len([p for p in selected_items if p.get('status') == 'F'])

        if error_count == 0:
            logger.info(f"Successfully compiled {success_count} plates")
        else:
            logger.warning(f"Compiled {success_count} plates, {error_count} errors")
