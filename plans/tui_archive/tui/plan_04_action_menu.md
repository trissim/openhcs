# plan_04_action_menu.md
## Component: Action Menu Pane

### Objective
Implement the Action Menu pane that displays buttons for key operations ([add], [pre-compile], [compile], [run], [save], [test], [settings]), wiring each to stubbed callbacks ready for later implementation.

### Plan

1. **Button Layout and Organization**
   - Implement a vertical layout with consistent button sizing
   - Group related actions with visual separators
   - Display keyboard shortcuts inline with button labels
   - Ensure adequate spacing for touch/mouse interaction
   - Provide clear visual hierarchy of operations

2. **Action Callbacks with Async Support**
   - Create non-blocking async callbacks for all operations
   - Use `get_app().create_background_task()` for long-running tasks
   - Implement state-dependent button enabling/disabling with Condition
   - Provide visual feedback during operations (spinners, progress)
   - Handle errors gracefully with user-friendly messages

3. **Compilation Integration with Validation**
   - Connect [pre-compile] to orchestrator initialization
   - Connect [compile] to pipeline-compile API with validation
   - Display detailed compilation status and errors in banner
   - Disable [run] button until compilation succeeds
   - Implement clear visual indicators of compilation state

4. **Save Warning and Error Handling**
   - Block [save] button if compilation fails with visual indication
   - Display warning banner with specific error details
   - Provide clear, actionable error messages
   - Include confirmation dialog for potentially destructive actions
   - Maintain consistent error display format

5. **Settings Dialog with Live Preview**
   - Implement modal settings dialog with categorized options
   - Include logging, display, editor, and behavior preferences
   - Save settings to configuration file with validation
   - Apply settings immediately with live preview
   - Provide reset to defaults option

### Findings

#### Key Considerations for Action Menu

1. **🔒 Button State Management**
   - Buttons must reflect valid state transitions only
   - Disabled states must clearly indicate why the action is unavailable
   - Implementation: `Button(text="Run", handler=run_handler, disabled=Condition(lambda: not self.state.is_compiled))`
   - Rationale: Prevents invalid operations and maintains declarative principles

2. **🔒 Compilation Integration (Clause 92)**
   - Pre-compile initializes orchestrator with plate configuration
   - Compile validates pipeline structure before execution
   - Implementation: `await self.context.compile_pipeline(self.state.selected_plate['id'])`
   - Rationale: Enforces validation before execution per Clause 92

3. **🔒 Error Handling and Display**
   - Errors must be displayed with specific, actionable information
   - Error banner must be consistently positioned above buttons
   - Implementation: `self.error_banner.text = HTML(f"<ansired>ERROR:</ansired> {error_message}")`
   - Rationale: Provides clear feedback for validation failures

4. **🔒 Visual Feedback for Async Operations**
   - All long-running operations must provide visual feedback
   - Status indicators must be consistent across the TUI
   - Implementation: `self.status_indicator.text = "◔ Compiling..." if self.is_compiling else ""`
   - Rationale: Maintains UI responsiveness during background tasks

5. **🔒 Settings Management**
   - Settings must be validated before being applied
   - Changes must be applied immediately with clear feedback
   - Implementation: `self.config.update_and_save(validated_settings)`
   - Rationale: Ensures configuration changes are valid and take effect

### Implementation Draft

```python
"""
Action Menu Pane for OpenHCS TUI.

This module implements the right pane of the OpenHCS TUI, which displays
buttons for key operations and handles pipeline compilation, execution,
and configuration.

🔒 Clause 92: Structural Validation First
All pipeline operations validate structure before execution.
"""
import asyncio
import os
from typing import List, Optional, Dict # Add Optional, Dict
from dataclasses import dataclass, field # Add dataclass, field

from prompt_toolkit.layout import HSplit, VSplit, Container
from prompt_toolkit.widgets import Button, Label, Box, Dialog, TextArea
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.filters import Condition
from prompt_toolkit.application import get_app
from prompt_toolkit.layout.containers import Float
from prompt_toolkit.widgets import Checkbox

from ezstitcher.core.context.processing_context import ProcessingContext
from ezstitcher.core.pipeline.pipeline import PipelineCompiler, PipelineExecutor

# 🔒 Clause 3: Declarative Primacy - Explicit state definition
@dataclass(slots=True)
class ActionMenuPaneState:
    """
    Defines the expected state structure for ActionMenuPane.
    All defaults are explicit and defined at compile time.
    """
    operation_status: Dict[str, str] = field(default_factory=lambda: {
        'compile': 'idle', 'run': 'idle', 'save': 'idle', 'test': 'idle'
    })
    vim_mode: bool = False
    editor_path: str = ""
    log_level: str = "INFO"
    
    is_compiled: bool = False
    error_message: Optional[str] = None  # Explicit None default per Clause 3
    selected_plate: Optional[Dict] = None
    # Add other state attributes ActionMenuPane interacts with if they are part of shared state.
    # For example, if the TUI application state has these and passes them through:
    # steps: List[Dict] = field(default_factory=list)
    # selected_step_id: Optional[str] = None


class ActionMenuPane:
    """
    Right pane for action menu in the OpenHCS TUI.

    Displays buttons for key operations ([add], [pre-compile], [compile],
    [run], [save], [test], [settings]), with state-dependent enabling/disabling
    and visual feedback for long-running operations.
    """
    VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] # Class attribute

    def __init__(self, state: ActionMenuPaneState, context: ProcessingContext):
        """
        Initialize the Action Menu pane.

        Args:
            state: An instance of ActionMenuPaneState. Its attributes are assumed
                   to be initialized with defaults as defined in the ActionMenuPaneState dataclass.
            context: The OpenHCS ProcessingContext.
        """
        self.state = state
        self.context = context
        self.compiler = PipelineCompiler()
        self.executor = PipelineExecutor()
        self.is_running: bool = False  # Internal component flag

        # The 'state' object is expected to arrive fully formed as per ActionMenuPaneState.
        # No conditional attribute creation (hasattr) or default setting within __init__ for shared state.
        # Example runtime validation (if editor_path *must* be set by a config loader):
        # if not self.state.editor_path:
        #     raise ValueError("ActionMenuPane: editor_path must be configured.")

        # Register for operation status changes
        self.state.add_observer('operation_status_changed', self._on_operation_status_changed)

        # Create UI components
        self.error_banner = self._create_error_banner()
        self.status_indicator = self._create_status_indicator()
        self.buttons = self._create_buttons()

        # Create container
        self.container = HSplit([
            self.error_banner,
            self.status_indicator,
            HSplit(self.buttons)
        ])

        # Register for events
        self.state.add_observer('plate_selected', self._on_plate_selected)
        self.state.add_observer('step_selected', self._on_step_selected)
        self.state.add_observer('steps_reordered', self._on_steps_reordered)

    def _create_error_banner(self) -> Label:
        """
        Create the error banner component.

        Returns:
            A Label for displaying error messages
        """
        return Label(
            text="[STATUS UNINITIALIZED]",
            dont_extend_height=True,
            style="bg:darkred fg:white"
        )

    def _create_status_indicator(self) -> Label:
        """
        Create the status indicator component.

        Returns:
            A Label for displaying operation status
        """
        return Label(
            text="[STATUS UNINITIALIZED]",
            dont_extend_height=True,
            style="bg:darkred fg:white"
        )

    def _update_error_banner(self):
        """Update the error banner with current error message."""
        if not self.state.error_message:
            self.error_banner.text = ""
            return

        self.error_banner.text = HTML(f"<ansired>ERROR:</ansired> {self.state.error_message}")

    def _update_status_indicator(self):
        """
        Update the status indicator with current operation status.

        Uses the centralized operation_status tracker in state to ensure
        UI accurately reflects the actual backend task status.
        """
        # Get current operation status
        op_status = self.state.operation_status

        # Define display text for each status and operation
        # 🔒 Clause 52: Semantic Representation - Using more descriptive status messages
        status_map = {
            'compile': {
                'running': "◔ Compiling pipeline...", 'pending': "⋯ Preparing compilation...",
                'error': "✗ Compilation failed", 'success': "✓ Compilation successful", 'idle': ""
            },
            'run': {
                'running': "◕ Running pipeline...", 'pending': "⋯ Preparing execution...",
                'error': "✗ Execution failed", 'success': "✓ Execution successful", 'idle': ""
            },
            'save': {
                'running': "◑ Saving pipeline...", 'pending': "⋯ Preparing to save...",
                'error': "✗ Save failed", 'success': "✓ Save successful", 'idle': ""
            },
            'test': {
                'running': "◐ Testing pipeline...", 'pending': "⋯ Preparing tests...",
                'error': "✗ Tests failed", 'success': "✓ Tests successful", 'idle': ""
            }
        }
        
        # Determine current message based on priority
        # 🔒 Clause 88: No Inferred Capabilities - Explicitly check for key existence
        current_message = ""
        priority_ops = ['compile', 'run', 'save', 'test'] # Operations in order of display priority
        
        for op in priority_ops:
            op_state = op_status.get(op, 'idle') # Default to 'idle' if key is missing
            if op_state != 'idle' and op_state != 'success': # Prioritize active/error states
                current_message = status_map.get(op, {}).get(op_state, f"Unknown {op} state: {op_state}")
                break
        
        if not current_message: # If no active/error states, show latest success or idle
            for op in priority_ops:
                op_state = op_status.get(op, 'idle')
                if op_state == 'success':
                    current_message = status_map.get(op, {}).get(op_state, "")
                    break # Show first success message
            if not current_message: # All are idle or unknown
                 # Check if any operation is not 'idle' and not in status_map (new unhandled status)
                unknown_status_ops = [
                    f"{op}:{op_status.get(op)}" for op in priority_ops
                    if op_status.get(op, 'idle') not in status_map.get(op, {})
                ]
                if unknown_status_ops:
                    current_message = f"Unknown status: {', '.join(unknown_status_ops)}"
                else: # All known and idle
                    current_message = ""


        self.status_indicator.text = current_message
        # Reset style to default if no specific error/warning style is needed
        if "✗" not in current_message and "Unknown status" not in current_message:
             self.status_indicator.style = ""
        elif "Unknown status" in current_message or "✗" in current_message:
             self.status_indicator.style = "bg:darkred fg:white"

    def _on_operation_status_changed(self, data):
        """
        Handle operation status change event.

        Args:
            data: Dictionary with operation and status
        """
        # 🔒 Clause 92: Structural Validation First
        VALID_OPERATIONS = {'compile', 'run', 'save', 'test'}
        VALID_STATUSES = {'idle', 'pending', 'running', 'success', 'error'}

        if not data or not isinstance(data, dict):
            self.state.error_message = "Invalid status change data: not a dict"
            self._update_ui()
            return

        operation = data.get('operation')
        status = data.get('status')

        if operation not in VALID_OPERATIONS:
            self.state.error_message = f"Invalid operation for status change: {operation}"
            self._update_ui()
            return
        
        if status not in VALID_STATUSES:
            self.state.error_message = f"Invalid status for {operation}: {status}"
            self._update_ui()
            return

        # Update operation status
        self.state.operation_status[operation] = status

        # Update UI
        self._update_ui()

    async def _reset_status_after_delay(self, operation: str, delay: float = 5.0):
        """Reset operation status after delay to avoid lingering error states."""
        await asyncio.sleep(delay)
        self.state.operation_status[operation] = 'idle'
        self.state.notify('operation_status_changed', operation)
        self.error_banner.text = ""

    def _create_buttons(self) -> List[Container]:
        """
        Create the action buttons with keyboard shortcuts.

        Returns:
            List of Button widgets in containers
        """
        # Create buttons with handlers and keyboard shortcuts
        add_button = Button(
            "Add (Ctrl+A)",
            handler=lambda: get_app().create_background_task(self._add_handler())
        )

        pre_compile_button = Button(
            "Pre-compile (F5)",
            handler=lambda: get_app().create_background_task(self._pre_compile_handler())
        )

        compile_button = Button(
            "Compile (F6)",
            handler=lambda: get_app().create_background_task(self._compile_handler())
        )

        run_button = Button(
            "Run (F7)",
            handler=lambda: get_app().create_background_task(self._run_handler()),
            disabled=Condition(lambda: not self.state.is_compiled or self.is_running)
        )

        save_button = Button(
            "Save (Ctrl+S)",
            handler=lambda: get_app().create_background_task(self._save_handler()),
            disabled=Condition(lambda: not self.state.is_compiled)
        )

        test_button = Button(
            "Test (F8)",
            handler=lambda: get_app().create_background_task(self._test_handler())
        )

        settings_button = Button(
            "Settings (F9)",
            handler=lambda: get_app().create_background_task(self._settings_handler())
        )

        # Create separators for visual grouping
        separator1 = Label("─" * 20)
        separator2 = Label("─" * 20)

        # Group buttons with spacing and separators
        return [
            Box(add_button, padding=1),
            separator1,
            Box(pre_compile_button, padding=1),
            Box(compile_button, padding=1),
            Box(run_button, padding=1),
            separator2,
            Box(save_button, padding=1),
            Box(test_button, padding=1),
            Box(settings_button, padding=1)
        ]

    def _on_plate_selected(self, plate):
        """
        Handle plate selection event.

        Args:
            plate: The selected plate
        """
        # Reset compilation state when plate changes
        self.state.is_compiled = False
        self.state.error_message = None
        self._update_ui()

    def _on_step_selected(self, step):
        """
        Handle step selection event.

        Args:
            step: The selected step
        """
        # Reset compilation state when steps change
        self.state.is_compiled = False
        self._update_ui()

    def _on_steps_reordered(self, data):
        """
        Handle steps reordered event.

        Args:
            data: Dictionary with pipeline_id and steps
        """
        # Reset compilation state when steps are reordered
        self.state.is_compiled = False
        self._update_ui()

    def _update_ui(self):
        """Update the UI to reflect current state."""
        self._update_error_banner()
        self._update_status_indicator()

    async def _add_handler(self):
        """Handle Add button click."""
        self.state.notify('add_requested')

    async def _pre_compile_handler(self):
        """
        Handle Pre-compile button click.

        Pre-compilation initializes the orchestrator for the selected plate,
        preparing it for full compilation.
        """
        # Check if plate is selected
        if not self.state.selected_plate:
            self.state.error_message = "No plate selected. Please select a plate first."
            self._update_ui()
            return

        # Fail fast if not implemented
        # TODO(plan_08_pipeline_compilation_logic.md): Implement orchestrator initialization
        raise NotImplementedError(
            "Pre-compile logic not implemented. See plan_08_pipeline_compilation_logic.md"
        )

        try:
            plate_id = self.state.selected_plate['id']
            
            # Update operation status to pending
            self.state.operation_status['compile'] = 'pending'
            self.state.notify('operation_status_changed', {
                'operation': 'compile',
                'status': 'pending'
            })
            self._update_ui()

            # Actual implementation would go here
            # await self.context.initialize_orchestrator(plate_id)
            
            # Update operation status to success
            self.state.error_message = None
            self.state.operation_status['compile'] = 'success'
            self.state.notify('operation_status_changed', {
                'operation': 'compile',
                'status': 'success'
            })
            self.state.notify('pre_compile_completed', {
                'plate_id': plate_id,
                'status': 'success'
            })

        except Exception as e:
            # Set error message and update operation status to error
            self.state.error_message = f"Pre-compilation failed: {str(e)}"
            self.state.operation_status['compile'] = 'error'
            self.state.notify('operation_status_changed', {
                'operation': 'compile',
                'status': 'error',
                'error': str(e)
            })
            self.state.notify('pre_compile_completed', {
                'plate_id': self.state.selected_plate['id'],
                'status': 'error',
                'error': str(e)
            })

        finally:
            # Update UI
            self._update_ui()

    async def _compile_handler(self):
        """
        Handle Compile button click.

        Compilation validates the pipeline structure and prepares it for execution.
        This implements Clause 92: Structural Validation First.
        """
        # Check if plate is selected
        if not self.state.selected_plate:
            self.state.error_message = "No plate selected. Please select a plate first."
            self._update_ui()
            return

        # Fail fast if not implemented
        # TODO(plan_08_pipeline_compilation_logic.md): Implement pipeline compilation
        raise NotImplementedError(
            "Compile logic not implemented. See plan_08_pipeline_compilation_logic.md"
        )

        try:
            plate_id = self.state.selected_plate['id']
            
            # Update operation status to pending
            self.state.operation_status['compile'] = 'pending'
            self.state.notify('operation_status_changed', {
                'operation': 'compile',
                'status': 'pending'
            })
            self._update_ui()

            # Actual implementation would go here
            # await self.context.compile_pipeline(plate_id)
            self._update_ui()

            # TODO(plan_08_pipeline_compilation_logic.md): Implement pipeline compilation
            await asyncio.sleep(0.1) # Simulate async work

            # Clear error and update operation status to success
            self.state.is_compiled = True
            self.state.error_message = None
            self.state.operation_status['compile'] = 'success'
            self.state.notify('operation_status_changed', {
                'operation': 'compile',
                'status': 'success'
            })
            self.state.notify('compile_completed', {
                'plate_id': plate_id,
                'status': 'success'
            })

        except Exception as e:
            # Set error message and update operation status to error
            self.state.is_compiled = False
            self.state.error_message = f"Compilation failed: {str(e)}"
            self.state.operation_status['compile'] = 'error'
            self.state.notify('operation_status_changed', {
                'operation': 'compile',
                'status': 'error',
                'error': str(e)
            })
            self.state.notify('compile_completed', {
                'plate_id': self.state.selected_plate['id'],
                'status': 'error',
                'error': str(e)
            })

        finally:
            # Update UI
            self._update_ui()

    async def _validate_pipeline(self) -> Tuple[bool, Optional[str]]:
        """
        Validate the pipeline structure.

        Returns:
            Tuple of (is_valid, error_message)

        Raises:
            NotImplementedError: This method is not implemented yet
        """
        # TODO(plan_09_validation_contract_integration.md): Implement structural validation
        raise NotImplementedError(
            "Structural validation must call FuncStepContractValidator and backend compiler. "
            "See plan_09_validation_contract_integration.md"
        )

    async def _run_handler(self):
        """
        Handle Run button click.

        Executes the compiled pipeline.
        """
        # Check if pipeline is compiled
        if not self.state.is_compiled:
            self.state.error_message = "Cannot run: Pipeline not compiled. Please compile first."
            self._update_ui()
            return

        # Fail fast if not implemented
        # TODO(plan_08_pipeline_compilation_logic.md): Implement pipeline execution
        raise NotImplementedError(
            "Execution logic not implemented. See plan_08_pipeline_compilation_logic.md"
        )

        try:
            plate_id = self.state.selected_plate['id']
            
            # Update operation status to pending
            self.state.operation_status['run'] = 'pending'
            self.state.notify('operation_status_changed', {
                'operation': 'run',
                'status': 'pending'
            })
            self._update_ui()

            # Actual implementation would go here
            # await self.context.execute_pipeline(plate_id)
            # self.is_running = True
            
            # Update operation status to success
            self.state.error_message = None
            self.state.operation_status['run'] = 'success'
            self.is_running = False
            self.state.notify('operation_status_changed', {
                'operation': 'run',
                'status': 'success'
            })
            self.state.notify('run_completed', {
                'plate_id': plate_id,
                'status': 'success'
            })

        except Exception as e:
            # Set error message and update operation status to error
            self.state.error_message = f"Execution failed: {str(e)}"
            self.state.operation_status['run'] = 'error'
            self.state.notify('operation_status_changed', {
                'operation': 'run',
                'status': 'error',
                'error': str(e)
            })
            self.state.notify('run_completed', {
                'plate_id': self.state.selected_plate['id'],
                'status': 'error',
                'error': str(e)
            })
        finally:
            self.is_running = False  # Always reset running state

        finally:
            # Update UI
            self._update_ui()

    async def _save_handler(self):
        """
        Handle Save button click.

        Saves the compiled pipeline configuration.
        """
        # Check if pipeline is compiled
        if not self.state.is_compiled:
            self.state.error_message = "Cannot save: Pipeline not compiled. Please compile first."
            self._update_ui()
            return

        # Fail fast if not implemented
        # TODO(plan_08_pipeline_compilation_logic.md): Implement pipeline saving
        raise NotImplementedError(
            "Save logic not implemented. See plan_08_pipeline_compilation_logic.md"
        )

        try:
            plate_id = self.state.selected_plate['id']
            
            # Update operation status to pending
            self.state.operation_status['save'] = 'pending'
            self.state.notify('operation_status_changed', {
                'operation': 'save',
                'status': 'pending'
            })
            self._update_ui()

            # Actual implementation would go here
            # await self.context.save_pipeline(plate_id)
            
            # Update operation status to success
            self.state.error_message = None
            self.state.operation_status['save'] = 'success'
            self.state.notify('operation_status_changed', {
                'operation': 'save',
                'status': 'success'
            })
            self.state.notify('save_completed', {
                'plate_id': plate_id,
                'status': 'success'
            })

        except Exception as e:
            # Set error message and update operation status to error
            self.state.error_message = f"Save failed: {str(e)}"
            self.state.operation_status['save'] = 'error'
            self.state.notify('operation_status_changed', {
                'operation': 'save',
                'status': 'error',
                'error': str(e)
            })
            self.state.notify('save_completed', {
                'plate_id': self.state.selected_plate['id'],
                'status': 'error',
                'error': str(e)
            })

        finally:
            # Update UI
            self._update_ui()

    async def _test_handler(self):
        """
        Handle Test button click.

        Runs tests on the pipeline.
        """
        try:
            # TODO(plan_10_pipeline_testing.md): Implement pipeline testing
            raise NotImplementedError(
                "Test logic not implemented. See plan_10_pipeline_testing.md"
            )

        except Exception as e:
            # Set error message and update operation status to error
            self.state.error_message = f"Test failed: {str(e)}"
            self.state.operation_status['test'] = 'error'
            self.state.notify('operation_status_changed', {
                'operation': 'test',
                'status': 'error',
                'error': str(e)
            })
            self.state.notify('test_completed', {
                'plate_id': self.state.selected_plate['id'] if self.state.selected_plate else None,
                'status': 'error',
                'error': str(e)
            })

        finally:
            # Update UI
            self._update_ui()

    async def _settings_handler(self):
        """
        Handle Settings button click.

        Shows a settings dialog for TUI configuration.
        """
        # Create settings dialog
        dialog = self._create_settings_dialog()

        # Show dialog: append once then focus
        app = get_app()
        app.layout.container.floats.append(Float(dialog))
        get_app().layout.focus(dialog)

        # Create future for result
        future = asyncio.Future()
        self._dialog_future = future

        # Wait for result
        try:
            result = await future

            if result:
                # Apply settings
                self.state.notify('settings_updated', result)

        finally:
            # Remove dialog from floats
            if dialog in app.layout.container.floats:
                app.layout.container.floats.remove(dialog)

    # TUI Settings Schema
    VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    def _create_settings_dialog(self) -> Dialog:
        """
        Create a settings dialog.

        Returns:
            A Dialog for configuring TUI settings
        """
        # TODO(plan_11_settings_management.md): Implement proper settings management
        # This is a placeholder that will be replaced with a proper implementation

        from prompt_toolkit.widgets import Checkbox, RadioList

        # Create settings fields with proper widgets
        vim_mode_checkbox = Checkbox(
            text="Enable Vim mode",
            checked=self.state.vim_mode
        )

        # Use a proper RadioList for log level selection
        log_level_radio = RadioList(
            values=[
                (level, level) for level in self.VALID_LOG_LEVELS
            ],
            default=self.state.log_level
        )

        # Editor path input (single-line)
        editor_path_input = TextArea(
            text=self.state.editor_path,
            multiline=False
        )

        # Create dialog with proper async handling
        dialog = Dialog(
            title="Settings",
            body=HSplit([
                Label("⚠️ Settings management not fully implemented - see plan_11_settings_management.md"),
                Label(""),
                vim_mode_checkbox,
                Label("Log level:"),
                log_level_radio,
                VSplit([
                    Label("Editor path:"),
                    editor_path_input
                ])
            ]),
            buttons=[
                Button("Save", handler=lambda: get_app().create_background_task(
                    self._apply_settings(
                        vim_mode_checkbox.checked,
                        log_level_radio.current_value,
                        editor_path_input.text
                    )
                )),
                Button("Cancel", handler=lambda: self._settings_dialog_cancel())
            ],
            width=60,
            modal=True
        )

        # Create dialog
        dialog = Dialog(
            title="Settings",
            body=HSplit([
                Label("⚠️ Settings management not fully implemented - see plan_11_settings_management.md"),
                Label(""),

                # Vim mode with proper checkbox
                vim_mode_checkbox,

                # Log level with proper radio list
                Label("Log level:"),
                log_level_radio,

                # Editor path
                VSplit([
                    Label("Editor path:"),
                    editor_path_input
                ]),

                # Validation message area
                Label(""),
                Label("Note: Settings will be validated before being applied")
            ]),
            buttons=[
                Button("Save", handler=lambda: self._validate_and_save_settings({
                    'vim_mode': vim_mode_checkbox.checked,
                    'log_level': log_level_radio.current_value,
                    'editor_path': editor_path_input.text
                })),
                Button("Cancel", handler=lambda: self._settings_dialog_cancel())
            ],
            width=60,
            modal=True
        )

        return dialog

    def _validate_and_save_settings(self, settings):
        """
        Validate settings before saving them.

        Args:
            settings: Dictionary of settings to validate
        """
        # Validate log level
        if settings['log_level'] not in self.VALID_LOG_LEVELS:
            # Show error and don't close dialog
            self.state.error_message = f"Invalid log level: {settings['log_level']}"
            self._update_ui()
            return

        # Validate editor path
        if not settings['editor_path'] or len(settings['editor_path'].strip()) == 0:
            # Show error and don't close dialog
            self.state.error_message = "Editor path cannot be empty"
            self._update_ui()
            return

        # All validation passed, save settings
        if hasattr(self, '_dialog_future') and not self._dialog_future.done():
            self._dialog_future.set_result(settings)

    def _settings_dialog_cancel(self):
        """Handle settings dialog Cancel button."""
        if hasattr(self, '_dialog_future') and not self._dialog_future.done():
            self._dialog_future.set_result(None)
```
