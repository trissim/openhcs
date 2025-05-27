"""
Application Controller for Hybrid TUI.

Simplified from TUI2's AppController with:
- No TUIState dependencies
- Direct component management
- Simplified lifecycle management
- Focus on core functionality

Manages the main application flow and component lifecycle.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.layout import Layout, HSplit, VSplit, Container, Dimension
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import Frame, Label, Button
from prompt_toolkit.key_binding import KeyBindings

from ..interfaces.component_interfaces import ControllerInterface
from .dual_editor_controller import DualEditorController
from ..commands import (
    ShowGlobalSettingsDialogCommand, ShowHelpCommand,
    ShowAddPlateDialogCommand, DeleteSelectedPlatesCommand,
    ShowEditPlateConfigDialogCommand, InitializePlatesCommand,
    CompilePlatesCommand, RunPlatesCommand,
    AddStepCommand, DeleteSelectedStepsCommand, ShowEditStepDialogCommand,
    LoadPipelineCommand, SavePipelineCommand
)

logger = logging.getLogger(__name__)

class AppController(ControllerInterface):
    """
    Main application controller for hybrid TUI.

    Manages application lifecycle, component creation, and high-level navigation.
    """

    def __init__(self, state=None, context=None):
        """Initialize the application controller."""
        self.current_editor: Optional[DualEditorController] = None
        self.main_layout: Optional[Container] = None
        self._app: Optional[Application] = None

        # Application state
        self.is_running = False
        self.initialization_complete = False

        # Command infrastructure
        self.state = state  # TUIState instance
        self.context = context  # ProcessingContext instance

        # Initialize command instances
        self._init_commands()

    def _init_commands(self):
        """Initialize command instances for UI actions."""
        # Top bar commands
        self.show_global_settings_cmd = ShowGlobalSettingsDialogCommand()
        self.show_help_cmd = ShowHelpCommand()

        # Plate manager commands
        self.show_add_plate_cmd = ShowAddPlateDialogCommand()
        self.delete_plates_cmd = DeleteSelectedPlatesCommand()
        self.show_edit_plate_config_cmd = ShowEditPlateConfigDialogCommand()
        self.init_plates_cmd = InitializePlatesCommand()
        self.compile_plates_cmd = CompilePlatesCommand()
        self.run_plates_cmd = RunPlatesCommand()

        # Pipeline editor commands
        self.add_step_cmd = AddStepCommand()
        self.delete_steps_cmd = DeleteSelectedStepsCommand()
        self.show_edit_step_cmd = ShowEditStepDialogCommand()
        self.load_pipeline_cmd = LoadPipelineCommand()
        self.save_pipeline_cmd = SavePipelineCommand()

    async def initialize_controller(self) -> None:
        """Initialize controller and create main layout."""
        try:
            self._create_main_layout()
            self._setup_key_bindings()
            self._setup_state_observers()

            self.initialization_complete = True

            logger.info("Simplified AppController initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AppController: {e}")
            raise

    def _setup_state_observers(self):
        """Set up state observers for dynamic layout updates."""
        if self.state:
            # Subscribe to step editing events to refresh layout
            self.state.subscribe('step_editing_started', self._on_step_editing_started)
            self.state.subscribe('step_editing_stopped', self._on_step_editing_stopped)

            # Subscribe to status changes to refresh status bar
            self.state.subscribe('status_changed', self._on_status_changed)

    async def _on_step_editing_started(self, data):
        """Handle step editing started event."""
        try:
            # Refresh the left pane to show dual editor
            self._refresh_left_pane()
        except Exception as e:
            logger.error(f"Error handling step editing started: {e}")

    async def _on_step_editing_stopped(self, data):
        """Handle step editing stopped event."""
        try:
            # Refresh the left pane to show plate list
            self._refresh_left_pane()
        except Exception as e:
            logger.error(f"Error handling step editing stopped: {e}")

    async def _on_status_changed(self, data):
        """Handle status change event."""
        try:
            # Refresh the UI to update status bar
            get_app().invalidate()
        except Exception as e:
            logger.error(f"Error handling status change: {e}")

    def _refresh_left_pane(self):
        """Refresh the left pane content."""
        try:
            if self.main_layout:
                # Find the main panes VSplit and update the left pane
                main_content = self.main_layout.content.children[4]  # Main panes area
                if hasattr(main_content, 'children') and len(main_content.children) >= 3:
                    # Update left pane (index 0)
                    main_content.children[0] = self._get_left_pane()

                    # Invalidate to trigger redraw
                    get_app().invalidate()
        except Exception as e:
            logger.error(f"Failed to refresh left pane: {e}")

    # Removed old complex initialization methods - using simplified TUI now

    async def cleanup_controller(self) -> None:
        """Clean up controller resources."""
        try:
            if self.current_editor:
                await self.current_editor.cleanup_controller()
                self.current_editor = None

            self.is_running = False
            logger.info("AppController cleaned up successfully")

        except Exception as e:
            logger.error(f"Failed to cleanup AppController: {e}")

    def get_container(self) -> Container:
        """Get the main container for this controller."""
        return self.main_layout

    def _create_main_layout(self):
        """Create the main application layout - completely simplified version."""
        # No complex components - just build the layout directly
        self._build_two_pane_layout()

    def _build_two_pane_layout(self):
        """Build the canonical 3-bar TUI layout with command-driven actions."""
        from ..utils.safe_formatting import SafeLabel

        # Create the complete layout implementing the canonical 3-bar structure
        self.main_layout = Frame(
            HSplit([
                # 1st Bar: Top Menu Bar (Global Settings, Help, OpenHCS V1.0, Quit)
                VSplit([
                    Button("Global Settings", handler=self._create_command_handler(self.show_global_settings_cmd)),
                    Button("Help", handler=self._create_command_handler(self.show_help_cmd)),
                    Window(width=Dimension(weight=1)),
                    SafeLabel("OpenHCS V1.0"),
                    Window(width=Dimension(weight=1)),
                    Button("Quit", handler=lambda: get_app().exit())
                ], height=Dimension.exact(1)),

                # 2nd Bar: Titles Bar (1 Plate Manager | 2 Pipeline Editor)
                VSplit([
                    Window(
                        content=FormattedTextControl([("class:title", " 1 Plate Manager ")]),
                        height=Dimension.exact(1),
                        char=' ',
                        style="class:title",
                        width=Dimension(weight=1)
                    ),
                    Window(width=Dimension.exact(1), char='│'),
                    Window(
                        content=FormattedTextControl([("class:title", " 2 Pipeline Editor ")]),
                        height=Dimension.exact(1),
                        char=' ',
                        style="class:title",
                        width=Dimension(weight=1)
                    )
                ], height=Dimension.exact(1)),

                # 3rd Bar: Contextual Buttons Bar
                VSplit([
                    # Left: Plate Manager buttons
                    VSplit([
                        Button("add", handler=self._create_command_handler(self.show_add_plate_cmd)),
                        Button("del", handler=self._create_command_handler(self.delete_plates_cmd)),
                        Button("edit", handler=self._create_command_handler(self.show_edit_plate_config_cmd)),
                        Button("init", handler=self._create_command_handler(self.init_plates_cmd)),
                        Button("compile", handler=self._create_command_handler(self.compile_plates_cmd)),
                        Button("run", handler=self._create_command_handler(self.run_plates_cmd)),
                    ], width=Dimension(weight=1)),

                    Window(width=Dimension.exact(1), char='│'),

                    # Right: Pipeline Editor buttons
                    VSplit([
                        Button("add", handler=self._create_command_handler(self.add_step_cmd)),
                        Button("del", handler=self._create_command_handler(self.delete_steps_cmd)),
                        Button("edit", handler=self._create_command_handler(self.show_edit_step_cmd)),
                        Button("load", handler=self._create_command_handler(self.load_pipeline_cmd)),
                        Button("save", handler=self._create_command_handler(self.save_pipeline_cmd)),
                    ], width=Dimension(weight=1))
                ], height=Dimension.exact(1)),

                # Horizontal separator
                Window(height=Dimension.exact(1), char='─'),

                # Main Panes: Left=Plate List, Right=Step List (or Dual Editor)
                VSplit([
                    # Left pane: Plate list (or dual editor when editing)
                    self._get_left_pane(),

                    # Vertical separator
                    Window(width=Dimension.exact(1), char='│'),

                    # Right pane: Step list
                    self._create_step_list_container()
                ], height=Dimension(weight=1)),

                # Horizontal separator for status bar
                Window(height=Dimension.exact(1), char='─'),

                # Bottom Bar: Status bar
                self._create_status_bar()
            ], padding=0)
        )

    def _create_status_bar(self):
        """Create a dynamic status bar that updates with state changes."""
        def get_status_text():
            if self.state:
                status_msg = getattr(self.state, 'status_message', 'Ready')
                priority = getattr(self.state, 'status_priority', 'info')

                # Add color based on priority
                if priority == 'error':
                    return [("class:error", status_msg)]
                elif priority == 'warning':
                    return [("class:warning", status_msg)]
                else:
                    return [("", status_msg)]
            else:
                return [("", "Ready")]

        return Window(
            content=FormattedTextControl(get_status_text),
            height=Dimension.exact(1)
        )

    def _create_command_handler(self, command):
        """Create an async command handler for button events."""
        def handler():
            if self.state and self.context:
                # Show immediate feedback
                command_name = command.__class__.__name__.replace('Command', '')
                logger.info(f"Executing command: {command_name}")

                # Update status to show action
                get_app().create_background_task(
                    self._execute_command_with_feedback(command, command_name)
                )
            else:
                logger.warning(f"Cannot execute command {command.__class__.__name__}: missing state or context")
        return handler

    async def _execute_command_with_feedback(self, command, command_name):
        """Execute command with user feedback."""
        try:
            # Update status to show action is happening
            await self.state.set_status(f"Executing {command_name}...", "info")

            # Execute the command
            await command.execute(self.state, self.context)

            # Show success feedback
            await self.state.set_status(f"{command_name} completed successfully", "info")

        except Exception as e:
            logger.error(f"Command {command_name} failed: {e}")
            await self.state.set_status(f"{command_name} failed: {e}", "error")

    def _get_left_pane(self):
        """Get the left pane content (plate list or dual editor)."""
        # Check if we're in step editing mode
        if (self.state and
            getattr(self.state, 'editing_step_config', False) and
            getattr(self.state, 'step_to_edit', None)):

            # Return dual step/func editor
            return self._create_dual_editor_container()
        else:
            # Return plate list
            return self._create_plate_list_container()

    def _create_dual_editor_container(self):
        """Create the dual step/func editor container."""
        try:
            from ..components.dual_step_func_editor import DualStepFuncEditor

            # Create dual editor with current step
            dual_editor = DualStepFuncEditor(
                state=self.state,
                step_to_edit=self.state.step_to_edit
            )

            return dual_editor.container

        except Exception as e:
            logger.error(f"Failed to create dual editor: {e}")
            from prompt_toolkit.widgets import Label
            return Label(f"Dual editor error: {e}")

    def _create_plate_list_container(self):
        """Create the interactive plate list container."""
        from ..components import PlateListView

        # Create some demo plate data
        demo_plates = [
            {"name": "Plate_001", "path": "/data/plates/plate_001", "status": "uninitialized"},
            {"name": "Plate_002", "path": "/data/plates/plate_002", "status": "initialized"},
            {"name": "Plate_003", "path": "/data/plates/plate_003", "status": "compiled"},
        ]

        # Create the plate list view
        self.plate_list_view = PlateListView(
            on_plate_selected=self._handle_plate_selected,
            on_plate_activated=self._handle_plate_activated
        )

        # Initialize with demo data
        import asyncio
        if hasattr(self, '_app') and self._app:
            get_app().create_background_task(
                self.plate_list_view.update_plate_list(demo_plates)
            )

        return self.plate_list_view

    def _create_step_list_container(self):
        """Create the interactive step list container."""
        from ..components import StepListView

        # Create some demo step data
        demo_steps = [
            {"name": "Load Images", "type": "LoadStep", "status": "completed"},
            {"name": "Normalize", "type": "NormalizeStep", "status": "running"},
            {"name": "Stitch", "type": "StitchStep", "status": "pending"},
        ]

        # Create the step list view
        self.step_list_view = StepListView(
            on_step_selected=self._handle_step_selected,
            on_step_activated=self._handle_step_activated,
            on_step_reorder_requested=self._handle_step_reorder
        )

        # Initialize with demo data
        import asyncio
        if hasattr(self, '_app') and self._app:
            get_app().create_background_task(
                self.step_list_view.update_step_list(demo_steps)
            )

        return self.step_list_view

    async def _handle_plate_selected(self, plate_data):
        """Handle plate selection."""
        if plate_data:
            logger.info(f"Plate selected: {plate_data.get('name', 'Unknown')}")
        else:
            logger.info("No plate selected")

    async def _handle_plate_activated(self, plate_data):
        """Handle plate activation (double-click/enter)."""
        logger.info(f"Plate activated: {plate_data.get('name', 'Unknown')}")

    async def _handle_step_selected(self, step_data):
        """Handle step selection."""
        if step_data:
            logger.info(f"Step selected: {step_data.get('name', 'Unknown')}")
        else:
            logger.info("No step selected")

    async def _handle_step_activated(self, step_data):
        """Handle step activation (double-click/enter)."""
        logger.info(f"Step activated: {step_data.get('name', 'Unknown')}")

    async def _handle_step_reorder(self, index, direction):
        """Handle step reordering."""
        logger.info(f"Step reorder requested: index {index}, direction {direction}")

    def _setup_key_bindings(self):
        """Setup global key bindings."""
        kb = KeyBindings()

        @kb.add('c-q')
        def quit_app(event):
            """Quit the application."""
            event.app.create_background_task(self._handle_quit())

        @kb.add('c-o')
        def open_step_editor(event):
            """Open step editor (demo)."""
            event.app.create_background_task(self._handle_open_demo_editor())

        @kb.add('escape')
        def close_current_editor(event):
            """Close current editor."""
            if self.current_editor:
                event.app.create_background_task(self._handle_close_editor())

        # Store key bindings for application
        self.key_bindings = kb

    async def open_step_editor(self, func_step: Any) -> None:
        """
        Open step editor for given FunctionStep.

        Args:
            func_step: FunctionStep object to edit
        """
        try:
            # Close current editor if open
            if self.current_editor:
                await self.current_editor.cleanup_controller()

            # Create new editor
            self.current_editor = DualEditorController(
                func_step=func_step,
                on_save=self._on_step_saved,
                on_cancel=self._on_step_cancelled
            )

            # Initialize editor
            await self.current_editor.initialize_controller()

            # Update layout
            self.main_layout = Frame(
                self.current_editor.get_container(),
                title=f"Editing: {getattr(func_step, 'name', 'Unnamed Step')}"
            )

            # Refresh UI
            get_app().invalidate()

            logger.info(f"Opened step editor for: {getattr(func_step, 'name', 'Unnamed')}")

        except Exception as e:
            logger.error(f"Failed to open step editor: {e}")
            from ..utils.dialogs import show_error_dialog
            await show_error_dialog("Error", f"Failed to open step editor: {e}")

    async def close_step_editor(self) -> None:
        """Close the current step editor."""
        try:
            if self.current_editor:
                # Check for unsaved changes
                if self.current_editor.has_unsaved_changes():
                    from ..utils.dialogs import show_confirmation_dialog

                    confirmed = await show_confirmation_dialog(
                        "Unsaved Changes",
                        "You have unsaved changes. Close anyway?"
                    )

                    if not confirmed:
                        return

                # Cleanup editor
                await self.current_editor.cleanup_controller()
                self.current_editor = None

                # Return to welcome screen
                self._create_main_layout()
                get_app().invalidate()

                logger.info("Step editor closed")

        except Exception as e:
            logger.error(f"Failed to close step editor: {e}")

    def _on_step_saved(self, func_step: Any):
        """Handle step save callback."""
        logger.info(f"Step saved: {getattr(func_step, 'name', 'Unnamed')}")
        # Could emit events, update other components, etc.

    def _on_step_cancelled(self):
        """Handle step cancel callback."""
        logger.info("Step editing cancelled")
        # Return to main view
        get_app().create_background_task(self.close_step_editor())

    async def _handle_quit(self):
        """Handle quit request."""
        try:
            # Check for unsaved changes
            if self.current_editor and self.current_editor.has_unsaved_changes():
                from ..utils.dialogs import show_confirmation_dialog

                confirmed = await show_confirmation_dialog(
                    "Unsaved Changes",
                    "You have unsaved changes. Quit anyway?"
                )

                if not confirmed:
                    return

            # Cleanup and quit
            await self.cleanup_controller()
            get_app().exit()

        except Exception as e:
            logger.error(f"Failed to quit application: {e}")

    async def _handle_open_demo_editor(self):
        """Handle demo editor opening."""
        try:
            # Create a demo FunctionStep for testing
            demo_step = self._create_demo_step()
            await self.open_step_editor(demo_step)

        except Exception as e:
            logger.error(f"Failed to open demo editor: {e}")
            from ..utils.dialogs import show_error_dialog
            await show_error_dialog("Error", f"Failed to open demo editor: {e}")

    async def _handle_close_editor(self):
        """Handle editor close request."""
        await self.close_step_editor()

    def _create_demo_step(self) -> Any:
        """Create a demo FunctionStep for testing."""
        # Create a simple object that mimics FunctionStep
        class DemoStep:
            def __init__(self):
                self.name = "Demo Step"
                self.variable_components = ["channel"]
                self.force_disk_output = False
                self.group_by = "site"
                self.input_dir = Path("/tmp/input")
                self.output_dir = Path("/tmp/output")
                self.func = None  # Will be set by function pattern editor

        return DemoStep()

    def create_application(self) -> Application:
        """Create and return the prompt_toolkit Application."""
        if not self.initialization_complete:
            raise RuntimeError("AppController must be initialized before creating application")

        # Create styles for visual distinction
        from prompt_toolkit.styles import Style
        button_style = Style.from_dict({
            'button': 'bg:#006600 #ffffff bold',           # Green background for all buttons
            'button.focused': 'bg:#ffaa00 #000000 bold',   # Orange background when focused
            'title': 'bg:#333333 #ffffff bold',            # Dark gray background for title bars
        })

        self._app = Application(
            layout=Layout(self.main_layout),
            key_bindings=self.key_bindings,
            style=button_style,
            full_screen=True,
            mouse_support=True
        )

        return self._app

    async def run_async(self) -> None:
        """Run the application asynchronously."""
        try:
            if not self.initialization_complete:
                await self.initialize_controller()

            app = self.create_application()
            self.is_running = True

            # Force component initialization after app is created
            await self._post_app_initialization()

            logger.info("Starting hybrid TUI application")
            await app.run_async()

        except Exception as e:
            logger.error(f"Failed to run application: {e}")
            raise
        finally:
            self.is_running = False
            await self.cleanup_controller()

    async def _post_app_initialization(self):
        """Initialize components after the app is created and running."""
        try:
            # Wait for app to be fully ready
            await asyncio.sleep(0.1)

            # Simple initialization - no complex components for now
            logger.info("Hybrid TUI initialized - simplified version")

            # Force UI refresh
            if hasattr(self._app, 'invalidate'):
                self._app.invalidate()

            logger.info("Post-app initialization completed successfully")

        except Exception as e:
            logger.error(f"Failed in post-app initialization: {e}")
            import traceback
            traceback.print_exc()

    def get_key_bindings(self) -> KeyBindings:
        """Get application key bindings."""
        return self.key_bindings

    # Legacy handlers removed - now using Command Pattern
    # All button actions are handled through command instances
