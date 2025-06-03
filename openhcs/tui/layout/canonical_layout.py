"""
Canonical TUI Layout Implementation.

Implements the exact 3-bar layout specified in tui_final.md:
- Row 1: [Global Settings] [Help] | OpenHCS V1.0
- Row 2: Plate Manager | Pipeline Editor  
- Row 3: Status bar

This is a simplified, working implementation that replaces the complex
dynamic container system with a clean, static layout.
"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Container, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.containers import Float, FloatContainer
from prompt_toolkit.widgets import Box, Button, Frame, Label
from prompt_toolkit.mouse_events import MouseEventType

# Removed FramedButton dependency - using standard Button
from openhcs.constants.constants import Backend

logger = logging.getLogger(__name__)


class CanonicalTUILayout:
    """
    Canonical TUI layout implementation following tui_final.md specification.
    
    Provides a clean, working 3-bar layout without complex dynamic containers.
    """
    
    def __init__(self, state, context, global_config, orchestrator_manager=None, storage_registry=None):
        """Initialize the canonical layout."""
        self.state = state
        self.context = context
        self.global_config = global_config
        self.orchestrator_manager = orchestrator_manager  # For orchestrator integration
        self.storage_registry = storage_registry  # For storage integration

        # Status message for dynamic updates
        self.status_message = "Ready"

        # Initialize coordination bridge (will be activated after components are created)
        self.coordination_bridge = None

        # Register commands that are needed for this layout
        self._register_commands()

        self.main_layout = self._create_canonical_layout()
        self.kb = self._create_key_bindings()

        self.application = Application(
            layout=Layout(self.main_layout),
            key_bindings=self.kb,
            mouse_support=True,
            full_screen=True
        )

        # Set up dialog event handling
        self.state.add_observer('show_dialog_requested', self._handle_dialog_request)

    async def _handle_dialog_request(self, event_data):
        """Handle dialog display requests from state."""
        dialog = event_data['dialog']
        result_future = event_data['result_future']

        # Show dialog using FloatContainer infrastructure
        self._show_dialog(dialog)

        # Wait for dialog completion and hide
        try:
            await result_future
        finally:
            self._hide_dialog()

    def _show_dialog(self, dialog):
        """Show a dialog by adding it to the layout."""
        from prompt_toolkit.layout.containers import Float
        from prompt_toolkit.application import get_app

        # Get the main layout (FloatContainer) and add the dialog as a float
        layout = get_app().layout
        if hasattr(layout, 'container') and hasattr(layout.container, 'floats'):
            float_dialog = Float(content=dialog)
            layout.container.floats.append(float_dialog)
            get_app().invalidate()

    def _hide_dialog(self):
        """Hide the current dialog by removing it from the layout."""
        from prompt_toolkit.application import get_app

        layout = get_app().layout
        if hasattr(layout, 'container') and hasattr(layout.container, 'floats'):
            # Remove the last float (dialog)
            if layout.container.floats:
                layout.container.floats.pop()
                get_app().invalidate()

    def _register_commands(self):
        """Register commands needed for this layout."""
        from openhcs.tui.commands import command_registry
        from openhcs.tui.commands.pipeline_step_commands import AddStepCommand, RemoveStepCommand

        # Register orchestrator commands with proper orchestrator integration
        if self.orchestrator_manager:
            # Use orchestrator-aware commands that can get orchestrators from the manager
            self._register_orchestrator_aware_commands()
        else:
            # Fallback to basic commands (will show "no orchestrators available" messages)
            self._register_basic_orchestrator_commands()

        # Register pipeline editor commands (notification-based commands)
        command_registry.register("add_step", AddStepCommand())
        command_registry.register("remove_step", RemoveStepCommand())

        logger.info("CanonicalTUILayout: Registered orchestrator commands with proper integration")

    def _register_orchestrator_aware_commands(self):
        """Register orchestrator commands that can access the orchestrator manager."""
        from openhcs.tui.commands import command_registry

        class OrchestratorAwareInitializeCommand:
            def __init__(self, state, context, orchestrator_manager):
                self.state = state
                self.context = context
                self.orchestrator_manager = orchestrator_manager

            async def execute(self, **kwargs):
                """Execute initialize with real orchestrators."""
                orchestrators_to_init = self._get_selected_orchestrators()
                if not orchestrators_to_init:
                    await self.state.notify('operation_status_changed', {
                        'message': 'Initialize: No plates selected',
                        'status': 'error'
                    })
                    return

                # Execute real initialization
                for orchestrator in orchestrators_to_init:
                    try:
                        orchestrator.initialize()
                        await self.state.notify('operation_status_changed', {
                            'message': f'Initialize: Initialized {len(orchestrators_to_init)} orchestrator(s)',
                            'status': 'success'
                        })
                    except Exception as e:
                        await self.state.notify('operation_status_changed', {
                            'message': f'Initialize: Error - {str(e)}',
                            'status': 'error'
                        })

            def _get_selected_orchestrators(self):
                """Get orchestrators for selected plates."""
                if not hasattr(self.state, 'selected_plate') or not self.state.selected_plate:
                    return []

                # Get orchestrator for selected plate
                plate_id = self.state.selected_plate.get('id')
                if plate_id and hasattr(self.orchestrator_manager, 'get_orchestrator'):
                    orchestrator = self.orchestrator_manager.get_orchestrator(plate_id)
                    return [orchestrator] if orchestrator else []
                return []

        # Register the orchestrator-aware commands
        command_registry.register("initialize_plates",
                                 OrchestratorAwareInitializeCommand(self.state, self.context, self.orchestrator_manager))

        self._register_basic_orchestrator_commands()

    def _register_basic_orchestrator_commands(self):
        """Register basic orchestrator commands (fallback when no orchestrator manager)."""
        from openhcs.tui.commands import command_registry
        from openhcs.tui.commands.pipeline_commands import (
            InitializePlatesCommand, CompilePlatesCommand, RunPlatesCommand
        )

        if not command_registry.get("initialize_plates"):
            command_registry.register("initialize_plates", InitializePlatesCommand(self.state, self.context))
        if not command_registry.get("compile_plates"):
            command_registry.register("compile_plates", CompilePlatesCommand(self.state, self.context))

        # Always register run command as fallback (no enhanced version yet)
        command_registry.register("run_plates", RunPlatesCommand(self.state, self.context))

    def _create_key_bindings(self) -> KeyBindings:
        """Create key bindings for the application."""
        kb = KeyBindings()
        
        @kb.add('c-c')
        def _(event):
            """Exit the application."""
            event.app.exit()
        
        @kb.add('q')
        def _(event):
            """Exit the application with 'q'."""
            event.app.exit()
            
        return kb
    
    def _create_canonical_layout(self) -> Container:
        """
        Create the canonical 3-bar layout as specified in tui_final.md.
        
        Layout:
        ┌─────────────────────────────────────────────────────────────────┐
        │ [Global Settings] [Help]                    OpenHCS V1.0       │
        ├─────────────────────────────────────────────────────────────────┤
        │ Plate Manager                    │ Pipeline Editor             │
        ├─────────────────────────────────────────────────────────────────┤
        │ [add][del][edit][init][compile][run] │ [add][del][edit][load][save] │
        ├─────────────────────────────────────────────────────────────────┤
        │ Plate List                       │ Step List                   │
        │                                  │                             │
        ├─────────────────────────────────────────────────────────────────┤
        │ Status: Ready                                                   │
        └─────────────────────────────────────────────────────────────────┘
        """
        
        # Row 1: Top bar with Quit, Global Settings, Help, and title
        top_bar = self._create_top_bar()

        # Row 2: Main content area (dual panes, each self-contained)
        main_content = self._create_main_content()

        # Row 3: Status bar
        status_bar = self._create_status_bar()
        
        # Complete layout - main content takes all available height
        # Import Dimension for explicit height control
        from prompt_toolkit.layout.dimension import Dimension

        # Wrap in FloatContainer to support modal dialogs
        from prompt_toolkit.layout.containers import FloatContainer

        main_layout = HSplit([
            top_bar,        # Row 1: Fixed height (determined by MenuBar)
            main_content,   # Row 2: Takes all remaining space
            status_bar      # Row 3: Fixed height (exactly 3 lines for framed status)
        ])

        return FloatContainer(
            content=main_layout,
            floats=[]  # Will contain modal dialogs
        )
    
    def _create_top_bar(self) -> Container:
        """Create the top bar using the actual MenuBar component with fixed height."""
        from openhcs.tui.layout.menu_bar import MenuBar
        from prompt_toolkit.layout.dimension import Dimension

        self.menu_bar = MenuBar(self.state, self.context)
        # Set fixed height for menu bar (3 lines for framed buttons)
        menu_container = self.menu_bar.container
        menu_container.height = Dimension.exact(3)
        return menu_container
    
    def _create_main_content(self) -> Container:
        """Create the main content area with dual panes, each self-contained."""
        from prompt_toolkit.layout.dimension import Dimension

        # Left pane: Plate Manager
        left_pane = self._create_plate_manager_pane()

        # Right pane: Pipeline Editor
        right_pane = self._create_pipeline_editor_pane()

        # Create VSplit with equal weight distribution
        main_vsplit = VSplit([
            left_pane,
            right_pane,
        ], style="class:main-content", padding=1)
        main_vsplit.height = Dimension(weight=1)  # Take all remaining space
        return main_vsplit

    def _get_proportional_width(self, percentage: float) -> Dimension:
        """
        Get width for a pane based on percentage of terminal width.

        Args:
            percentage: 0.0 to 1.0 (0% to 100%)
            - 0.0 hides the pane
            - 1.0 takes full width
            - 0.5 takes half width

        Returns:
            Dimension with calculated preferred width and weight=0
        """
        from prompt_toolkit.application import get_app
        if get_app().output:
            terminal_width = get_app().output.get_size().columns
            calculated_width = int(terminal_width * percentage)
            return Dimension(preferred=calculated_width, weight=0)  # weight=0 prevents further distribution
        return Dimension(preferred=int(80 * percentage), weight=0)  # fallback for 80-char terminal

    def _create_plate_manager_pane(self) -> Container:
        """Create the complete Plate Manager pane using PRODUCTION PlateManagerPane."""
        from openhcs.tui.panes.plate_manager import PlateManagerPane
        from prompt_toolkit.layout.containers import DynamicContainer
        import asyncio

        if not self.storage_registry:
            logger.warning("No storage registry provided - PlateManagerPane may not function properly")

        self.plate_manager = PlateManagerPane(
            state=self.state,
            filemanager=self.context.filemanager
        )

        self._plate_manager_container = None

        def get_plate_manager_container():
            if self._plate_manager_container:
                return self._plate_manager_container
            else:
                return Frame(
                    HSplit([Label("Loading Plate Manager...")]),
                    title="Plate Manager (Loading)"
                )

        dynamic_container = DynamicContainer(get_plate_manager_container)
        dynamic_container.width = Dimension(weight=1)  # Equal weight for 50/50 split

        async def initialize_plate_manager():
            # PlateManagerPane is ready immediately - no async initialization needed
            self._plate_manager_container = self.plate_manager.container
            await self._initialize_coordination_bridge()

            from prompt_toolkit.application import get_app
            get_app().invalidate()
            logger.info("PlateManagerPane: Initialization complete")

        asyncio.create_task(initialize_plate_manager())
        return dynamic_container

    def _create_pipeline_editor_pane(self) -> Container:
        """Create the complete Pipeline Editor pane using PRODUCTION PipelineEditorPane."""
        from openhcs.tui.panes.pipeline_editor import PipelineEditorPane
        from prompt_toolkit.layout.containers import DynamicContainer
        import asyncio

        self._pipeline_editor_container = None

        def get_pipeline_editor_container():
            if self._pipeline_editor_container:
                return self._pipeline_editor_container
            else:
                return Frame(
                    HSplit([Label("Loading Pipeline Editor...")]),
                    title="Pipeline Editor (Loading)"
                )

        dynamic_container = DynamicContainer(get_pipeline_editor_container)
        dynamic_container.width = Dimension(weight=1)  # Equal weight for 50/50 split

        async def create_pipeline_editor():
            from prompt_toolkit.widgets import Label, Frame
            from prompt_toolkit.layout import HSplit
            from prompt_toolkit.application import get_app

            self.pipeline_editor = await PipelineEditorPane.create(self.state, self.context)
            # Use the pipeline editor container directly - it already has proper title and structure
            self._pipeline_editor_container = self.pipeline_editor.container

            get_app().invalidate()
            logger.info("PipelineEditorPane: Async initialization complete")

        asyncio.create_task(create_pipeline_editor())
        return dynamic_container
    
    def _create_status_bar(self) -> Container:
        """Create the status bar using PRODUCTION StatusBar with fixed height."""
        from openhcs.tui.layout.status_bar import StatusBar
        from prompt_toolkit.layout.dimension import Dimension

        self.status_bar = StatusBar(tui_state=self.state)
        # Wrap in Frame with fixed height (3 lines: top border, content, bottom border)
        framed_status = Frame(self.status_bar.container, title="Status")
        framed_status.height = Dimension.exact(3)
        return framed_status

    async def _initialize_coordination_bridge(self):
        """Initialize the plate-orchestrator coordination bridge."""
        if not self.orchestrator_manager or not hasattr(self, 'plate_manager'):
            logger.info("Coordination bridge not initialized - missing orchestrator_manager or plate_manager")
            return
        logger.info("Coordination bridge skipped - using direct integration")



    async def shutdown(self):
        """Shutdown the canonical layout and all components."""
        try:
            # Shutdown coordination bridge
            if self.coordination_bridge:
                await self.coordination_bridge.shutdown()
                logger.info("Coordination bridge shut down")

            # Shutdown commands (enhanced compilation command was removed)
            from openhcs.tui.commands import command_registry
            compile_command = command_registry.get("compile_plates")
            if compile_command and hasattr(compile_command, 'shutdown'):
                await compile_command.shutdown()
                logger.info("Compilation command shut down")

            # Shutdown plate manager if it has a shutdown method
            if hasattr(self, 'plate_manager') and hasattr(self.plate_manager, 'shutdown'):
                await self.plate_manager.shutdown()
                logger.info("Plate manager shut down")

            # Shutdown pipeline editor if it has a shutdown method
            if hasattr(self, 'pipeline_editor') and hasattr(self.pipeline_editor, 'shutdown'):
                await self.pipeline_editor.shutdown()
                logger.info("Pipeline editor shut down")

            logger.info("CanonicalTUILayout: Shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)

    async def run_async(self):
        """Run the application asynchronously."""
        await self.application.run_async()

    # Layout class should only handle layout, not menu business logic
