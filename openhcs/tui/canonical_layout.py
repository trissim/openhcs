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

        # Store the original layout for dialog management
        self.main_layout = self._create_canonical_layout()

        # Create key bindings
        self.kb = self._create_key_bindings()

        # Create the application - CRITICAL: mouse_support=True for visual programming
        self.application = Application(
            layout=Layout(self.main_layout),
            key_bindings=self.kb,
            mouse_support=True,  # ESSENTIAL for clickable buttons in visual programming
            full_screen=True
        )

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

        # Create orchestrator-aware command wrappers
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

        # Use enhanced compilation command for better validation and error handling
        from openhcs.tui.enhanced_compilation_command import EnhancedCompilePlatesCommand
        enhanced_compile_command = EnhancedCompilePlatesCommand(orchestrator_manager=self.orchestrator_manager)
        command_registry.register("compile_plates", enhanced_compile_command)

        # TODO: Add similar wrapper for run command
        self._register_basic_orchestrator_commands()  # Fallback for run command for now

    def _register_basic_orchestrator_commands(self):
        """Register basic orchestrator commands (fallback when no orchestrator manager)."""
        from openhcs.tui.commands import command_registry
        from openhcs.tui.commands.pipeline_commands import (
            InitializePlatesCommand, CompilePlatesCommand, RunPlatesCommand
        )

        # Register basic commands (these will show "no orchestrators available" messages)
        # Note: Don't override initialize_plates and compile_plates if they're already registered
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
        
        # Complete layout
        return HSplit([
            top_bar,        # Row 1: [Quit] [Global Settings] [Help] | OpenHCS V1.0
            main_content,   # Row 2: Dual panes (each with title, buttons, list)
            status_bar      # Row 3: Status bar
        ])
    
    def _create_top_bar(self) -> Container:
        """Create the top menu bar using PRODUCTION MenuBar."""
        try:
            # Import the production menu bar
            from openhcs.tui.menu_bar import MenuBar

            # Create the production menu bar
            self.menu_bar = MenuBar(state=self.state, context=self.context)

            # Return the production container
            return self.menu_bar.container

        except ImportError as e:
            logger.error(f"Failed to import MenuBar: {e}")
            # Fallback to simple container
            return Frame(
                VSplit([
                    Button(text="Quit", handler=self._handle_quit),
                    Button(text="Global Settings", handler=self._handle_global_settings),
                    Button(text="Help", handler=self._handle_help),
                    Window(width=Dimension(weight=1)),  # Spacer
                    Label("OpenHCS V1.0 (MenuBar Fallback)"),
                ], padding=1),
                height=3
            )
        except Exception as e:
            logger.error(f"Error creating MenuBar: {e}")
            # Fallback to simple container
            return Frame(
                VSplit([
                    Label(f"MenuBar Error: {str(e)}"),
                    Window(width=Dimension(weight=1)),  # Spacer
                    Label("OpenHCS V1.0"),
                ], padding=1),
                height=3
            )
    
    def _create_main_content(self) -> Container:
        """Create the main content area with dual panes, each self-contained."""
        # Left pane: Plate Manager
        left_pane = self._create_plate_manager_pane()

        # Right pane: Pipeline Editor
        right_pane = self._create_pipeline_editor_pane()

        return VSplit([
            left_pane,
            right_pane,
        ], style="class:main-content")

    def _create_plate_manager_pane(self) -> Container:
        """Create the complete Plate Manager pane using PRODUCTION PlateManagerPane."""
        try:
            # Import the production plate manager
            from openhcs.tui.plate_manager_refactored import PlateManagerPane

            # Use the real storage registry from the launcher
            storage_registry = self.storage_registry

            if not storage_registry:
                logger.warning("No storage registry provided - PlateManagerPane may not function properly")

            # Create the production plate manager with REAL storage registry
            self.plate_manager = PlateManagerPane(
                state=self.state,
                context=self.context,
                storage_registry=storage_registry
            )

            # Create a dynamic container that shows loading state until initialization completes
            from prompt_toolkit.layout.containers import DynamicContainer

            self._plate_manager_container = None

            def get_plate_manager_container():
                """Get the current plate manager container."""
                if self._plate_manager_container:
                    return self._plate_manager_container
                else:
                    return Frame(
                        HSplit([Label("Loading Plate Manager...")]),
                        title="Plate Manager (Loading)"
                    )

            # Create the dynamic container
            dynamic_container = DynamicContainer(get_plate_manager_container)

            # Initialize the plate manager asynchronously and update container when ready
            import asyncio

            async def initialize_plate_manager():
                """Initialize plate manager and update the container."""
                try:
                    await self.plate_manager.initialize_and_refresh()
                    self._plate_manager_container = self.plate_manager.get_container()

                    # Initialize coordination bridge if orchestrator manager is available
                    await self._initialize_coordination_bridge()

                    # Invalidate the application to refresh the UI
                    from prompt_toolkit.application import get_app
                    get_app().invalidate()

                    logger.info("PlateManagerPane: Async initialization complete")

                except Exception as e:
                    logger.error(f"Error initializing PlateManagerPane: {e}")
                    self._plate_manager_container = Frame(
                        HSplit([Label(f"PlateManager Init Error: {str(e)}")]),
                        title="Plate Manager (Init Error)"
                    )
                    # Invalidate the application to show the error
                    from prompt_toolkit.application import get_app
                    get_app().invalidate()

            # Start the async initialization
            asyncio.create_task(initialize_plate_manager())

            # Return the dynamic container immediately
            return dynamic_container

        except ImportError as e:
            logger.error(f"Failed to import PlateManagerPane: {e}")
            # Fallback to simple container
            return Frame(
                HSplit([Label("PlateManagerPane not available")]),
                title="Plate Manager (Fallback)"
            )
        except Exception as e:
            logger.error(f"Error creating PlateManagerPane: {e}")
            # Fallback to simple container
            return Frame(
                HSplit([Label(f"PlateManager Error: {str(e)}")]),
                title="Plate Manager (Error)"
            )

    def _create_pipeline_editor_pane(self) -> Container:
        """Create the complete Pipeline Editor pane using PRODUCTION PipelineEditorPane."""
        try:
            # Import the production pipeline editor
            from openhcs.tui.pipeline_editor import PipelineEditorPane
            from prompt_toolkit.layout.containers import DynamicContainer

            # Create a dynamic container that will be updated when the async editor is ready
            self._pipeline_editor_container = None

            def get_pipeline_editor_container():
                """Get the current pipeline editor container."""
                if self._pipeline_editor_container:
                    return self._pipeline_editor_container
                else:
                    return Frame(
                        HSplit([Label("Loading Pipeline Editor...")]),
                        title="Pipeline Editor (Loading)"
                    )

            # Create the dynamic container
            dynamic_container = DynamicContainer(get_pipeline_editor_container)

            # Create the production pipeline editor asynchronously
            import asyncio

            async def create_pipeline_editor():
                """Create pipeline editor asynchronously and update the container."""
                try:
                    self.pipeline_editor = await PipelineEditorPane.create(self.state, self.context)
                    self._pipeline_editor_container = self.pipeline_editor.container

                    # Invalidate the application to refresh the UI
                    from prompt_toolkit.application import get_app
                    get_app().invalidate()

                    logger.info("PipelineEditorPane: Async initialization complete")

                except Exception as e:
                    logger.error(f"Error creating PipelineEditorPane: {e}")
                    self._pipeline_editor_container = Frame(
                        HSplit([Label(f"PipelineEditor Error: {str(e)}")]),
                        title="Pipeline Editor (Error)"
                    )
                    # Invalidate the application to show the error
                    from prompt_toolkit.application import get_app
                    get_app().invalidate()

            # Start the async creation
            asyncio.create_task(create_pipeline_editor())

            # Return the dynamic container immediately
            return dynamic_container

        except ImportError as e:
            logger.error(f"Failed to import PipelineEditorPane: {e}")
            # Fallback to simple container
            return Frame(
                HSplit([Label("PipelineEditorPane not available")]),
                title="Pipeline Editor (Fallback)"
            )
        except Exception as e:
            logger.error(f"Error creating PipelineEditorPane: {e}")
            # Fallback to simple container
            return Frame(
                HSplit([Label(f"PipelineEditor Error: {str(e)}")]),
                title="Pipeline Editor (Error)"
            )
    
    def _create_status_bar(self) -> Container:
        """Create the status bar using PRODUCTION StatusBar."""
        try:
            # Import the production status bar
            from openhcs.tui.status_bar import StatusBar

            # Create the production status bar
            self.status_bar = StatusBar(tui_state=self.state)

            # Return the production container
            return self.status_bar.container

        except ImportError as e:
            logger.error(f"Failed to import StatusBar: {e}")
            # Fallback to simple container
            return Frame(
                Window(
                    content=FormattedTextControl(
                        lambda: f"Status: {self.status_message} (StatusBar Fallback)"
                    ),
                    height=1
                ),
                height=3
            )
        except Exception as e:
            logger.error(f"Error creating StatusBar: {e}")
            # Fallback to simple container
            return Frame(
                Window(
                    content=FormattedTextControl(
                        lambda: f"StatusBar Error: {str(e)}"
                    ),
                    height=1
                ),
                height=3
            )
    
    # NOTE: Top bar button handlers are now handled by the PRODUCTION MenuBar
    # The production component has its own menu structure and command integration

    # NOTE: Plate manager button handlers are now handled by the PRODUCTION PlateManagerPane
    # The production component has its own MVC architecture with proper button handling

    # NOTE: Pipeline editor button handlers are now handled by the PRODUCTION PipelineEditorPane
    # The production component has its own command integration and button handling

    def _handle_edit_step(self):
        """Handle Edit Step button."""
        logger.info("Edit Step clicked")

        # Check if we have a pipeline with steps
        if not hasattr(self.state, 'current_pipeline_definition') or not self.state.current_pipeline_definition:
            self._update_status("Edit Step: No steps to edit")
            return

        # Check if a step is selected
        selected_index = getattr(self.state, 'selected_step_index', -1)
        if selected_index < 0 or selected_index >= len(self.state.current_pipeline_definition):
            self._update_status("Edit Step: No step selected - click on a step first")
            return

        try:
            # Get the selected step
            selected_step = self.state.current_pipeline_definition[selected_index]

            # Import the PRODUCTION dual step/func editor
            from openhcs.tui.dual_step_func_editor import DualStepFuncEditorPane

            # Create the PRODUCTION editor with the selected step
            self.step_editor = DualStepFuncEditorPane(
                state=self.state,
                func_step=selected_step
            )

            # Replace the plate manager pane with the step editor
            self._show_step_editor()

            step_info = f"Step {selected_index + 1}"
            if hasattr(selected_step, 'name') and selected_step.name:
                step_info = f"{step_info}: {selected_step.name}"
            self._update_status(f"Edit Step: Editing {step_info}")

        except ImportError:
            self._update_status("Edit Step: Step editor not available")
            logger.error("DualStepFuncEditor not found - step editor not implemented")
        except Exception as e:
            logger.error(f"Error opening step editor: {e}")
            self._update_status(f"Edit Step: Error - {str(e)}")

    def _show_step_editor(self):
        """Replace the plate manager pane with the step editor."""
        try:
            # Store the original main content for restoration
            if not hasattr(self, '_original_main_content'):
                self._original_main_content = self._create_main_content()

            # Create PRODUCTION step editor if not already created
            if not hasattr(self, 'step_editor'):
                try:
                    # Import the production step editor
                    from openhcs.tui.dual_step_func_editor import DualStepFuncEditorPane

                    # Create the production step editor
                    self.step_editor = DualStepFuncEditorPane(
                        state=self.state,
                        context=self.context,
                        on_save=self._on_step_editor_save,
                        on_cancel=self._on_step_editor_cancel
                    )

                    logger.info("DualStepFuncEditorPane: Created successfully")

                except ImportError as e:
                    logger.error(f"Failed to import DualStepFuncEditorPane: {e}")
                    # Create fallback container
                    from prompt_toolkit.widgets import Label
                    from prompt_toolkit.layout.containers import Frame, HSplit

                    class FallbackStepEditor:
                        def __init__(self):
                            self._container = Frame(
                                HSplit([Label("Step Editor not available - DualStepFuncEditorPane import failed")]),
                                title="Step Editor (Fallback)"
                            )

                    self.step_editor = FallbackStepEditor()

                except Exception as e:
                    logger.error(f"Error creating DualStepFuncEditorPane: {e}")
                    # Create error container
                    from prompt_toolkit.widgets import Label
                    from prompt_toolkit.layout.containers import Frame, HSplit

                    class ErrorStepEditor:
                        def __init__(self, error):
                            self._container = Frame(
                                HSplit([Label(f"Step Editor Error: {str(error)}")]),
                                title="Step Editor (Error)"
                            )

                    self.step_editor = ErrorStepEditor(e)

            # Create new main content with step editor replacing plate manager
            right_pane = self._create_pipeline_editor_pane()  # Keep pipeline editor

            # Replace left pane with PRODUCTION step editor
            main_content_with_editor = VSplit([
                self.step_editor._container,  # PRODUCTION step editor replaces plate manager
                right_pane,
            ], style="class:main-content")

            # Update the layout
            new_layout = HSplit([
                self._create_top_bar(),
                main_content_with_editor,
                self._create_status_bar()
            ])

            self.application.layout = Layout(new_layout)
            self.application.invalidate()

        except Exception as e:
            logger.error(f"Error showing step editor: {e}")
            self._update_status(f"Step Editor: Error - {str(e)}")

    def _on_step_editor_save(self, updated_step, step_index):
        """Handle step editor save - update the pipeline and restore layout."""
        try:
            # Update the step in the pipeline
            if hasattr(self.state, 'current_pipeline_definition') and self.state.current_pipeline_definition:
                if 0 <= step_index < len(self.state.current_pipeline_definition):
                    self.state.current_pipeline_definition[step_index] = updated_step
                    self._update_status(f"Edit Step: Saved changes to step {step_index + 1}")
                else:
                    self._update_status("Edit Step: Invalid step index")

            # Restore the original layout
            self._restore_main_layout()

        except Exception as e:
            logger.error(f"Error saving step: {e}")
            self._update_status(f"Edit Step: Save error - {str(e)}")

    def _on_step_editor_cancel(self):
        """Handle step editor cancel - restore layout without saving."""
        try:
            self._restore_main_layout()
            self._update_status("Edit Step: Cancelled")

        except Exception as e:
            logger.error(f"Error cancelling step editor: {e}")
            self._update_status(f"Edit Step: Cancel error - {str(e)}")

    def _restore_main_layout(self):
        """Restore the original main layout."""
        try:
            # Restore the original layout
            restored_layout = HSplit([
                self._create_top_bar(),
                self._create_main_content(),
                self._create_status_bar()
            ])

            self.application.layout = Layout(restored_layout)
            self.application.invalidate()

            # Clean up step editor reference
            if hasattr(self, 'step_editor'):
                delattr(self, 'step_editor')
            if hasattr(self, '_original_main_content'):
                delattr(self, '_original_main_content')

        except Exception as e:
            logger.error(f"Error restoring layout: {e}")

    # NOTE: Load/Save pipeline handlers are now handled by the PRODUCTION PipelineEditorPane
    # The production component has its own load/save command integration

    def _show_dialog(self, dialog_container: Container):
        """Show a dialog as a floating window - Improved version."""
        try:
            # Store current layout for restoration
            if not hasattr(self, '_previous_layout'):
                self._previous_layout = self.application.layout

            # Create a float container with the dialog
            float_container = FloatContainer(
                content=self.main_layout,
                floats=[
                    Float(
                        content=Frame(dialog_container, title="Dialog"),
                        transparent=False,
                        width=80,
                        height=20,
                    )
                ]
            )

            # Update the application layout
            self.application.layout = Layout(float_container)
            self.application.invalidate()

        except Exception as e:
            logger.error(f"Error showing dialog: {e}")
            self._update_status(f"Dialog Error: {str(e)}")

    def _hide_dialog(self):
        """Hide the current dialog and restore main layout - Improved version."""
        try:
            # Restore previous layout or default to main layout
            if hasattr(self, '_previous_layout'):
                self.application.layout = self._previous_layout
                delattr(self, '_previous_layout')
            else:
                self.application.layout = Layout(self.main_layout)

            self.application.invalidate()

        except Exception as e:
            logger.error(f"Error hiding dialog: {e}")
            # Fallback to main layout
            self.application.layout = Layout(self.main_layout)
            self.application.invalidate()

    # NOTE: List display methods are now handled by the PRODUCTION components
    # PlateManagerPane and PipelineEditorPane have their own list rendering

    def _update_status(self, message: str):
        """Update the status bar with a message."""
        self.status_message = message
        logger.info(f"Status: {message}")
        # Trigger a redraw
        if hasattr(self.application, 'invalidate'):
            self.application.invalidate()

    async def _initialize_coordination_bridge(self):
        """Initialize the plate-orchestrator coordination bridge."""
        try:
            # Only initialize if we have both orchestrator manager and plate manager
            if not self.orchestrator_manager or not hasattr(self, 'plate_manager'):
                logger.info("Coordination bridge not initialized - missing orchestrator_manager or plate_manager")
                return

            # Import the coordination bridge
            from openhcs.tui.plate_orchestrator_bridge import PlateOrchestratorCoordinationBridge

            # Create the coordination bridge
            self.coordination_bridge = PlateOrchestratorCoordinationBridge(
                plate_manager_pane=self.plate_manager,
                orchestrator_manager=self.orchestrator_manager,
                tui_state=self.state
            )

            # Initialize the bridge (registers event observers)
            await self.coordination_bridge.initialize()

            logger.info("PlateOrchestratorCoordinationBridge: Successfully initialized")

        except ImportError as e:
            logger.error(f"Failed to import PlateOrchestratorCoordinationBridge: {e}")
        except Exception as e:
            logger.error(f"Error initializing coordination bridge: {e}", exc_info=True)

    async def shutdown(self):
        """Shutdown the canonical layout and all components."""
        try:
            # Shutdown coordination bridge
            if self.coordination_bridge:
                await self.coordination_bridge.shutdown()
                logger.info("Coordination bridge shut down")

            # Shutdown enhanced compilation command
            from openhcs.tui.commands import command_registry
            enhanced_compile_command = command_registry.get("compile_plates")
            if enhanced_compile_command and hasattr(enhanced_compile_command, 'shutdown'):
                await enhanced_compile_command.shutdown()
                logger.info("Enhanced compilation command shut down")

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
