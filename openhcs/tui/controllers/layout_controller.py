"""
Layout Controller - UI Layout Management.

Manages the overall TUI layout structure and component coordination.
Separates layout logic from application logic.

🔒 Clause 295: Component Boundaries
Clear separation between layout management and business logic.
"""
import asyncio
import logging
from typing import Any, Dict, Optional

from prompt_toolkit.layout import Container, HSplit, VSplit, Window, Dimension
from prompt_toolkit.layout.containers import DynamicContainer
from prompt_toolkit.widgets import Box, Label, Frame
from prompt_toolkit.key_binding import KeyBindings

from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.config import GlobalPipelineConfig
from openhcs.tui.tui_architecture import TUIState
from openhcs.tui.components import FramedButton
from openhcs.tui.controllers.plate_manager_controller import PlateManagerController
from openhcs.tui.controllers.menu_controller import MenuController
from openhcs.tui.services.plate_manager_service import PlateManagerService
from openhcs.tui.services.menu_service import MenuService
from openhcs.tui.views.plate_manager_view import PlateManagerView
from openhcs.tui.views.menu_view import MenuView

logger = logging.getLogger(__name__)


class LayoutController:
    """
    Controller for managing the TUI layout structure.
    
    Coordinates:
    - Overall layout structure
    - Component placement
    - Layout updates
    - Key binding management
    """
    
    def __init__(self, state: TUIState, context: ProcessingContext, global_config: GlobalPipelineConfig):
        self.state = state
        self.context = context
        self.global_config = global_config
        
        # Component controllers
        self.plate_manager_controller = None
        self.menu_controller = None
        
        # Component views
        self.plate_manager_view = None
        self.menu_view = None
        
        # Layout components
        self.root_container = None
        self.key_bindings = None
        
        # Layout state
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the layout controller and all components."""
        if self.is_initialized:
            return
        
        try:
            logger.info("LayoutController: Starting initialization...")
            
            # Initialize services
            await self._initialize_services()
            
            # Initialize controllers
            await self._initialize_controllers()
            
            # Initialize views
            await self._initialize_views()
            
            # Create layout
            self._create_layout()
            
            # Create key bindings
            self._create_key_bindings()
            
            self.is_initialized = True
            logger.info("LayoutController: Initialization complete")
            
        except Exception as e:
            logger.error(f"LayoutController: Initialization failed: {e}", exc_info=True)
            raise
    
    async def _initialize_services(self):
        """Initialize business logic services."""
        # Initialize plate manager service
        from openhcs.io.base import storage_registry
        storage_reg = storage_registry()
        self.plate_manager_service = PlateManagerService(self.context, storage_reg)
        
        # Initialize menu service
        self.menu_service = MenuService(self.state, self.context)
    
    async def _initialize_controllers(self):
        """Initialize component controllers."""
        # Initialize validation service for plate manager
        from openhcs.tui.services.plate_validation import PlateValidationService
        validation_service = PlateValidationService(self.state, self.context)
        
        # Initialize plate manager controller
        self.plate_manager_controller = PlateManagerController(
            self.state, 
            self.plate_manager_service, 
            validation_service
        )
        
        # Initialize menu controller
        self.menu_controller = MenuController(self.state, self.menu_service)
    
    async def _initialize_views(self):
        """Initialize UI view components."""
        # Initialize plate manager view
        self.plate_manager_view = PlateManagerView(self.plate_manager_controller)
        
        # Create menu structure (simplified for now)
        menu_structure = {
            'File': {
                'mnemonic': 'F',
                'items': [
                    {'type': 'command', 'label': 'New Pipeline', 'command': 'new_pipeline'},
                    {'type': 'command', 'label': 'Open Pipeline', 'command': 'open_pipeline'},
                    {'type': 'separator'},
                    {'type': 'command', 'label': 'Save Pipeline', 'command': 'save_pipeline'},
                    {'type': 'command', 'label': 'Save As...', 'command': 'save_pipeline_as'},
                    {'type': 'separator'},
                    {'type': 'command', 'label': 'Exit', 'command': 'exit'},
                ]
            },
            'Edit': {
                'mnemonic': 'E',
                'items': [
                    {'type': 'command', 'label': 'Undo', 'command': 'undo'},
                    {'type': 'command', 'label': 'Redo', 'command': 'redo'},
                    {'type': 'separator'},
                    {'type': 'command', 'label': 'Cut', 'command': 'cut'},
                    {'type': 'command', 'label': 'Copy', 'command': 'copy'},
                    {'type': 'command', 'label': 'Paste', 'command': 'paste'},
                ]
            },
            'Pipeline': {
                'mnemonic': 'P',
                'items': [
                    {'type': 'command', 'label': 'Add Step', 'command': 'add_step'},
                    {'type': 'command', 'label': 'Remove Step', 'command': 'remove_step'},
                    {'type': 'separator'},
                    {'type': 'command', 'label': 'Validate', 'command': 'validate_pipeline'},
                ]
            },
            'Run': {
                'mnemonic': 'R',
                'items': [
                    {'type': 'command', 'label': 'Run Pipeline', 'command': 'run_pipeline'},
                    {'type': 'command', 'label': 'Stop Pipeline', 'command': 'stop_pipeline'},
                ]
            },
            'Help': {
                'mnemonic': 'H',
                'items': [
                    {'type': 'command', 'label': 'Documentation', 'command': 'documentation'},
                    {'type': 'command', 'label': 'Keyboard Shortcuts', 'command': 'keyboard_shortcuts'},
                    {'type': 'separator'},
                    {'type': 'command', 'label': 'About', 'command': 'about'},
                ]
            }
        }
        
        # Initialize menu view
        self.menu_view = MenuView(self.menu_controller, menu_structure)
    
    def _create_layout(self):
        """Create the main layout structure."""
        # Top bar with menu and global buttons
        top_bar = VSplit([
            Label(" OpenHCS ", style="class:menu-bar"),
            Window(width=Dimension(weight=1)),  # Spacer
            FramedButton(text="Settings", handler=self._handle_settings),
            FramedButton(text="Help", handler=self._handle_help),
        ], height=1, style="class:menu-bar")
        
        # Left pane - Plate Manager
        left_pane = Frame(
            self.plate_manager_view.get_container(),
            title="Plate Manager",
            style="class:left-pane-frame"
        )
        
        # Right pane - Pipeline Editor (placeholder for now)
        right_pane_content = Box(
            Label("Pipeline Editor\n\nSelect a plate to begin editing pipelines."),
            padding=1
        )
        
        right_pane = Frame(
            right_pane_content,
            title="Pipeline Editor",
            style="class:right-pane-frame"
        )
        
        # Main content area
        main_content = VSplit([
            left_pane,
            right_pane
        ], style="class:main-content")
        
        # Status bar (placeholder)
        status_bar = Box(
            Label("Ready", style="class:status-bar"),
            height=1,
            style="class:status-bar"
        )
        
        # Complete layout
        self.root_container = HSplit([
            top_bar,        # Row 1: Menu bar
            main_content,   # Row 2: Main content
            status_bar      # Row 3: Status bar
        ])
    
    def _create_key_bindings(self):
        """Create global key bindings."""
        self.key_bindings = KeyBindings()
        
        # Global shortcuts
        @self.key_bindings.add('c-q')
        def _(event):
            """Quit application."""
            from prompt_toolkit.application import get_app
            app = get_app()
            app.create_background_task(self.state.notify('exit_requested', {}))
        
        @self.key_bindings.add('c-s')
        def _(event):
            """Save pipeline."""
            from prompt_toolkit.application import get_app
            app = get_app()
            app.create_background_task(self.menu_service.execute_command('save_pipeline'))
        
        @self.key_bindings.add('c-n')
        def _(event):
            """New pipeline."""
            from prompt_toolkit.application import get_app
            app = get_app()
            app.create_background_task(self.menu_service.execute_command('new_pipeline'))
    
    async def _handle_settings(self):
        """Handle settings button click."""
        await self.state.notify('show_dialog_requested', {
            'type': 'info',
            'data': {
                'title': 'Settings',
                'message': 'Settings dialog not yet implemented.'
            }
        })
    
    async def _handle_help(self):
        """Handle help button click."""
        await self.state.notify('show_dialog_requested', {
            'type': 'info',
            'data': {
                'title': 'Help',
                'message': 'Help system not yet implemented.'
            }
        })
    
    async def initialize_components(self):
        """Initialize components that require async setup."""
        try:
            # Initialize plate manager
            await self.plate_manager_controller.refresh_plates()
            
            logger.info("LayoutController: Component initialization complete")
            
        except Exception as e:
            logger.error(f"LayoutController: Component initialization failed: {e}", exc_info=True)
            raise
    
    def get_root_container(self) -> Container:
        """Get the root layout container."""
        if not self.root_container:
            raise RuntimeError("Layout not initialized")
        return self.root_container
    
    def get_key_bindings(self) -> KeyBindings:
        """Get the global key bindings."""
        if not self.key_bindings:
            raise RuntimeError("Key bindings not initialized")
        return self.key_bindings
    
    async def shutdown(self):
        """Shutdown the layout controller and all components."""
        logger.info("LayoutController: Shutting down...")
        
        try:
            # Shutdown controllers
            if self.plate_manager_controller:
                await self.plate_manager_controller.shutdown()
            
            if self.menu_controller:
                await self.menu_controller.shutdown()
            
            logger.info("LayoutController: Shutdown complete")
            
        except Exception as e:
            logger.error(f"LayoutController: Shutdown error: {e}", exc_info=True)
