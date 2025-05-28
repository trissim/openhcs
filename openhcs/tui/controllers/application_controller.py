"""
Application Controller - Main TUI Coordination Layer.

Coordinates the entire TUI application lifecycle, component initialization,
and high-level state management.

🔒 Clause 295: Component Boundaries
Clear separation between application coordination and component implementation.
"""
import asyncio
import logging
from typing import Any, Dict, Optional

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.layout import Layout

from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.config.global_config import GlobalPipelineConfig
from openhcs.tui.tui_architecture import TUIState
from openhcs.tui.services.dialog_service import DialogService
from openhcs.tui.controllers.layout_controller import LayoutController

logger = logging.getLogger(__name__)


class ApplicationController:
    """
    Main application controller for the OpenHCS TUI.
    
    Coordinates:
    - Application lifecycle
    - Component initialization
    - High-level state management
    - Dialog coordination
    - Layout management
    """
    
    def __init__(self, 
                 initial_context: ProcessingContext,
                 state: TUIState,
                 global_config: GlobalPipelineConfig):
        """
        Initialize the application controller.
        
        Args:
            initial_context: Pre-configured ProcessingContext
            state: Shared TUIState instance
            global_config: Shared GlobalPipelineConfig instance
        """
        self.state = state
        self.context = initial_context
        self.global_config = global_config
        
        # Component controllers
        self.layout_controller = None
        self.dialog_service = None
        
        # Application state
        self.application = None
        self.is_initialized = False
        self.is_shutting_down = False
        
        # Initialize services
        self._initialize_services()
        
        # Register event handlers
        self._register_event_handlers()
    
    def _initialize_services(self):
        """Initialize core services."""
        # Initialize dialog service
        self.dialog_service = DialogService(self.state)
        
        # Initialize layout controller
        self.layout_controller = LayoutController(self.state, self.context, self.global_config)
    
    def _register_event_handlers(self):
        """Register application-level event handlers."""
        self.state.add_observer('exit_requested', self._handle_exit_request)
        self.state.add_observer('error', self._handle_error)
        self.state.add_observer('show_dialog_requested', self._handle_show_dialog_request)
    
    async def initialize(self):
        """Initialize the application and all components."""
        if self.is_initialized:
            logger.warning("ApplicationController: Already initialized")
            return
        
        try:
            logger.info("ApplicationController: Starting initialization...")
            
            # Initialize layout controller
            await self.layout_controller.initialize()
            
            # Create the application
            self.application = Application(
                layout=Layout(self.layout_controller.get_root_container()),
                key_bindings=self.layout_controller.get_key_bindings(),
                mouse_support=True,
                full_screen=True
            )
            
            # Initialize components asynchronously
            await self._initialize_components_async()
            
            self.is_initialized = True
            logger.info("ApplicationController: Initialization complete")
            
        except Exception as e:
            logger.error(f"ApplicationController: Initialization failed: {e}", exc_info=True)
            await self._handle_error({'message': f"Initialization failed: {str(e)}", 'source': 'ApplicationController'})
            raise
    
    async def _initialize_components_async(self):
        """Initialize components that require async setup."""
        try:
            # Initialize layout components
            await self.layout_controller.initialize_components()
            
            # Notify that initialization is complete
            await self.state.notify('application_initialized', {})
            
        except Exception as e:
            logger.error(f"ApplicationController: Component initialization failed: {e}", exc_info=True)
            raise
    
    async def run(self):
        """Run the TUI application."""
        if not self.is_initialized:
            await self.initialize()
        
        if not self.application:
            raise RuntimeError("Application not initialized")
        
        try:
            logger.info("ApplicationController: Starting TUI application...")
            await self.application.run_async()
        except Exception as e:
            logger.error(f"ApplicationController: Application run failed: {e}", exc_info=True)
            raise
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the application and clean up resources."""
        if self.is_shutting_down:
            return
        
        self.is_shutting_down = True
        logger.info("ApplicationController: Starting shutdown...")
        
        try:
            # Shutdown layout controller
            if self.layout_controller:
                await self.layout_controller.shutdown()
            
            # Shutdown dialog service
            if self.dialog_service:
                await self.dialog_service.shutdown()
            
            # Unregister event handlers
            self.state.remove_observer('exit_requested', self._handle_exit_request)
            self.state.remove_observer('error', self._handle_error)
            self.state.remove_observer('show_dialog_requested', self._handle_show_dialog_request)
            
            logger.info("ApplicationController: Shutdown complete")
            
        except Exception as e:
            logger.error(f"ApplicationController: Shutdown error: {e}", exc_info=True)
    
    async def _handle_exit_request(self, data):
        """Handle application exit requests."""
        logger.info("ApplicationController: Exit requested")
        
        # Show confirmation dialog if needed
        if not data.get('force', False):
            confirmed = await self.dialog_service.show_confirmation_dialog(
                title="Exit OpenHCS",
                message="Are you sure you want to exit?",
                default_yes=False
            )
            
            if not confirmed:
                logger.info("ApplicationController: Exit cancelled by user")
                return
        
        # Exit the application
        if self.application:
            self.application.exit()
    
    async def _handle_error(self, data):
        """Handle application errors."""
        message = data.get('message', 'Unknown error')
        source = data.get('source', 'Unknown')
        details = data.get('details', '')
        
        logger.error(f"ApplicationController: Error from {source}: {message}")
        
        # Show error dialog
        await self.dialog_service.show_error_dialog(
            title=f"Error - {source}",
            message=message,
            details=details
        )
    
    async def _handle_show_dialog_request(self, data):
        """Handle dialog show requests."""
        dialog_type = data.get('type')
        dialog_data = data.get('data', {})
        
        if dialog_type == 'error':
            await self.dialog_service.show_error_dialog(
                title=dialog_data.get('title', 'Error'),
                message=dialog_data.get('message', ''),
                details=dialog_data.get('details', '')
            )
        elif dialog_type == 'confirmation':
            result = await self.dialog_service.show_confirmation_dialog(
                title=dialog_data.get('title', 'Confirm'),
                message=dialog_data.get('message', ''),
                default_yes=dialog_data.get('default_yes', False)
            )
            # Notify result if callback provided
            callback = data.get('callback')
            if callback:
                await callback(result)
        elif dialog_type == 'info':
            await self.dialog_service.show_info_dialog(
                title=dialog_data.get('title', 'Information'),
                message=dialog_data.get('message', '')
            )
        else:
            logger.warning(f"ApplicationController: Unknown dialog type: {dialog_type}")
    
    def get_application(self) -> Optional[Application]:
        """Get the prompt_toolkit Application instance."""
        return self.application
    
    def get_state(self) -> TUIState:
        """Get the TUI state."""
        return self.state
    
    def get_context(self) -> ProcessingContext:
        """Get the processing context."""
        return self.context
    
    def is_running(self) -> bool:
        """Check if the application is running."""
        return self.is_initialized and not self.is_shutting_down
