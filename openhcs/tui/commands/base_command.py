"""
Base Command Classes - Simplified Command Architecture.

Provides base classes for commands with clean separation of concerns.
Reduces code duplication and improves maintainability.

🔒 Clause 295: Component Boundaries
Clear separation between command interface and implementation.
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from openhcs.tui.tui_architecture import TUIState
    from openhcs.core.context.processing_context import ProcessingContext

logger = logging.getLogger(__name__)


class BaseCommand(ABC):
    """
    Base class for all TUI commands.
    
    Provides common functionality and enforces consistent interface.
    """
    
    def __init__(self, state: "TUIState" = None, context: "ProcessingContext" = None):
        """
        Initialize the base command.
        
        Args:
            state: Optional TUIState instance
            context: Optional ProcessingContext instance
        """
        self.state = state
        self.context = context
    
    @abstractmethod
    async def execute(self, state: "TUIState" = None, context: "ProcessingContext" = None, **kwargs: Any) -> None:
        """
        Execute the command.
        
        Args:
            state: TUIState instance (uses self.state if not provided)
            context: ProcessingContext instance (uses self.context if not provided)
            **kwargs: Additional command-specific arguments
        """
        pass
    
    def can_execute(self, state: "TUIState") -> bool:
        """
        Check if the command can be executed in the current state.
        
        Args:
            state: TUIState instance to check
            
        Returns:
            True if command can be executed, False otherwise
        """
        return True  # Default implementation allows execution
    
    def _get_state(self, state: "TUIState" = None) -> "TUIState":
        """Get the state to use for execution."""
        result_state = state or self.state
        if result_state is None:
            raise ValueError(f"{self.__class__.__name__}: No state available for execution")
        return result_state
    
    def _get_context(self, context: "ProcessingContext" = None) -> "ProcessingContext":
        """Get the context to use for execution."""
        result_context = context or self.context
        if result_context is None:
            raise ValueError(f"{self.__class__.__name__}: No context available for execution")
        return result_context
    
    async def _notify_error(self, state: "TUIState", message: str, details: str = ""):
        """Notify about an error."""
        await state.notify('error', {
            'source': self.__class__.__name__,
            'message': message,
            'details': details
        })
    
    async def _notify_success(self, state: "TUIState", operation: str, message: str):
        """Notify about successful operation."""
        await state.notify('operation_status_changed', {
            'operation': operation,
            'status': 'success',
            'message': message,
            'source': self.__class__.__name__
        })
    
    async def _notify_info(self, state: "TUIState", operation: str, message: str):
        """Notify about informational status."""
        await state.notify('operation_status_changed', {
            'operation': operation,
            'status': 'info',
            'message': message,
            'source': self.__class__.__name__
        })


class ServiceCommand(BaseCommand):
    """
    Base class for commands that use services.
    
    Provides service injection and management.
    """
    
    def __init__(self, state: "TUIState" = None, context: "ProcessingContext" = None, service=None):
        """
        Initialize the service command.
        
        Args:
            state: Optional TUIState instance
            context: Optional ProcessingContext instance
            service: Service instance to use
        """
        super().__init__(state, context)
        self.service = service
    
    def _get_service(self, service=None):
        """Get the service to use for execution."""
        result_service = service or self.service
        if result_service is None:
            raise ValueError(f"{self.__class__.__name__}: No service available for execution")
        return result_service


class DialogCommand(BaseCommand):
    """
    Base class for commands that show dialogs.
    
    Provides dialog management functionality.
    """
    
    async def _show_error_dialog(self, title: str, message: str, details: str = ""):
        """Show an error dialog."""
        await self.state.notify('show_dialog_requested', {
            'type': 'error',
            'data': {
                'title': title,
                'message': message,
                'details': details
            }
        })
    
    async def _show_info_dialog(self, title: str, message: str):
        """Show an information dialog."""
        await self.state.notify('show_dialog_requested', {
            'type': 'info',
            'data': {
                'title': title,
                'message': message
            }
        })
    
    async def _show_confirmation_dialog(self, title: str, message: str, default_yes: bool = False) -> bool:
        """
        Show a confirmation dialog.
        
        Args:
            title: Dialog title
            message: Confirmation message
            default_yes: Whether "Yes" should be the default
            
        Returns:
            True if user confirmed, False otherwise
        """
        result_future = asyncio.Future()
        
        await self.state.notify('show_dialog_requested', {
            'type': 'confirmation',
            'data': {
                'title': title,
                'message': message,
                'default_yes': default_yes
            },
            'callback': lambda result: result_future.set_result(result)
        })
        
        return await result_future


class PlateCommand(ServiceCommand):
    """
    Base class for plate-related commands.
    
    Provides plate-specific functionality and validation.
    """
    
    def _get_selected_plate(self, state: "TUIState") -> Optional[Dict[str, Any]]:
        """Get the currently selected plate."""
        return getattr(state, 'selected_plate', None)
    
    def _get_active_orchestrator(self, state: "TUIState") -> Optional[Any]:
        """Get the active orchestrator."""
        return getattr(state, 'active_orchestrator', None)
    
    def _validate_plate_selected(self, state: "TUIState") -> bool:
        """Validate that a plate is selected."""
        return self._get_selected_plate(state) is not None
    
    def _validate_orchestrator_active(self, state: "TUIState") -> bool:
        """Validate that an orchestrator is active."""
        return self._get_active_orchestrator(state) is not None
    
    async def _handle_no_plate_selected(self, state: "TUIState"):
        """Handle case when no plate is selected."""
        await self._show_error_dialog(
            title="No Plate Selected",
            message="Please select a plate first."
        )
    
    async def _handle_no_orchestrator_active(self, state: "TUIState"):
        """Handle case when no orchestrator is active."""
        await self._show_error_dialog(
            title="No Active Plate",
            message="Please select and initialize a plate first."
        )


class PipelineCommand(PlateCommand):
    """
    Base class for pipeline-related commands.
    
    Provides pipeline-specific functionality and validation.
    """
    
    def _get_pipeline_definition(self, orchestrator: Any) -> Optional[List[Any]]:
        """Get the pipeline definition from an orchestrator."""
        return getattr(orchestrator, 'pipeline_definition', None)
    
    def _validate_pipeline_exists(self, orchestrator: Any) -> bool:
        """Validate that a pipeline exists in the orchestrator."""
        pipeline_def = self._get_pipeline_definition(orchestrator)
        return pipeline_def is not None and len(pipeline_def) > 0
    
    def _validate_pipeline_compiled(self, state: "TUIState", plate_id: str) -> bool:
        """Validate that a pipeline is compiled."""
        compiled_contexts = getattr(state, 'compiled_contexts', {})
        return plate_id in compiled_contexts and compiled_contexts[plate_id] is not None
    
    async def _handle_no_pipeline(self, state: "TUIState"):
        """Handle case when no pipeline exists."""
        await self._show_error_dialog(
            title="No Pipeline",
            message="Please create a pipeline first by adding steps."
        )
    
    async def _handle_pipeline_not_compiled(self, state: "TUIState"):
        """Handle case when pipeline is not compiled."""
        await self._show_error_dialog(
            title="Pipeline Not Compiled",
            message="Please compile the pipeline first."
        )
