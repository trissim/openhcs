"""
TUI State Management for OpenHCS.

Implements centralized state management with observer pattern for
event-driven communication between UI components.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class PlateData:
    """Data structure for plate information."""
    id: str
    name: str
    path: str
    status: str = 'uninitialized'  # uninitialized, initialized, compiled, running, completed, error
    orchestrator: Optional[Any] = None

@dataclass 
class StepData:
    """Data structure for step information."""
    id: str
    name: str
    type: str
    status: str = 'pending'  # pending, ready, running, completed, error
    step_object: Optional[Any] = None

class TUIState:
    """
    Centralized state manager for the OpenHCS TUI.
    
    Implements the observer pattern for event-driven communication
    between UI components while maintaining separation of concerns.
    All state changes flow through this class to ensure consistency.
    """
    
    def __init__(self):
        """Initialize the TUI state."""
        # Core application state
        self.is_running: bool = False
        self.error_message: Optional[str] = None
        
        # Plate management state
        self.plates: Dict[str, PlateData] = {}
        self.focused_plate: Optional[str] = None
        self.selected_plates: Set[str] = set()
        
        # Pipeline/step management state  
        self.steps: Dict[str, StepData] = {}  # Steps for the focused plate
        self.focused_step: Optional[str] = None
        self.selected_steps: Set[str] = set()
        
        # Editor state
        self.editing_step_config: bool = False
        self.step_to_edit: Optional[Any] = None
        
        # Dialog state
        self.active_dialog: Optional[str] = None
        self.dialog_data: Dict[str, Any] = {}
        
        # Status and messaging
        self.status_message: str = "Ready"
        self.status_priority: str = "info"  # info, warning, error
        
        # Observer pattern implementation
        self.observers: Dict[str, List[Callable]] = {}
        self._notification_lock = asyncio.Lock()
        
    async def notify(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Notify all observers of a state change event.
        
        Args:
            event: Event name (e.g., 'plate_selected', 'step_added')
            data: Optional event data
        """
        async with self._notification_lock:
            if event in self.observers:
                # Create tasks for all observers to run concurrently
                tasks = []
                for observer in self.observers[event]:
                    try:
                        if asyncio.iscoroutinefunction(observer):
                            tasks.append(observer(data or {}))
                        else:
                            # Handle sync observers by running in executor
                            loop = asyncio.get_event_loop()
                            tasks.append(loop.run_in_executor(None, observer, data or {}))
                    except Exception as e:
                        logger.error(f"Error creating task for observer {observer}: {e}")
                
                # Wait for all observers to complete
                if tasks:
                    try:
                        await asyncio.gather(*tasks, return_exceptions=True)
                    except Exception as e:
                        logger.error(f"Error in observer notification for event '{event}': {e}")
    
    def subscribe(self, event: str, observer: Callable) -> None:
        """
        Subscribe an observer to an event.
        
        Args:
            event: Event name to observe
            observer: Callback function (can be sync or async)
        """
        if event not in self.observers:
            self.observers[event] = []
        self.observers[event].append(observer)
        
    def unsubscribe(self, event: str, observer: Callable) -> None:
        """
        Unsubscribe an observer from an event.
        
        Args:
            event: Event name
            observer: Callback function to remove
        """
        if event in self.observers and observer in self.observers[event]:
            self.observers[event].remove(observer)
            
    # Plate management methods
    async def add_plate(self, plate_data: PlateData) -> None:
        """Add a new plate to the state."""
        self.plates[plate_data.id] = plate_data
        await self.notify('plate_added', {'plate': plate_data})
        
    async def remove_plate(self, plate_id: str) -> None:
        """Remove a plate from the state."""
        if plate_id in self.plates:
            plate_data = self.plates.pop(plate_id)
            
            # Clean up related state
            if self.focused_plate == plate_id:
                self.focused_plate = None
            self.selected_plates.discard(plate_id)
            
            await self.notify('plate_removed', {'plate_id': plate_id, 'plate': plate_data})
            
    async def set_focused_plate(self, plate_id: Optional[str]) -> None:
        """Set the focused plate."""
        if self.focused_plate != plate_id:
            old_focused = self.focused_plate
            self.focused_plate = plate_id
            
            # Clear step state when changing plates
            self.steps.clear()
            self.focused_step = None
            self.selected_steps.clear()
            
            await self.notify('plate_focus_changed', {
                'old_focused': old_focused,
                'new_focused': plate_id
            })
            
    async def update_plate_status(self, plate_id: str, status: str, message: Optional[str] = None) -> None:
        """Update a plate's status."""
        if plate_id in self.plates:
            old_status = self.plates[plate_id].status
            self.plates[plate_id].status = status
            
            await self.notify('plate_status_changed', {
                'plate_id': plate_id,
                'old_status': old_status,
                'new_status': status,
                'message': message
            })
            
    # Step management methods
    async def add_step(self, step_data: StepData) -> None:
        """Add a new step to the current pipeline."""
        self.steps[step_data.id] = step_data
        await self.notify('step_added', {'step': step_data})
        
    async def remove_step(self, step_id: str) -> None:
        """Remove a step from the current pipeline."""
        if step_id in self.steps:
            step_data = self.steps.pop(step_id)
            
            # Clean up related state
            if self.focused_step == step_id:
                self.focused_step = None
            self.selected_steps.discard(step_id)
            
            await self.notify('step_removed', {'step_id': step_id, 'step': step_data})
            
    async def set_focused_step(self, step_id: Optional[str]) -> None:
        """Set the focused step."""
        if self.focused_step != step_id:
            old_focused = self.focused_step
            self.focused_step = step_id
            
            await self.notify('step_focus_changed', {
                'old_focused': old_focused,
                'new_focused': step_id
            })
            
    # Editor state methods
    async def start_step_editing(self, step_to_edit: Any) -> None:
        """Start editing a step."""
        self.editing_step_config = True
        self.step_to_edit = step_to_edit
        
        await self.notify('step_editing_started', {
            'step_to_edit': step_to_edit
        })
        
    async def stop_step_editing(self) -> None:
        """Stop editing a step."""
        self.editing_step_config = False
        old_step = self.step_to_edit
        self.step_to_edit = None
        
        await self.notify('step_editing_stopped', {
            'step_was_editing': old_step
        })
        
    # Status and messaging methods
    async def set_status(self, message: str, priority: str = "info") -> None:
        """Set the status message."""
        self.status_message = message
        self.status_priority = priority
        
        await self.notify('status_changed', {
            'message': message,
            'priority': priority
        })
        
    async def set_error(self, error_message: str) -> None:
        """Set an error message."""
        self.error_message = error_message
        await self.set_status(f"Error: {error_message}", "error")
        
    async def clear_error(self) -> None:
        """Clear the current error."""
        self.error_message = None
        await self.set_status("Ready", "info")
