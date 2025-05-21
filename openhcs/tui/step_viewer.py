import asyncio
import uuid
from typing import Any, Dict, List, Optional

from prompt_toolkit.application import get_app
from prompt_toolkit.filters import has_focus
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, VSplit
from prompt_toolkit.widgets import Button, Frame, Label, TextArea

from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.steps.function_step import FunctionStep


class StepViewerPane:
    """
    Middle pane for viewing steps in the OpenHCS TUI.

    This pane displays a list of FuncStep declarations for the selected plate,
    with visual separators between pipelines. It supports keyboard navigation
    and click selection.
    """
    def __init__(self, state, context: ProcessingContext):
        """
        Initialize the Step Viewer pane.
        
        🔒 Clause 12: Explicit Error Handling
        Constructor must be synchronous in Python.

        Args:
            state: The TUI state manager
            context: The OpenHCS ProcessingContext
        """
        self.state = state
        self.context = context
        self.steps: List[Dict[str, Any]] = []
        self.pipelines: List[Dict[str, Any]] = []
        self.selected_index = 0
        self.steps_lock = asyncio.Lock()  # Thread safety for step operations
        
        # Placeholder for UI components that will be created in setup()
        self.step_list = None
        self.edit_button = None
        self.add_button = None
        self.remove_button = None
        self.move_up_button = None
        self.move_down_button = None
        self.kb = None
        self.container = None
        
        # Flag to track initialization state
        self._ui_initialized = False
    
    @classmethod
    async def create(cls, state, context: ProcessingContext):
        """
        Factory method to create and initialize a StepViewerPane asynchronously.
        
        🔒 Clause 12: Explicit Error Handling
        Use factory pattern for async initialization.
        
        Args:
            state: The TUI state manager
            context: The OpenHCS ProcessingContext
            
        Returns:
            Initialized StepViewerPane instance
        """
        # Create instance with synchronous constructor
        instance = cls(state, context)
        
        # Initialize UI components asynchronously
        await instance.setup()
        
        return instance
    
    async def setup(self):
        """
        Set up UI components asynchronously.
        
        This method must be called after instantiation to initialize
        UI components that require async operations.
        """
        if self._ui_initialized:
            return
            
        # Create UI components asynchronously
        self.step_list = await self._create_step_list()
        self.edit_button = Button("Edit Step", handler=lambda: get_app().create_background_task(self._edit_step()))
        self.add_button = Button("Add Step", handler=lambda: get_app().create_background_task(self._add_step()))
        self.remove_button = Button("Remove Step", handler=lambda: get_app().create_background_task(self._remove_step()))
        self.move_up_button = Button("↑ Move Up", handler=lambda: get_app().create_background_task(self._move_step_up()))
        self.move_down_button = Button("↓ Move Down", handler=lambda: get_app().create_background_task(self._move_step_down()))

        # Create key bindings
        self.kb = self._create_key_bindings()

        # Create container
        self.container = HSplit([
            self.step_list,
            VSplit([
                self.edit_button,
                self.add_button,
                self.remove_button,
                self.move_up_button,
                self.move_down_button
            ])
        ])

        # Register for events - use wrapper for async methods
        self.state.add_observer('plate_selected', lambda plate: get_app().create_background_task(self._on_plate_selected(plate)))
        self.state.add_observer('steps_updated', lambda _: get_app().create_background_task(self._refresh_steps()))
        
        # Mark as initialized
        self._ui_initialized = True

    async def _create_step_list(self) -> TextArea:
        """
        Create the step list component.

        Returns:
            A TextArea containing the formatted step list
        """
        text_area = TextArea(
            text=await self._format_step_list(),
            read_only=True,
            scrollbar=True,
            wrap_lines=False
        )
        return text_area

    async def _format_step_list(self) -> str:
        """
        Format the step list for display.

        Returns:
            Formatted text for the step list
        """
        # Thread-safe access to steps
        async with self.steps_lock:
            if not self.steps:
                return "No steps available. Select a plate first."

            lines = []
            current_pipeline = None

            for i, step in enumerate(self.steps):
                # Add pipeline separator if needed
                pipeline_id = step.get('pipeline_id')
                if pipeline_id != current_pipeline:
                    current_pipeline = pipeline_id
                    # 🔒 Clause 24: Performance Optimization
                    # Use O(1) lookup instead of O(N) search
                    pipeline_info = self.pipeline_lookup.get(pipeline_id)
                    if pipeline_info:
                        pipeline_name = pipeline_info.get('name', f"Pipeline {pipeline_id}")
                        lines.append(f"─── {pipeline_name} ───")

                # Format step line
                status_icon = self._get_status_icon(step['status'])
                # 🔒 Clause 52: Semantic Representation
                # Use Unicode characters for selection indicators
                selected = "▶" if i == self.selected_index else " "
                name = step['name']
                func_name = self._get_function_name(step)
                
                # 🔒 Clause 87: Explicit Validation Layer
                # Safe access to memory_type with fallback for rendering
                # Validation happens in _load_steps_for_plate, not in the render loop
                memory_type = step.get('output_memory_type', '[INVALID]')

                line = f"{selected} {status_icon} {name}: {func_name} → {memory_type}"
                lines.append(line)

            return "\n".join(lines)

    def _get_status_icon(self, status: str) -> str:
        """
        Get the status icon for a step.

        Args:
            status: The step status

        Returns:
            A status icon character
        """
        icons = {
            'pending': "○",
            'validated': "●",
            'error': "✗"
        }
        if status not in icons:
            raise ValueError(f"Unknown step status: {status}")
        return icons[status]

    def _get_function_name(self, step: Dict[str, Any]) -> str:
        """
        Get a display name for the function pattern.

        Args:
            step: The step dictionary

        Returns:
            A display name for the function pattern
        """
        # 🔒 Clause 88: No Inferred Capabilities
        # 🔒 Clause 92: Structural Validation First
        # Safe access to func field with fallback for rendering
        # Validation happens in _load_steps_for_plate, not in the render loop
        if 'func' not in step:
            return '[MISSING FUNC]'
        
        func = step['func']
        if func is None:
            return '[NULL FUNC]'

        # Handle different function pattern types
        if callable(func):
            return func.__name__
        elif isinstance(func, tuple) and len(func) == 2 and callable(func[0]):
            return f"{func[0].__name__}(...)"
        elif isinstance(func, list):
            return f"[{len(func)} functions]"
        elif isinstance(func, dict):
            return f"{{{len(func)} components}}"
        else:
            return str(func)

    def _create_key_bindings(self) -> KeyBindings:
        """
        Create key bindings for the step list.

        Returns:
            KeyBindings object
        """
        kb = KeyBindings()

        @kb.add('up', filter=has_focus(self.step_list))
        def _(event):
            """Move selection up."""
            if self.steps:
                self.selected_index = max(0, self.selected_index - 1)
                get_app().create_background_task(self._update_selection())

        @kb.add('down', filter=has_focus(self.step_list))
        def _(event):
            """Move selection down."""
            if self.steps:
                self.selected_index = min(len(self.steps) - 1, self.selected_index + 1)
                get_app().create_background_task(self._update_selection())

        @kb.add('enter', filter=has_focus(self.step_list))
        def _(event):
            """Select the current step."""
            if self.steps and 0 <= self.selected_index < len(self.steps):
                get_app().create_background_task(self._select_step(self.selected_index))

        @kb.add('shift+up', filter=has_focus(self.step_list))
        def _(event):
            """Move the selected step up."""
            get_app().create_background_task(self._move_step_up())

        @kb.add('shift+down', filter=has_focus(self.step_list))
        def _(event):
            """Move the selected step down."""
            get_app().create_background_task(self._move_step_down())

        return kb

    async def _update_selection(self):
        """Update the step list to reflect the current selection."""
        self.step_list.text = await self._format_step_list()
        if self.steps and 0 <= self.selected_index < len(self.steps):
            get_app().create_background_task(self._select_step(self.selected_index))

    async def _select_step(self, index: int):
        """
        Select a step and emit selection event.

        Args:
            index: The index of the step to select
        """
        async with self.steps_lock:
            if 0 <= index < len(self.steps):
                step = self.steps[index]
                self.state.set_selected_step(step)

    async def _on_plate_selected(self, plate):
        """
        Handle plate selection event.

        Args:
            plate: The selected plate
        """
        if plate:
            # In a real implementation, this would query the ProcessingContext
            # for the steps associated with the plate
            # For now, we'll use placeholder data
            await self._load_steps_for_plate(plate['id'])

    async def _load_steps_for_plate(self, plate_id: str):
        """
        Load steps for the specified plate.

        Args:
            plate_id: The ID of the plate
        """
        # Get pipelines and build lookup dict for O(1) access
        self.pipelines = self.context.list_pipelines_for_plate(plate_id)
        
        # 🔒 Clause 24: Performance Optimization
        # Build pipeline lookup dict for O(1) access instead of O(N) search
        self.pipeline_lookup = {p['id']: p for p in self.pipelines}

        # Thread-safe step loading
        async with self.steps_lock:
            self.steps = self.context.list_steps_for_plate(plate_id)
            
            # 🔒 Clause 92 & 241: Structural Validation First
            # Validate steps before rendering using explicit checks, not assert
            invalid_steps = []
            for i, step in enumerate(self.steps):
                # Check for required fields
                required_fields = ['func', 'output_memory_type', 'name', 'status']
                missing_fields = [field for field in required_fields if field not in step]
                
                if missing_fields:
                    step_id = step.get('id', f'at index {i}')
                    invalid_steps.append((step_id, missing_fields))
                    
                # Check for null func field
                if 'func' in step and step['func'] is None:
                    invalid_steps.append((step.get('id', f'at index {i}'), ['func is None']))
            
            # Report validation errors through state's error channel
            if invalid_steps:
                error_msg = "Invalid steps detected:\n" + "\n".join(
                    f"- Step {step_id}: missing {', '.join(fields)}"
                    for step_id, fields in invalid_steps
                )
                self.state.notify('error', {'message': error_msg, 'source': 'StepViewerPane'})
                
                # Mark invalid steps with error status for display
                for step_id, _ in invalid_steps:
                    for step in self.steps:
                        if step.get('id') == step_id:
                            step['status'] = 'error'
                            # Ensure required fields exist for rendering
                            for field in required_fields:
                                if field not in step:
                                    step[field] = '[INVALID]'

            self.selected_index = 0
            get_app().create_background_task(self._update_selection())

    async def _edit_step(self):
        """Edit the selected step."""
        async with self.steps_lock:
            if self.steps and 0 <= self.selected_index < len(self.steps):
                step = self.steps[self.selected_index]
                self.state.notify('edit_step', step)

    async def _add_step(self):
        """Add a new step."""
        # This would typically open a dialog to configure the new step
        if not self.pipelines:
            self.state.notify('error', {
                'message': "Cannot add step: no pipelines available",
                'source': 'StepViewerPane'
            })
            return
            
        # 🔒 Clause 231: Deferred-Default Enforcement
        # Initialize with valid placeholders to avoid impossible UX loop
        # In a real implementation, these would be replaced with user input from a dialog
        new_step = {
            'id': str(uuid.uuid4()),  # Generate unique ID
            'name': "New Step",  # Generic name, will be configured by dialog
            'pipeline_id': self.pipelines[0]['id'],
            'func': lambda x: x,  # Valid placeholder function
            'input_memory_type': 'placeholder',  # Valid placeholder
            'output_memory_type': 'placeholder',  # Valid placeholder
            'status': 'pending',
        }
        
        # Show dialog to configure step (not implemented here)
        # In a real implementation, this would open a dialog to get user input
        # and update the new_step dict with the user's values
        
        # Thread-safe step addition
        async with self.steps_lock:
            # 🔒 Clause 92 & 241: Structural Validation First
            # Validate step has required fields before adding using explicit checks
            required_fields = ['func', 'input_memory_type', 'output_memory_type']
            missing_fields = []
            
            for field in required_fields:
                if field not in new_step or new_step[field] is None:
                    missing_fields.append(field)
            
            if missing_fields:
                error_msg = f"Cannot add step: missing required fields: {', '.join(missing_fields)}"
                self.state.notify('error', {'message': error_msg, 'source': 'StepViewerPane'})
                return
                    
            self.steps.append(new_step)
            self.selected_index = len(self.steps) - 1
            get_app().create_background_task(self._update_selection())

    async def _remove_step(self):
        """Remove the selected step."""
        if not self.steps or not (0 <= self.selected_index < len(self.steps)):
            return
            
        # Thread-safe step removal
        async with self.steps_lock:
            del self.steps[self.selected_index]
            if self.selected_index >= len(self.steps) and self.selected_index > 0:
                self.selected_index -= 1
            get_app().create_background_task(self._update_selection())

    async def _refresh_steps(self, _=None):
        """Refresh the step list."""
        await self._update_selection()

    async def _move_step_up(self):
        """
        Move the selected step up in the pipeline.

        This reorders steps within the same pipeline but does not
        allow moving steps across pipeline boundaries.
        """
        if not self.steps or self.selected_index <= 0:
            return

        # Thread-safe step reordering
        async with self.steps_lock:
            # Get current step and the one above it
            current_step = self.steps[self.selected_index]
            prev_step = self.steps[self.selected_index - 1]

            # Check if they're in the same pipeline
            if current_step['pipeline_id'] != prev_step['pipeline_id']:
                # Can't move across pipeline boundaries
                return

            # Swap the steps
            self.steps[self.selected_index], self.steps[self.selected_index - 1] = \
                self.steps[self.selected_index - 1], self.steps[self.selected_index]

            # Update selection
            self.selected_index -= 1
            get_app().create_background_task(self._update_selection())

            # Notify that steps have been reordered
            self.state.notify('steps_reordered', {
                'pipeline_id': current_step['pipeline_id'],
                'steps': self.steps
            })

    async def _move_step_down(self):
        """
        Move the selected step down in the pipeline.

        This reorders steps within the same pipeline but does not
        allow moving steps across pipeline boundaries.
        """
        if not self.steps or self.selected_index >= len(self.steps) - 1:
            return

        # Thread-safe step reordering
        async with self.steps_lock:
            # Get current step and the one below it
            current_step = self.steps[self.selected_index]
            next_step = self.steps[self.selected_index + 1]

            # Check if they're in the same pipeline
            if current_step['pipeline_id'] != next_step['pipeline_id']:
                # Can't move across pipeline boundaries
                return

            # Swap the steps
            self.steps[self.selected_index], self.steps[self.selected_index + 1] = \
                self.steps[self.selected_index + 1], self.steps[self.selected_index]

            # Update selection
            self.selected_index += 1
            get_app().create_background_task(self._update_selection())

            # Notify that steps have been reordered
            self.state.notify('steps_reordered', {
                'pipeline_id': current_step['pipeline_id'],
                'steps': self.steps
            })
```
