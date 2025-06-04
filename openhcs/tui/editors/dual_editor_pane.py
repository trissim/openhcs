"""
Dual Editor Pane for OpenHCS TUI.

Clean tab coordinator that composes StepParameterEditor and FunctionPatternEditor.
Follows the same composition pattern as ListManagerPane.
"""
import copy
import logging
from typing import Any, Callable, Optional

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import HSplit, VSplit, Container, DynamicContainer, Window, Dimension
from prompt_toolkit.widgets import Button, Label, Frame

from openhcs.core.steps.function_step import FunctionStep
from .step_parameter_editor import StepParameterEditor
from .function_pattern_editor import FunctionPatternEditor

logger = logging.getLogger(__name__)

# Use Button directly - no defensive programming


class DualEditorPane:
    """
    Clean dual editor pane - just tab coordination.
    
    Composes StepParameterEditor + FunctionPatternEditor.
    No god class responsibilities, just simple coordination.
    """
    
    def __init__(self, 
                 state: Any,
                 func_step: FunctionStep,
                 on_save: Optional[Callable[[FunctionStep], None]] = None,
                 on_cancel: Optional[Callable[[], None]] = None):
        """
        Initialize dual editor pane.
        
        Args:
            state: TUI state
            func_step: The FunctionStep to edit
            on_save: Callback when save is requested
            on_cancel: Callback when cancel is requested
        """
        self.state = state
        self.original_step = func_step
        self.editing_step = copy.deepcopy(func_step)
        self.on_save = on_save
        self.on_cancel = on_cancel
        
        # Tab state
        self.current_tab = "step"  # "step" or "func"
        
        # Change tracking
        self.has_changes = False
        
        # Create editors
        self.step_editor = StepParameterEditor(
            step=self.editing_step,
            on_change=self._on_step_change,
            on_load=self._on_load_step,
            on_save_as=self._on_save_step_as
        )
        
        self.func_editor = FunctionPatternEditor(
            state=self.state,
            initial_pattern=self.editing_step.func,
            change_callback=self._on_func_change
        )
        
        # Create UI
        self._build_ui()
    
    @property
    def container(self) -> Container:
        """Get the container for this pane."""
        return self._container
    
    def _build_ui(self):
        """Build the dual editor UI."""
        # Tab buttons
        self.step_tab_btn = Button("Step",
            handler=lambda: self._switch_tab("step"),
            width=10)
        self.func_tab_btn = Button("Func",
            handler=lambda: self._switch_tab("func"),
            width=10)

        # Action buttons
        self.save_btn = Button("Save", handler=self._on_save_clicked, width=len("Save") + 2)
        self.save_btn.disabled = True  # Disabled until changes

        self.cancel_btn = Button("Cancel", handler=self._on_cancel_clicked, width=len("Cancel") + 2)
        
        # Tab bar
        tab_bar = VSplit([
            self.step_tab_btn,
            self.func_tab_btn,
            Window(width=Dimension(weight=1), char=' '),  # Spacer
            self.save_btn,
            self.cancel_btn,
        ], height=1, padding=1)
        
        # Dynamic content area
        def get_current_editor():
            if self.current_tab == "step":
                return HSplit([
                    VSplit([
                        Label(" Step Settings Editor ", style="class:frame.title"),
                        Window(width=Dimension(weight=1), char=' '),
                    ], height=1, style="class:frame.title"),
                    self.step_editor.container
                ])
            else:  # "func"
                return HSplit([
                    VSplit([
                        Label(" Function Pattern Editor ", style="class:frame.title"),
                        Window(width=Dimension(weight=1), char=' '),
                    ], height=1, style="class:frame.title"),
                    self.func_editor.container
                ])
        
        content_area = DynamicContainer(get_current_editor)
        
        # Main container
        self._container = HSplit([
            tab_bar,
            Frame(content_area)
        ])
        
        self._update_tab_styles()
    
    def _switch_tab(self, tab_name: str):
        """Switch to the specified tab."""
        self.current_tab = tab_name
        self._update_tab_styles()
        get_app().invalidate()
    
    def _update_tab_styles(self):
        """Update tab button styles to show active tab."""
        if self.step_tab_btn:
            self.step_tab_btn.style = "class:button.focused" if self.current_tab == "step" else "class:button"
        if self.func_tab_btn:
            self.func_tab_btn.style = "class:button.focused" if self.current_tab == "func" else "class:button"
    
    def _on_step_change(self, param_name: str, value: Any):
        """Handle step parameter change."""
        # Update the editing step
        setattr(self.editing_step, param_name, value)
        
        # Check for changes
        self._update_change_state()
    
    def _on_func_change(self):
        """Handle function pattern change."""
        # Update the editing step with new pattern
        self.editing_step.func = self.func_editor.get_pattern()
        
        # Check for changes
        self._update_change_state()
    
    def _update_change_state(self):
        """Update change state and save button."""
        # Simple change detection - compare with original
        step_changed = self._detect_step_changes()
        func_changed = self._detect_func_changes()
        
        self.has_changes = step_changed or func_changed
        
        if self.save_btn:
            self.save_btn.disabled = not self.has_changes
        
        get_app().invalidate()
    
    def _detect_step_changes(self) -> bool:
        """Detect if step parameters have changed."""
        # Compare key attributes
        for attr in ['name', 'description', 'variable_components', 'group_by']:
            original_val = getattr(self.original_step, attr, None)
            editing_val = getattr(self.editing_step, attr, None)
            if original_val != editing_val:
                return True
        return False
    
    def _detect_func_changes(self) -> bool:
        """Detect if function pattern has changed."""
        return (copy.deepcopy(self.editing_step.func) != 
                copy.deepcopy(self.original_step.func))
    
    def _on_save_clicked(self):
        """Handle save button click."""
        if self.on_save and self.has_changes:
            # Ensure func pattern is up to date
            self.editing_step.func = self.func_editor.get_pattern()
            
            # Call save callback
            self.on_save(copy.deepcopy(self.editing_step))
            
            # Update original and reset change state
            self.original_step = copy.deepcopy(self.editing_step)
            self.has_changes = False
            self.save_btn.disabled = True
            get_app().invalidate()
    
    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        if self.on_cancel:
            self.on_cancel()
    
    def _on_load_step(self):
        """Handle load step button click."""
        # This would be implemented by the parent component
        # For now, just log
        logger.info("Load step requested - should be handled by parent")
    
    def _on_save_step_as(self):
        """Handle save step as button click."""
        # This would be implemented by the parent component
        # For now, just log
        logger.info("Save step as requested - should be handled by parent")
    
    def get_buttons_container(self) -> Container:
        """Get the buttons container for external use."""
        return VSplit([
            self.step_tab_btn,
            self.func_tab_btn,
            Window(width=Dimension(weight=1), char=' '),
            self.save_btn,
            self.cancel_btn,
        ], height=1, padding=1)

    def get_created_step(self) -> FunctionStep:
        """Get the current edited step with latest changes."""
        # Ensure func pattern is up to date
        self.editing_step.func = self.func_editor.get_pattern()
        return copy.deepcopy(self.editing_step)
