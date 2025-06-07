"""DualEditor modal screen - port of dual_editor_pane.py to Textual."""

from typing import Optional, Callable
from textual.screen import ModalScreen
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Static, TabbedContent, TabPane

from openhcs.core.steps.function_step import FunctionStep
from openhcs.textual_tui.services.pattern_data_manager import PatternDataManager
from openhcs.textual_tui.services.function_registry_service import FunctionRegistryService
from openhcs.textual_tui.widgets.step_parameter_editor import StepParameterEditorWidget
from openhcs.textual_tui.widgets.function_list_editor import FunctionListEditorWidget


class DualEditorScreen(ModalScreen):
    """
    Modal screen for editing steps and functions.
    
    Ports the tab switching and data flow logic from dual_editor_pane.py
    """
    
    def __init__(self, step_data: Optional[FunctionStep] = None, is_new: bool = False):
        super().__init__()
        self.step_data = step_data
        self.is_new = is_new
        
        # Initialize services (reuse existing business logic)
        self.pattern_manager = PatternDataManager()
        self.registry_service = FunctionRegistryService()
        
        # Create working copy of step data
        if self.step_data:
            self.editing_step = self.pattern_manager.clone_pattern(self.step_data)
        else:
            # Create new step with default function
            default_func = self.registry_service.find_default_function()
            self.editing_step = FunctionStep(
                func=default_func,
                name="New Step",
                variable_components=[],
                group_by=""
            )

    def compose(self) -> ComposeResult:
        """Compose the dual editor modal."""
        with Container(id="dual_editor_dialog"):
            yield Static("Step Editor", id="dialog_title")
            
            with TabbedContent():
                with TabPane("Step", id="step_tab"):
                    yield StepParameterEditorWidget(self.editing_step)
                
                with TabPane("Functions", id="func_tab"):
                    yield FunctionListEditorWidget(self.editing_step.func)
            
            with Horizontal(id="dialog_buttons"):
                yield Button("Save", id="save_btn", variant="primary", compact=True)
                yield Button("Cancel", id="cancel_btn", compact=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses - port from dual_editor_pane.py logic."""
        if event.button.id == "save_btn":
            # Return edited step data
            self.dismiss(self.editing_step)
        elif event.button.id == "cancel_btn":
            # Return None to indicate cancellation
            self.dismiss(None)
