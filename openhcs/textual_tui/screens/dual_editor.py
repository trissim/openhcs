"""DualEditor modal screen - enhanced port of dual_editor_pane.py to Textual."""

import logging
from typing import Optional, Callable, Union, List, Dict
from textual.screen import ModalScreen
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Static, TabbedContent, TabPane, Label, Input, Select
from textual.reactive import reactive
from textual.widgets._tabbed_content import ContentTabs, ContentSwitcher, ContentTab
from textual.widgets._tabs import Tab
from textual import events          # NEW
from textual.widgets import Tabs    # NEW – for type hint below (optional)
from itertools import zip_longest

from openhcs.core.steps.function_step import FunctionStep
from openhcs.textual_tui.services.pattern_data_manager import PatternDataManager
from openhcs.textual_tui.services.function_registry_service import FunctionRegistryService
# Updated import to get the message class as well
from openhcs.textual_tui.widgets.step_parameter_editor import (
    StepParameterEditorWidget,
    StepParameterEditorWidget)
from openhcs.textual_tui.widgets.function_list_editor import FunctionListEditorWidget

logger = logging.getLogger(__name__)


class ButtonTab(Tab):
    """A fake tab that acts like a button."""

    class ButtonClicked(Message):
        """A button tab was clicked."""
        def __init__(self, button_id: str) -> None:
            self.button_id = button_id
            super().__init__()

    def __init__(self, label: str, button_id: str, disabled: bool = False):
        # Use a unique ID for the button tab
        super().__init__(label, id=f"button-{button_id}", disabled=disabled)
        self.button_id = button_id
        # Add button-like styling
        self.add_class("button-tab")

    def _on_click(self, event: events.Click) -> None:
        """Send the ButtonClicked message and swallow the click."""
        event.stop()                       # don't let real tab logic run
        event.prevent_default()            # prevent default tab behavior
        self.post_message(self.ButtonClicked(self.button_id))


class TabbedContentWithButtons(TabbedContent):
    """Custom TabbedContent that adds Save/Close buttons to the tab bar."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_callback = None
        self.close_callback = None

    def set_callbacks(self, save_callback: Optional[Callable] = None, close_callback: Optional[Callable] = None):
        """Set the callbacks for save and close buttons."""
        self.save_callback = save_callback
        self.close_callback = close_callback

    def compose(self) -> ComposeResult:
        """Compose the tabbed content with button tabs in the tab bar."""
        # Wrap content in a `TabPane` if required (copied from parent)
        pane_content = [
            self._set_id(
                (
                    content
                    if isinstance(content, TabPane)
                    else TabPane(title or self.render_str(f"Tab {index}"), content)
                ),
                self._generate_tab_id(),
            )
            for index, (title, content) in enumerate(
                zip_longest(self.titles, self._tab_content), 1
            )
        ]

        # Get a tab for each pane (copied from parent)
        tabs = [
            ContentTab(
                content._title,
                content.id or "",
                disabled=content.disabled,
            )
            for content in pane_content
        ]

        # ── tab strip ───────────────────────────────────────────────────
        # 1️⃣ regular content-selecting tabs
        # 2️⃣ an elastic spacer tab (width: 1fr in CSS)
        # 3️⃣ our two action "button-tabs"

        spacer = Tab("", id="spacer_tab", disabled=True)        # visual gap
        spacer.add_class("spacer-tab")

        tabs.extend([
            spacer,
            ButtonTab("Save", "save", disabled=True),
            ButtonTab("Close", "close"),
        ])

        # yield the single ContentTabs row (must be an immediate child)
        yield ContentTabs(*tabs,
                          active=self._initial or None,
                          tabbed_content=self)

        # Yield the content switcher and panes (copied from parent)
        with ContentSwitcher(initial=self._initial or None):
            yield from pane_content

    def _on_tabs_tab_activated(
        self, event: Tabs.TabActivated
    ) -> None:
        """Override to prevent button tabs from being activated as content tabs."""
        # Check if the activated tab is a button tab by ID (more reliable than isinstance)
        if hasattr(event, 'tab') and event.tab.id and event.tab.id.startswith('button-'):
            # Don't activate button tabs as content tabs
            event.stop()
            return

        # For regular tabs, use the normal behavior
        super()._on_tabs_tab_activated(event)

    def on_button_tab_button_clicked(self, event: ButtonTab.ButtonClicked) -> None:
        """Handle button tab clicks."""
        if event.button_id == "save" and self.save_callback:
            self.save_callback()
        elif event.button_id == "close" and self.close_callback:
            self.close_callback()


class DualEditorScreen(ModalScreen):
    """
    Enhanced modal screen for editing steps and functions.

    Ports the complete functionality from dual_editor_pane.py with:
    - Change tracking and validation
    - Tab switching with proper state management
    - Integration with all visual programming services
    - Proper error handling and user feedback
    """

    CSS_PATH = "dual_editor.css"

    # Reactive state for change tracking
    has_changes = reactive(False)
    current_tab = reactive("step")

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
            # Create new step with empty function list (user adds functions manually)
            self.editing_step = FunctionStep(
                func=[],  # Start with empty function list
                name="New Step",
                # Let variable_components use FunctionStep's default [VariableComponents.SITE]
                group_by=""
            )

        # Store original for change detection
        self.original_step = self.pattern_manager.clone_pattern(self.editing_step)

        # Editor widgets (will be created in compose)
        self.step_editor = None
        self.func_editor = None

    def compose(self) -> ComposeResult:
        """Compose the enhanced dual editor modal."""
        with Container(id="dual_editor_dialog") as container:
            container.styles.border = ("solid", "white")
            # Dialog title with change indicator
            title_text = "Visual Programming - Edit Step" if not self.is_new else "Visual Programming - New Step"
            yield Static(title_text, id="dialog_title")

            # Status bar for feedback
            yield Static("Ready", id="status_bar")

            # Custom tabbed content with buttons
            with TabbedContentWithButtons(id="editor_tabs") as tabbed_content:
                tabbed_content.set_callbacks(
                    save_callback=self._handle_save,
                    close_callback=self._handle_cancel
                )

                with TabPane("Step Settings", id="step_tab"):
                    # Create step editor with correct constructor
                    self.step_editor = StepParameterEditorWidget(self.editing_step)
                    yield self.step_editor

                with TabPane("Function Pattern", id="func_tab"):
                    # Create function editor with validated function data
                    func_data = self._validate_function_data(self.editing_step.func)
                    self.func_editor = FunctionListEditorWidget(func_data)
                    yield self.func_editor

    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        # Update change tracking to set initial Save button state
        self._update_change_tracking()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes from child widgets."""
        logger.debug(f"Input changed: {event}")

        # Update change tracking when any input changes
        self._update_change_tracking()
        self._update_status("Modified step parameters")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes from child widgets."""
        logger.debug(f"Select changed: {event}")

        # Update change tracking when any select changes
        self._update_change_tracking()
        self._update_status("Modified step parameters")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses from child widgets."""
        # Handle Save/Close buttons
        if event.button.id == "save_button":
            self._handle_save()
        elif event.button.id == "close_button":
            self._handle_cancel()
        else:
            # Handle other button presses from child widgets
            logger.debug(f"Child widget button pressed: {event.button.id}")
            self._update_change_tracking()
            self._update_status("Modified step configuration")

    def on_step_parameter_editor_widget_step_parameter_changed(
        self, event: StepParameterEditorWidget.StepParameterChanged # Listen for the specific message
    ) -> None:
        """Handle parameter changes from the step editor widget."""
        logger.debug("Received StepParameterChanged from child StepParameterEditorWidget")
        self._update_change_tracking()
        self._update_status("Modified step parameters (via message)")

    def on_function_list_editor_widget_function_pattern_changed(
        self, event: FunctionListEditorWidget.FunctionPatternChanged
    ) -> None:
        """Handle function pattern changes from the function editor widget."""
        logger.debug("Received FunctionPatternChanged from child FunctionListEditorWidget")
        # Use current_pattern property to get List or Dict format
        self.editing_step.func = self.func_editor.current_pattern
        self._update_change_tracking()
        self._update_status("Modified function pattern")



    def _update_change_tracking(self) -> None:
        """Update change tracking state."""
        # Compare current editing step with original
        has_changes = not self._steps_equal(self.editing_step, self.original_step)
        self.has_changes = has_changes

        # For new steps, always enable save button since they need to be saved
        # For existing steps, enable only when there are changes
        save_should_be_enabled = self.is_new or has_changes

        # Update save button state
        try:
            # Find the save button tab by looking for button tabs
            tabs = self.query(ButtonTab)
            for tab in tabs:
                if tab.button_id == "save":
                    tab.disabled = not save_should_be_enabled
                    logger.debug(f"Save button disabled={not save_should_be_enabled} (is_new={self.is_new}, has_changes={has_changes})")
                    break
        except Exception as e:
            logger.debug(f"Error updating save button state: {e}")

    def _validate_function_data(self, func_data) -> Union[List, callable, None]:
        """Validate and normalize function data for FunctionListEditorWidget."""
        if func_data is None:
            return None
        elif callable(func_data):
            return func_data
        elif isinstance(func_data, (list, dict)):
            return func_data
        else:
            logger.warning(f"Invalid function data type: {type(func_data)}, using None")
            return None

    def _steps_equal(self, step1: FunctionStep, step2: FunctionStep) -> bool:
        """Compare two FunctionSteps for equality."""
        return (
            step1.name == step2.name and
            step1.func == step2.func and
            step1.variable_components == step2.variable_components and
            step1.group_by == step2.group_by
        )

    def _update_status(self, message: str) -> None:
        """Update status bar message."""
        try:
            status_bar = self.query_one("#status_bar", Static)
            status_bar.update(message)
        except Exception:
            pass

    def watch_has_changes(self, has_changes: bool) -> None:
        """React to changes in has_changes state."""
        # Update dialog title to show unsaved changes
        try:
            title = self.query_one("#dialog_title", Static)
            base_title = "Visual Programming - Edit Step" if not self.is_new else "Visual Programming - New Step"
            if has_changes:
                title.update(f"{base_title} *")
            else:
                title.update(base_title)
        except Exception:
            pass



    def _handle_save(self) -> None:
        """Handle save button with validation."""
        # Validate step data
        if not self.editing_step.name or not self.editing_step.name.strip():
            self._update_status("Error: Step name cannot be empty")
            return

        if not self.editing_step.func:
            self._update_status("Error: Function pattern cannot be empty")
            return

        # Save successful
        self._update_status("Saved successfully")
        self.dismiss(self.editing_step)

    def _handle_cancel(self) -> None:
        """Handle cancel button with change confirmation."""
        if self.has_changes:
            # TODO: Add confirmation dialog for unsaved changes
            self._update_status("Cancelled - changes discarded")
        else:
            self._update_status("Cancelled")

        self.dismiss(None)
