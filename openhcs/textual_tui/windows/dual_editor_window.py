"""DualEditor window - enhanced port of dual_editor_pane.py to Textual."""

import logging
from typing import Optional, Callable, Union, List, Dict
# ModalScreen import removed - using BaseOpenHCSWindow instead
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
from openhcs.textual_tui.windows.base_window import BaseOpenHCSWindow
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

    def _get_content_switcher(self) -> Optional[ContentSwitcher]:
        """Safely get the content switcher."""
        try:
            return self.query_one(ContentSwitcher)
        except:
            return None

    def _safe_switch_content(self, content_id: str) -> bool:
        """Safely switch content, returning True if successful."""
        switcher = self._get_content_switcher()
        if not switcher:
            return False

        try:
            # Check if the content actually exists before switching
            switcher.get_child_by_id(content_id)
            switcher.current = content_id
            return True
        except:
            # Content doesn't exist (probably a button tab), ignore silently
            logger.debug(f"Ignoring switch to non-existent content: {content_id}")
            return False

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

        # Create button tabs with explicit styling to distinguish them
        save_button = ButtonTab("Save", "save", disabled=False)  # Always enabled
        close_button = ButtonTab("Close", "close")

        # Add additional classes for safety
        save_button.add_class("action-button")
        close_button.add_class("action-button")

        tabs.extend([
            spacer,
            save_button,
            close_button,
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
        """Override to safely handle tab activation, including button tabs."""
        # Check if the activated tab is a button tab by ID
        if hasattr(event, 'tab') and event.tab.id and event.tab.id.startswith('button-'):
            # Don't activate button tabs as content tabs
            event.stop()
            return

        # For regular tabs, use safe switching instead of the default behavior
        if hasattr(event, 'tab') and event.tab.id:
            content_id = ContentTab.sans_prefix(event.tab.id)
            if self._safe_switch_content(content_id):
                # Successfully switched, stop the event to prevent default handling
                event.stop()
                return

        # If we get here, let the default behavior handle it
        super()._on_tabs_tab_activated(event)

    def on_button_tab_button_clicked(self, event: ButtonTab.ButtonClicked) -> None:
        """Handle button tab clicks."""
        if event.button_id == "save" and self.save_callback:
            self.save_callback()
        elif event.button_id == "close" and self.close_callback:
            self.close_callback()

    def on_resize(self, event) -> None:
        """Handle window resize to readjust tab layout."""
        # Refresh the tabs to recalculate their layout
        try:
            tabs = self.query_one(ContentTabs)
            tabs.refresh()
        except:
            pass  # Tabs not ready yet




class DualEditorWindow(BaseOpenHCSWindow):
    """
    Enhanced modal screen for editing steps and functions.

    Ports the complete functionality from dual_editor_pane.py with:
    - Change tracking and validation
    - Tab switching with proper state management
    - Integration with all visual programming services
    - Proper error handling and user feedback
    """

    DEFAULT_CSS = """
    DualEditorWindow {
        width: 80; height: 20;
        min-width: 80; min-height: 20;
    }
    """

    # Reactive state for change tracking
    has_changes = reactive(False)
    current_tab = reactive("step")

    def __init__(self, step_data: Optional[FunctionStep] = None, is_new: bool = False,
                 on_save_callback: Optional[Callable] = None):
        super().__init__(
            window_id="dual_editor",
            title="Dual Editor",
            mode="temporary"
        )
        self.step_data = step_data
        self.is_new = is_new
        self.on_save_callback = on_save_callback

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
                name="New Step"
                # Let variable_components and group_by use FunctionStep's defaults
            )

        # Store original for change detection
        self.original_step = self.pattern_manager.clone_pattern(self.editing_step)

        # Editor widgets (will be created in compose)
        self.step_editor = None
        self.func_editor = None

    def compose(self) -> ComposeResult:
        """Compose the enhanced dual editor modal."""
        # Custom tabbed content with buttons - start directly with tabs
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
                # Create function editor with validated function data and step identifier
                func_data = self._validate_function_data(self.editing_step.func)
                step_id = getattr(self.editing_step, 'name', 'unknown_step')
                self.func_editor = FunctionListEditorWidget(func_data, step_identifier=step_id)

                # Initialize step configuration settings in function editor
                self.func_editor.current_group_by = self.editing_step.group_by
                self.func_editor.current_variable_components = self.editing_step.variable_components or []

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
        # Note: Save/Close buttons are handled by TabbedContentWithButtons.on_button_tab_button_clicked
        # This method only handles buttons from child widgets like StepParameterEditorWidget
        logger.debug(f"Child widget button pressed: {event.button.id}")
        self._update_change_tracking()
        self._update_status("Modified step configuration")

    def on_step_parameter_editor_widget_step_parameter_changed(
        self, event: StepParameterEditorWidget.StepParameterChanged # Listen for the specific message
    ) -> None:
        """Handle parameter changes from the step editor widget."""
        logger.debug("Received StepParameterChanged from child StepParameterEditorWidget")

        # Sync step configuration settings to function editor for dynamic component selection
        self.func_editor.current_group_by = self.editing_step.group_by
        self.func_editor.current_variable_components = self.editing_step.variable_components or []

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

        # Always keep save button enabled - user requested this for better UX
        # No need to disable save button based on changes

        # Update save button state - always enabled
        try:
            # Find the save button tab by looking for button tabs
            # Use try_query to avoid NoMatches exceptions during widget lifecycle
            tabs = self.query(ButtonTab)
            if tabs:  # Check if any tabs were found
                for tab in tabs:
                    if hasattr(tab, 'button_id') and tab.button_id == "save":
                        tab.disabled = False  # Always enabled
                        logger.debug(f"Save button always enabled (user preference)")
                        break
            else:
                logger.debug("No ButtonTab widgets found yet (widget still mounting?)")
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
        """Update status message - now just logs since we removed the status bar."""
        logger.debug(f"Status: {message}")

    def watch_has_changes(self, has_changes: bool) -> None:
        """React to changes in has_changes state."""
        # Update window title to show unsaved changes
        base_title = "Dual Editor - Edit Step" if not self.is_new else "Dual Editor - New Step"
        if has_changes:
            self.title = f"{base_title} *"
        else:
            self.title = base_title



    def _handle_save(self) -> None:
        """Handle save button with validation and type conversion."""
        # Sync current UI values to editing_step before validation
        self._sync_ui_to_editing_step()

        # Debug logging to see what's happening with step name
        logger.debug(f"Step name validation - editing_step.name: '{self.editing_step.name}', type: {type(self.editing_step.name)}")

        # Validate step data
        if not self.editing_step.name or not self.editing_step.name.strip():
            self._update_status("Error: Step name cannot be empty")
            return

        if not self.editing_step.func:
            self._update_status("Error: Function pattern cannot be empty")
            return

        # Validate and convert function parameter types
        validation_errors = self._validate_and_convert_function_parameters()
        if validation_errors:
            # Show error dialog with specific validation errors
            from openhcs.textual_tui.app import ErrorDialog
            error_message = "Parameter validation failed. Please fix the following issues:"
            error_details = "\n".join(f"• {error}" for error in validation_errors)
            error_dialog = ErrorDialog(error_message, error_details)
            self.app.push_screen(error_dialog)
            self._update_status("Error: Invalid parameter values")
            return

        # Save successful
        self._update_status("Saved successfully")
        if self.on_save_callback:
            self.on_save_callback(self.editing_step)
        self.close_window()

    def _validate_and_convert_function_parameters(self) -> List[str]:
        """
        Validate and convert all function parameters using type hints.

        Returns:
            List of error messages. Empty list if all parameters are valid.
        """
        errors = []

        try:
            # Get current function pattern from the editor
            current_pattern = self.func_editor.current_pattern

            # Handle different pattern types (list or dict)
            functions_to_validate = []
            if isinstance(current_pattern, list):
                functions_to_validate = current_pattern
            elif isinstance(current_pattern, dict):
                # Flatten all functions from all channels
                for channel_functions in current_pattern.values():
                    if isinstance(channel_functions, list):
                        functions_to_validate.extend(channel_functions)

            # Validate each function
            for func_index, func_item in enumerate(functions_to_validate):
                if isinstance(func_item, tuple) and len(func_item) == 2:
                    func, kwargs = func_item

                    # Get expected parameter types from function signature
                    from openhcs.textual_tui.widgets.shared.signature_analyzer import SignatureAnalyzer
                    from openhcs.textual_tui.widgets.shared.parameter_form_manager import ParameterFormManager

                    param_info = SignatureAnalyzer.analyze(func)

                    # Validate each parameter
                    for param_name, info in param_info.items():
                        if param_name in kwargs:
                            current_value = kwargs[param_name]
                            expected_type = info.param_type

                            # Skip if value is already the correct type (not a string)
                            if not isinstance(current_value, str):
                                continue

                            # Try to convert using the enhanced type converter
                            try:
                                converted_value = ParameterFormManager.convert_string_to_type(
                                    current_value, expected_type, strict=True
                                )
                                # Update the kwargs with the converted value
                                kwargs[param_name] = converted_value

                            except ValueError as e:
                                # Collect the error message
                                func_name = getattr(func, '__name__', str(func))
                                errors.append(f"Function '{func_name}', parameter '{param_name}': {str(e)}")

        except Exception as e:
            # Catch any unexpected errors during validation
            errors.append(f"Validation error: {str(e)}")

        return errors

    def _sync_ui_to_editing_step(self) -> None:
        """Sync current UI values to the editing_step object before validation."""
        try:
            # Sync step editor values (name, group_by, variable_components)
            if self.step_editor:
                # The step editor should have already updated editing_step via messages,
                # but let's make sure by getting current values
                pass  # StepParameterEditorWidget updates editing_step directly

            # Sync function editor values (func pattern)
            if self.func_editor:
                self.editing_step.func = self.func_editor.current_pattern

        except Exception as e:
            # Log but don't fail - validation will catch issues
            logger.debug(f"Error syncing UI to editing_step: {e}")

    def _handle_cancel(self) -> None:
        """Handle cancel button with change confirmation."""
        if self.has_changes:
            # TODO: Add confirmation dialog for unsaved changes
            self._update_status("Cancelled - changes discarded")
        else:
            self._update_status("Cancelled")

        if self.on_save_callback:
            self.on_save_callback(None)
        self.close_window()
