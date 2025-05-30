"""
Help Dialog for OpenHCS TUI.

This module implements a simple modal dialog to display help information.
"""
import asyncio
import logging
from typing import Optional

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import HSplit, Float, Dimension, FormattedTextControl
from prompt_toolkit.widgets import Button, Dialog, Label

logger = logging.getLogger(__name__)

# Define SafeButton locally to avoid circular imports
class SafeButton(Button):
    """Safe wrapper around Button that handles formatting errors."""
    
    def __init__(self, text="", handler=None, width=None, **kwargs):
        # Sanitize text before passing to parent
        if text is not None:
            text = str(text).replace('{', '{{').replace('}', '}}').replace(':', ' ')
        super().__init__(text=text, handler=handler, width=width, **kwargs)
    
    def _get_text_fragments(self):
        """FAIL FAST: No fallback text formatting - if formatting fails, crash immediately."""
        return super()._get_text_fragments()

DEFAULT_HELP_TEXT = """
OpenHCS TUI - Help

- Use arrow keys or Vim keys (j, k) to navigate lists.
- Use Enter to select or activate items.
- Buttons can be clicked with the mouse or navigated to with Tab.

Global Settings:
  Access via the top bar to configure system-wide parameters.
  Changes here affect all plates and pipelines.

Plate Manager (Pane 1):
  - Add: Discover new plates (directories) to be managed by OpenHCS.
  - Del: Remove a plate from OpenHCS management (does not delete files).
  - Edit: Modify plate-specific configurations (e.g., custom metadata).
  - Init: Prepares the plate's workspace, discovers images, and validates metadata.
  - Compile: Validates the pipeline and prepares it for execution against the plate.
  - Run: Executes the compiled pipeline on the selected plate.

Pipeline Editor (Pane 2):
  - Add: Add a new processing step to the current pipeline.
  - Del: Remove the selected step from the pipeline.
  - Edit: Modify the parameters of the selected step using the Step/Function Editor.
  - Load: Load a pipeline definition from a .pipeline file.
  - Save: Save the current pipeline definition to a .pipeline file.

Step/Function Editor (activated when editing a step):
  - Step Settings View: Edit general parameters of the step (name, input/output, etc.).
  - Func Pattern View: Edit the specific function and its arguments for the step.
  - Load/Save .step: Load or save individual step configurations.

More detailed documentation can be found at [link to docs if available].
"""

class HelpDialog:
    """
    A simple modal dialog to display help text.
    """
    def __init__(self, help_text: Optional[str] = None):
        """
        Initialize the HelpDialog.

        Args:
            help_text: Optional custom help text. If None, uses default.
        """
        self.help_text_content = help_text or DEFAULT_HELP_TEXT
        self.future: Optional[asyncio.Future[None]] = None

        # UI Elements
        # Using FormattedTextControl for potentially better formatting control later if needed
        # For simple text, Label is also fine.
        help_body = FormattedTextControl(text=self.help_text_content, focusable=True, show_cursor=False)

        ok_button = SafeButton("OK", handler=self._handle_ok)

        self.dialog = Dialog(
            title="OpenHCS Help",
            body=HSplit([
                # Using a simple Label for now, can be replaced with ScrollablePane if help text is very long
                Label(text=self.help_text_content, dont_extend_height=False, dont_extend_width=False)
            ], width=Dimension(preferred=80), height=Dimension(preferred=25)), # Fixed size for now
            buttons=[ok_button],
            modal=True, # Ensures it's modal
            width=Dimension(preferred=80, max=100),
            height=Dimension(preferred=25, max=30)
        )

    def _handle_ok(self):
        """Handles the OK button press."""
        if self.future and not self.future.done():
            self.future.set_result(None)

    async def show(self) -> None:
        """
        Displays the help dialog modally and waits until it's closed.
        """
        if not self._validate_dialog_ready():
            return

        app = get_app()
        self.future = asyncio.Future()

        previous_focus = self._store_previous_focus(app)
        float_ = self._create_and_display_dialog(app)

        try:
            await self.future
        finally:
            self._cleanup_dialog(app, float_, previous_focus)

    def _validate_dialog_ready(self) -> bool:
        """Validate that dialog is ready to be shown."""
        if not self.dialog:
            logger.error("HelpDialog: Dialog not built before show() called.")
            return False
        return True

    def _store_previous_focus(self, app):
        """Store previous focus to restore it later."""
        return app.layout.current_window if hasattr(app.layout, 'current_window') else None

    def _create_and_display_dialog(self, app) -> Float:
        """Create float container and display dialog."""
        float_ = Float(content=self.dialog)
        setattr(self.dialog, '__ohcs_float__', float_)

        app.layout.container.floats.append(float_)
        app.layout.focus(self.dialog)

        return float_

    def _cleanup_dialog(self, app, float_: Float, previous_focus) -> None:
        """Clean up dialog and restore focus."""
        self._remove_float_container(app, float_)
        self._restore_focus(app, previous_focus)

    def _remove_float_container(self, app, float_: Float) -> None:
        """Remove float container from layout."""
        if float_ in app.layout.container.floats:
            app.layout.container.floats.remove(float_)

    def _restore_focus(self, app, previous_focus) -> None:
        """Restore focus to previous widget or fallback."""
        if previous_focus and hasattr(app.layout, 'walk'):
            self._try_restore_previous_focus(app, previous_focus)
        elif hasattr(app.layout, 'focus_last'):
            app.layout.focus_last()

    def _try_restore_previous_focus(self, app, previous_focus) -> None:
        """Try to restore focus to previous widget."""
        try:
            if self._is_widget_still_in_layout(app, previous_focus):
                app.layout.focus(previous_focus)
            else:
                app.layout.focus_last()
        except Exception as e:
            self._handle_focus_restoration_error(app, e)

    def _is_widget_still_in_layout(self, app, widget) -> bool:
        """Check if widget is still part of the layout."""
        for elem in app.layout.walk(skip_hidden=True):
            if elem == widget:
                return True
        return False

    def _handle_focus_restoration_error(self, app, error: Exception) -> None:
        """Handle errors during focus restoration."""
        logger.warning(f"HelpDialog: Could not restore focus: {error}")
        try:
            app.layout.focus_last()
        except Exception:
            pass  # Silently ignore if even this fails

# Example of how ShowHelpCommand might use this:
# from .help_dialog import HelpDialog
# class ShowHelpCommand(Command):
#     async def execute(self, app_state: Optional[Any] = None, context: Optional[Any] = None, **kwargs):
#         dialog = HelpDialog()
#         await dialog.show()
#         logger.info("Help dialog closed.")
