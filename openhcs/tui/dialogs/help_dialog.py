"""
Help Dialog for OpenHCS TUI.

This module implements a simple modal dialog to display help information.
"""
import asyncio
import logging
from typing import Optional

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import HSplit, Float, Dimension
from prompt_toolkit.widgets import Button, Dialog, Label, FormattedTextControl

logger = logging.getLogger(__name__)

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
        
        ok_button = Button("OK", handler=self._handle_ok)

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
        if not self.dialog:
            # This should not happen if __init__ is called correctly
            logger.error("HelpDialog: Dialog not built before show() called.")
            return

        app = get_app()
        self.future = asyncio.Future()

        # Store previous focus to restore it later
        previous_focus = app.layout.current_window if hasattr(app.layout, 'current_window') else None
        
        # Create a Float to display the dialog
        float_ = Float(content=self.dialog)
        # Store the float on the dialog to remove it later (optional, but good practice)
        setattr(self.dialog, '__ohcs_float__', float_)

        app.layout.container.floats.append(float_)
        app.layout.focus(self.dialog) # Focus the dialog itself

        try:
            await self.future
        finally:
            # Clean up: remove the float and restore focus
            if float_ in app.layout.container.floats:
                app.layout.container.floats.remove(float_)
            
            if previous_focus and hasattr(app.layout, 'walk'): # Check if layout has walk method
                try:
                    # Check if previous_focus widget is still part of the layout
                    is_still_in_layout = False
                    for elem in app.layout.walk(skip_hidden=True): # Iterate over visible elements
                        if elem == previous_focus:
                            is_still_in_layout = True
                            break
                    if is_still_in_layout:
                        app.layout.focus(previous_focus)
                    else: # Fallback if previous focus target is gone
                        app.layout.focus_last() 
                except Exception as e: # Broad exception for focus restoration issues
                    logger.warning(f"HelpDialog: Could not restore focus: {e}")
                    try:
                        app.layout.focus_last() # Try to focus something else
                    except Exception:
                        pass # Silently ignore if even this fails
            elif hasattr(app.layout, 'focus_last'): # Fallback if walk isn't available or previous_focus is None
                 app.layout.focus_last()

# Example of how ShowHelpCommand might use this:
# from .help_dialog import HelpDialog
# class ShowHelpCommand(Command):
#     async def execute(self, app_state: Optional[Any] = None, context: Optional[Any] = None, **kwargs):
#         dialog = HelpDialog()
#         await dialog.show()
#         logger.info("Help dialog closed.")
```
