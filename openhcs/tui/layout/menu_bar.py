"""
Simple MenuBar with 3 buttons and OpenHCS label.
"""
import logging
from typing import TYPE_CHECKING

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import Container, VSplit, Window, Dimension
from prompt_toolkit.widgets import Label

from openhcs.tui.components import FramedButton
from openhcs.tui.components.config_editor import ConfigEditor
from openhcs.core.config import GlobalPipelineConfig
from prompt_toolkit.widgets import Dialog, Label, Button
from prompt_toolkit.layout.containers import HSplit
from openhcs.tui.utils.button_utils import dialog_button

if TYPE_CHECKING:
    from openhcs.core.context.processing_context import ProcessingContext

logger = logging.getLogger(__name__)


class MenuBar(Container):
    """Simple menu bar with 3 buttons and OpenHCS label."""

    def __init__(self, state, context: 'ProcessingContext'):
        """Initialize the simple menu bar."""
        self.state = state
        self.context = context

        # Create buttons with dynamic width calculation
        from openhcs.tui.utils.unified_task_manager import get_task_manager

        self.global_config_button = FramedButton(
            "Global Config",
            handler=lambda: get_task_manager().fire_and_forget(self._handle_global_config(), "global_config"),
            width=len("Global Config") + 2
        )

        self.help_button = FramedButton(
            "Help",
            handler=lambda: get_task_manager().fire_and_forget(self._handle_help(), "help"),
            width=len("Help") + 2
        )

        self.quit_button = FramedButton(
            "Quit",
            handler=lambda: get_task_manager().fire_and_forget(self._handle_quit(), "quit"),
            width=len("Quit") + 2
        )

        # Create 3-line layout with buttons and centered connecting line
        self.container = self._create_three_line_menu()

        logger.info("MenuBar: Simple menu bar initialized")

    def _create_three_line_menu(self) -> Container:
        """Create 3-line menu with dynamic text and buttons."""
        from prompt_toolkit.layout.controls import FormattedTextControl

        def get_menubar_text():
            """Build the menu bar text with centered title and connecting line."""
            try:
                width = get_app().output.get_size().columns
            except:
                width = 80  # Fallback width

            title = " OpenHCS v1.0 "
            title_len = len(title)

            # Calculate how much space for the line on each side
            available_space = width - title_len
            left_space = available_space // 2
            right_space = available_space - left_space

            # Build the line: ─────── OpenHCS v1.0 ───────
            line = '─' * left_space + title + '─' * right_space

            return [
                ('', '\n'),  # Line 1: empty
                ('class:title', line),  # Line 2: connecting line with title
                ('', '\n'),  # Line 3: empty
            ]

        # Create the text-based menu bar
        menubar_text = Window(
            FormattedTextControl(get_menubar_text),
            height=3,
        )

        # Overlay buttons using FloatContainer
        from prompt_toolkit.layout.containers import FloatContainer, Float

        return FloatContainer(
            content=menubar_text,
            floats=[
                # Left buttons
                Float(
                    content=VSplit([
                        self.global_config_button,
                        Window(width=1, char=' '),
                        self.help_button,
                    ]),
                    left=0,
                    top=0
                ),
                # Right button
                Float(
                    content=self.quit_button,
                    right=0,
                    top=0
                ),
            ]
        )

    async def _handle_global_config(self):
        """Handle Global Config button."""
        logger.info("Global Config button clicked")

        # Create config editor using new API
        config_editor = ConfigEditor(
            config_class=GlobalPipelineConfig,
            current_config=self.context.global_config,
            backend=getattr(self.context, 'backend', 'disk'),
            scope="global",
            on_config_change=self._on_global_config_change,
            on_reset_field=self._on_global_config_reset_field,
            on_reset_all=self._on_global_config_reset_all
        )

        # Build the UI container
        config_container = config_editor.build_ui()

        def save_and_close():
            # Get updated config from editor
            updated_config = config_editor.get_current_config()
            logger.info(f"Global config updated: {updated_config}")
            self._hide_dialog()

        def cancel_and_close():
            self._hide_dialog()

        # Create dialog with config editor
        settings_dialog = Dialog(
            title="Global Settings",
            body=HSplit([
                config_container,
                VSplit([
                    dialog_button("Save", handler=save_and_close),
                    Window(width=2, char=' '),  # Spacer
                    dialog_button("Cancel", handler=cancel_and_close)
                ], height=1)
            ]),
            buttons=[],
            modal=True
        )

        self._show_dialog(settings_dialog)

    async def _handle_help(self):
        """Handle Help button."""
        logger.info("Help button clicked")

        def close_dialog():
            self._hide_dialog()

        # Create help dialog content
        help_content = HSplit([
            Label("OpenHCS - Open High-Content Screening"),
            Label(""),
            Label("🔬 Visual Programming for Cell Biology Research"),
            Label(""),
            Label("Key Features:"),
            Label("• GPU-accelerated image processing"),
            Label("• Visual pipeline building"),
            Label("• Multi-backend storage support"),
            Label("• Real-time parameter editing"),
            Label(""),
            Label("Workflow:"),
            Label("1. Add Plate → Select microscopy data"),
            Label("2. Edit Step → Visual function selection"),
            Label("3. Compile → Create execution plan"),
            Label("4. Run → Process images"),
            Label(""),
            Label("For detailed documentation, see Nature Methods publication."),
            Label(""),
            dialog_button("Close", handler=close_dialog)
        ])

        help_dialog = Dialog(
            title="OpenHCS Help",
            body=help_content,
            buttons=[],
            modal=True
        )

        self._show_dialog(help_dialog)

    async def _handle_quit(self):
        """Handle Quit button."""
        get_app().exit()

    def _show_dialog(self, dialog):
        """Show a dialog by adding it to the layout."""
        # Get the current layout and add the dialog as a float
        layout = get_app().layout
        if hasattr(layout, 'container') and hasattr(layout.container, 'floats'):
            from prompt_toolkit.layout.containers import Float
            float_dialog = Float(content=dialog)
            layout.container.floats.append(float_dialog)
            get_app().invalidate()

    def _hide_dialog(self):
        """Hide the current dialog by removing it from the layout."""
        layout = get_app().layout
        if hasattr(layout, 'container') and hasattr(layout.container, 'floats'):
            # Remove the last float (dialog)
            if layout.container.floats:
                layout.container.floats.pop()
                get_app().invalidate()

    async def _on_global_config_change(self, field_name: str, value, scope: str):
        """Handle global config field change."""
        logger.info(f"Global config field changed: {field_name} = {value}")

    async def _on_global_config_reset_field(self, field_name: str, scope: str):
        """Handle global config field reset."""
        logger.info(f"Global config field reset: {field_name}")

    async def _on_global_config_reset_all(self, scope: str):
        """Handle global config reset all."""
        logger.info("Global config reset all")

    def __pt_container__(self) -> Container:
        """Return the container to render."""
        return self.container

    # Implement abstract methods by delegating to the internal container
    def get_children(self):
        return self.container.get_children()

    def preferred_width(self, max_available_width):
        return self.container.preferred_width(max_available_width)

    def preferred_height(self, max_available_height, width):
        return self.container.preferred_height(max_available_height, width)

    def reset(self):
        self.container.reset()

    def write_to_screen(self, screen, mouse_handlers, write_position,
                        parent_style, erase_bg, z_index):
        self.container.write_to_screen(screen, mouse_handlers, write_position,
                                       parent_style, erase_bg, z_index)

    def mouse_handler(self, mouse_event):
        """Handle mouse events for the menu bar."""
        return self.container.mouse_handler(mouse_event)

    async def shutdown(self):
        """Cleanup method."""
        logger.info("MenuBar: Shutdown complete")
