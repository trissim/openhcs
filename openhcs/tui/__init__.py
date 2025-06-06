"""
Terminal User Interface (TUI) for OpenHCS.

This package provides a TUI for interacting with OpenHCS, featuring:
- Clean layout architecture
- Modular editor components
- Pipeline visualization
- Step configuration
"""

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL DECLARATIVE STYLING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# This section globally overrides prompt_toolkit components for consistent styling
# across the entire TUI without manual configuration in every file.

import prompt_toolkit.widgets
import prompt_toolkit.layout.containers
import prompt_toolkit.layout
from prompt_toolkit.widgets import Button as OriginalButton, TextArea as OriginalTextArea, Frame as OriginalFrame, Dialog as OriginalDialog, Label as OriginalLabel
from prompt_toolkit.layout.containers import VSplit as OriginalVSplit
from prompt_toolkit.layout import ScrollablePane as OriginalScrollablePane


class _StyledButton(OriginalButton):
    """Button with [ ] symbols and async handler support."""
    def __init__(self, text, handler=None, width=12, left_symbol='[', right_symbol=']'):
        # Auto-wrap handler for async capability
        if handler:
            from openhcs.tui.utils.async_handler import make_async_capable
            handler = make_async_capable(handler, f"button_{text}")

        super().__init__(text, handler, width, left_symbol, right_symbol)


class _StyledTextArea(OriginalTextArea):
    """TextArea with sensible defaults for single-line input fields."""
    def __init__(self, text='', multiline=None, height=None, **kwargs):
        # Smart defaults: if height=1 specified, assume single-line
        if height == 1 and multiline is None:
            multiline = False
        # If multiline not specified and height=1, make it single-line
        if multiline is None and height == 1:
            multiline = False

        # Don't override style - let the original TextArea handle it
        super().__init__(text=text, multiline=multiline, height=height, **kwargs)


class _StyledFrame(OriginalFrame):
    """Frame with consistent styling AND intelligent width management to prevent wall collapse."""
    def __init__(self, body, title='', style='class:frame', width=None, height=None, **kwargs):
        from prompt_toolkit.layout.dimension import Dimension

        # Only apply automatic width management for main content frames
        # Don't override if width/height are explicitly provided
        if width is None and self._should_auto_manage_width(title, style):
            width = Dimension(weight=1)
        if height is None and self._should_auto_manage_height(title, style):
            height = Dimension(weight=1)

        super().__init__(body, title=title, style=style, width=width, height=height, **kwargs)

    def _should_auto_manage_width(self, title, style):
        """Determine if this frame should get automatic width management."""
        # Apply to main content frames (like PlateManager, PipelineEditor)
        main_content_titles = ['Plate Manager', 'Pipeline Editor', 'Function Pattern Editor']
        if any(main_title in str(title) for main_title in main_content_titles):
            return True
        # Don't apply to menu bars, status bars, or button containers
        if 'menu' in str(title).lower() or 'status' in str(title).lower():
            return False
        return False

    def _should_auto_manage_height(self, title, style):
        """Determine if this frame should get automatic height management."""
        # Apply to main content frames
        main_content_titles = ['Plate Manager', 'Pipeline Editor', 'Function Pattern Editor']
        if any(main_title in str(title) for main_title in main_content_titles):
            return True
        return False


# Global dialog styling parameters
DIALOG_CONTENT_PADDING_LEFT = 3
DIALOG_CONTENT_PADDING_RIGHT = 3
DIALOG_CONTENT_PADDING_TOP = 1
DIALOG_CONTENT_PADDING_BOTTOM = 1
DIALOG_FRAME_BORDER_WIDTH = 4  # 2 chars for left border + 2 chars for right border
DIALOG_MIN_WIDTH = 60
DIALOG_MAX_WIDTH_RATIO = 0.7  # 70% of terminal width


class _ResponsiveDialog(OriginalDialog):
    """Dialog with content-aware width management - use the calculated content width."""
    def __init__(self, title='', body=None, buttons=None, width=None, **kwargs):
        from prompt_toolkit.widgets import Box

        # Calculate the actual content width (longest line + padding)
        if width is None:
            content_width = self._calculate_actual_content_width(title, body, buttons)
            # Add padding to get the total width - this same value used everywhere
            total_padding = DIALOG_CONTENT_PADDING_LEFT + DIALOG_CONTENT_PADDING_RIGHT
            width = content_width + total_padding

        # Add content padding using Box wrapper
        if body is not None:
            body = Box(
                body,
                padding_left=DIALOG_CONTENT_PADDING_LEFT,
                padding_right=DIALOG_CONTENT_PADDING_RIGHT,
                padding_top=DIALOG_CONTENT_PADDING_TOP,
                padding_bottom=DIALOG_CONTENT_PADDING_BOTTOM
            )

        super().__init__(title=title, body=body, buttons=buttons, width=width, **kwargs)

    def _calculate_actual_content_width(self, title, body, buttons):
        """Calculate the actual width needed by finding the longest text line."""
        max_width = len(str(title)) if title else 0

        # Analyze body content for actual text width
        if body is not None:
            body_width = self._analyze_body_text_width(body)
            max_width = max(max_width, body_width)

        # Account for buttons
        if buttons:
            button_width = sum(len(str(getattr(btn, 'text', str(btn)))) + 4 for btn in buttons) + len(buttons) * 2
            max_width = max(max_width, button_width)

        return max_width

    def _analyze_body_text_width(self, body):
        """Recursively analyze body content to find the longest text line."""
        max_width = 0

        # Handle different body types
        if hasattr(body, 'text'):
            # Label or similar text widget
            text = str(body.text)
            lines = text.split('\n')
            if lines:
                max_width = max(len(line.rstrip()) for line in lines)  # Strip right whitespace
        elif hasattr(body, 'children'):
            # Container with children - recursively check
            for child in body.children:
                child_width = self._analyze_body_text_width(child)
                max_width = max(max_width, child_width)
        elif hasattr(body, 'content'):
            # Window or similar - check content
            max_width = self._analyze_body_text_width(body.content)

        return max_width




class _StyledVSplit(OriginalVSplit):
    """VSplit with intelligent width management for dual-pane layouts."""
    def __init__(self, children, height=None, style='', **kwargs):
        from prompt_toolkit.layout.dimension import Dimension

        # Only apply height management for main content areas
        if height is None and self._is_main_content_vsplit(children, style):
            height = Dimension(weight=1)

        # Only apply equal weight to children in main content dual-pane layouts
        if self._is_dual_pane_layout(children):
            for child in children:
                if hasattr(child, 'width') and getattr(child, 'width', None) is None:
                    child.width = Dimension(weight=1)

        super().__init__(children, height=height, style=style, **kwargs)

    def _is_main_content_vsplit(self, children, style):
        """Check if this is a main content VSplit that needs height management."""
        # Look for main-content style or dual-pane pattern
        if 'main-content' in style:
            return True
        # Check if it's a dual-pane layout (exactly 2 children)
        return len(children) == 2

    def _is_dual_pane_layout(self, children):
        """Check if this is a dual-pane layout that needs equal width distribution."""
        # Only apply to exactly 2 children (dual pane)
        return len(children) == 2


class _StyledScrollablePane(OriginalScrollablePane):
    """ScrollablePane with intelligent height management."""
    def __init__(self, content, height=None, show_scrollbar=True, display_arrows=True, **kwargs):
        from prompt_toolkit.layout.dimension import Dimension

        # Only apply automatic height management if no height specified
        # and it looks like a main content scrollable area
        if height is None and self._should_auto_manage_height(content):
            height = Dimension(weight=1)

        super().__init__(content, height=height, show_scrollbar=show_scrollbar,
                        display_arrows=display_arrows, **kwargs)

    def _should_auto_manage_height(self, content):
        """Determine if this scrollable pane should get automatic height management."""
        # Apply to content that looks like main scrollable areas
        # Don't apply to small lists or button containers
        return True  # For now, be conservative and apply to all


class _ResponsiveLabel(OriginalLabel):
    """Label with intelligent width management to prevent dialog content collapse."""
    def __init__(self, text='', style='', width=None, dont_extend_width=False, dont_extend_height=False, **kwargs):
        from prompt_toolkit.layout.dimension import Dimension

        # If width is a small fixed value (like 25 in ConfigEditor), make it more flexible
        if isinstance(width, int) and width <= 40:
            # Convert small fixed widths to minimum widths with some flexibility
            width = Dimension(min=width, preferred=width + 10, max=width + 30)

        super().__init__(text=text, style=style, width=width,
                        dont_extend_width=dont_extend_width,
                        dont_extend_height=dont_extend_height, **kwargs)


# Apply global styling overrides
prompt_toolkit.widgets.Button = _StyledButton
prompt_toolkit.widgets.TextArea = _StyledTextArea
prompt_toolkit.widgets.Label = _ResponsiveLabel
prompt_toolkit.widgets.Frame = _StyledFrame
prompt_toolkit.widgets.Dialog = _ResponsiveDialog

# Apply global layout container overrides to prevent wall collapse
prompt_toolkit.layout.containers.VSplit = _StyledVSplit
prompt_toolkit.layout.ScrollablePane = _StyledScrollablePane

# Install async handler architecture for sync→async bridging (Frame and Window only)
from openhcs.tui.utils.async_handler import install_frame_and_window_async_handlers
install_frame_and_window_async_handlers()

# Width management is now handled by class overrides above
import logging
logger = logging.getLogger(__name__)
logger.info("Automatic width management enabled via class overrides - all Frame/VSplit/ScrollablePane containers will prevent wall collapse")

# Import key components from organized submodules
from openhcs.tui.layout import CanonicalTUILayout, SimpleOpenHCSTUILauncher
from openhcs.tui.editors import StepParameterEditor, DualEditorPane
# FunctionPatternEditor replaced by FunctionPatternView in views module

__all__ = [
    'CanonicalTUILayout',
    'SimpleOpenHCSTUILauncher',
    'StepParameterEditor',
    'DualEditorPane',
    # FunctionPatternEditor replaced by FunctionPatternView in views module
]
