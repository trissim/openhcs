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

# Registry of components with special styling requirements
UI_STYLING_EXCEPTIONS = {
    'ScrollablePane': {
        'skip_auto_style': ['height'],
        'reason': 'Must calculate height from content - see prompt_toolkit docs'
    }
}


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
    """Frame with intelligent width management to prevent wall collapse (semantic styling only)."""
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

        # Apply to frames marked with dialog-content style
        if 'class:dialog-content' in style:
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
    """Dialog that lets prompt-toolkit handle sizing naturally."""
    def __init__(self, title='', body=None, buttons=None, width=None, **kwargs):
        from prompt_toolkit.widgets import Box

        # Let prompt-toolkit handle width naturally - don't override unless explicitly set
        # This allows Float centering to work with the dialog's actual rendered size

        # Add minimal content padding using Box wrapper for better readability
        if body is not None:
            body = Box(
                body,
                padding_left=1,
                padding_right=1,
                padding_top=1,
                padding_bottom=1
            )

        super().__init__(title=title, body=body, buttons=buttons, width=width, **kwargs)






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
    """ScrollablePane with OpenHCS styling that respects behavioral requirements."""
    def __init__(self, content, height=None, show_scrollbar=True, display_arrows=True, **kwargs):
        from prompt_toolkit.layout.dimension import Dimension

        # Check registry - don't auto-style height for ScrollablePane
        exceptions = UI_STYLING_EXCEPTIONS.get('ScrollablePane', {})
        skip_params = exceptions.get('skip_auto_style', [])

        # DON'T set height if it's in skip list
        if height is None and 'height' not in skip_params:
            height = Dimension(weight=1)
        # For ScrollablePane, height stays None to allow natural scaling

        super().__init__(
            content=content,
            height=height,
            show_scrollbar=show_scrollbar,
            display_arrows=display_arrows,
            **kwargs
        )


class _StyledLabel(OriginalLabel):
    """Label with semantic styling only - no width manipulation."""
    def __init__(self, text='', style='', width=None, dont_extend_width=False, dont_extend_height=False, **kwargs):
        # Pure pass-through - no behavioral changes
        super().__init__(text=text, style=style, width=width,
                        dont_extend_width=dont_extend_width,
                        dont_extend_height=dont_extend_height, **kwargs)


# Apply global styling overrides
prompt_toolkit.widgets.Button = _StyledButton
prompt_toolkit.widgets.TextArea = _StyledTextArea
prompt_toolkit.widgets.Label = _StyledLabel
prompt_toolkit.widgets.Frame = _StyledFrame  # RE-ENABLED
prompt_toolkit.widgets.Dialog = _ResponsiveDialog

# Apply global layout container overrides to prevent wall collapse
# prompt_toolkit.layout.containers.VSplit = _StyledVSplit  # DISABLED
prompt_toolkit.layout.ScrollablePane = _StyledScrollablePane  # RE-ENABLED with registry pattern

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
