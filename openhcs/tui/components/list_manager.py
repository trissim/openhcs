"""
Clean List Manager Components for OpenHCS TUI.

Provides reusable list management infrastructure with proper MVC separation:
- ListConfig: Declarative configuration
- ListModel: Pure data model with observer pattern
- ListView: UI view that observes model changes
- ListManagerPane: Coordinator that composes model and view
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import Container, HSplit, VSplit, DynamicContainer, Dimension, Window
from prompt_toolkit.widgets import Button, Frame, Label

from .interactive_list_item import InteractiveListItem
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout import ScrollablePane
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.filters import Condition
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ButtonConfig:
    """Configuration for a button in the button bar."""
    text: str
    handler: Callable[[], Any]
    width: Optional[int] = None
    enabled_func: Optional[Callable[[], bool]] = None
    
    def is_enabled(self) -> bool:
        """Check if button should be enabled."""
        return self.enabled_func() if self.enabled_func else True


@dataclass
class ListConfig:
    """Configuration for a list manager pane."""
    title: str
    frame_title: Optional[str] = None
    button_configs: List[ButtonConfig] = field(default_factory=list)
    display_func: Optional[Callable[[Dict[str, Any], bool], str]] = None
    can_move_up_func: Optional[Callable[[int, Dict[str, Any]], bool]] = None
    can_move_down_func: Optional[Callable[[int, Dict[str, Any]], bool]] = None
    empty_message: str = "No items available."
    allow_multi_select: bool = False  # Enable checkbox multi-selection
    bulk_button_configs: List[ButtonConfig] = field(default_factory=list)  # Bulk operation buttons


class ListModel:
    """Model for list state - pure data, no UI concerns."""
    
    def __init__(self):
        self.items: List[Dict[str, Any]] = []
        self.selected_index: int = 0  # Currently focused item
        self.selected_indices: set[int] = set()  # Track selected items by index
        self._observers: List[Callable] = []
    
    def add_observer(self, callback: Callable):
        """Add observer for model changes."""
        self._observers.append(callback)
    
    def _notify_observers(self):
        """Notify all observers of model changes."""
        for callback in self._observers:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in model observer: {e}", exc_info=True)
    
    def set_items(self, items: List[Dict[str, Any]]):
        """Set items and notify observers."""
        self.items = items
        self.selected_index = min(self.selected_index, len(items) - 1) if items else 0
        self._notify_observers()
    
    def select_item(self, index: int):
        """Select item by index (single selection for compatibility)."""
        if 0 <= index < len(self.items):
            self.selected_index = index
            self._notify_observers()
    
    def move_item_up(self, index: int) -> bool:
        """Move item up, return True if moved."""
        if index > 0 and index < len(self.items):
            self.items[index], self.items[index - 1] = self.items[index - 1], self.items[index]
            self.selected_index = index - 1
            self._notify_observers()
            return True
        return False
    
    def move_item_down(self, index: int) -> bool:
        """Move item down, return True if moved."""
        if index >= 0 and index < len(self.items) - 1:
            self.items[index], self.items[index + 1] = self.items[index + 1], self.items[index]
            self.selected_index = index + 1
            self._notify_observers()
            return True
        return False
    
    def remove_item(self, index: int) -> bool:
        """Remove item, return True if removed."""
        if 0 <= index < len(self.items):
            self.items.pop(index)
            if self.selected_index >= len(self.items) and self.items:
                self.selected_index = len(self.items) - 1
            self._notify_observers()
            return True
        return False

    def toggle_selection(self, index: int) -> None:
        """Toggle selection state of item at index."""
        if not (0 <= index < len(self.items)):
            return

        if index in self.selected_indices:
            self.selected_indices.remove(index)
        else:
            self.selected_indices.add(index)
        self._notify_observers()

    def get_selected_items(self) -> List[Dict[str, Any]]:
        """Get list of currently selected items."""
        selected_items = []
        for index in self.selected_indices:
            if 0 <= index < len(self.items):
                selected_items.append(self.items[index])
        return selected_items

    def clear_selection(self) -> None:
        """Clear all selections."""
        if self.selected_indices:
            self.selected_indices.clear()
            self._notify_observers()

    def set_focused_index(self, index: int) -> None:
        """Set focused index with bounds checking."""
        if 0 <= index < len(self.items):
            self.selected_index = index
            self._notify_observers()

    # Aliases for PlateManager compatibility
    def get_checked_items(self) -> List[Dict[str, Any]]:
        """Get all checked items (alias for get_selected_items)."""
        return self.get_selected_items()

    def get_all_items(self) -> List[Dict[str, Any]]:
        """Get all items in the list."""
        return self.items.copy() if self.items else []

    def clear_all_checks(self) -> None:
        """Clear all checkbox selections (alias for clear_selection)."""
        self.clear_selection()

    def remove_item_by_data(self, item_data: Dict[str, Any]) -> bool:
        """Remove item by data reference."""
        try:
            index = self.items.index(item_data)
            return self.remove_item(index)
        except ValueError:
            return False

    def notify_observers(self) -> None:
        """Public method to notify observers (alias for _notify_observers)."""
        self._notify_observers()


class ListView:
    """View for list display - observes model, updates UI automatically."""
    
    def __init__(self, model: ListModel, config: ListConfig):
        self.model = model
        self.config = config
        self.list_control: Optional[FormattedTextControl] = None  # Will be created in _create_current_list
        self.allow_multi_select: bool = getattr(config, 'allow_multi_select', False)

        # UI components (built once, updated automatically)
        self.container = self._build_container()

        # Observe model changes
        self.model.add_observer(self._on_model_changed)
    
    def _build_container(self) -> Container:
        """Build container once - updates automatically via FormattedTextControl dynamic text."""
        # Button bar (height 3 for framed buttons)
        button_bar = self._create_button_bar()

        # Create FormattedTextControl with dynamic text (like file browser)
        # This avoids DynamicContainer which breaks mouse event handling
        self.list_control = FormattedTextControl(
            text=self._generate_list_text,
            focusable=True,
            key_bindings=self._create_list_key_bindings()
        )

        # CRITICAL: Store Window reference for focus management
        self.scrollable_window = Window(content=self.list_control)

        # Create ScrollablePane (like file browser)
        self.scrollable_pane = ScrollablePane(
            self.scrollable_window,
            height=Dimension(weight=1),
            show_scrollbar=True,
            display_arrows=True
        )

        # Force content to expand to full width
        frame_content = HSplit([
            button_bar,
            self.scrollable_pane
        ], height=Dimension(weight=1), width=Dimension(weight=1))

        return Frame(
            frame_content,
            title=self.config.frame_title or self.config.title,
            height=Dimension(weight=1),
            width=Dimension(weight=1)
        )
    
    def _create_button_bar(self) -> Container:
        """Create button bar with regular and bulk operation buttons."""
        from openhcs.tui.components.framed_button import FramedButton
        from prompt_toolkit.layout import Window
        from prompt_toolkit.layout.containers import ConditionalContainer
        from prompt_toolkit.filters import Condition

        regular_buttons = self._create_regular_buttons()

        if self.allow_multi_select and self.config.bulk_button_configs:
            bulk_buttons = self._create_bulk_buttons()
            # Show bulk buttons only when items are selected
            bulk_container = ConditionalContainer(
                bulk_buttons,
                filter=Condition(lambda: len(self.model.selected_indices) > 0)
            )
            return HSplit([regular_buttons, bulk_container])

        return regular_buttons

    def _create_regular_buttons(self) -> Container:
        """Create regular button bar."""
        from openhcs.tui.components.framed_button import FramedButton
        from prompt_toolkit.layout import Window

        if not self.config.button_configs:
            # Return empty container with height 3 if no buttons
            return Window(height=Dimension.exact(3), char=' ')

        buttons = []
        for config in self.config.button_configs:
            # Create framed button (height 3)
            framed_button = FramedButton(
                text=config.text,
                handler=self._wrap_handler(config.handler),
                width=config.width
            )
            if not config.is_enabled():
                framed_button.disabled = True
            buttons.append(framed_button)

        # Add spacers between buttons for even distribution
        spaced_buttons = []
        for i, button in enumerate(buttons):
            if i > 0:
                # Add flexible spacer between buttons
                spaced_buttons.append(Window(width=Dimension(weight=1), char=' '))
            spaced_buttons.append(button)

        # Add spacers at start and end for centering
        if spaced_buttons:
            spaced_buttons.insert(0, Window(width=Dimension(weight=1), char=' '))
            spaced_buttons.append(Window(width=Dimension(weight=1), char=' '))

        return VSplit(spaced_buttons, height=Dimension.exact(3))

    def _create_bulk_buttons(self) -> Container:
        """Create bulk operation buttons."""
        from openhcs.tui.components.framed_button import FramedButton
        from prompt_toolkit.layout import Window

        buttons = []
        for config in self.config.bulk_button_configs:
            # Create button with selection count
            selection_count = len(self.model.selected_indices)
            button_text = f"{config.text} ({selection_count})"

            framed_button = FramedButton(
                text=button_text,
                handler=self._wrap_handler(config.handler),
                width=len(button_text) + 2
            )
            if not config.is_enabled():
                framed_button.disabled = True
            buttons.append(framed_button)

        # Add spacers between buttons for even distribution
        spaced_buttons = []
        for i, button in enumerate(buttons):
            if i > 0:
                spaced_buttons.append(Window(width=Dimension(weight=1), char=' '))
            spaced_buttons.append(button)

        # Add spacers at start and end for centering
        if spaced_buttons:
            spaced_buttons.insert(0, Window(width=Dimension(weight=1), char=' '))
            spaced_buttons.append(Window(width=Dimension(weight=1), char=' '))

        return VSplit(spaced_buttons, height=Dimension.exact(3))



    def _generate_list_text(self) -> List[Tuple]:
        """Generate formatted text for all list items.

        Returns:
            List of (style, text, mouse_handler) tuples for FormattedTextControl
        """
        if not self.model.items:
            # Return empty message with proper formatting
            return [("", self.config.empty_message)]

        lines = []

        for i, item_data in enumerate(self.model.items):
            is_selected = i in self.model.selected_indices
            is_focused = i == self.model.selected_index

            # Build checkbox: [x] or [ ]
            checkbox = "[x]" if is_selected else "[ ]"

            # Get display text using config function or fallback to str
            if self.config.display_func:
                display_text = self.config.display_func(item_data, is_selected)
            else:
                display_text = str(item_data)

            # Don't truncate display text - let it extend to the scrollbar
            # The ScrollablePane will handle wrapping and scrolling as needed

            # Format line: "^/v [x] display_text" with proper spacing
            line_text = f"^/v {checkbox} {display_text}"

            # Add consistent spacing and focus styling
            line_text = f" {line_text}"  # Always start with space for consistent alignment
            if is_focused:
                style = "reverse"  # Use inversion highlight for focus
            else:
                style = ""

            # Create mouse handler for this specific line
            def make_handler(line_index):
                def mouse_handler(mouse_event):
                    """Handle mouse events on this specific list item."""
                    # Handle mouse wheel events FIRST (copied from FileManagerBrowser)
                    if mouse_event.event_type == MouseEventType.SCROLL_UP:
                        if self.model.items:
                            new_index = max(0, self.model.selected_index - 3)
                            self.model.set_focused_index(new_index)
                        return True
                    elif mouse_event.event_type == MouseEventType.SCROLL_DOWN:
                        if self.model.items:
                            new_index = min(len(self.model.items) - 1, self.model.selected_index + 3)
                            self.model.set_focused_index(new_index)
                        return True

                    # Handle regular mouse clicks
                    elif mouse_event.event_type == MouseEventType.MOUSE_UP:
                        x_pos = mouse_event.position.x

                        # Update focused index
                        self.model.set_focused_index(line_index)

                        # Handle click based on x position
                        # Layout: " ^/v [x] display_text" (consistent spacing)
                        if x_pos == 1:  # Up button "^"
                            if self._can_move_up(line_index, self.model.items[line_index]):
                                self.model.move_item_up(line_index)
                        elif x_pos == 3:  # Down button "v"
                            if self._can_move_down(line_index, self.model.items[line_index]):
                                self.model.move_item_down(line_index)
                        elif 5 <= x_pos <= 8:  # Checkbox area "[x] "
                            if self.allow_multi_select:
                                self.model.toggle_selection(line_index)
                        else:  # Item text area or focus area
                            self._handle_select(line_index)

                        return True

                    return False  # Event not handled
                return mouse_handler

            # Add line with mouse handler
            lines.append((style, line_text, make_handler(i)))
            if i < len(self.model.items) - 1:  # Add newline except for last item
                lines.append(("", "\n"))  # Newline (2-tuple like FileManagerBrowser)

        return lines

    def _create_list_key_bindings(self) -> KeyBindings:
        """Create key bindings for list navigation and selection.

        Returns:
            KeyBindings object with navigation and selection keys
        """
        kb = KeyBindings()

        @kb.add('up')
        def move_up(event):
            """Move focus up one item."""
            if self.model.selected_index > 0:
                self.model.set_focused_index(self.model.selected_index - 1)

        @kb.add('down')
        def move_down(event):
            """Move focus down one item."""
            if self.model.selected_index < len(self.model.items) - 1:
                self.model.set_focused_index(self.model.selected_index + 1)

        @kb.add('space')
        def toggle_selection(event):
            """Toggle selection of focused item."""
            if self.allow_multi_select:
                self.model.toggle_selection(self.model.selected_index)

        @kb.add('enter')
        def select_item(event):
            """Select/activate focused item."""
            self._handle_select(self.model.selected_index)

        # Mouse wheel support for smooth scrolling (copied from FileManagerBrowser)
        @kb.add('<scroll-up>')
        def scroll_up(event):
            """Handle mouse wheel scroll up."""
            # Move focus up by 3 items (same as FileManager)
            if self.model.items:
                new_index = max(0, self.model.selected_index - 3)
                self.model.set_focused_index(new_index)

        @kb.add('<scroll-down>')
        def scroll_down(event):
            """Handle mouse wheel scroll down."""
            # Move focus down by 3 items (same as FileManager)
            if self.model.items:
                new_index = min(len(self.model.items) - 1, self.model.selected_index + 3)
                self.model.set_focused_index(new_index)

        return kb

    def _create_item_widget(self, index: int, item_data: Dict[str, Any]) -> InteractiveListItem:
        """Create InteractiveListItem for an item."""
        is_selected = (index == self.model.selected_index)

        return InteractiveListItem(
            item_data=item_data,
            item_index=index,
            is_selected=is_selected,
            display_text_func=self.config.display_func,
            on_select=self._handle_select,
            on_move_up=self._handle_move_up,
            on_move_down=self._handle_move_down,
            can_move_up=self._can_move_up(index, item_data),
            can_move_down=self._can_move_down(index, item_data)
        )
    
    def _can_move_up(self, index: int, item_data: Dict[str, Any]) -> bool:
        """Check if item can move up."""
        if self.config.can_move_up_func:
            return self.config.can_move_up_func(index, item_data)
        return index > 0
    
    def _can_move_down(self, index: int, item_data: Dict[str, Any]) -> bool:
        """Check if item can move down."""
        if self.config.can_move_down_func:
            return self.config.can_move_down_func(index, item_data)
        return index < len(self.model.items) - 1
    
    def _handle_select(self, index: int):
        """Handle item selection."""
        self.model.select_item(index)
    
    def _handle_move_up(self, index: int):
        """Handle move up."""
        self.model.move_item_up(index)
    
    def _handle_move_down(self, index: int):
        """Handle move down."""
        self.model.move_item_down(index)
    
    def _wrap_handler(self, handler: Callable) -> Callable:
        """Wrap handler for async support."""
        from openhcs.tui.utils.unified_task_manager import get_task_manager
        def wrapped():
            from openhcs.tui.utils.unified_task_manager import get_task_manager
            if asyncio.iscoroutinefunction(handler):
                get_task_manager().fire_and_forget(handler(), "list_handler")
            else:
                handler()
        return wrapped
    
    def _on_model_changed(self):
        """Called when model changes - triggers UI update."""
        # 1. Update cursor position (safe)
        self._ensure_focused_visible()

        # 2. BULLETPROOF focus restoration
        self._restore_focus_bulletproof()

        # 3. Invalidate last
        get_app().invalidate()

    def _ensure_focused_visible(self) -> None:
        """Ensure the focused item is visible by setting cursor position."""
        if not hasattr(self, 'list_control') or not self.list_control:
            return

        try:
            from prompt_toolkit.data_structures import Point

            # Calculate cursor position based on focused item
            if not self.model.items:
                cursor_y = 0
            else:
                cursor_y = max(0, min(self.model.selected_index, len(self.model.items) - 1))

            cursor_pos = Point(x=0, y=cursor_y)

            # Override the get_cursor_position method of FormattedTextControl
            def get_cursor_position():
                return cursor_pos

            # Monkey patch the method (following FileManagerBrowser pattern)
            self.list_control.get_cursor_position = get_cursor_position

            # Also make the control show cursor
            self.list_control.show_cursor = True

        except Exception as e:
            pass  # Silently ignore cursor positioning errors

    def _restore_focus_bulletproof(self):
        """Restore focus with ALL edge cases handled."""
        try:
            # Check 1: App exists and is running
            app = get_app()
            if not app or not hasattr(app, 'layout'):
                return False

            # Check 2: Window exists and is valid
            if not hasattr(self, 'scrollable_window') or not self.scrollable_window:
                return False

            # Check 3: Window is in the layout tree
            if not self._is_window_in_layout(self.scrollable_window, app.layout):
                return False

            # Check 4: Window is focusable
            if not getattr(self.scrollable_window, 'content', None):
                return False

            # Check 5: FormattedTextControl exists and is focusable
            control = self.scrollable_window.content
            if not hasattr(control, 'focusable'):
                return False
            # Note: focusable is a Filter, we just check it exists

            # SAFE focus call with exception handling - focus the FormattedTextControl directly
            app.layout.focus(self.list_control)
            return True

        except Exception as e:
            # Silently handle focus restoration errors
            return False

    def _is_window_in_layout(self, window, layout):
        """Check if window is actually in the layout tree."""
        try:
            # Traverse layout tree to find window
            def find_window(container):
                if container == window:
                    return True
                if hasattr(container, 'children'):
                    for child in container.children:
                        if find_window(child):
                            return True
                if hasattr(container, 'content') and container.content == window:
                    return True
                return False

            return find_window(layout.container)
        except:
            return False


class ListManagerPane:
    """
    Clean list manager pane - pure composition, no inheritance.
    
    Coordinates model, view, and business logic without architectural anti-patterns.
    """
    
    def __init__(self, config: ListConfig, backend: Any):
        """Pure setup - no work in constructor."""
        self.config = config
        self.backend = backend
        
        # Clean MVC components
        self.model = ListModel()
        self.view = ListView(self.model, config)
        
        # Observe model for business logic
        self.model.add_observer(self._on_model_changed)
    
    @property
    def container(self) -> Container:
        """Get the UI container."""
        return self.view.container
    
    def load_items(self, items: List[Dict[str, Any]]):
        """Load items - just update model, view updates automatically."""
        self.model.set_items(items)
    
    def get_selected_item(self) -> Optional[Dict[str, Any]]:
        """Get currently selected item."""
        if 0 <= self.model.selected_index < len(self.model.items):
            return self.model.items[self.model.selected_index]
        return None
    
    def _on_model_changed(self):
        """Handle model changes for business logic."""
        # Override in subclasses for business logic
        pass
