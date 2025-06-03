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


class ListModel:
    """Model for list state - pure data, no UI concerns."""
    
    def __init__(self):
        self.items: List[Dict[str, Any]] = []
        self.selected_index: int = 0
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
        """Select item by index."""
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


class ListView:
    """View for list display - observes model, updates UI automatically."""
    
    def __init__(self, model: ListModel, config: ListConfig):
        self.model = model
        self.config = config
        
        # UI components (built once, updated automatically)
        self.container = self._build_container()
        
        # Observe model changes
        self.model.add_observer(self._on_model_changed)
    
    def _build_container(self) -> Container:
        """Build container once - updates automatically via DynamicContainer."""
        # Button bar (height 3 for framed buttons)
        button_bar = self._create_button_bar()

        # Dynamic list that updates when model changes
        def get_current_list():
            return self._create_current_list()

        dynamic_list = DynamicContainer(get_current_list)
        dynamic_list.height = Dimension(weight=1)

        # Force content to expand to full width
        frame_content = HSplit([
            button_bar,
            dynamic_list
        ], height=Dimension(weight=1), width=Dimension(weight=1))

        return Frame(
            frame_content,
            title=self.config.frame_title or self.config.title,
            height=Dimension(weight=1),
            width=Dimension(weight=1)
        )
    
    def _create_button_bar(self) -> Container:
        """Create button bar with framed buttons, evenly spaced horizontally."""
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
    
    def _create_current_list(self) -> HSplit:
        """Create current list state - called automatically when model changes."""
        if not self.model.items:
            from prompt_toolkit.layout import Window
            from prompt_toolkit.layout.controls import FormattedTextControl

            # Use Window with text wrapping instead of Label to constrain width
            empty_window = Window(
                FormattedTextControl(
                    self.config.empty_message,
                    focusable=False,
                ),
                width=Dimension(weight=1),  # Take proportional space
                wrap_lines=True,  # KEY: Wrap long lines instead of expanding
                dont_extend_width=True,  # Don't expand beyond allocated width
            )
            return HSplit([empty_window])

        item_widgets = []
        for index, item_data in enumerate(self.model.items):
            widget = self._create_item_widget(index, item_data)
            item_widgets.append(widget)

        return HSplit(item_widgets)
    
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
        def wrapped():
            if asyncio.iscoroutinefunction(handler):
                get_app().create_background_task(handler())
            else:
                handler()
        return wrapped
    
    def _on_model_changed(self):
        """Called when model changes - triggers UI update."""
        get_app().invalidate()


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
