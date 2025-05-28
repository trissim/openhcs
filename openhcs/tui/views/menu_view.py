"""
Menu View Component.

Pure UI component responsible only for rendering and user input handling.
Delegates all business logic to the controller.

🔒 Clause 295: Component Boundaries
Clear separation between view and controller layers.
"""
import logging
from typing import Any, Dict, List, Optional

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import Container, VSplit, HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import Box, Label
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent

logger = logging.getLogger(__name__)


class MenuView(Container):
    """
    View component for the menu bar.
    
    Handles:
    - Menu bar rendering
    - Mouse and keyboard input
    - Visual state updates
    """
    
    def __init__(self, controller, menu_structure: Dict[str, Any]):
        """
        Initialize the menu view.
        
        Args:
            controller: Menu controller instance
            menu_structure: Dictionary defining menu structure
        """
        self.controller = controller
        self.menu_structure = menu_structure
        
        # UI components
        self.menu_labels = []
        self.submenu_container = None
        self.container = None
        
        # UI state
        self.active_menu = None
        self.active_submenu = None
        self.active_item_index = None
        
        # Build UI
        self._build_ui()
        
        # Create key bindings
        self.kb = self._create_key_bindings()
        
        # Register key bindings with application
        app = get_app()
        app.key_bindings.merge(self.kb)
        
        # Register for UI updates
        self.controller.state.add_observer('menu_activated', self._handle_menu_activated)
        self.controller.state.add_observer('menu_closed', self._handle_menu_closed)
    
    def _build_ui(self):
        """Build the UI components."""
        # Create menu labels
        self.menu_labels = self._create_menu_labels()
        
        # Create main container
        self.container = VSplit(self.menu_labels)
    
    def _create_menu_labels(self) -> List[Container]:
        """Create labels for top-level menus."""
        labels = []
        
        for menu_name, menu_data in self.menu_structure.items():
            # Get mnemonic (first character)
            mnemonic = menu_data.get('mnemonic', menu_name[0])
            
            # Create label with mnemonic highlighting
            label_text = f"[{mnemonic}]{menu_name[1:]}"
            
            # Create label
            label = Label(
                text=label_text,
                dont_extend_height=True,
                style="class:menu-bar"
            )
            
            # Add mouse handler
            def create_mouse_handler(menu):
                def menu_mouse_handler(mouse_event):
                    if mouse_event.event_type == MouseEventType.MOUSE_UP:
                        app = get_app()
                        app.create_background_task(self.controller.handle_mouse_click(menu))
                        return True
                    return False
                return menu_mouse_handler
            
            # Wrap in box with padding
            boxed_label = Box(label, padding=1)
            
            # Store reference for mouse handling
            boxed_label._menu_name = menu_name
            
            labels.append(boxed_label)
        
        return labels
    
    def _create_key_bindings(self) -> KeyBindings:
        """Create key bindings for menu navigation."""
        kb = KeyBindings()
        
        # Add menu activation key bindings
        for menu_name, menu_data in self.menu_structure.items():
            mnemonic = menu_data.get('mnemonic', menu_name[0])
            
            # Create handler for this menu
            def create_handler(menu):
                def handler(event: KeyPressEvent):
                    app = get_app()
                    app.create_background_task(self.controller.handle_key_binding(mnemonic.lower()))
                return handler
            
            # Bind Alt+key
            key = mnemonic.lower()
            kb.add('escape', key)(create_handler(menu_name))
        
        # Escape to close menu
        @kb.add('escape')
        def _(event: KeyPressEvent):
            app = get_app()
            app.create_background_task(self.controller.close_menu())
        
        # Arrow key navigation
        @kb.add('left')
        def _(event: KeyPressEvent):
            app = get_app()
            app.create_background_task(self.controller.navigate_menu(-1))
        
        @kb.add('right')
        def _(event: KeyPressEvent):
            app = get_app()
            app.create_background_task(self.controller.navigate_menu(1))
        
        @kb.add('up')
        def _(event: KeyPressEvent):
            app = get_app()
            app.create_background_task(self.controller.navigate_submenu(-1))
        
        @kb.add('down')
        def _(event: KeyPressEvent):
            app = get_app()
            app.create_background_task(self.controller.navigate_submenu(1))
        
        # Enter to select
        @kb.add('enter')
        def _(event: KeyPressEvent):
            if self.active_menu and self.active_submenu and self.active_item_index is not None:
                # Get selected item
                if 0 <= self.active_item_index < len(self.active_submenu):
                    item = self.active_submenu[self.active_item_index]
                    command_name = item.get('command')
                    if command_name:
                        app = get_app()
                        app.create_background_task(self.controller.select_menu_item(command_name))
        
        return kb
    
    async def _handle_menu_activated(self, data):
        """Handle menu activation events."""
        menu_name = data.get('menu_name')
        is_open = data.get('is_open', False)
        
        if is_open and menu_name:
            self.active_menu = menu_name
            self.active_submenu = self.menu_structure.get(menu_name, {}).get('items', [])
            self.active_item_index = 0 if self.active_submenu else None
            
            # Update visual state
            await self._update_menu_display()
    
    async def _handle_menu_closed(self, data):
        """Handle menu close events."""
        self.active_menu = None
        self.active_submenu = None
        self.active_item_index = None
        
        # Update visual state
        await self._update_menu_display()
    
    async def _update_menu_display(self):
        """Update the menu display."""
        # Update label styles to show active state
        for i, label_box in enumerate(self.menu_labels):
            menu_name = getattr(label_box, '_menu_name', None)
            if menu_name == self.active_menu:
                # Highlight active menu
                label_box.style = "class:menu-bar-active"
            else:
                label_box.style = "class:menu-bar"
        
        # Force redraw
        app = get_app()
        app.invalidate()
    
    def get_enabled_style(self, command_name: str) -> str:
        """Get the style for a menu item based on its enabled state."""
        if self.controller.is_command_enabled(command_name):
            return "class:menu-item"
        else:
            return "class:menu-item-disabled"
    
    def handle_mouse_event(self, mouse_event):
        """Handle mouse events for the menu bar."""
        if mouse_event.event_type == MouseEventType.MOUSE_UP:
            # Check which menu was clicked
            for label_box in self.menu_labels:
                menu_name = getattr(label_box, '_menu_name', None)
                if menu_name and mouse_event.is_mouse_over(label_box):
                    app = get_app()
                    app.create_background_task(self.controller.handle_mouse_click(menu_name))
                    return True
        return False
    
    def __pt_container__(self) -> Container:
        """Return the container to render."""
        return self.container
    
    # Implement Container abstract methods
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
        """Handle mouse events."""
        return self.handle_mouse_event(mouse_event)
