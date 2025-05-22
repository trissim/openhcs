import asyncio
from asyncio import Lock
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (Any, Callable, ClassVar, Dict, FrozenSet, List, Optional,
                    Tuple, Union, TYPE_CHECKING) # Added TYPE_CHECKING
import logging # ADDED
import json # ADDED for _on_save_pipeline

# For handler implementations
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.context.processing_context import ProcessingContext
if TYPE_CHECKING:
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
    # TUIState is typically passed in __init__, so direct import might not be needed here
    # from openhcs.tui.tui_architecture import TUIState


# import yaml # YAML dependency is being removed
from prompt_toolkit.application import get_app
from prompt_toolkit.filters import Condition, has_focus
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import (Container, FormattedTextControl, HSplit,
                                   VSplit, Window)
from prompt_toolkit.layout.containers import (AnyContainer,
                                              ConditionalContainer, Float)
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.widgets import Box, Frame, Label

logger = logging.getLogger(__name__) # ADDED


class MissingStateError(Exception):
    """
    Error raised when a required state attribute is missing.
    
    🔒 Clause 88: No Inferred Capabilities
    Explicitly fails when required state is missing instead of using defaults.
    """
    def __init__(self, attribute_name: str):
        super().__init__(f"Required state attribute missing: {attribute_name}")
        self.attribute_name = attribute_name


class ReentrantLock:
    """
    A reentrant lock wrapper around asyncio.Lock.
    
    🔒 Clause 317: Runtime Correctness
    Prevents deadlocks by allowing the same task to acquire the lock multiple times.
    """
    def __init__(self):
        self._lock = Lock()
        self._owner = None
        self._count = 0
    
    async def __aenter__(self):
        """Acquire the lock."""
        # Get current task
        current_task = asyncio.current_task()
        
        # If we already own the lock, just increment the count
        if self._owner == current_task:
            self._count += 1
            return self
        
        # Otherwise, acquire the lock
        await self._lock.acquire()
        self._owner = current_task
        self._count = 1
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release the lock."""
        # Decrement the count
        self._count -= 1
        
        # If count is 0, release the lock
        if self._count == 0:
            self._owner = None
            self._lock.release()

# Define layout contract
class LayoutContract:
    """
    Defines the required interface for layout containers.
    
    🔒 Clause 245: Declarative Enforcement
    Explicitly declares required capabilities of layout containers.
    """
    @staticmethod
    def validate_layout_container(container: Any) -> None:
        """
        Validate that a container meets the layout contract.
        
        Args:
            container: The container to validate
            
        Raises:
            ValueError: If container doesn't meet the contract
        """
        if not hasattr(container, 'floats'):
            raise ValueError("Layout container must have 'floats' attribute")
        
        if not isinstance(container.floats, list):
            raise ValueError("Layout container 'floats' must be a list")


class MenuItemType(Enum):
    """Types of menu items."""
    COMMAND = auto()    # Regular command
    SUBMENU = auto()    # Submenu
    CHECKBOX = auto()   # Checkable item
    SEPARATOR = auto()  # Separator line


@dataclass
class MenuItemSchema:
    """
    Schema for validating menu items.
    
    🔒 Clause 3: Declarative Primacy
    Defines the structure of menu items declaratively.
    """
    VALID_TYPES: ClassVar[FrozenSet[str]] = frozenset(["command", "submenu", "checkbox", "separator"])
    
    @staticmethod
    def validate_menu_item(item: Dict[str, Any]) -> None:
        """
        Validate a menu item dictionary.
        
        Args:
            item: Menu item dictionary
            
        Raises:
            ValueError: If item is invalid
        """
        if "type" not in item:
            raise ValueError("Menu item must have 'type' field")
            
        item_type = item["type"].lower()
        if item_type not in MenuItemSchema.VALID_TYPES:
            raise ValueError(f"Invalid menu item type: {item_type}")
            
        if item_type != "separator" and "label" not in item:
            raise ValueError(f"{item_type} menu item must have 'label' field")
            
        if item_type == "submenu" and "children" not in item:
            raise ValueError("Submenu must have 'children' field")
            
        if "children" in item and not isinstance(item["children"], list):
            raise ValueError("Children must be a list")
            
        # Validate children recursively
        if "children" in item:
            for child in item["children"]:
                MenuItemSchema.validate_menu_item(child)


@dataclass
class MenuStructureSchema:
    """
    Schema for validating menu structure.
    
    🔒 Clause 3: Declarative Primacy
    Defines the structure of the menu declaratively.
    """
    @staticmethod
    def validate_menu_structure(structure: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Validate a menu structure dictionary.
        
        Args:
            structure: Menu structure dictionary
            
        Raises:
            ValueError: If structure is invalid
        """
        if not isinstance(structure, dict):
            raise ValueError("Menu structure must be a dictionary")
            
        if not structure:
            raise ValueError("Menu structure cannot be empty")
            
        for menu_name, items in structure.items():
            if not isinstance(items, list):
                raise ValueError(f"Menu '{menu_name}' items must be a list")
                
            for item in items:
                MenuItemSchema.validate_menu_item(item)


class MenuItem:
    """
    Represents a single menu item.

    Menu items can be commands, submenus, checkboxes, or separators.
    
    🔒 Clause 3: Declarative Primacy
    Created from declarative structure rather than procedural code.
    """
    def __init__(
        self,
        type: MenuItemType,
        label: str = "",
        handler: Optional[Callable] = None,
        shortcut: Optional[str] = None,
        enabled: Union[bool, Condition] = True,
        checked: Union[bool, Condition] = False,
        children: Optional[List['MenuItem']] = None
    ):
        """
        Initialize a menu item.

        Args:
            type: The type of menu item
            label: The display label
            handler: Callback function for command items
            shortcut: Keyboard shortcut (e.g., "Ctrl+S")
            enabled: Whether the item is enabled
            checked: Whether the item is checked (for checkbox items)
            children: List of child menu items (for submenu items)
        """
        self.type = type
        self.label = label
        self.handler = handler
        self.shortcut = shortcut
        self.enabled = enabled
        self.checked = checked
        self.children = children or []

    def is_enabled(self) -> bool:
        """Check if the menu item is enabled."""
        if isinstance(self.enabled, Condition):
            return self.enabled()
        return self.enabled

    def is_checked(self) -> bool:
        """Check if the menu item is checked."""
        if isinstance(self.checked, Condition):
            return self.checked()
        return self.checked

    def set_checked(self, checked: bool) -> None:
        """Set the checked state of the menu item."""
        if not isinstance(self.checked, Condition):
            self.checked = checked
            
    @classmethod
    def from_dict(cls, item_dict: Dict[str, Any], handler_map: Dict[str, Callable]) -> 'MenuItem':
        """
        Create a MenuItem from a dictionary.
        
        Args:
            item_dict: Dictionary with menu item data
            handler_map: Map of handler names to handler functions
            
        Returns:
            MenuItem instance
        """
        item_type_str = item_dict["type"].upper()
        item_type = MenuItemType[item_type_str]
        
        # For separators, just return a separator
        if item_type == MenuItemType.SEPARATOR:
            return cls(type=item_type)
            
        # Get basic properties
        label = item_dict["label"]
        shortcut = item_dict.get("shortcut")
        
        # Get handler if specified
        handler = None
        handler_name = item_dict.get("handler")
        if handler_name and handler_name in handler_map:
            handler = handler_map[handler_name]
            
        # Get enabled condition
        enabled = item_dict.get("enabled", True)
        
        # Get checked condition
        checked = item_dict.get("checked", False)
        
        # Process children for submenus
        children = None
        if "children" in item_dict:
            children = [
                cls.from_dict(child, handler_map)
                for child in item_dict["children"]
            ]
            
        return cls(
            type=item_type,
            label=label,
            handler=handler,
            shortcut=shortcut,
            enabled=enabled,
            checked=checked,
            children=children
        )

# Python-defined menu structure
# This replaces the content that would have been in menu.yaml
# Ensure handler and condition names used here match those in _create_handler_map and _create_condition_map
_DEFAULT_MENU_STRUCTURE = {
    "File": {
        "mnemonic": "F", # For Alt+F activation
        "items": [
            {"type": "command", "label": "&New Pipeline", "handler": "_on_new_pipeline", "shortcut": "Ctrl+N"},
            {"type": "command", "label": "&Open Pipeline...", "handler": "_on_open_pipeline", "shortcut": "Ctrl+O"},
            {"type": "command", "label": "&Save Pipeline", "handler": "_on_save_pipeline", "enabled": "is_compiled"},
            {"type": "separator"},
            {"type": "command", "label": "E&xit", "handler": "_on_exit"}
        ]
    },
    "Edit": {
        "mnemonic": "E",
        "items": [
            {"type": "command", "label": "&Add Step", "handler": "_on_add_step", "enabled": "has_selected_step"},
            {"type": "command", "label": "Edit Ste&p", "handler": "_on_edit_step", "enabled": "has_selected_step"},
            {"type": "command", "label": "&Remove Step", "handler": "_on_remove_step", "enabled": "has_selected_step"},
        ]
    },
    "View": {
        "mnemonic": "V",
        "items": [
            {"type": "checkbox", "label": "&Vim Mode", "handler": "_on_toggle_vim_mode", "checked": "vim_mode"},
            {"type": "checkbox", "label": "&Log Drawer", "handler": "_on_toggle_log_drawer", "checked": "log_drawer_expanded"},
            {"type": "separator"},
            {"type": "submenu", "label": "&Theme", "children": [
                {"type": "checkbox", "label": "&Light", "handler": "_on_set_theme_light", "checked": "theme_is_light"},
                {"type": "checkbox", "label": "&Dark", "handler": "_on_set_theme_dark", "checked": "theme_is_dark"},
                {"type": "checkbox", "label": "&System", "handler": "_on_set_theme_system", "checked": "theme_is_system"},
            ]}
        ]
    },
    "Pipeline": {
        "mnemonic": "P",
        "items": [
            {"type": "command", "label": "Pre-&compile", "handler": "_on_pre_compile"},
            {"type": "command", "label": "&Compile", "handler": "_on_compile"},
            {"type": "command", "label": "&Run", "handler": "_on_run", "enabled": "is_compiled"},
            {"type": "separator"},
            {"type": "command", "label": "Se&ttings...", "handler": "_on_settings"}
        ]
    },
    "Help": {
        "mnemonic": "H",
        "items": [
            {"type": "command", "label": "&Keyboard Shortcuts", "handler": "_on_keyboard_shortcuts"},
            {"type": "command", "label": "&About", "handler": "_on_about"}
        ]
    }
}


class MenuBar(Container):
    """
    Top menu bar for the OpenHCS TUI.

    Provides access to application-wide commands and settings
    through a consistent, keyboard-navigable interface.
    
    🔒 Clause 3: Declarative Primacy
    Menu structure is loaded from a declarative YAML definition.
    
    🔒 Clause 245: Declarative Enforcement
    Layout contract is explicitly validated.
    """
    def __init__(self, state):
        """
        Initialize the menu bar.

        Args:
            state: The TUI state manager
        """
        self.state = state
        
        # Initialize state with thread safety
        self.active_menu: Optional[str] = None
        self.active_submenu: Optional[List[MenuItem]] = None
        self.active_item_index: Optional[int] = None
        self.menu_lock = ReentrantLock()

        # Create handler map
        self.handler_map = self._create_handler_map()
        
        # Create condition map
        self.condition_map = self._create_condition_map()

        # Load menu structure from YAML
        self.menu_structure = self._load_menu_structure()
        
        # Create UI components
        self.menu_labels = self._create_menu_labels()
        self.submenu_float = self._create_submenu_float()
        
        # Create container
        self.container = VSplit(self.menu_labels)
        
        # Create key bindings
        self.kb = self._create_key_bindings()
        
        # Register key bindings with application
        app = get_app()
        app.key_bindings.merge(self.kb)
        
        # Register for events
        self.state.add_observer('operation_status_changed', self._on_operation_status_changed)
        self.state.add_observer('plate_selected', self._on_plate_selected)
        self.state.add_observer('is_compiled_changed', self._on_is_compiled_changed)
        
    def _create_handler_map(self) -> Dict[str, Callable]:
        """
        Create a map of handler names to handler methods.
        
        Returns:
            Dictionary mapping handler names to methods
        """
        return {
            # File menu
            "_on_new_pipeline": self._on_new_pipeline,
            "_on_open_pipeline": self._on_open_pipeline,
            "_on_save_pipeline": self._on_save_pipeline,
            "_on_save_pipeline_as": self._on_save_pipeline_as,
            "_on_exit": self._on_exit,
            
            # Edit menu
            "_on_add_step": self._on_add_step,
            "_on_edit_step": self._on_edit_step,
            "_on_remove_step": self._on_remove_step,
            "_on_move_step_up": self._on_move_step_up,
            "_on_move_step_down": self._on_move_step_down,
            
            # View menu
            "_on_toggle_log_drawer": self._on_toggle_log_drawer,
            "_on_toggle_vim_mode": self._on_toggle_vim_mode,
            "_on_set_theme_light": lambda: self._on_set_theme("light"),
            "_on_set_theme_dark": lambda: self._on_set_theme("dark"),
            "_on_set_theme_system": lambda: self._on_set_theme("system"),
            
            # Pipeline menu
            "_on_pre_compile": self._on_pre_compile,
            "_on_compile": self._on_compile,
            "_on_run": self._on_run,
            "_on_test": self._on_test,
            "_on_settings": self._on_settings,
            
            # Help menu
            "_on_documentation": self._on_documentation,
            "_on_keyboard_shortcuts": self._on_keyboard_shortcuts,
            "_on_about": self._on_about
        }
        
    def _create_condition_map(self) -> Dict[str, Condition]:
        """
        Create a map of condition names to Condition objects.
        
        Returns:
            Dictionary mapping condition names to Condition objects
        """
        return {
            # Compilation state
            "is_compiled": Condition(lambda: self.state.is_compiled),
            
            # Step selection
            "has_selected_step": Condition(lambda: self.state.selected_step is not None),
            
            # View settings
            "log_drawer_expanded": Condition(lambda: self._get_required_state('log_drawer_expanded')),
            "vim_mode": Condition(lambda: self._get_required_state('vim_mode')),
            
            # Theme settings
            "theme_is_light": Condition(lambda: self._get_required_state('theme') == 'light'),
            "theme_is_dark": Condition(lambda: self._get_required_state('theme') == 'dark'),
            "theme_is_system": Condition(lambda: self._get_required_state('theme') == 'system')
        }
        
    def _get_required_state(self, attribute_name: str) -> Any:
        """
        Get a required state attribute, raising an error if it doesn't exist.
        
        Args:
            attribute_name: Name of the state attribute
            
        Returns:
            Value of the state attribute
            
        Raises:
            MissingStateError: If the attribute doesn't exist
        """
        if not hasattr(self.state, attribute_name):
            raise MissingStateError(attribute_name)
        return getattr(self.state, attribute_name)
        
    def _load_menu_structure(self) -> Dict[str, List[MenuItem]]:
        """
        Load menu structure from YAML and convert to MenuItem objects.
        
        Returns:
            Dictionary mapping menu names to lists of MenuItem objects
        """
        # Use the Python-defined structure
        raw_structure = _DEFAULT_MENU_STRUCTURE

        # Validate the Python-defined structure first
        try:
            MenuStructureSchema.validate_menu_structure(raw_structure)
        except ValueError as e:
            # Handle validation error, e.g., log and use an empty menu or raise
            # For now, let's log and raise, as a malformed default is a critical dev error.
            # In a production setting, might fall back to a minimal safe menu.
            # logger.error(f"Invalid _DEFAULT_MENU_STRUCTURE: {e}") # Assuming logger is available
            print(f"CRITICAL: Invalid _DEFAULT_MENU_STRUCTURE in menu_bar.py: {e}") # Fallback print
            raise RuntimeError(f"Invalid _DEFAULT_MENU_STRUCTURE: {e}") from e
            # menu_structure = {}
            # return menu_structure

        # Convert to MenuItem objects
        menu_structure = {}
        for menu_name, menu_data in raw_structure.items():
            # Validate menu data
            if not isinstance(menu_data, dict):
                raise ValueError(f"Menu '{menu_name}' data must be a dictionary")
                
            if "items" not in menu_data:
                raise ValueError(f"Menu '{menu_name}' must have 'items' field")
                
            if "mnemonic" not in menu_data:
                raise ValueError(f"Menu '{menu_name}' must have 'mnemonic' field")
                
            # Process items
            menu_items = []
            for item in menu_data["items"]:
                # Process conditions
                if "enabled" in item and isinstance(item["enabled"], str):
                    condition_name = item["enabled"]
                    if condition_name in self.condition_map:
                        item["enabled"] = self.condition_map[condition_name]
                
                if "checked" in item and isinstance(item["checked"], str):
                    condition_name = item["checked"]
                    if condition_name in self.condition_map:
                        item["checked"] = self.condition_map[condition_name]
                
                # Create MenuItem
                menu_item = MenuItem.from_dict(item, self.handler_map)
                menu_items.append(menu_item)
            
            menu_structure[menu_name] = menu_items
        
        return menu_structure

   
    
    # Removed static method MenuBar.load_menu_structure() as YAML loading is no longer used.
    # The _DEFAULT_MENU_STRUCTURE defined above is now the source of the menu definition.
    def _create_menu_labels(self) -> List[Label]:
        """
        Create the menu labels for the top bar.

        Returns:
            List of Label widgets for menu categories
        """
        labels = []
        
        for menu_name in self.menu_structure.keys():
            # Create label with mnemonic (underlined first character)
            mnemonic = menu_name[0]
            label_text = f"[{mnemonic}]{menu_name[1:]}"
            
            # Create label
            label = Label(
                text=label_text,
                dont_extend_height=True,
                style=lambda: f"class:menu-bar{'.' + self.active_menu if self.active_menu == menu_name else ''}"
            )
            
            # Add mouse handler
            original_mouse_handler = label.mouse_handler
            
            def create_mouse_handler(menu):
                def menu_mouse_handler(mouse_event):
                    if mouse_event.event_type == MouseEventType.MOUSE_UP:
                        get_app().create_background_task(self._activate_menu(menu))
                        return True
                    return original_mouse_handler(mouse_event)
                return menu_mouse_handler
            
            label.mouse_handler = create_mouse_handler(menu_name)
            
            # Add to list
            labels.append(Box(label, padding=1))
        
        return labels

    def _create_submenu_float(self) -> Float:
        """
        Create the floating container for submenus.

        Returns:
            A Float container for displaying submenus
        """
        # Create empty container initially
        submenu_container = HSplit([])
        
        # Create float
        return Float(
            Frame(
                submenu_container,
                style="class:menu"
            ),
            transparent=False
        )

    def _create_key_bindings(self) -> KeyBindings:
        """
        Create key bindings for menu navigation.

        Returns:
            KeyBindings object with menu navigation bindings
        """
        kb = KeyBindings()
        
        # Add menu activation key bindings from explicit mnemonics
        for menu_name, items in self.menu_structure.items():
            # Get mnemonic from menu data
            menu_data = MenuItem.load_menu_structure()
            if menu_name not in menu_data:
                raise ValueError(f"Menu '{menu_name}' not found in menu data")
                
            # Get mnemonic from menu data
            mnemonic = menu_data[menu_name].get('mnemonic')
            if not mnemonic:
                raise ValueError(f"Menu '{menu_name}' has no mnemonic defined")
                
            # Create a closure to capture menu_name
            def create_handler(menu):
                def handler(event: KeyPressEvent):
                    get_app().create_background_task(self._activate_menu(menu))
                return handler
            
            # Add the key binding with the created handler
            kb.add(f'a-{mnemonic.lower()}')(create_handler(menu_name))
        
        # Escape to close menu
        @kb.add('escape')
        def _(event: KeyPressEvent):
            get_app().create_background_task(self._close_menu())
        
        # Left/right navigation between menus
        # Create menu navigation conditions
        menu_active = StateConditionRegistry.create_condition(
            StateConditionType.MENU_ACTIVE, self)
        submenu_active = StateConditionRegistry.create_condition(
            StateConditionType.SUBMENU_ACTIVE, self)
        
        @kb.add('left', filter=menu_active)
        def _(event: KeyPressEvent):
            get_app().create_background_task(self._navigate_menu(-1))
        
        @kb.add('right', filter=menu_active)
        def _(event: KeyPressEvent):
            get_app().create_background_task(self._navigate_menu(1))
        
        # Up/down navigation within submenu
        @kb.add('up', filter=submenu_active)
        def _(event: KeyPressEvent):
            get_app().create_background_task(self._navigate_submenu(-1))
        
        @kb.add('down', filter=submenu_active)
        def _(event: KeyPressEvent):
            get_app().create_background_task(self._navigate_submenu(1))
        
        # Enter to select menu item
        @kb.add('enter', filter=submenu_active)
        def _(event: KeyPressEvent):
            get_app().create_background_task(self._select_current_item())
        
        return kb

    async def _activate_menu(self, menu_name: str) -> None:
        """
        Activate a menu category.

        Args:
            menu_name: The name of the menu to activate
        """
        async with self.menu_lock:
            # If menu is already active, close it
            if self.active_menu == menu_name:
                # Since we're using a ReentrantLock, we can safely call _close_menu
                # without releasing the lock first
                await self._close_menu()
                return
            
            # Set active menu
            self.active_menu = menu_name
            
            # Get menu items
            menu_items = self.menu_structure.get(menu_name, [])
            
            # Set active submenu
            self.active_submenu = menu_items
            
            # Create submenu container
            submenu_container = self._create_submenu_container(menu_items)
            
            # Update float content
            self.submenu_float.content.body = submenu_container
            
            # Add float to layout
            app = get_app()
            if self.submenu_float not in app.layout.container.floats:
                app.layout.container.floats.append(self.submenu_float)
            
            # Force UI refresh
            app.invalidate()

    async def _close_menu(self) -> None:
        """Close the active menu."""
        async with self.menu_lock:
            # Clear active menu
            self.active_menu = None
            self.active_submenu = None
            
            # Remove float from layout
            app = get_app()
            
            # Validate layout container
            LayoutContract.validate_layout_container(app.layout.container)
            
            # Remove float
            if self.submenu_float in app.layout.container.floats:
                app.layout.container.floats.remove(self.submenu_float)
            
            # Force UI refresh
            app.invalidate()

    async def _navigate_menu(self, delta: int) -> None:
        """
        Navigate between menu categories.

        Args:
            delta: Direction to navigate (-1 for left, 1 for right)
        """
        async with self.menu_lock:
            if self.active_menu is None:
                return
            
            # Get menu names
            menu_names = list(self.menu_structure.keys())
            
            # Find current index
            current_index = menu_names.index(self.active_menu)
            
            # Calculate new index
            new_index = (current_index + delta) % len(menu_names)
            
            # Activate new menu
            await self._activate_menu(menu_names[new_index])

    async def _navigate_submenu(self, delta: int) -> None:
        """
        Navigate within a submenu.

        Args:
            delta: Direction to navigate (-1 for up, 1 for down)
        """
        async with self.menu_lock:
            if self.active_submenu is None or not self.active_submenu:
                return
                
            # Initialize index if not set
            if self.active_item_index is None:
                self.active_item_index = 0
                
            # Count valid (non-separator) items
            valid_indices = []
            for i, item in enumerate(self.active_submenu):
                if item.type != MenuItemType.SEPARATOR:
                    valid_indices.append(i)
                    
            if not valid_indices:
                return
                
            # Find current index in valid indices
            try:
                current_valid_index = valid_indices.index(self.active_item_index)
            except ValueError:
                # Current index is not valid, reset to first valid index
                self.active_item_index = valid_indices[0]
                get_app().invalidate()
                return
                
            # Calculate new valid index
            new_valid_index = (current_valid_index + delta) % len(valid_indices)
            
            # Set new index
            self.active_item_index = valid_indices[new_valid_index]
            
            # Force UI refresh
            get_app().invalidate()

    async def _select_current_item(self) -> None:
        """Select the currently highlighted menu item."""
        async with self.menu_lock:
            if (self.active_submenu is None or
                self.active_item_index is None or
                self.active_item_index >= len(self.active_submenu)):
                return
                
            # Get selected item
            item = self.active_submenu[self.active_item_index]
            
            # Handle item
            await self._handle_menu_item(item)

    def _create_submenu_container(self, menu_items: List[MenuItem]) -> Container:
        """
        Create a container for displaying a submenu.

        Args:
            menu_items: List of menu items to display

        Returns:
            A Container for the submenu
        """
        # Create labels for each menu item
        labels = []
        
        for i, item in enumerate(menu_items):
            if item.type == MenuItemType.SEPARATOR:
                # Add separator
                labels.append(Label("─" * 20))
            else:
                # Create label text
                label_text = item.label
                
                # Add checkbox indicator
                if item.type == MenuItemType.CHECKBOX:
                    label_text = f"[{'X' if item.is_checked() else ' '}] {label_text}"
                
                # Add submenu indicator
                if item.type == MenuItemType.SUBMENU:
                    label_text = f"{label_text} ►"
                
                # Add shortcut
                if item.shortcut:
                    # Pad to align shortcuts
                    padding = " " * (20 - len(label_text))
                    label_text = f"{label_text}{padding}{item.shortcut}"
                
                # Create label
                label = Label(
                    text=label_text,
                    style=lambda: f"class:menu-item{'.selected' if self.active_item_index == i else ''}{'.disabled' if not item.is_enabled() else ''}"
                )
                
                # Add mouse handler for clickable items
                if item.type != MenuItemType.SUBMENU and item.is_enabled():
                    original_mouse_handler = label.mouse_handler
                    
                    def create_mouse_handler(menu_item):
                        def item_mouse_handler(mouse_event):
                            if mouse_event.event_type == MouseEventType.MOUSE_UP:
                                if menu_item.handler:
                                    get_app().create_background_task(self._handle_menu_item(menu_item))
                                return True
                            return original_mouse_handler(mouse_event)
                        return item_mouse_handler
                    
                    label.mouse_handler = create_mouse_handler(item)
                
                # Add to list
                labels.append(Box(label, padding=1))
        
        # Create container
        return HSplit(labels)

    async def _handle_menu_item(self, item: MenuItem) -> None:
        """
        Handle selection of a menu item.

        Args:
            item: The selected menu item
        """
        # Close menu
        await self._close_menu()
        
        # Toggle checkbox items
        if item.type == MenuItemType.CHECKBOX:
            item.set_checked(not item.is_checked())
        
        # Call handler
        if item.handler:
            await item.handler()

    # Menu command handlers
    
    async def _on_new_pipeline(self) -> None:
        """Handle New Pipeline command."""
        logger.warning("MenuBar: File > New Pipeline selected (Not Implemented).")
        if hasattr(self.state, 'notify'):
            self.state.notify('operation_status_changed', {'operation': 'new_pipeline', 'status': 'info', 'message': 'File > New Pipeline: Not Implemented', 'source': 'MenuBar'})

    async def _on_open_pipeline(self) -> None:
        """Handle Open Pipeline command."""
        logger.warning("MenuBar: File > Open Pipeline selected (Not Implemented).")
        if hasattr(self.state, 'notify'):
            self.state.notify('operation_status_changed', {'operation': 'open_pipeline', 'status': 'info', 'message': 'File > Open Pipeline: Not Implemented', 'source': 'MenuBar'})

    async def _on_save_pipeline(self) -> None: # This was implemented in msg_idx 235, ensuring it's the correct version
        """Handle Save Pipeline command."""
        logger.info("MenuBar: File > Save Pipeline selected.")
        active_orchestrator: Optional['PipelineOrchestrator'] = getattr(self.state, 'active_orchestrator', None)
        selected_plate: Optional[Dict[str, Any]] = getattr(self.state, 'selected_plate', None)

        if not active_orchestrator or not selected_plate:
            err_msg = "No active plate selected to save pipeline for."
            logger.warning(err_msg)
            if hasattr(self.state, 'notify'):
                self.state.notify('operation_status_changed', {'operation': 'save_pipeline', 'status': 'error', 'message': err_msg, 'source': 'MenuBar'})
            return

        pipeline_definition: Optional[List[AbstractStep]] = getattr(active_orchestrator, 'pipeline_definition', None)
        if not pipeline_definition:
            err_msg = f"No pipeline definition loaded for plate '{selected_plate.get('name', 'Unknown')}' to save."
            logger.warning(err_msg)
            if hasattr(self.state, 'notify'):
                self.state.notify('operation_status_changed', {'operation': 'save_pipeline', 'status': 'error', 'message': err_msg, 'source': 'MenuBar'})
            return

        try:
            plate_dir_path_str = selected_plate.get('path')
            if not plate_dir_path_str:
                raise ValueError("Selected plate information is missing a valid 'path' attribute.")
            
            plate_dir_path = Path(plate_dir_path_str)
            # Ensure json is imported if not already (it was added in a previous step for this file)
            import json
            filename = getattr(active_orchestrator, 'DEFAULT_PIPELINE_FILENAME', 'pipeline_definition.json')
            save_path = plate_dir_path / filename
            
            pipeline_dicts = [step.to_dict() for step in pipeline_definition]
            json_content = json.dumps(pipeline_dicts, indent=2)

            logger.info(f"Attempting to save pipeline definition from MenuBar for plate '{selected_plate.get('id')}' to {save_path}")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(json_content)

            save_success_msg = f"Pipeline saved to {save_path} (via MenuBar)"
            logger.info(save_success_msg)
            if hasattr(self.state, 'notify'):
                self.state.notify('operation_status_changed', {
                    'operation': 'save_pipeline',
                    'status': 'success',
                    'message': save_success_msg,
                    'source': 'MenuBar'
                })
        except Exception as e:
            save_fail_msg = f"Save Pipeline Error (MenuBar): {str(e)}"
            logger.error(save_fail_msg, exc_info=True)
            if hasattr(self.state, 'notify'):
                self.state.notify('operation_status_changed', {
                    'operation': 'save_pipeline',
                    'status': 'error',
                    'message': save_fail_msg,
                    'source': 'MenuBar'
                })

    async def _on_save_pipeline_as(self) -> None:
        """Handle Save Pipeline As command."""
        logger.warning("MenuBar: File > Save Pipeline As selected (Not Implemented).")
        if hasattr(self.state, 'notify'):
            self.state.notify('operation_status_changed', {'operation': 'save_pipeline_as', 'status': 'info', 'message': 'File > Save Pipeline As: Not Implemented', 'source': 'MenuBar'})

    async def _on_exit(self) -> None:
        """Handle Exit command."""
        self.state.notify('menu_command', {'command': 'exit'})

    async def _on_add_step(self) -> None:
        """Handle Add Step command."""
        logger.warning("MenuBar: Edit > Add Step selected (Not Implemented).")
        if hasattr(self.state, 'notify'):
            self.state.notify('operation_status_changed', {'operation': 'add_step', 'status': 'info', 'message': 'Edit > Add Step: Not Implemented', 'source': 'MenuBar'})

    async def _on_edit_step(self) -> None:
        """Handle Edit Step command: Activates the FunctionPatternEditor for the selected step."""
        logger.info("MenuBar: Edit > Edit Step selected.")
        selected_step = getattr(self.state, 'selected_step', None)

        if selected_step:
            # These attributes ('editing_pattern', 'selected_step_for_editing')
            # were added to TUIState in tui_architecture.py
            self.state.selected_step_for_editing = selected_step
            self.state.editing_pattern = True # This will trigger OpenHCSTUI._get_left_pane() to show FPE
            
            logger.info(f"MenuBar: Set editing_pattern=True, selected_step_for_editing='{selected_step.get('name', 'N/A')}'.")
            
            if hasattr(self.state, 'notify'):
                # Notify that editing state has changed, perhaps for other UI elements to react
                self.state.notify('editing_state_changed', {'editing_pattern': True, 'step': selected_step})
                # Also provide immediate feedback via status bar
                self.state.notify('operation_status_changed', {
                    'operation': 'edit_step_start',
                    'status': 'info',
                    'message': f"Editing pattern for step: {selected_step.get('name', 'N/A')}",
                    'source': 'MenuBar'
                })
            
            # Force a redraw so that OpenHCSTUI._get_left_pane() re-evaluates and shows the FPE
            app = get_app()
            if app:
                app.invalidate()
        else:
            logger.warning("MenuBar: Edit Step clicked, but no step is selected in TUIState.")
            if hasattr(self.state, 'notify'):
                self.state.notify('operation_status_changed', {
                    'operation': 'edit_step_attempt',
                    'status': 'warning',
                    'message': "No step selected to edit. Please select a step in the Step Viewer.",
                    'source': 'MenuBar'
                })

    async def _on_remove_step(self) -> None:
        """Handle Remove Step command."""
        logger.info("MenuBar: Edit > Remove Step selected.")
        selected_step_dict: Optional[Dict[str, Any]] = getattr(self.state, 'selected_step', None)
        active_orchestrator: Optional['PipelineOrchestrator'] = getattr(self.state, 'active_orchestrator', None)

        if not selected_step_dict:
            msg = "No step selected to remove."
            logger.warning(msg)
            if hasattr(self.state, 'notify'): self.state.notify('operation_status_changed', {'operation': 'remove_step', 'status': 'warning', 'message': msg, 'source': 'MenuBar'})
            return

        if not active_orchestrator or not hasattr(active_orchestrator, 'pipeline_definition') or active_orchestrator.pipeline_definition is None:
            msg = "No active pipeline definition to remove step from."
            logger.warning(msg)
            if hasattr(self.state, 'notify'): self.state.notify('operation_status_changed', {'operation': 'remove_step', 'status': 'error', 'message': msg, 'source': 'MenuBar'})
            return
        
        step_uid_to_remove = selected_step_dict.get('uid')
        if not step_uid_to_remove:
            msg = "Selected step has no UID, cannot remove."
            logger.error(msg)
            if hasattr(self.state, 'notify'): self.state.notify('operation_status_changed', {'operation': 'remove_step', 'status': 'error', 'message': msg, 'source': 'MenuBar'})
            return

        current_pipeline: List[AbstractStep] = active_orchestrator.pipeline_definition
        original_length = len(current_pipeline)
        
        active_orchestrator.pipeline_definition = [step for step in current_pipeline if step.uid != step_uid_to_remove]

        if len(active_orchestrator.pipeline_definition) < original_length:
            self.state.selected_step = None # Clear selection
            self.state.selected_step_for_editing = None # Clear if it was being edited
            if hasattr(self.state, 'notify'):
                self.state.notify('pipeline_definition_changed', {'orchestrator_id': active_orchestrator.plate_path}) # Notify StepViewerPane
                self.state.notify('operation_status_changed', {
                    'operation': 'remove_step',
                    'status': 'success',
                    'message': f"Step '{selected_step_dict.get('name', step_uid_to_remove)}' removed.",
                    'source': 'MenuBar'
                })
            logger.info(f"Step '{step_uid_to_remove}' removed from pipeline.")
        else:
            msg = f"Step with UID '{step_uid_to_remove}' not found in pipeline."
            logger.warning(msg)
            if hasattr(self.state, 'notify'): self.state.notify('operation_status_changed', {'operation': 'remove_step', 'status': 'warning', 'message': msg, 'source': 'MenuBar'})
            
    async def _on_move_step_up(self) -> None:
        """Handle Move Step Up command."""
        logger.warning("MenuBar: Edit > Move Step Up selected (Not Implemented).")
        if hasattr(self.state, 'notify'):
            self.state.notify('operation_status_changed', {'operation': 'move_step_up', 'status': 'info', 'message': 'Edit > Move Step Up: Not Implemented', 'source': 'MenuBar'})

    async def _on_move_step_down(self) -> None:
        """Handle Move Step Down command."""
        logger.warning("MenuBar: Edit > Move Step Down selected (Not Implemented).")
        if hasattr(self.state, 'notify'):
            self.state.notify('operation_status_changed', {'operation': 'move_step_down', 'status': 'info', 'message': 'Edit > Move Step Down: Not Implemented', 'source': 'MenuBar'})

    async def _on_toggle_log_drawer(self) -> None:
        """Handle Toggle Log Drawer command."""
        self.state.notify('menu_command', {'command': 'toggle_log_drawer'})

    async def _on_toggle_vim_mode(self) -> None:
        """Handle Toggle Vim Mode command."""
        self.state.notify('menu_command', {'command': 'toggle_vim_mode'})

    async def _on_set_theme(self, theme: str) -> None:
        """
        Handle Set Theme command.
        
        Args:
            theme: The theme to set ('light', 'dark', or 'system')
        """
        self.state.notify('menu_command', {'command': 'set_theme', 'theme': theme})

    async def _on_pre_compile(self) -> None:
        """Handle Pre-compile command."""
        self.state.notify('menu_command', {'command': 'pre_compile'})

    async def _on_compile(self) -> None:
        """Handle Compile command."""
        self.state.notify('menu_command', {'command': 'compile'})

    async def _on_run(self) -> None:
        """Handle Run command."""
        self.state.notify('menu_command', {'command': 'run'})

    async def _on_test(self) -> None:
        """Handle Test command by adding a predefined test plate."""
        logger.info("MenuBar: Pipeline > Test selected. Adding predefined test plate.")
        test_plate_relative_path = "tests/integration/tests_data/opera_phenix_pipeline/test_main_3d[OperaPhenix]/zstack_plate"
        test_plate_path_str = test_plate_relative_path
        test_plate_backend = "OperaPhenix"

        logger.info(f"MenuBar: Attempting to add test plate: {test_plate_path_str} (Backend: {test_plate_backend})")
        if hasattr(self.state, 'notify'):
            self.state.notify('add_predefined_plate', {
                'path': test_plate_path_str,
                'backend': test_plate_backend
            })
            self.state.notify('operation_status_changed', {
                'operation': 'add_test_plate',
                'status': 'pending',
                'message': f"Adding test plate '{Path(test_plate_path_str).name}' (from Menu)...",
                'source': 'MenuBar'
            })
        else:
            logger.error("MenuBar: Cannot add test plate: TUIState.notify not available.")

    async def _on_settings(self) -> None:
        """Handle Settings command."""
        self.state.notify('menu_command', {'command': 'settings'})

    async def _on_documentation(self) -> None:
        """Handle Documentation command."""
        logger.info("MenuBar: Help > Documentation selected.")
        if hasattr(self.state, 'notify'):
            self.state.notify('operation_status_changed', {'operation': 'show_documentation', 'status': 'info', 'message': 'Help > Documentation: Not Implemented (placeholder)', 'source': 'MenuBar'})

    async def _on_keyboard_shortcuts(self) -> None:
        """Handle Keyboard Shortcuts command."""
        logger.info("MenuBar: Help > Keyboard Shortcuts selected.")
        if hasattr(self.state, 'notify'):
            self.state.notify('operation_status_changed', {'operation': 'show_shortcuts', 'status': 'info', 'message': 'Help > Keyboard Shortcuts: Not Implemented (placeholder)', 'source': 'MenuBar'})

    async def _on_about(self) -> None:
        """Handle About command."""
        logger.info("MenuBar: Help > About selected.")
        if hasattr(self.state, 'notify'):
            self.state.notify('operation_status_changed', {'operation': 'show_about', 'status': 'info', 'message': 'Help > About: Not Implemented (placeholder)', 'source': 'MenuBar'})

    # Event handlers
    
    async def _on_operation_status_changed(self, data) -> None:
        """
        Handle operation status change event.

        Args:
            data: Dictionary with operation and status
        """
        # Force UI refresh to update menu item enabling
        get_app().invalidate()

    async def _on_plate_selected(self, data) -> None:
        """
        Handle plate selection event.

        Args:
            data: The selected plate
        """
        # Force UI refresh to update menu item enabling
        get_app().invalidate()

    async def _on_is_compiled_changed(self, data) -> None:
        """
        Handle is_compiled flag change event.

        Args:
            data: The new is_compiled value
        """
        # Force UI refresh to update menu item enabling
        get_app().invalidate()

    def __pt_container__(self) -> Container:
        """Return the container to render."""
        return self.container
