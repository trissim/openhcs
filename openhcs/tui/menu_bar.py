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
from openhcs.core.config import GlobalPipelineConfig # Added for type hinting and use
# Dialog import moved to commands.py to avoid potential circularity if dialogs use commands
# from openhcs.tui.dialogs.global_settings_editor import GlobalSettingsEditorDialog

# Import Commands
from openhcs.tui.commands import Command, ShowGlobalSettingsDialogCommand, ShowHelpCommand

if TYPE_CHECKING:
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
    from openhcs.tui.tui_architecture import TUIState


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

        for menu_name, menu_data in structure.items():
            if not isinstance(menu_data, dict):
                raise ValueError(f"Menu '{menu_name}' data must be a dictionary")
            if "mnemonic" not in menu_data:
                raise ValueError(f"Menu '{menu_name}' must have a 'mnemonic' field")
            if "items" not in menu_data:
                raise ValueError(f"Menu '{menu_name}' must have an 'items' field")

            items = menu_data["items"]
            if not isinstance(items, list):
                raise ValueError(f"Menu '{menu_name}' 'items' field must be a list")

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

# Import modular menu structure and handlers
from .menu_structure import DEFAULT_MENU_STRUCTURE as _DEFAULT_MENU_STRUCTURE, MENU_CONDITIONS, MENU_HANDLERS
from .menu_handlers import MenuHandlers


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
    def __init__(self, state: 'TUIState', context: 'ProcessingContext'): # Added context
        """
        Initialize the menu bar.

        Args:
            state: The TUI state manager
            context: The main ProcessingContext (or initial context)
        """
        self.state = state
        self.context = context # Store context for commands

        # Initialize modular menu handlers
        self.menu_handlers = MenuHandlers(state)

        # Initialize state with thread safety
        self.active_menu: Optional[str] = None
        self.active_submenu: Optional[List[MenuItem]] = None
        self.active_item_index: Optional[int] = None
        self.menu_lock = ReentrantLock()

        # Create handler map (now uses modular MENU_HANDLERS)
        self.handler_map = self._create_handler_map()

        # Create condition map
        self.condition_map = self._create_condition_map()

        # Load menu structure (uses _DEFAULT_MENU_STRUCTURE)
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

    def _create_handler_map(self) -> Dict[str, Union[Callable, Command]]:
        """
        Create a map of handler names to handler methods using modular structure.

        Returns:
            Dictionary mapping handler names to callables or Commands
        """
        # Create handler map using modular MENU_HANDLERS mapping
        handler_map = {}

        for handler_name, method_name in MENU_HANDLERS.items():
            # Get the method from menu_handlers instance
            if hasattr(self.menu_handlers, method_name):
                handler_map[handler_name] = getattr(self.menu_handlers, method_name)
            else:
                # Fallback to legacy handlers if method doesn't exist yet
                legacy_method = f"_on_{method_name.replace('_', '')}"
                if hasattr(self, legacy_method):
                    handler_map[handler_name] = getattr(self, legacy_method)
                else:
                    # Create placeholder handler
                    handler_map[handler_name] = lambda: logger.warning(f"Handler {method_name} not implemented")

        return handler_map

    def _create_condition_map(self) -> Dict[str, Condition]:
        """
        Create a map of condition names to Condition objects using modular structure.

        Returns:
            Dictionary mapping condition names to Condition objects
        """
        # Create condition map using modular MENU_CONDITIONS mapping
        condition_map = {}

        for condition_name, condition_expr in MENU_CONDITIONS.items():
            # Create condition lambda from expression string
            try:
                # Simple evaluation for basic state checks
                if condition_expr.startswith("state."):
                    attr_path = condition_expr[6:]  # Remove "state."
                    if "==" in attr_path:
                        attr_name, value = attr_path.split("==", 1)
                        attr_name = attr_name.strip()
                        value = value.strip().strip("'\"")
                        condition_map[condition_name] = Condition(
                            lambda attr=attr_name, val=value: self._get_required_state(attr) == val
                        )
                    else:
                        condition_map[condition_name] = Condition(
                            lambda attr=attr_path: self._get_required_state(attr)
                        )
                else:
                    # Fallback for complex expressions
                    condition_map[condition_name] = Condition(lambda: True)
            except Exception:
                # Safe fallback
                condition_map[condition_name] = Condition(lambda: True)

        return condition_map

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
        raw_structure = _DEFAULT_MENU_STRUCTURE
        self._validate_menu_structure(raw_structure)
        return self._convert_to_menu_items(raw_structure)

    def _validate_menu_structure(self, raw_structure: Dict[str, Any]) -> None:
        """Validate the menu structure."""
        try:
            MenuStructureSchema.validate_menu_structure(raw_structure)
        except ValueError as e:
            self._handle_menu_validation_error(e)

    def _handle_menu_validation_error(self, error: ValueError) -> None:
        """Handle menu validation errors."""
        error_message = f"Invalid _DEFAULT_MENU_STRUCTURE: {error}"
        print(f"CRITICAL: Invalid _DEFAULT_MENU_STRUCTURE in menu_bar.py: {error}")
        raise RuntimeError(error_message) from error

    def _convert_to_menu_items(self, raw_structure: Dict[str, Any]) -> Dict[str, List[MenuItem]]:
        """Convert raw structure to MenuItem objects."""
        menu_structure = {}
        for menu_name, menu_data in raw_structure.items():
            self._validate_menu_data(menu_name, menu_data)
            menu_items = self._process_menu_items(menu_data["items"])
            menu_structure[menu_name] = {
                "mnemonic": menu_data["mnemonic"],
                "items": menu_items
            }
        return menu_structure

    def _validate_menu_data(self, menu_name: str, menu_data: Any) -> None:
        """Validate individual menu data."""
        if not isinstance(menu_data, dict):
            raise ValueError(f"Menu '{menu_name}' data must be a dictionary")
        if "items" not in menu_data:
            raise ValueError(f"Menu '{menu_name}' must have 'items' field")
        if "mnemonic" not in menu_data:
            raise ValueError(f"Menu '{menu_name}' must have 'mnemonic' field")

    def _process_menu_items(self, items: List[Dict[str, Any]]) -> List[MenuItem]:
        """Process menu items and convert to MenuItem objects."""
        menu_items = []
        for item in items:
            self._process_item_conditions(item)
            menu_item = MenuItem.from_dict(item, self.handler_map)
            menu_items.append(menu_item)
        return menu_items

    def _process_item_conditions(self, item: Dict[str, Any]) -> None:
        """Process conditions for a menu item."""
        if "enabled" in item and isinstance(item["enabled"], str):
            condition_name = item["enabled"]
            if condition_name in self.condition_map:
                item["enabled"] = self.condition_map[condition_name]

        if "checked" in item and isinstance(item["checked"], str):
            condition_name = item["checked"]
            if condition_name in self.condition_map:
                item["checked"] = self.condition_map[condition_name]



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
                style="class:menu-bar"
            )

            # Add mouse handler

            def create_mouse_handler(menu):
                def menu_mouse_handler(mouse_event):
                    if mouse_event.event_type == MouseEventType.MOUSE_UP:
                        get_app().create_background_task(self._activate_menu(menu))
                        return True
                    return original_mouse_handler(mouse_event)
                return menu_mouse_handler


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
            mnemonic = items.get('mnemonic')
            if not mnemonic:
                raise ValueError(f"Menu '{menu_name}' has no mnemonic defined")

            # Closure to capture the current menu_name
            def create_handler(menu):
                def handler(event: KeyPressEvent):
                    get_app().create_background_task(self._activate_menu(menu))
                return handler

            # Use explicit ESC prefix for Alt (Meta) keys:
            key = mnemonic.lower()
            logger.debug(f"Adding key binding: (escape, {key})")
            kb.add('escape', key)(create_handler(menu_name))

        return kb

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
        labels = []

        for i, item in enumerate(menu_items):
            label_widget = self._create_menu_item_widget(item, i)
            labels.append(label_widget)

        return HSplit(labels)

    def _create_menu_item_widget(self, item: MenuItem, index: int) -> Any:
        """Create a widget for a single menu item."""
        if item.type == MenuItemType.SEPARATOR:
            return self._create_separator_widget()
        else:
            return self._create_interactive_menu_item_widget(item, index)

    def _create_separator_widget(self) -> Label:
        """Create a separator widget."""
        return Label("─" * 20)

    def _create_interactive_menu_item_widget(self, item: MenuItem, index: int) -> Any:
        """Create an interactive menu item widget."""
        label_text = self._build_menu_item_text(item)
        label = self._create_menu_item_label(label_text, item, index)

        if self._should_add_mouse_handler(item):
            self._add_mouse_handler_to_label(label, item)

        return Box(label, padding=1)

    def _build_menu_item_text(self, item: MenuItem) -> str:
        """Build the display text for a menu item."""
        label_text = item.label

        # Add checkbox indicator
        if item.type == MenuItemType.CHECKBOX:
            checkbox_indicator = 'X' if item.is_checked() else ' '
            label_text = f"[{checkbox_indicator}] {label_text}"

        # Add submenu indicator
        if item.type == MenuItemType.SUBMENU:
            label_text = f"{label_text} ►"

        # Add shortcut
        if item.shortcut:
            padding = " " * (20 - len(label_text))
            label_text = f"{label_text}{padding}{item.shortcut}"

        return label_text

    def _create_menu_item_label(self, label_text: str, item: MenuItem, index: int) -> Label:
        """Create a label for a menu item."""
        return Label(
            text=label_text,
            style=lambda: self._get_menu_item_style(item, index)
        )

    def _get_menu_item_style(self, item: MenuItem, index: int) -> str:
        """Get the style string for a menu item."""
        base_style = "class:menu-item"

        if self.active_item_index == index:
            base_style += ".selected"

        if not item.is_enabled():
            base_style += ".disabled"

        return base_style

    def _should_add_mouse_handler(self, item: MenuItem) -> bool:
        """Check if a mouse handler should be added to the item."""
        return item.type != MenuItemType.SUBMENU and item.is_enabled()

    def _add_mouse_handler_to_label(self, label: Label, item: MenuItem) -> None:
        """Add mouse handler to a label."""
        def create_mouse_handler(menu_item):
            def item_mouse_handler(mouse_event):
                if mouse_event.event_type == MouseEventType.MOUSE_UP:
                    if menu_item.handler:
                        get_app().create_background_task(self._handle_menu_item(menu_item))
                    return True
                return False  # Changed from original_mouse_handler which was undefined
            return item_mouse_handler

        # Note: This would need to be properly implemented with the label's mouse handler system

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

    async def _on_save_pipeline(self) -> None:
        """Handle Save Pipeline command."""
        logger.info("MenuBar: File > Save Pipeline selected.")

        # Validate prerequisites
        if not self._validate_save_prerequisites():
            return

        # Get validated data
        active_orchestrator = getattr(self.state, 'active_orchestrator', None)
        selected_plate = getattr(self.state, 'selected_plate', None)
        pipeline_definition = getattr(active_orchestrator, 'pipeline_definition', None)

        try:
            save_path = self._prepare_save_path(selected_plate, active_orchestrator)
            self._save_pipeline_to_file(pipeline_definition, save_path, selected_plate)
        except Exception as e:
            self._handle_save_error(e)

    def _validate_save_prerequisites(self) -> bool:
        """Validate prerequisites for saving pipeline."""
        active_orchestrator = getattr(self.state, 'active_orchestrator', None)
        selected_plate = getattr(self.state, 'selected_plate', None)

        if not active_orchestrator or not selected_plate:
            self._notify_save_error("No active plate selected to save pipeline for.")
            return False

        pipeline_definition = getattr(active_orchestrator, 'pipeline_definition', None)
        if not pipeline_definition:
            plate_name = selected_plate.get('name', 'Unknown')
            self._notify_save_error(f"No pipeline definition loaded for plate '{plate_name}' to save.")
            return False

        return True

    def _prepare_save_path(self, selected_plate: Dict[str, Any], active_orchestrator: 'PipelineOrchestrator') -> Path:
        """Prepare the save path for the pipeline file."""
        plate_dir_path_str = selected_plate.get('path')
        if not plate_dir_path_str:
            raise ValueError("Selected plate information is missing a valid 'path' attribute.")

        plate_dir_path = Path(plate_dir_path_str)
        filename = getattr(active_orchestrator, 'DEFAULT_PIPELINE_FILENAME', 'pipeline_definition.json')
        return plate_dir_path / filename

    def _save_pipeline_to_file(self, pipeline_definition: List[AbstractStep], save_path: Path, selected_plate: Dict[str, Any]) -> None:
        """Save pipeline definition to file."""
        # Convert pipeline to JSON
        pipeline_dicts = [step.to_dict() for step in pipeline_definition]
        json_content = json.dumps(pipeline_dicts, indent=2)

        # Create directory and save file
        logger.info(f"Attempting to save pipeline definition from MenuBar for plate '{selected_plate.get('id')}' to {save_path}")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(json_content)

        # Notify success
        save_success_msg = f"Pipeline saved to {save_path} (via MenuBar)"
        logger.info(save_success_msg)
        self._notify_save_success(save_success_msg)

    def _notify_save_error(self, message: str) -> None:
        """Notify save error."""
        logger.warning(message)
        if hasattr(self.state, 'notify'):
            asyncio.create_task(self.state.notify('operation_status_changed', {
                'operation': 'save_pipeline',
                'status': 'error',
                'message': message,
                'source': 'MenuBar'
            }))

    def _notify_save_success(self, message: str) -> None:
        """Notify save success."""
        if hasattr(self.state, 'notify'):
            asyncio.create_task(self.state.notify('operation_status_changed', {
                'operation': 'save_pipeline',
                'status': 'success',
                'message': message,
                'source': 'MenuBar'
            }))

    def _handle_save_error(self, error: Exception) -> None:
        """Handle save errors."""
        save_fail_msg = f"Save Pipeline Error (MenuBar): {str(error)}"
        logger.error(save_fail_msg, exc_info=True)
        if hasattr(self.state, 'notify'):
            asyncio.create_task(self.state.notify('operation_status_changed', {
                'operation': 'save_pipeline',
                'status': 'error',
                'message': save_fail_msg,
                'source': 'MenuBar'
            }))

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

        # Validate preconditions
        selected_step_dict, active_orchestrator = self._validate_remove_step_preconditions()
        if not selected_step_dict or not active_orchestrator:
            return

        # Get step UID and remove from pipeline
        step_uid_to_remove = self._get_step_uid_for_removal(selected_step_dict)
        if not step_uid_to_remove:
            return

        # Perform removal and handle result
        removal_successful = self._remove_step_from_pipeline(active_orchestrator, step_uid_to_remove)
        self._handle_removal_result(removal_successful, selected_step_dict, step_uid_to_remove, active_orchestrator)

    def _validate_remove_step_preconditions(self) -> Tuple[Optional[Dict[str, Any]], Optional['PipelineOrchestrator']]:
        """Validate preconditions for step removal."""
        selected_step_dict = getattr(self.state, 'selected_step', None)
        active_orchestrator = getattr(self.state, 'active_orchestrator', None)

        if not selected_step_dict:
            self._notify_error("remove_step", "warning", "No step selected to remove.")
            return None, None

        if not active_orchestrator or not hasattr(active_orchestrator, 'pipeline_definition') or active_orchestrator.pipeline_definition is None:
            self._notify_error("remove_step", "error", "No active pipeline definition to remove step from.")
            return None, None

        return selected_step_dict, active_orchestrator

    def _get_step_uid_for_removal(self, selected_step_dict: Dict[str, Any]) -> Optional[str]:
        """Get and validate step UID for removal."""
        step_uid_to_remove = selected_step_dict.get('step_id')
        if not step_uid_to_remove:
            self._notify_error("remove_step", "error", "Selected step has no step_id, cannot remove.")
            return None
        return step_uid_to_remove

    def _remove_step_from_pipeline(self, active_orchestrator: 'PipelineOrchestrator', step_uid_to_remove: str) -> bool:
        """Remove step from pipeline and return success status."""
        current_pipeline = active_orchestrator.pipeline_definition
        original_length = len(current_pipeline)

        active_orchestrator.pipeline_definition = [step for step in current_pipeline if step.step_id != step_uid_to_remove]

        return len(active_orchestrator.pipeline_definition) < original_length

    def _handle_removal_result(self, success: bool, selected_step_dict: Dict[str, Any], step_uid: str, orchestrator: 'PipelineOrchestrator') -> None:
        """Handle the result of step removal."""
        if success:
            self._handle_successful_removal(selected_step_dict, step_uid, orchestrator)
        else:
            self._notify_error("remove_step", "warning", f"Step with UID '{step_uid}' not found in pipeline.")

    def _handle_successful_removal(self, selected_step_dict: Dict[str, Any], step_uid: str, orchestrator: 'PipelineOrchestrator') -> None:
        """Handle successful step removal."""
        self.state.selected_step = None
        self.state.selected_step_for_editing = None

        if hasattr(self.state, 'notify'):
            self.state.notify('pipeline_definition_changed', {'orchestrator_id': orchestrator.plate_path})
            self.state.notify('operation_status_changed', {
                'operation': 'remove_step',
                'status': 'success',
                'message': f"Step '{selected_step_dict.get('name', step_uid)}' removed.",
                'source': 'MenuBar'
            })
        logger.info(f"Step '{step_uid}' removed from pipeline.")

    def _notify_error(self, operation: str, status: str, message: str) -> None:
        """Helper to notify errors and log them."""
        logger.warning(message) if status == "warning" else logger.error(message)
        if hasattr(self.state, 'notify'):
            self.state.notify('operation_status_changed', {
                'operation': operation, 'status': status, 'message': message, 'source': 'MenuBar'
            })

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

    # _on_settings, _on_documentation, _on_keyboard_shortcuts, _on_about are removed
    # as their functionality is now handled by ShowGlobalSettingsDialogCommand and ShowHelpCommand.

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

    async def shutdown(self):
        """
        Explicit cleanup method for deterministic resource release.
        Unregisters observers from TUIState.
        """
        logger.info("MenuBar: Shutting down...")
        # Unregister observers
        self.state.remove_observer('operation_status_changed', self._on_operation_status_changed)
        self.state.remove_observer('plate_selected', self._on_plate_selected)
        self.state.remove_observer('is_compiled_changed', self._on_is_compiled_changed)
        logger.info("MenuBar: Observers unregistered.")
        logger.info("MenuBar: Shutdown complete.")

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
        if mouse_event.event_type == MouseEventType.MOUSE_UP:
            # Check if a top-level menu was clicked
            for i, label in enumerate(self.menu_labels):
                if mouse_event.is_mouse_over(label):
                    menu_name = list(self.menu_structure.keys())[i]
                    get_app().create_background_task(self._activate_menu(menu_name))
                    return True

            # Check if a submenu item was clicked
            if self.active_submenu and mouse_event.is_mouse_over(self.active_submenu_container):
                # Delegate to the submenu's mouse handler if it has one
                if hasattr(self.active_submenu_container, 'mouse_handler'):
                    return self.active_submenu_container.mouse_handler(mouse_event)
                return True # Event handled, but no specific item handler
        return False # Event not handled
