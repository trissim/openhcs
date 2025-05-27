from prompt_toolkit.layout import Container
import ast
import inspect # Added for ParameterEditor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from prompt_toolkit.application import get_app
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.layout import HSplit, ScrollablePane, VSplit, Container # Added Container
from prompt_toolkit.widgets import (Box, Button, Dialog, Label,
                                    TextArea, RadioList as Dropdown)
from prompt_toolkit.key_binding import KeyBindings # Added KeyBindings

class InteractiveListItem:
    """
    A custom prompt_toolkit widget that represents an interactive item in a list.
    It displays item data, handles selection, and provides optional move up/down buttons.
    """
    def __init__(self, item_data: Dict[str, Any], item_index: int, is_selected: bool,
                 display_text_func: Callable[[Dict[str, Any], bool], str],
                 on_select: Callable[[int], None],
                 on_move_up: Optional[Callable[[int], None]] = None,
                 on_move_down: Optional[Callable[[int], None]] = None,
                 can_move_up: bool = False,
                 can_move_down: bool = False):
        self.item_data = item_data
        self.item_index = item_index
        self.is_selected = is_selected
        self.display_text_func = display_text_func
        self.on_select = on_select
        self.on_move_up = on_move_up
        self.on_move_down = on_move_down
        self.can_move_up = can_move_up
        self.can_move_down = can_move_down

        self.label = Label(self._get_display_text(), dont_extend_width=True)
        # Making label focusable and adding keybindings for selection
        # self.label.control.is_focusable = True # This might not be needed if item itself is focusable
        # self.label.control.key_bindings = self._create_key_bindings()

        self.move_up_button = Button("▲", handler=self._handle_move_up_click)
        self.move_down_button = Button("▼", handler=self._handle_move_down_click)

        self.container = self._create_container()

    def _get_display_text(self) -> str:
        """Generate the display text for the item."""
        return self.display_text_func(self.item_data, self.is_selected)

    # def _create_key_bindings(self): # May not be needed directly on label
    #     kb = KeyBindings()
    #     @kb.add('enter')
    #     def _(event):
    #         self.on_select(self.item_index)
    #     return kb

    def _handle_move_up_click(self):
        if self.on_move_up and self.can_move_up:
            # Ensure background task creation if on_move_up is async
            # get_app().create_background_task(self.on_move_up(self.item_index))
            self.on_move_up(self.item_index)


    def _handle_move_down_click(self):
        if self.on_move_down and self.can_move_down:
            # get_app().create_background_task(self.on_move_down(self.item_index))
            self.on_move_down(self.item_index)


    def _create_container(self) -> VSplit: # Changed to VSplit for better layout control
        """Create the main container for the list item."""
        style = "class:list-item.selected" if self.is_selected else "class:list-item"

        move_buttons_children = []
        if self.on_move_up:
            self.move_up_button.text = "▲" if self.can_move_up else " "
            self.move_up_button.handler = self._handle_move_up_click if self.can_move_up else None
            move_buttons_children.append(self.move_up_button)
        if self.on_move_down:
            self.move_down_button.text = "▼" if self.can_move_down else " "
            self.move_down_button.handler = self._handle_move_down_click if self.can_move_down else None
            move_buttons_children.append(self.move_down_button)

        buttons_container = VSplit(move_buttons_children, width=3, padding=0) if move_buttons_children else VSplit([], width=0)

        # Create the item index display
        index_display = f"{self.item_index + 1}: "

        # Get the main display text from the callback
        display_text = self.display_text_func(self.item_data, self.is_selected)

        # Combine parts for the button text
        full_text = f"{index_display}{display_text}"

        # Use a Button for the main item to make it easily clickable and focusable
        item_button = Button(
            text=full_text,
            handler=lambda: self.on_select(self.item_index),
            width=None # Allow button to take available width
        )
        if self.is_selected:
             item_button.style = "class:list-item.selected" # Apply selection style to button
        else:
             item_button.style = "class:list-item"

        return VSplit([
            item_button,
            buttons_container
        ], style=style, padding=0)


    def __pt_container__(self):
        return self.container

class GroupedDropdown(Dropdown):
    """Dropdown with support for disabled group headers."""

    def __init__(self, options, default=None):
        # Filter out headers for internal options list
        self.all_options = options
        self.selectable_options = [(value, text) for value, text in options if value != 'header']
        super().__init__(options=self.selectable_options, default=default)

    def _get_text_fragments(self):
        """Override to display all options including headers."""
        result = []
        for i, (value, text) in enumerate(self.all_options):
            if value == 'header':
                # Header style
                result.append(('class:dropdown.header', text))
                result.append(('', '\n'))
            elif value == self.current_value:
                # Selected item
                result.append(('class:dropdown-selected', f"> {text}"))
                result.append(('', '\n'))
            else:
                # Normal item
                result.append(('', f"  {text}"))
                result.append(('', '\n'))
        return result

class ParameterEditor:
    """
    A UI component for editing parameters of a given function.
    It dynamically generates input fields based on function signature.
    """
    def __init__(self,
                 func: Optional[Callable],
                 current_kwargs: Dict[str, Any],
                 func_index: int, # Added func_index
                 on_parameter_change: Callable[[str, Any, int], None], # Callback: (param_name, new_value, func_idx)
                 on_reset_parameter: Callable[[str, int], None],       # Callback: (param_name, func_idx)
                 on_reset_all_parameters: Callable[[int], None]      # Callback: (func_idx)
                ):
        self.func = func
        self.current_kwargs = current_kwargs
        self.func_index = func_index # Store func_index
        self.on_parameter_change = on_parameter_change
        self.on_reset_parameter = on_reset_parameter
        self.on_reset_all_parameters = on_reset_all_parameters

        self.container = self._build_ui()

    def update_function(self, func: Optional[Callable], new_kwargs: Dict[str, Any], func_index: int): # Added func_index
        """Updates the editor with a new function, its current arguments, and its index."""
        self.func = func
        self.current_kwargs = new_kwargs
        self.func_index = func_index # Update func_index
        self.container = self._build_ui() # Rebuild UI for the new function
        # Request a redraw if this component is part of a larger UI
        get_app().invalidate()

    def _build_ui(self) -> Container:
        if not self.func:
            return Label("Select a function to edit parameters.")

        params_info = self._get_function_parameters(self.func)
        param_fields = []

        for p_info in params_info:
            name = p_info['name']
            default = p_info['default']
            # Use current_kwargs if available, otherwise default
            current_value = self.current_kwargs.get(name, default)
            required = p_info['required']
            is_special = p_info.get('is_special', False)

            param_fields.append(
                self._create_parameter_field(name, default, current_value, required, is_special)
            )

        reset_all_button = Button(
            "Reset All Params",
            handler=lambda: get_app().create_background_task(self.on_reset_all_parameters(self.func_index))
        )
        param_fields.append(reset_all_button)

        return HSplit(param_fields)

    def _get_function_parameters(self, func: Callable) -> List[Dict]:
        """Get parameters for a function through introspection."""
        params_list = []
        sig = inspect.signature(func)
        for name, param in sig.parameters.items():
            if name in ('self', 'cls'):
                continue
            default_value = param.default if param.default is not inspect.Parameter.empty else None
            param_type = param.annotation if param.annotation is not inspect.Parameter.empty else Any

            # Check if special parameter (assuming special_inputs is an attribute on the func)
            is_special = hasattr(func, 'special_inputs') and name in getattr(func, 'special_inputs', [])

            params_list.append({
                'name': name,
                'default': default_value,
                'type': param_type,
                'required': param.default is inspect.Parameter.empty,
                'is_special': is_special
            })
        return params_list

    def _create_parameter_field(self, name: str, default: Any, current_value: Any, required: bool, is_special: bool) -> Container:
        """Create a UI field for a single parameter."""
        label_text = f"{name}{'*' if required else ''}"
        if is_special:
            label_text = f"{label_text} [S]"
        label = Label(label_text + ": ", width=25) # Increased width for better readability

        input_field = self._create_input_field(name, current_value)

        reset_button = Button(
            "Reset",
            handler=lambda: get_app().create_background_task(self.on_reset_parameter(name, self.func_index))
        )
        return VSplit([label, input_field, Box(reset_button, width=8)], padding=1)

    def _create_input_field(self, name: str, value: Any) -> TextArea:
        """Create an input field appropriate for the parameter type."""
        value_str = str(value) if value is not None else ""
        text_area = TextArea(text=value_str, multiline=False, height=1)

        def on_text_changed_handler(buffer):
            # This now calls the on_parameter_change callback passed during instantiation
            # The actual parsing and state update will happen in the parent component (FunctionPatternEditor)
            get_app().create_background_task(self.on_parameter_change(name, buffer.text, self.func_index))

        text_area.buffer.on_text_changed += on_text_changed_handler
        return text_area

    def __pt_container__(self):
        return self.container