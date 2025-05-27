# Code Reference Analysis

Found 16 files with references:

## openhcs/tui/components.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 17 | InteractiveListItem.__init__ | `Dict` |
| 18 | InteractiveListItem.__init__ | `Dict` |
| 140 | ParameterEditor.__init__ | `Dict` |
| 155 | ParameterEditor.update_function | `Dict` |
| 191 | ParameterEditor._get_function_parameters | `List` |
| 191 | ParameterEditor._get_function_parameters | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 3 | `from typing import Any, Callable, Dict, List, Optional, Tuple, Union` |
| 9 | `TextArea, RadioList as Dropdown)` |
| 12 | `class InteractiveListItem:` |
| 17 | `def __init__(self, item_data: Dict[str, Any], item_index: int, is_selected: bool...` |
| 18 | `display_text_func: Callable[[Dict[str, Any], bool], str],` |
| 140 | `current_kwargs: Dict[str, Any],` |
| 155 | `def update_function(self, func: Optional[Callable], new_kwargs: Dict[str, Any], ...` |
| 191 | `def _get_function_parameters(self, func: Callable) -> List[Dict]:` |

## openhcs/tui/file_browser.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 48 | FileManagerBrowser.__init__ | `List` |
| 55 | FileManagerBrowser.__init__ | `List` |
| 67 | FileManagerBrowser.__init__ | `List` |
| 67 | FileManagerBrowser.__init__ | `Dict` |
| 69 | FileManagerBrowser.__init__ | `List` |
| 348 | FileManagerBrowser._handle_ok | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 10 | `from typing import Any, Callable, Dict, List, Optional, Union, Coroutine # Added...` |
| 31 | `# from openhcs.tui.components import InteractiveListItem # Not used in current b...` |
| 48 | `on_path_selected: Callable[[List[Path]], Coroutine[Any, Any, None]], # Expects a...` |
| 55 | `filter_extensions: Optional[List[str]] = None # e.g., [".h5", ".zarr"]` |
| 67 | `self.current_listing: List[Dict[str, Any]] = [] # List of {'name': str, 'path': ...` |
| 69 | `self.selected_item_indices: List[int] = [] # For multi-selection` |
| 348 | `selected_paths: List[Path] = []` |

## openhcs/tui/tui_launcher.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 56 | OpenHCSTUILauncher.__init__ | `Dict` |
| 147 | OpenHCSTUILauncher._on_plate_added | `Dict` |
| 191 | OpenHCSTUILauncher._on_plate_removed | `Dict` |
| 213 | OpenHCSTUILauncher._on_plate_selected | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 11 | `from typing import Dict, Optional, Any` |
| 56 | `self.orchestrators: Dict[str, PipelineOrchestrator] = {}` |
| 147 | `async def _on_plate_added(self, plate_info: Dict[str, Any]):` |
| 191 | `async def _on_plate_removed(self, plate_info: Dict[str, Any]):` |
| 213 | `async def _on_plate_selected(self, plate_info: Dict[str, Any]):` |

## openhcs/tui/dual_step_func_editor.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 56 | DualStepFuncEditorPane.__init__ | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 15 | `from typing import Any, Callable, Dict, List, Optional, Union as TypingUnion, ge...` |
| 22 | `from prompt_toolkit.widgets import Box, Button, Frame, Label, TextArea, RadioLis...` |
| 56 | `self.step_param_inputs: Dict[str, Any] = {}` |
| 159 | `widget = CheckboxList(values=[(param_name, "")])` |
| 162 | `# For CheckboxList, changes are typically handled on save or via a dedicated cal...` |
| 167 | `# Determine initial selection for RadioList` |
| 176 | `widget = RadioList(values=options, default=initial_selection)` |
| 177 | `# The handler for RadioList is set directly on the widget instance` |
| 191 | `widget = RadioList(values=options, default=initial_selection)` |
| 302 | `# Handle variable_components specifically if it was changed by RadioList` |
| 349 | `if param_name == "variable_components" and isinstance(widget, RadioList):` |
| 355 | `elif param_name == "group_by" and isinstance(widget, RadioList):` |
| 361 | `elif isinstance(widget, CheckboxList) and actual_type is bool: # Existing bool h...` |
| 503 | `elif isinstance(associated_widget, CheckboxList):` |
| 504 | `# Assuming param_name_to_reset is the value for boolean CheckboxList` |
| 506 | `elif isinstance(associated_widget, RadioList):` |
| 524 | `logger.warning(f"Cannot reset RadioList for unknown enum param: {param_name_to_r...` |
| 530 | `self._something_changed(param_name=param_name_to_reset, widget_value=original_va...` |

## openhcs/tui/menu_bar.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 136 | MenuItemSchema.validate_menu_item | `Dict` |
| 177 | MenuStructureSchema.validate_menu_structure | `Dict` |
| 177 | MenuStructureSchema.validate_menu_structure | `List` |
| 177 | MenuStructureSchema.validate_menu_structure | `Dict` |
| 226 | MenuItem.__init__ | `List` |
| 266 | MenuItem.from_dict | `Dict` |
| 266 | MenuItem.from_dict | `Dict` |
| 399 | MenuBar.__init__ | `List` |
| 431 | MenuBar._create_handler_map | `Dict` |
| 471 | MenuBar._create_condition_map | `Dict` |
| 512 | MenuBar._load_menu_structure | `Dict` |
| 512 | MenuBar._load_menu_structure | `List` |
| 577 | MenuBar._create_menu_labels | `List` |
| 834 | MenuBar._create_submenu_container | `List` |
| 930 | MenuBar._on_save_pipeline | `Dict` |
| 939 | MenuBar._on_save_pipeline | `List` |
| 1043 | MenuBar._on_remove_step | `Dict` |
| 1065 | MenuBar._on_remove_step | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 6 | `from typing import (Any, Callable, ClassVar, Dict, FrozenSet, List, Optional,` |
| 136 | `def validate_menu_item(item: Dict[str, Any]) -> None:` |
| 177 | `def validate_menu_structure(structure: Dict[str, List[Dict[str, Any]]]) -> None:` |
| 226 | `children: Optional[List['MenuItem']] = None` |
| 238 | `children: List of child menu items (for submenu items)` |
| 266 | `def from_dict(cls, item_dict: Dict[str, Any], handler_map: Dict[str, Callable]) ...` |
| 271 | `item_dict: Dictionary with menu item data` |
| 399 | `self.active_submenu: Optional[List[MenuItem]] = None` |
| 431 | `def _create_handler_map(self) -> Dict[str, Union[Callable, Command]]: # Return t...` |
| 436 | `Dictionary mapping handler names to callables or Commands` |
| 471 | `def _create_condition_map(self) -> Dict[str, Condition]:` |
| 476 | `Dictionary mapping condition names to Condition objects` |
| 512 | `def _load_menu_structure(self) -> Dict[str, List[MenuItem]]:` |
| 517 | `Dictionary mapping menu names to lists of MenuItem objects` |
| 577 | `def _create_menu_labels(self) -> List[Label]:` |
| 582 | `List of Label widgets for menu categories` |
| 834 | `def _create_submenu_container(self, menu_items: List[MenuItem]) -> Container:` |
| 839 | `menu_items: List of menu items to display` |
| 930 | `selected_plate: Optional[Dict[str, Any]] = getattr(self.state, 'selected_plate',...` |
| 939 | `pipeline_definition: Optional[List[AbstractStep]] = getattr(active_orchestrator,...` |
| 1043 | `selected_step_dict: Optional[Dict[str, Any]] = getattr(self.state, 'selected_ste...` |
| 1065 | `current_pipeline: List[AbstractStep] = active_orchestrator.pipeline_definition` |
| 1160 | `data: Dictionary with operation and status` |

## openhcs/tui/function_pattern_editor.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 90 | FunctionPatternEditor.__init__ | `List` |
| 90 | FunctionPatternEditor.__init__ | `Dict` |
| 149 | FunctionPatternEditor._extract_pattern | `List` |
| 149 | FunctionPatternEditor._extract_pattern | `Dict` |
| 155 | FunctionPatternEditor._clone_pattern | `List` |
| 155 | FunctionPatternEditor._clone_pattern | `Dict` |
| 173 | FunctionPatternEditor._get_initial_func_for_param_editor | `Dict` |
| 298 | FunctionPatternEditor.get_pattern | `List` |
| 298 | FunctionPatternEditor.get_pattern | `Dict` |
| 450 | FunctionPatternEditor._get_current_functions | `List` |
| 473 | FunctionPatternEditor._extract_func_and_kwargs | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 9 | `When converting function patterns from List to Dict, use None as the key` |
| 22 | `from typing import Any, Callable, Dict, List, Optional, Tuple, Union` |
| 27 | `from prompt_toolkit.widgets import (Box, Button, Dialog, Label, TextArea, RadioL...` |
| 54 | `Dictionary of function metadata` |
| 90 | `def __init__(self, state: Any, initial_pattern: Union[List, Dict, None] = None, ...` |
| 149 | `def _extract_pattern(self, step) -> Union[List, Dict]:` |
| 155 | `def _clone_pattern(self, pattern) -> Union[List, Dict]:` |
| 173 | `def _get_initial_func_for_param_editor(self) -> Tuple[Optional[Callable], Dict[s...` |
| 298 | `def get_pattern(self) -> Union[List, Dict]:` |
| 366 | `"Convert to Dict Pattern",` |
| 450 | `def _get_current_functions(self) -> List:` |
| 473 | `def _extract_func_and_kwargs(self, func_item) -> Tuple[Optional[Callable], Dict]...` |

## openhcs/tui/plate_manager_core.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 56 | PlateEventHandler.on_plate_added | `Dict` |
| 57 | PlateEventHandler.on_plate_removed | `Dict` |
| 58 | PlateEventHandler.on_plate_selected | `Dict` |
| 72 | PlateManagerPane.__init__ | `List` |
| 72 | PlateManagerPane.__init__ | `Dict` |
| 154 | PlateManagerPane._get_selected_plate_data_for_action | `List` |
| 154 | PlateManagerPane._get_selected_plate_data_for_action | `Dict` |
| 169 | PlateManagerPane._get_selected_orchestrators_for_action | `List` |
| 163 | PlateManagerPane._get_selected_orchestrators_for_action | `List` |
| 222 | PlateManagerPane._handle_add_predefined_plate | `Dict` |
| 303 | PlateManagerPane._get_plate_display_text | `Dict` |
| 439 | PlateManagerPane._handle_add_dialog_result | `Dict` |
| 495 | PlateManagerPane._handle_delete_plates_request | `Dict` |
| 535 | PlateManagerPane._handle_remove_dialog_result | `Dict` |
| 551 | PlateManagerPane._handle_validation_result | `Dict` |
| 619 | PlateManagerPane._select_plate | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 33 | `from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union` |
| 42 | `from .components import InteractiveListItem # Import the new component` |
| 56 | `async def on_plate_added(self, plate: Dict[str, Any]) -> None: ...` |
| 57 | `async def on_plate_removed(self, plate: Dict[str, Any]) -> None: ...` |
| 58 | `async def on_plate_selected(self, plate: Dict[str, Any]) -> None: ...` |
| 72 | `self.plates: List[Dict[str, Any]] = []` |
| 154 | `def _get_selected_plate_data_for_action(self) -> Optional[List[Dict[str, Any]]]:` |
| 163 | `def _get_selected_orchestrators_for_action(self) -> List["PipelineOrchestrator"]...` |
| 166 | `# e.g. if multi-selection is supported by InteractiveListItem or a similar mecha...` |
| 169 | `orchestrators: List["PipelineOrchestrator"] = []` |
| 222 | `async def _handle_add_predefined_plate(self, data: Optional[Dict[str, Any]] = No...` |
| 279 | `"""Builds the HSplit container holding individual InteractiveListItem widgets fo...` |
| 291 | `item_widget = InteractiveListItem(` |
| 303 | `def _get_plate_display_text(self, plate_data: Dict[str, Any], is_selected: bool)...` |
| 340 | `# The ^/v symbols for reordering are best handled by InteractiveListItem itself` |
| 347 | `# --- New callback handlers for InteractiveListItem ---` |
| 439 | `async def _handle_add_dialog_result(self, result: Dict[str, Any]):` |
| 495 | `async def _handle_delete_plates_request(self, data: Dict[str, Any]):` |
| 535 | `async def _handle_remove_dialog_result(self, plate_to_remove: Dict[str, Any]):` |
| 551 | `async def _handle_validation_result(self, validated_plate: Dict[str, Any]):` |
| 619 | `plate_to_select: Optional[Dict[str, Any]] = None` |

## openhcs/tui/tui_architecture.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 83 | TUIState.__init__ | `Dict` |
| 84 | TUIState.__init__ | `Dict` |
| 91 | TUIState.__init__ | `Dict` |
| 96 | TUIState.__init__ | `Dict` |
| 114 | TUIState.__init__ | `Dict` |
| 114 | TUIState.__init__ | `List` |
| 143 | TUIState.set_selected_plate | `Dict` |
| 157 | TUIState.set_selected_step | `Dict` |
| 486 | OpenHCSTUI._handle_show_edit_plate_config_request | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 25 | `from typing import Any, Callable, Container, Dict, List, Optional, Union, TYPE_C...` |
| 83 | `self.selected_plate: Optional[Dict[str, Any]] = None` |
| 84 | `self.selected_step: Optional[Dict[str, Any]] = None` |
| 91 | `self.compiled_contexts: Optional[Dict[str, ProcessingContext]] = None` |
| 96 | `self.step_to_edit_config: Optional[Dict[str, Any]] = None # Renamed from selecte...` |
| 114 | `self.observers: Dict[str, List[Callable]] = {}` |
| 143 | `async def set_selected_plate(self, plate: Dict[str, Any]) -> None:` |
| 157 | `async def set_selected_step(self, step: Dict[str, Any]) -> None:` |
| 486 | `async def _handle_show_edit_plate_config_request(self, data: Dict[str, Any]):` |

## openhcs/tui/commands.py
**Imports target:** True

### String References
| Line | Reference |
| ---- | --------- |
| 8 | `from typing import Protocol, Any, TYPE_CHECKING, List # Added List` |

## openhcs/tui/step_viewer.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 42 | PipelineEditorPane.__init__ | `List` |
| 42 | PipelineEditorPane.__init__ | `Dict` |
| 43 | PipelineEditorPane.__init__ | `List` |
| 43 | PipelineEditorPane.__init__ | `Dict` |
| 139 | PipelineEditorPane._get_selected_steps_for_action | `List` |
| 139 | PipelineEditorPane._get_selected_steps_for_action | `Dict` |
| 205 | PipelineEditorPane._get_step_display_text | `Dict` |
| 264 | PipelineEditorPane._get_function_name | `Dict` |
| 282 | PipelineEditorPane._handle_edit_step_request | `Dict` |
| 466 | PipelineEditorPane._handle_step_pattern_saved | `Dict` |
| 607 | PipelineEditorPane._move_step_up | `List` |
| 661 | PipelineEditorPane._move_step_down | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 3 | `from typing import Any, Dict, List, Optional` |
| 14 | `from .components import InteractiveListItem` |
| 42 | `self.steps: List[Dict[str, Any]] = []` |
| 43 | `self.pipelines: List[Dict[str, Any]] = [] # This might be simplified if only one...` |
| 48 | `self.step_items_container_widget: Optional[HSplit] = None # Will hold HSplit of ...` |
| 139 | `def _get_selected_steps_for_action(self) -> List[Dict[str, Any]]:` |
| 191 | `item_widget = InteractiveListItem(` |
| 205 | `def _get_step_display_text(self, step_data: Dict[str, Any], is_selected: bool) -...` |
| 208 | `Reordering symbols (^/v) should be handled by InteractiveListItem.` |
| 215 | `# InteractiveListItem will handle its own selection highlighting.` |
| 216 | `# The ^/v symbols for reordering are also best handled by InteractiveListItem.` |
| 222 | `# --- New callback handlers for InteractiveListItem ---` |
| 264 | `def _get_function_name(self, step: Dict[str, Any]) -> str:` |
| 282 | `async def _handle_edit_step_request(self, data: Optional[Dict[str, Any]]) -> Non...` |
| 372 | `# If on_select is implemented in InteractiveListItem to call _edit_step,` |
| 394 | `# Rebuild the HSplit with new InteractiveListItem instances` |
| 466 | `async def _handle_step_pattern_saved(self, data: Dict[str, Any]):` |
| 607 | `pipeline: List[FunctionStep] = self.state.active_orchestrator.pipeline_definitio...` |
| 661 | `pipeline: List[FunctionStep] = self.state.active_orchestrator.pipeline_definitio...` |

## openhcs/tui/status_bar.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 107 | StatusBarSchema.validate_log_entry | `Dict` |
| 131 | StatusBarState.None | `Dict` |
| 157 | StatusBarState.with_log_entry | `Dict` |
| 167 | LogFormatter.None | `Dict` |
| 176 | LogFormatter.format_log_entry | `Dict` |
| 201 | LogFormatter.format_log_entries | `Dict` |
| 203 | LogFormatter.format_log_entries | `List` |
| 344 | StatusBar._on_operation_status_changed | `Dict` |
| 370 | StatusBar._on_error_event | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 31 | `from typing import List, Optional, Dict, Any, Deque, ClassVar, Tuple, Union` |
| 107 | `def validate_log_entry(entry: Dict[str, Any]) -> None:` |
| 131 | `log_buffer: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=S...` |
| 157 | `def with_log_entry(self, entry: Dict[str, Any]) -> 'StatusBarState':` |
| 167 | `LEVEL_STYLES: ClassVar[Dict[LogLevel, Tuple[str, str]]] = {` |
| 176 | `def format_log_entry(cls, entry: Dict[str, Any]) -> FormattedText:` |
| 201 | `def format_log_entries(cls, entries: Deque[Dict[str, Any]]) -> FormattedText:` |
| 203 | `result_fragments: List[Tuple[str, str]] = []` |
| 248 | `# Listen for requests to toggle the log drawer (e.g., from MenuBar)` |
| 344 | `async def _on_operation_status_changed(self, data: Dict[str, Any]):` |
| 370 | `async def _on_error_event(self, error_data: Dict[str, Any]):` |

## openhcs/tui/dialogs/global_settings_editor.py
**Imports target:** False

### String References
| Line | Reference |
| ---- | --------- |
| 13 | `from prompt_toolkit.widgets import Button, Dialog, Label, TextArea, RadioList, C...` |
| 55 | `# Example: VFS default_storage_backend (using RadioList as Dropdown)` |
| 57 | `self.vfs_backend_selector = RadioList(` |
| 65 | `self.microscope_selector = RadioList(` |

## openhcs/tui/dialogs/plate_config_editor.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 54 | PlateConfigEditorPane.__init__ | `Dict` |
| 125 | PlateConfigEditorPane._create_config_widgets | `List` |
| 123 | PlateConfigEditorPane._create_config_widgets | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 9 | `from typing import Any, Dict, Optional, TYPE_CHECKING` |
| 15 | `from prompt_toolkit.widgets import Box, Button, Frame, Label, TextArea, RadioLis...` |
| 54 | `self.config_param_inputs: Dict[str, Any] = {} # To store UI input widgets` |
| 97 | `elif current_value_type == bool: # For Checkbox or RadioList representing bool` |
| 102 | `elif issubclass(current_value_type, Enum): # For RadioList returning string memb...` |
| 123 | `def _create_config_widgets(self, config_obj: Any, parent_path: str = "") -> List...` |
| 125 | `widgets: List[Any] = []` |
| 170 | `input_widget = RadioList(values=enum_values, current_value=current_value.name)` |
| 171 | `# RadioList changes are typically polled. For live update:` |
| 173 | `#     # This is tricky as RadioList doesn't have a direct on_change.` |
| 174 | `#     # This would require a custom RadioList or polling.` |
| 235 | `# This loop explicitly processes RadioLists and Checkboxes.` |
| 247 | `if widget_identifier == "radiolist" and isinstance(widget, RadioList):` |

## openhcs/tui/dialogs/plate_dialog_manager.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 208 | PlateDialogManager.browser_on_path_selected | `List` |
| 509 | PlateDialogManager.show_remove_plate_dialog | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 24 | `from typing import Any, Dict, List, Optional, Protocol, Coroutine # Added List` |
| 32 | `from prompt_toolkit.widgets import RadioList as Dropdown` |
| 208 | `async def browser_on_path_selected(selected_paths: List[Path]): # Expects a List...` |
| 509 | `async def show_remove_plate_dialog(self, plate: Dict[str, Any]):` |

## openhcs/tui/services/external_editor_service.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 21 | ExternalEditorService.edit_pattern_in_external_editor | `List` |
| 21 | ExternalEditorService.edit_pattern_in_external_editor | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 5 | `from typing import Any, Dict, List, Optional, Tuple, Union` |
| 21 | `async def edit_pattern_in_external_editor(self, initial_content: str) -> Tuple[b...` |
| 30 | `A tuple: (success: bool, pattern: Optional[Union[List, Dict]], error_message: Op...` |

## openhcs/tui/services/plate_validation.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 15 | ValidationResultCallback.__call__ | `Dict` |
| 74 | PlateValidationService.validate_plate | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 6 | `from typing import Any, Callable, Dict, Optional, Protocol` |
| 15 | `async def __call__(self, result: Dict[str, Any]) -> None: ...` |
| 74 | `async def validate_plate(self, path: str, backend: str, /) -> Dict[str, Any]:` |
