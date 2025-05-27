# Detailed Code Definition Matrix
Generated on: 2025-05-22 22:44:08

### Detailed Matrix for `openhcs/tui/components.py`

| Definition Type | Name | Parent Class | Parameters | Return Type | Lines |
| --- | --- | --- | --- | --- | --- |
| class | InteractiveListItem |  |  |  | 10-145 |
| method | __init__ | InteractiveListItem | self: Any, item_data: Any, item_index: int, is_selected: bool, display_text_func: Optional[Callable[<complex_annotation>, str]], on_select: Optional[Callable[<complex_annotation>, <complex_annotation>]], on_move_up: Optional[Callable[<complex_annotation>, <complex_annotation>]], on_move_down: Optional[Callable[<complex_annotation>, <complex_annotation>]], can_move_up: bool, can_move_down: bool |  | 15-94 |
| method | _get_display_text | InteractiveListItem | self: Any | str | 97-99 |
| method | _handle_move_up_click | InteractiveListItem | self: Any |  | 101-103 |
| method | _handle_move_down_click | InteractiveListItem | self: Any |  | 105-107 |
| method | _handle_mouse | InteractiveListItem | self: Any, mouse_event: Any |  | 109-115 |
| method | _get_key_bindings | InteractiveListItem | self: Any |  | 117-120 |
| method | update_selection_style | InteractiveListItem | self: Any |  | 122-129 |
| method | update_data | InteractiveListItem | self: Any, item_data: Any, is_selected: bool, can_move_up: bool, can_move_down: bool |  | 137-145 |

### Detailed Matrix for `openhcs/tui/dual_step_func_editor.py`

| Definition Type | Name | Parent Class | Parameters | Return Type | Lines |
| --- | --- | --- | --- | --- | --- |
| class | DualStepFuncEditorPane |  |  |  | 27-361 |
| method | __init__ | DualStepFuncEditorPane | self: Any, state: Any, func_step: FunctionStep |  | 32-60 |
| method | _func_pattern_changed | DualStepFuncEditorPane | self: Any |  | 62-68 |
| method | _initialize_ui | DualStepFuncEditorPane | self: Any |  | 70-125 |
| method | get_current_view_container | DualStepFuncEditorPane |  |  | 113-117 |
| method | _get_current_view_title | DualStepFuncEditorPane | self: Any | str | 128-131 |
| method | _create_step_settings_view | DualStepFuncEditorPane | self: Any | ScrollablePane | 133-199 |
| method | _create_func_pattern_view | DualStepFuncEditorPane | self: Any | Any | 201-210 |
| method | _switch_view | DualStepFuncEditorPane | self: Any, view_name: str |  | 212-217 |
| method | _update_button_styles | DualStepFuncEditorPane | self: Any |  | 219-224 |
| method | _something_changed | DualStepFuncEditorPane | self: Any, param_name: Optional[str], widget_value: Any |  | 226-278 |
| method | _save_changes | DualStepFuncEditorPane | self: Any |  | 280-322 |
| method | _close_editor | DualStepFuncEditorPane | self: Any |  | 324-328 |
| method | container | DualStepFuncEditorPane | self: Any | Container | 333-337 |
| method | shutdown | DualStepFuncEditorPane | self: Any |  | 339-342 |
| method | _load_step_object | DualStepFuncEditorPane | self: Any |  | 344-351 |
| method | _save_step_object_as | DualStepFuncEditorPane | self: Any |  | 354-361 |

### Detailed Matrix for `openhcs/tui/function_pattern_editor.py`

| Definition Type | Name | Parent Class | Parameters | Return Type | Lines |
| --- | --- | --- | --- | --- | --- |
| function | get_function_info |  | func: Any |  | 37-73 |
| class | PatternValidationError |  |  |  | 76-78 |
| function | _validate_pattern_file |  | content: str | Tuple[bool, Optional[Any], Optional[str]] | 81-132 |
| class | GroupedDropdown |  |  |  | 135-160 |
| method | __init__ | GroupedDropdown | self: Any, options: Any, default: Any |  | 138-142 |
| method | _get_text_fragments | GroupedDropdown | self: Any |  | 144-160 |
| class | FunctionPatternEditor |  |  |  | 163-992 |
| method | __init__ | FunctionPatternEditor | self: Any, state: Any, initial_pattern: Union[List, Dict, <complex_annotation>], change_callback: Optional[Callable] |  | 166-200 |
| method | container | FunctionPatternEditor | self: Any | Container | 203-205 |
| method | _extract_pattern | FunctionPatternEditor | self: Any, step: Any | Union[List, Dict] | 207-211 |
| method | _clone_pattern | FunctionPatternEditor | self: Any, pattern: Any | Union[List, Dict] | 213-222 |
| method | _clone_list_item | FunctionPatternEditor | self: Any, item: Any |  | 224-229 |
| method | _create_header | FunctionPatternEditor | self: Any |  | 231-247 |
| method | _notify_change | FunctionPatternEditor | self: Any |  | 249-252 |
| method | get_pattern | FunctionPatternEditor | self: Any | Union[List, Dict] | 254-265 |
| method | _refresh_key_selector | FunctionPatternEditor | self: Any |  | 267-325 |
| method | on_key_change | FunctionPatternEditor | key: Any |  | 293-294 |
| method | _refresh_function_list | FunctionPatternEditor | self: Any |  | 328-332 |
| method | _build_function_list | FunctionPatternEditor | self: Any, functions: Any |  | 334-352 |
| method | _create_function_item | FunctionPatternEditor | self: Any, index: Any, func: Any, kwargs: Any |  | 354-395 |
| method | _get_current_functions | FunctionPatternEditor | self: Any | List | 397-418 |
| method | _extract_func_and_kwargs | FunctionPatternEditor | self: Any, func_item: Any | Tuple[Optional[Callable], Dict] | 420-427 |
| method | _create_function_dropdown | FunctionPatternEditor | self: Any, index: Any, current_func: Any |  | 429-483 |
| method | on_selection_change | FunctionPatternEditor | func: Any |  | 476-479 |
| method | _switch_key | FunctionPatternEditor | self: Any, key: Any |  | 485-508 |
| method | _convert_list_to_dict_pattern | FunctionPatternEditor | self: Any |  | 510-518 |
| method | _add_key | FunctionPatternEditor | self: Any |  | 520-540 |
| method | _remove_key | FunctionPatternEditor | self: Any |  | 542-572 |
| method | _edit_in_vim | FunctionPatternEditor | self: Any |  | 574-646 |
| method | _show_error | FunctionPatternEditor | self: Any, message: str |  | 648-668 |
| method | _update_function | FunctionPatternEditor | self: Any, index: Any, func: Any |  | 670-695 |
| method | _update_pattern_functions | FunctionPatternEditor | self: Any, functions: Any |  | 697-715 |
| method | _create_parameter_editor | FunctionPatternEditor | self: Any, func: Any, kwargs: Any, func_index: Any |  | 717-747 |
| method | _create_parameter_field | FunctionPatternEditor | self: Any, name: Any, default: Any, current_value: Any, required: Any, is_special: Any, func_index: Any |  | 749-771 |
| method | _create_input_field | FunctionPatternEditor | self: Any, name: Any, value: Any, func_index: Any |  | 773-791 |
| method | on_text_changed | FunctionPatternEditor | buffer: Any |  | 786-787 |
| method | _get_function_parameters | FunctionPatternEditor | self: Any, func: Any | List[Dict] | 793-827 |
| method | _update_parameter | FunctionPatternEditor | self: Any, name: Any, value_str: Any, func_index: Any |  | 829-849 |
| method | _parse_parameter_value | FunctionPatternEditor | self: Any, value_str: Any |  | 851-871 |
| method | _reset_parameter | FunctionPatternEditor | self: Any, name: Any, default: Any, func_index: Any |  | 873-889 |
| method | _reset_all_parameters | FunctionPatternEditor | self: Any, func_index: Any |  | 892-905 |
| method | _move_function_up | FunctionPatternEditor | self: Any, index: Any |  | 908-915 |
| method | _move_function_down | FunctionPatternEditor | self: Any, index: Any |  | 918-925 |
| method | _delete_function | FunctionPatternEditor | self: Any, index: Any |  | 928-935 |
| method | _add_function | FunctionPatternEditor | self: Any |  | 938-956 |
| method | _validate_pattern | FunctionPatternEditor | self: Any | bool | 962-992 |

### Detailed Matrix for `openhcs/tui/__init__.py`

No definitions found or error in parsing.
### Detailed Matrix for `openhcs/tui/__main__.py`

| Definition Type | Name | Parent Class | Parameters | Return Type | Lines |
| --- | --- | --- | --- | --- | --- |
| function | main |  |  |  | 23-117 |

### Detailed Matrix for `openhcs/tui/menu_bar.py`

| Definition Type | Name | Parent Class | Parameters | Return Type | Lines |
| --- | --- | --- | --- | --- | --- |
| class | MissingStateError |  |  |  | 38-47 |
| method | __init__ | MissingStateError | self: Any, attribute_name: str |  | 45-47 |
| class | ReentrantLock |  |  |  | 50-86 |
| method | __init__ | ReentrantLock | self: Any |  | 57-60 |
| method | __aenter__ | ReentrantLock | self: Any |  | 62-76 |
| method | __aexit__ | ReentrantLock | self: Any, exc_type: Any, exc_val: Any, exc_tb: Any |  | 78-86 |
| class | LayoutContract |  |  |  | 89-111 |
| method | validate_layout_container | LayoutContract | container: Any | <complex_annotation> | 97-111 |
| class | MenuItemType |  |  |  | 114-119 |
| class | MenuItemSchema |  |  |  | 123-162 |
| method | validate_menu_item | MenuItemSchema | item: Dict[str, Any] | <complex_annotation> | 133-162 |
| class | MenuStructureSchema |  |  |  | 166-203 |
| method | validate_menu_structure | MenuStructureSchema | structure: Dict[str, List[Dict[str, Any]]] | <complex_annotation> | 174-203 |
| class | MenuItem |  |  |  | 206-313 |
| method | __init__ | MenuItem | self: Any, type: MenuItemType, label: str, handler: Optional[Callable], shortcut: Optional[str], enabled: Union[bool, Condition], checked: Union[bool, Condition], children: Optional[List[<complex_annotation>]] |  | 215-243 |
| method | is_enabled | MenuItem | self: Any | bool | 245-249 |
| method | is_checked | MenuItem | self: Any | bool | 251-255 |
| method | set_checked | MenuItem | self: Any, checked: bool | <complex_annotation> | 257-260 |
| method | from_dict | MenuItem | cls: Any, item_dict: Dict[str, Any], handler_map: Dict[str, Callable] | <complex_annotation> | 263-313 |
| class | MenuBar |  |  |  | 370-1252 |
| method | __init__ | MenuBar | self: Any, state: Any |  | 383-424 |
| method | _create_handler_map | MenuBar | self: Any | Dict[str, Callable] | 426-466 |
| method | _create_condition_map | MenuBar | self: Any | Dict[str, Condition] | 468-490 |
| method | _get_required_state | MenuBar | self: Any, attribute_name: str | Any | 492-507 |
| method | _load_menu_structure | MenuBar | self: Any | Dict[str, List[MenuItem]] | 509-568 |
| method | _create_menu_labels | MenuBar | self: Any | List[Label] | 574-609 |
| method | create_mouse_handler | MenuBar | menu: Any |  | 597-603 |
| method | menu_mouse_handler | MenuBar | mouse_event: Any |  | 598-602 |
| method | _create_submenu_float | MenuBar | self: Any | Float | 611-628 |
| method | _create_key_bindings | MenuBar | self: Any | KeyBindings | 630-692 |
| method | create_handler | MenuBar | menu: Any |  | 646-649 |
| method | handler | MenuBar | event: KeyPressEvent |  | 647-648 |
| method | _ | MenuBar | event: KeyPressEvent |  | 660-661 |
| method | _ | MenuBar | event: KeyPressEvent |  | 671-672 |
| method | _ | MenuBar | event: KeyPressEvent |  | 675-676 |
| method | _ | MenuBar | event: KeyPressEvent |  | 680-681 |
| method | _ | MenuBar | event: KeyPressEvent |  | 684-685 |
| method | _ | MenuBar | event: KeyPressEvent |  | 689-690 |
| method | _activate_menu | MenuBar | self: Any, menu_name: str | <complex_annotation> | 694-730 |
| method | _close_menu | MenuBar | self: Any | <complex_annotation> | 732-750 |
| method | _navigate_menu | MenuBar | self: Any, delta: int | <complex_annotation> | 752-773 |
| method | _navigate_submenu | MenuBar | self: Any, delta: int | <complex_annotation> | 775-815 |
| method | _select_current_item | MenuBar | self: Any | <complex_annotation> | 817-829 |
| method | _create_submenu_container | MenuBar | self: Any, menu_items: List[MenuItem] | Container | 831-889 |
| method | create_mouse_handler | MenuBar | menu_item: Any |  | 875-882 |
| method | item_mouse_handler | MenuBar | mouse_event: Any |  | 876-881 |
| method | _handle_menu_item | MenuBar | self: Any, item: MenuItem | <complex_annotation> | 891-907 |
| method | _on_new_pipeline | MenuBar | self: Any | <complex_annotation> | 911-915 |
| method | _on_open_pipeline | MenuBar | self: Any | <complex_annotation> | 917-921 |
| method | _on_save_pipeline | MenuBar | self: Any | <complex_annotation> | 923-981 |
| method | _on_save_pipeline_as | MenuBar | self: Any | <complex_annotation> | 983-987 |
| method | _on_exit | MenuBar | self: Any | <complex_annotation> | 989-991 |
| method | _on_add_step | MenuBar | self: Any | <complex_annotation> | 993-997 |
| method | _on_edit_step | MenuBar | self: Any | <complex_annotation> | 999-1035 |
| method | _on_remove_step | MenuBar | self: Any | <complex_annotation> | 1037-1082 |
| method | _on_move_step_up | MenuBar | self: Any | <complex_annotation> | 1084-1088 |
| method | _on_move_step_down | MenuBar | self: Any | <complex_annotation> | 1090-1094 |
| method | _on_toggle_log_drawer | MenuBar | self: Any | <complex_annotation> | 1096-1098 |
| method | _on_toggle_vim_mode | MenuBar | self: Any | <complex_annotation> | 1100-1102 |
| method | _on_set_theme | MenuBar | self: Any, theme: str | <complex_annotation> | 1104-1111 |
| method | _on_pre_compile | MenuBar | self: Any | <complex_annotation> | 1113-1115 |
| method | _on_compile | MenuBar | self: Any | <complex_annotation> | 1117-1119 |
| method | _on_run | MenuBar | self: Any | <complex_annotation> | 1121-1123 |
| method | _on_test | MenuBar | self: Any | <complex_annotation> | 1125-1145 |
| method | _on_settings | MenuBar | self: Any | <complex_annotation> | 1147-1149 |
| method | _on_documentation | MenuBar | self: Any | <complex_annotation> | 1151-1155 |
| method | _on_keyboard_shortcuts | MenuBar | self: Any | <complex_annotation> | 1157-1161 |
| method | _on_about | MenuBar | self: Any | <complex_annotation> | 1163-1167 |
| method | _on_operation_status_changed | MenuBar | self: Any, data: Any | <complex_annotation> | 1171-1179 |
| method | _on_plate_selected | MenuBar | self: Any, data: Any | <complex_annotation> | 1181-1189 |
| method | _on_is_compiled_changed | MenuBar | self: Any, data: Any | <complex_annotation> | 1191-1199 |
| method | shutdown | MenuBar | self: Any |  | 1201-1212 |
| method | __pt_container__ | MenuBar | self: Any | Container | 1214-1216 |
| method | get_children | MenuBar | self: Any |  | 1219-1220 |
| method | preferred_width | MenuBar | self: Any, max_available_width: Any |  | 1222-1223 |
| method | preferred_height | MenuBar | self: Any, max_available_height: Any, width: Any |  | 1225-1226 |
| method | reset | MenuBar | self: Any |  | 1228-1229 |
| method | write_to_screen | MenuBar | self: Any, screen: Any, mouse_handlers: Any, write_position: Any, parent_style: Any, erase_bg: Any, z_index: Any |  | 1231-1234 |
| method | mouse_handler | MenuBar | self: Any, mouse_event: Any |  | 1236-1252 |

### Detailed Matrix for `openhcs/tui/plate_manager_core.py`

| Definition Type | Name | Parent Class | Parameters | Return Type | Lines |
| --- | --- | --- | --- | --- | --- |
| class | PlateEventHandler |  |  |  | 54-59 |
| method | on_plate_added | PlateEventHandler | self: Any, plate: Dict[str, Any] | <complex_annotation> | 56-56 |
| method | on_plate_removed | PlateEventHandler | self: Any, plate: Dict[str, Any] | <complex_annotation> | 57-57 |
| method | on_plate_selected | PlateEventHandler | self: Any, plate: Dict[str, Any] | <complex_annotation> | 58-58 |
| method | on_plate_status_changed | PlateEventHandler | self: Any, plate_id: str, status: str | <complex_annotation> | 59-59 |
| class | PlateManagerPane |  |  |  | 62-582 |
| method | __init__ | PlateManagerPane | self: Any, state: Any, context: ProcessingContext, storage_registry: Any |  | 66-105 |
| method | _initialize_ui | PlateManagerPane | self: Any |  | 108-132 |
| method | container | PlateManagerPane | self: Any | Container | 136-141 |
| method | get_buttons_container | PlateManagerPane | self: Any | Container | 143-158 |
| method | register_with_app | PlateManagerPane | self: Any |  | 160-166 |
| method | _handle_request_show_add_plate_dialog | PlateManagerPane | self: Any, data: Any |  | 169-171 |
| method | _handle_add_predefined_plate | PlateManagerPane | self: Any, data: Optional[Dict[str, Any]] |  | 173-186 |
| method | _on_filemanager_available | PlateManagerPane | self: Any, data: Any |  | 188-195 |
| method | _initialize_and_refresh | PlateManagerPane | self: Any |  | 198-208 |
| method | _update_selection | PlateManagerPane | self: Any |  | 211-227 |
| method | _build_plate_items_container | PlateManagerPane | self: Any | HSplit | 229-252 |
| method | _get_plate_display_text | PlateManagerPane | self: Any, plate_data: Dict[str, Any], is_selected: bool | str | 254-275 |
| method | _handle_plate_item_select | PlateManagerPane | self: Any, index: int |  | 283-287 |
| method | _handle_plate_item_move_up | PlateManagerPane | self: Any, index: int |  | 289-293 |
| method | _handle_plate_item_move_down | PlateManagerPane | self: Any, index: int |  | 295-299 |
| method | _update_selection_and_notify_order | PlateManagerPane | self: Any |  | 301-305 |
| method | _create_key_bindings | PlateManagerPane | self: Any | KeyBindings | 308-338 |
| method | _ | PlateManagerPane | event: Any |  | 314-315 |
| method | _ | PlateManagerPane | event: Any |  | 317-318 |
| method | _ | PlateManagerPane | event: Any |  | 322-323 |
| method | _ | PlateManagerPane | event: Any |  | 325-326 |
| method | _ | PlateManagerPane | event: Any |  | 329-329 |
| method | _ | PlateManagerPane | event: Any |  | 331-331 |
| method | _ | PlateManagerPane | event: Any |  | 334-337 |
| method | _move_plate_up | PlateManagerPane | self: Any |  | 340-348 |
| method | _move_plate_down | PlateManagerPane | self: Any |  | 350-358 |
| method | _show_add_plate_dialog | PlateManagerPane | self: Any |  | 360-361 |
| method | _show_remove_plate_dialog | PlateManagerPane | self: Any |  | 363-366 |
| method | _handle_add_dialog_result | PlateManagerPane | self: Any, result: Dict[str, Any] |  | 368-416 |
| method | _handle_remove_dialog_result | PlateManagerPane | self: Any, plate_to_remove: Dict[str, Any] |  | 418-424 |
| method | _handle_validation_result | PlateManagerPane | self: Any, validated_plate: Dict[str, Any] |  | 426-445 |
| method | _handle_error | PlateManagerPane | self: Any, message: str, details: str |  | 447-453 |
| method | _on_edit_plate_clicked | PlateManagerPane | self: Any |  | 455-467 |
| method | _on_init_plate_clicked | PlateManagerPane | self: Any |  | 469-471 |
| method | _on_compile_plate_clicked | PlateManagerPane | self: Any |  | 472-474 |
| method | _on_run_plate_clicked | PlateManagerPane | self: Any |  | 475-477 |
| method | _move_selection | PlateManagerPane | self: Any, delta: int | <complex_annotation> | 479-486 |
| method | _ensure_selection_visible | PlateManagerPane | self: Any | <complex_annotation> | 488-490 |
| method | _select_plate | PlateManagerPane | self: Any, index: int | <complex_annotation> | 492-513 |
| method | _update_plate_status | PlateManagerPane | self: Any, data: Any |  | 516-523 |
| method | _update_plate_status_locally_and_notify | PlateManagerPane | self: Any, plate_id: str, new_status: str, message: Optional[str], notify_state: bool |  | 525-544 |
| method | _refresh_plates | PlateManagerPane | self: Any, _: Any |  | 546-569 |
| method | shutdown | PlateManagerPane | self: Any |  | 571-582 |

### Detailed Matrix for `openhcs/tui/status_bar.py`

| Definition Type | Name | Parent Class | Parameters | Return Type | Lines |
| --- | --- | --- | --- | --- | --- |
| class | Priority |  |  |  | 49-55 |
| class | LogLevel |  |  |  | 58-82 |
| method | from_string | LogLevel | cls: Any, level_str: str | <complex_annotation> | 67-72 |
| method | from_logging_level | LogLevel | cls: Any, level_no: int | <complex_annotation> | 75-82 |
| class | StatusBarSchema |  |  |  | 96-118 |
| method | validate_priority | StatusBarSchema | priority: Priority | <complex_annotation> | 101-104 |
| method | validate_log_entry | StatusBarSchema | entry: Dict[str, Any] | <complex_annotation> | 107-118 |
| class | StatusBarState |  |  |  | 122-162 |
| method | __post_init__ | StatusBarState | self: Any |  | 133-137 |
| method | with_status_message | StatusBarState | self: Any, message: str, priority: Priority | <complex_annotation> | 139-145 |
| method | with_operation_status | StatusBarState | self: Any, operation_status: str | <complex_annotation> | 147-151 |
| method | with_drawer_expanded | StatusBarState | self: Any, expanded: bool | <complex_annotation> | 153-155 |
| method | with_log_entry | StatusBarState | self: Any, entry: Dict[str, Any] | <complex_annotation> | 157-162 |
| class | LogFormatter |  |  |  | 165-213 |
| method | format_log_entry | LogFormatter | cls: Any, entry: Dict[str, Any] | FormattedText | 176-198 |
| method | format_log_entries | LogFormatter | cls: Any, entries: Deque[Dict[str, Any]] | FormattedText | 201-213 |
| class | StatusBar |  |  |  | 216-449 |
| method | __init__ | StatusBar | self: Any, tui_state: Any, max_log_entries: int |  | 219-252 |
| method | _create_status_label | StatusBar | self: Any | Label | 254-271 |
| method | get_display_text | StatusBar |  | FormattedText | 256-263 |
| method | _create_log_drawer_content | StatusBar | self: Any | Label | 273-279 |
| method | _toggle_drawer | StatusBar | self: Any | <complex_annotation> | 281-287 |
| method | set_status_message | StatusBar | self: Any, message: str, priority: Priority, operation_status: Optional[str] | <complex_annotation> | 289-296 |
| method | add_log_entry | StatusBar | self: Any, message: str, level: Union[str, LogLevel], source: Optional[str] | <complex_annotation> | 298-321 |
| method | _setup_logging_handler | StatusBar | self: Any |  | 323-341 |
| method | _on_operation_status_changed | StatusBar | self: Any, data: Dict[str, Any] |  | 344-367 |
| method | _on_error_event | StatusBar | self: Any, error_data: Dict[str, Any] |  | 370-381 |
| method | _on_tui_log_level_changed | StatusBar | self: Any, new_level_str: str |  | 383-396 |
| method | __pt_container__ | StatusBar | self: Any |  | 399-400 |
| method | get_children | StatusBar | self: Any |  | 403-404 |
| method | preferred_width | StatusBar | self: Any, max_available_width: Any |  | 406-407 |
| method | preferred_height | StatusBar | self: Any, max_available_height: Any, width: Any |  | 409-410 |
| method | reset | StatusBar | self: Any |  | 412-413 |
| method | write_to_screen | StatusBar | self: Any, screen: Any, mouse_handlers: Any, write_position: Any, parent_style: Any, erase_bg: Any, z_index: Any |  | 415-418 |
| method | mouse_handler | StatusBar | self: Any, mouse_event: Any |  | 420-425 |
| method | shutdown | StatusBar | self: Any |  | 428-449 |
| class | TUIStatusBarLogHandler |  |  |  | 452-491 |
| method | __init__ | TUIStatusBarLogHandler | self: Any, status_bar_instance: StatusBar, level: Any |  | 454-456 |
| method | emit | TUIStatusBarLogHandler | self: Any, record: logging.LogRecord |  | 458-491 |

### Detailed Matrix for `openhcs/tui/step_viewer.py`

| Definition Type | Name | Parent Class | Parameters | Return Type | Lines |
| --- | --- | --- | --- | --- | --- |
| class | PipelineEditorPane |  |  |  | 18-644 |
| method | __init__ | PipelineEditorPane | self: Any, state: Any, context: ProcessingContext |  | 25-55 |
| method | create | PipelineEditorPane | cls: Any, state: Any, context: ProcessingContext |  | 58-78 |
| method | setup | PipelineEditorPane | self: Any |  | 80-116 |
| method | container | PipelineEditorPane | self: Any | Container | 119-121 |
| method | get_buttons_container | PipelineEditorPane | self: Any | Container | 123-131 |
| method | _build_step_items_container | PipelineEditorPane | self: Any | HSplit | 133-171 |
| method | _get_step_display_text | PipelineEditorPane | self: Any, step_data: Dict[str, Any], is_selected: bool | str | 173-184 |
| method | _handle_item_select | PipelineEditorPane | self: Any, index: int |  | 197-201 |
| method | _handle_item_move_up | PipelineEditorPane | self: Any, index: int |  | 206-210 |
| method | _handle_item_move_down | PipelineEditorPane | self: Any, index: int |  | 212-216 |
| method | _get_status_icon | PipelineEditorPane | self: Any, status: str | str | 219-236 |
| method | _get_function_name | PipelineEditorPane | self: Any, step: Dict[str, Any] | str | 238-269 |
| method | _create_key_bindings | PipelineEditorPane | self: Any | KeyBindings | 271-318 |
| method | _ | PipelineEditorPane | event: Any |  | 286-290 |
| method | _ | PipelineEditorPane | event: Any |  | 293-297 |
| method | _ | PipelineEditorPane | event: Any |  | 300-306 |
| method | _ | PipelineEditorPane | event: Any |  | 309-311 |
| method | _ | PipelineEditorPane | event: Any |  | 314-316 |
| method | _update_selection | PipelineEditorPane | self: Any |  | 320-335 |
| method | _select_step | PipelineEditorPane | self: Any, index: int |  | 337-347 |
| method | _load_pipeline | PipelineEditorPane | self: Any |  | 349-380 |
| method | _save_pipeline | PipelineEditorPane | self: Any |  | 382-406 |
| method | _handle_step_pattern_saved | PipelineEditorPane | self: Any, data: Dict[str, Any] |  | 408-424 |
| method | _on_plate_selected | PipelineEditorPane | self: Any, plate: Any |  | 427-438 |
| method | _load_steps_for_plate | PipelineEditorPane | self: Any, plate_id: str |  | 440-493 |
| method | _edit_step | PipelineEditorPane | self: Any |  | 495-500 |
| method | _add_step | PipelineEditorPane | self: Any |  | 502-547 |
| method | _remove_step | PipelineEditorPane | self: Any |  | 549-559 |
| method | _refresh_steps | PipelineEditorPane | self: Any, _: Any |  | 561-563 |
| method | _move_step_up | PipelineEditorPane | self: Any |  | 565-598 |
| method | _move_step_down | PipelineEditorPane | self: Any |  | 600-633 |
| method | shutdown | PipelineEditorPane | self: Any |  | 635-644 |

### Detailed Matrix for `openhcs/tui/tui_architecture.py`

| Definition Type | Name | Parent Class | Parameters | Return Type | Lines |
| --- | --- | --- | --- | --- | --- |
| class | Clause5Violation |  |  |  | 48-54 |
| class | TUIState |  |  |  | 95-186 |
| method | __init__ | TUIState | self: Any |  | 102-132 |
| method | add_observer | TUIState | self: Any, event_type: str, callback: Callable | <complex_annotation> | 134-144 |
| method | notify | TUIState | self: Any, event_type: str, data: Any | <complex_annotation> | 146-159 |
| method | set_selected_plate | TUIState | self: Any, plate: Dict[str, Any] | <complex_annotation> | 161-173 |
| method | set_selected_step | TUIState | self: Any, step: Dict[str, Any] | <complex_annotation> | 175-186 |
| class | OpenHCSTUI |  |  |  | 189-557 |
| method | __init__ | OpenHCSTUI | self: Any, initial_context: ProcessingContext, state: TUIState, global_config: GlobalPipelineConfig |  | 196-244 |
| method | _create_key_bindings | OpenHCSTUI | self: Any | KeyBindings | 246-272 |
| method | _ | OpenHCSTUI | event: Any |  | 257-259 |
| method | _ | OpenHCSTUI | event: Any |  | 263-265 |
| method | _ | OpenHCSTUI | event: Any |  | 268-270 |
| method | _validate_components_present | OpenHCSTUI | self: Any |  | 274-323 |
| method | _create_root_container | OpenHCSTUI | self: Any | Container | 325-359 |
| method | _get_left_pane | OpenHCSTUI | self: Any | Container | 361-416 |
| method | _get_step_viewer | OpenHCSTUI | self: Any | Container | 418-433 |
| method | _get_status_bar | OpenHCSTUI | self: Any | Container | 435-447 |
| method | _get_menu_bar | OpenHCSTUI | self: Any | Container | 449-461 |
| method | _on_launcher_config_rebound | OpenHCSTUI | self: Any, new_core_config: GlobalPipelineConfig | <complex_annotation> | 463-482 |
| method | shutdown_components | OpenHCSTUI | self: Any |  | 484-517 |
| method | _async_initialize_step_viewer | OpenHCSTUI | self: Any |  | 519-534 |
| method | get_children | OpenHCSTUI | self: Any |  | 538-539 |
| method | preferred_width | OpenHCSTUI | self: Any, max_available_width: Any |  | 541-542 |
| method | preferred_height | OpenHCSTUI | self: Any, max_available_height: Any, width: Any |  | 544-545 |
| method | reset | OpenHCSTUI | self: Any |  | 547-548 |
| method | write_to_screen | OpenHCSTUI | self: Any, screen: Any, mouse_handlers: Any, write_position: Any, parent_style: Any, erase_bg: Any, z_index: Any |  | 550-553 |
| method | mouse_handler | OpenHCSTUI | self: Any, mouse_event: Any |  | 555-557 |

### Detailed Matrix for `openhcs/tui/tui_launcher.py`

| Definition Type | Name | Parent Class | Parameters | Return Type | Lines |
| --- | --- | --- | --- | --- | --- |
| class | OpenHCSTUILauncher |  |  |  | 27-310 |
| method | __init__ | OpenHCSTUILauncher | self: Any, core_global_config: GlobalPipelineConfig, common_output_directory: Optional[str], tui_config_path: Optional[str] |  | 32-96 |
| method | _handle_global_config_update | OpenHCSTUILauncher | self: Any, new_config: GlobalPipelineConfig |  | 98-145 |
| method | _on_plate_added | OpenHCSTUILauncher | self: Any, plate_info: Dict[str, Any] |  | 147-189 |
| method | _on_plate_removed | OpenHCSTUILauncher | self: Any, plate_info: Dict[str, Any] |  | 191-211 |
| method | _on_plate_selected | OpenHCSTUILauncher | self: Any, plate_info: Dict[str, Any] |  | 213-226 |
| method | run | OpenHCSTUILauncher | self: Any |  | 231-259 |
| method | _cleanup | OpenHCSTUILauncher | self: Any |  | 261-310 |

