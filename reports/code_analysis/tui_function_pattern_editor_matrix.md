# Detailed Code Definition Matrix
Generated on: 2025-05-22 18:08:36

### Detailed Matrix for `openhcs/tui/function_pattern_editor.py`

| Definition Type | Name | Parent Class | Parameters | Return Type | Lines |
| --- | --- | --- | --- | --- | --- |
| function | get_function_info |  | func: Any |  | 37-73 |
| class | PatternValidationError |  |  |  | 76-78 |
| function | _validate_pattern_file |  | content: str | Tuple[bool, Optional[Any], Optional[str]] | 81-132 |
| class | GroupedDropdown |  |  |  | 135-160 |
| method | __init__ | GroupedDropdown | self: Any, options: Any, default: Any |  | 138-142 |
| method | _get_text_fragments | GroupedDropdown | self: Any |  | 144-160 |
| class | FunctionPatternEditor |  |  |  | 163-972 |
| method | __init__ | FunctionPatternEditor | self: Any, state: Any, step: Any |  | 166-193 |
| method | container | FunctionPatternEditor | self: Any | Container | 196-198 |
| method | _extract_pattern | FunctionPatternEditor | self: Any, step: Any | Union[List, Dict] | 200-204 |
| method | _clone_pattern | FunctionPatternEditor | self: Any, pattern: Any | Union[List, Dict] | 206-215 |
| method | _clone_list_item | FunctionPatternEditor | self: Any, item: Any |  | 217-222 |
| method | _create_header | FunctionPatternEditor | self: Any |  | 224-240 |
| method | _refresh_key_selector | FunctionPatternEditor | self: Any |  | 242-296 |
| method | on_key_change | FunctionPatternEditor | key: Any |  | 268-269 |
| method | _refresh_function_list | FunctionPatternEditor | self: Any |  | 298-302 |
| method | _build_function_list | FunctionPatternEditor | self: Any, functions: Any |  | 304-322 |
| method | _create_function_item | FunctionPatternEditor | self: Any, index: Any, func: Any, kwargs: Any |  | 324-365 |
| method | _get_current_functions | FunctionPatternEditor | self: Any | List | 367-388 |
| method | _extract_func_and_kwargs | FunctionPatternEditor | self: Any, func_item: Any | Tuple[Optional[Callable], Dict] | 390-397 |
| method | _create_function_dropdown | FunctionPatternEditor | self: Any, index: Any, current_func: Any |  | 399-453 |
| method | on_selection_change | FunctionPatternEditor | func: Any |  | 446-449 |
| method | _switch_key | FunctionPatternEditor | self: Any, key: Any |  | 455-477 |
| method | _add_key | FunctionPatternEditor | self: Any |  | 479-499 |
| method | _remove_key | FunctionPatternEditor | self: Any |  | 501-531 |
| method | _edit_in_vim | FunctionPatternEditor | self: Any |  | 533-605 |
| method | _show_error | FunctionPatternEditor | self: Any, message: str |  | 607-627 |
| method | _update_function | FunctionPatternEditor | self: Any, index: Any, func: Any |  | 629-654 |
| method | _update_pattern_functions | FunctionPatternEditor | self: Any, functions: Any |  | 656-674 |
| method | _create_parameter_editor | FunctionPatternEditor | self: Any, func: Any, kwargs: Any, func_index: Any |  | 676-706 |
| method | _create_parameter_field | FunctionPatternEditor | self: Any, name: Any, default: Any, current_value: Any, required: Any, is_special: Any, func_index: Any |  | 708-730 |
| method | _create_input_field | FunctionPatternEditor | self: Any, name: Any, value: Any, func_index: Any |  | 732-750 |
| method | on_text_changed | FunctionPatternEditor | buffer: Any |  | 745-746 |
| method | _get_function_parameters | FunctionPatternEditor | self: Any, func: Any | List[Dict] | 752-786 |
| method | _update_parameter | FunctionPatternEditor | self: Any, name: Any, value_str: Any, func_index: Any |  | 788-806 |
| method | _parse_parameter_value | FunctionPatternEditor | self: Any, value_str: Any |  | 808-828 |
| method | _reset_parameter | FunctionPatternEditor | self: Any, name: Any, default: Any, func_index: Any |  | 830-851 |
| method | _reset_all_parameters | FunctionPatternEditor | self: Any, func_index: Any |  | 853-871 |
| method | _move_function_up | FunctionPatternEditor | self: Any, index: Any |  | 873-880 |
| method | _move_function_down | FunctionPatternEditor | self: Any, index: Any |  | 882-889 |
| method | _delete_function | FunctionPatternEditor | self: Any, index: Any |  | 891-898 |
| method | _add_function | FunctionPatternEditor | self: Any |  | 900-920 |
| method | _save_pattern | FunctionPatternEditor | self: Any |  | 922-936 |
| method | _cancel_editing | FunctionPatternEditor | self: Any |  | 938-940 |
| method | _validate_pattern | FunctionPatternEditor | self: Any | bool | 942-972 |

