# Detailed Code Definition Matrix
Generated on: 2025-05-22 18:08:36

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

