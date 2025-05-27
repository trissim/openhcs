# Detailed Code Definition Matrix
Generated on: 2025-05-22 18:08:36

### Detailed Matrix for `openhcs/tui/tui_architecture.py`

| Definition Type | Name | Parent Class | Parameters | Return Type | Lines |
| --- | --- | --- | --- | --- | --- |
| class | Clause5Violation |  |  |  | 49-55 |
| class | TUIState |  |  |  | 96-185 |
| method | __init__ | TUIState | self: Any |  | 103-131 |
| method | add_observer | TUIState | self: Any, event_type: str, callback: Callable | <complex_annotation> | 133-143 |
| method | notify | TUIState | self: Any, event_type: str, data: Any | <complex_annotation> | 145-158 |
| method | set_selected_plate | TUIState | self: Any, plate: Dict[str, Any] | <complex_annotation> | 160-172 |
| method | set_selected_step | TUIState | self: Any, step: Dict[str, Any] | <complex_annotation> | 174-185 |
| class | OpenHCSTUI |  |  |  | 188-530 |
| method | __init__ | OpenHCSTUI | self: Any, initial_context: ProcessingContext, state: TUIState, global_config: GlobalPipelineConfig |  | 195-244 |
| method | _create_key_bindings | OpenHCSTUI | self: Any | KeyBindings | 246-272 |
| method | _ | OpenHCSTUI | event: Any |  | 257-259 |
| method | _ | OpenHCSTUI | event: Any |  | 263-265 |
| method | _ | OpenHCSTUI | event: Any |  | 268-270 |
| method | _validate_components_present | OpenHCSTUI | self: Any |  | 274-324 |
| method | _get_left_pane | OpenHCSTUI | self: Any | Container | 326-375 |
| method | _get_step_viewer | OpenHCSTUI | self: Any | Container | 377-392 |
| method | _get_action_menu | OpenHCSTUI | self: Any | Container | 394-406 |
| method | _get_status_bar | OpenHCSTUI | self: Any | Container | 408-420 |
| method | _get_menu_bar | OpenHCSTUI | self: Any | Container | 422-434 |
| method | _on_launcher_config_rebound | OpenHCSTUI | self: Any, new_core_config: GlobalPipelineConfig | <complex_annotation> | 436-455 |
| method | shutdown_components | OpenHCSTUI | self: Any |  | 457-490 |
| method | _async_initialize_step_viewer | OpenHCSTUI | self: Any |  | 492-507 |
| method | get_children | OpenHCSTUI | self: Any |  | 511-512 |
| method | preferred_width | OpenHCSTUI | self: Any, max_available_width: Any |  | 514-515 |
| method | preferred_height | OpenHCSTUI | self: Any, max_available_height: Any, width: Any |  | 517-518 |
| method | reset | OpenHCSTUI | self: Any |  | 520-521 |
| method | write_to_screen | OpenHCSTUI | self: Any, screen: Any, mouse_handlers: Any, write_position: Any, parent_style: Any, erase_bg: Any, z_index: Any |  | 523-526 |
| method | mouse_handler | OpenHCSTUI | self: Any, mouse_event: Any |  | 528-530 |

