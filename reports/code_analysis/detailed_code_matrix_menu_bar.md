# Detailed Code Definition Matrix
Generated on: 2025-05-22 22:45:31

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

