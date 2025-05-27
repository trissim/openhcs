# Detailed Code Definition Matrix
Generated on: 2025-05-22 18:08:36

### Detailed Matrix for `openhcs/tui/menu_bar.py`

| Definition Type | Name | Parent Class | Parameters | Return Type | Lines |
| --- | --- | --- | --- | --- | --- |
| class | MissingStateError |  |  |  | 35-44 |
| method | __init__ | MissingStateError | self: Any, attribute_name: str |  | 42-44 |
| class | ReentrantLock |  |  |  | 47-83 |
| method | __init__ | ReentrantLock | self: Any |  | 54-57 |
| method | __aenter__ | ReentrantLock | self: Any |  | 59-73 |
| method | __aexit__ | ReentrantLock | self: Any, exc_type: Any, exc_val: Any, exc_tb: Any |  | 75-83 |
| class | LayoutContract |  |  |  | 86-108 |
| method | validate_layout_container | LayoutContract | container: Any | <complex_annotation> | 94-108 |
| class | MenuItemType |  |  |  | 111-116 |
| class | MenuItemSchema |  |  |  | 120-159 |
| method | validate_menu_item | MenuItemSchema | item: Dict[str, Any] | <complex_annotation> | 130-159 |
| class | MenuStructureSchema |  |  |  | 163-200 |
| method | validate_menu_structure | MenuStructureSchema | structure: Dict[str, List[Dict[str, Any]]] | <complex_annotation> | 171-200 |
| class | MenuItem |  |  |  | 203-310 |
| method | __init__ | MenuItem | self: Any, type: MenuItemType, label: str, handler: Optional[Callable], shortcut: Optional[str], enabled: Union[bool, Condition], checked: Union[bool, Condition], children: Optional[List[<complex_annotation>]] |  | 212-240 |
| method | is_enabled | MenuItem | self: Any | bool | 242-246 |
| method | is_checked | MenuItem | self: Any | bool | 248-252 |
| method | set_checked | MenuItem | self: Any, checked: bool | <complex_annotation> | 254-257 |
| method | from_dict | MenuItem | cls: Any, item_dict: Dict[str, Any], handler_map: Dict[str, Callable] | <complex_annotation> | 260-310 |
| class | MenuBar |  |  |  | 367-1249 |
| method | __init__ | MenuBar | self: Any, state: Any |  | 380-421 |
| method | _create_handler_map | MenuBar | self: Any | Dict[str, Callable] | 423-463 |
| method | _create_condition_map | MenuBar | self: Any | Dict[str, Condition] | 465-487 |
| method | _get_required_state | MenuBar | self: Any, attribute_name: str | Any | 489-504 |
| method | _load_menu_structure | MenuBar | self: Any | Dict[str, List[MenuItem]] | 506-565 |
| method | _create_menu_labels | MenuBar | self: Any | List[Label] | 571-606 |
| method | create_mouse_handler | MenuBar | menu: Any |  | 594-600 |
| method | menu_mouse_handler | MenuBar | mouse_event: Any |  | 595-599 |
| method | _create_submenu_float | MenuBar | self: Any | Float | 608-625 |
| method | _create_key_bindings | MenuBar | self: Any | KeyBindings | 627-689 |
| method | create_handler | MenuBar | menu: Any |  | 643-646 |
| method | handler | MenuBar | event: KeyPressEvent |  | 644-645 |
| method | _ | MenuBar | event: KeyPressEvent |  | 657-658 |
| method | _ | MenuBar | event: KeyPressEvent |  | 668-669 |
| method | _ | MenuBar | event: KeyPressEvent |  | 672-673 |
| method | _ | MenuBar | event: KeyPressEvent |  | 677-678 |
| method | _ | MenuBar | event: KeyPressEvent |  | 681-682 |
| method | _ | MenuBar | event: KeyPressEvent |  | 686-687 |
| method | _activate_menu | MenuBar | self: Any, menu_name: str | <complex_annotation> | 691-727 |
| method | _close_menu | MenuBar | self: Any | <complex_annotation> | 729-747 |
| method | _navigate_menu | MenuBar | self: Any, delta: int | <complex_annotation> | 749-770 |
| method | _navigate_submenu | MenuBar | self: Any, delta: int | <complex_annotation> | 772-812 |
| method | _select_current_item | MenuBar | self: Any | <complex_annotation> | 814-826 |
| method | _create_submenu_container | MenuBar | self: Any, menu_items: List[MenuItem] | Container | 828-886 |
| method | create_mouse_handler | MenuBar | menu_item: Any |  | 872-879 |
| method | item_mouse_handler | MenuBar | mouse_event: Any |  | 873-878 |
| method | _handle_menu_item | MenuBar | self: Any, item: MenuItem | <complex_annotation> | 888-904 |
| method | _on_new_pipeline | MenuBar | self: Any | <complex_annotation> | 908-912 |
| method | _on_open_pipeline | MenuBar | self: Any | <complex_annotation> | 914-918 |
| method | _on_save_pipeline | MenuBar | self: Any | <complex_annotation> | 920-978 |
| method | _on_save_pipeline_as | MenuBar | self: Any | <complex_annotation> | 980-984 |
| method | _on_exit | MenuBar | self: Any | <complex_annotation> | 986-988 |
| method | _on_add_step | MenuBar | self: Any | <complex_annotation> | 990-994 |
| method | _on_edit_step | MenuBar | self: Any | <complex_annotation> | 996-1032 |
| method | _on_remove_step | MenuBar | self: Any | <complex_annotation> | 1034-1079 |
| method | _on_move_step_up | MenuBar | self: Any | <complex_annotation> | 1081-1085 |
| method | _on_move_step_down | MenuBar | self: Any | <complex_annotation> | 1087-1091 |
| method | _on_toggle_log_drawer | MenuBar | self: Any | <complex_annotation> | 1093-1095 |
| method | _on_toggle_vim_mode | MenuBar | self: Any | <complex_annotation> | 1097-1099 |
| method | _on_set_theme | MenuBar | self: Any, theme: str | <complex_annotation> | 1101-1108 |
| method | _on_pre_compile | MenuBar | self: Any | <complex_annotation> | 1110-1112 |
| method | _on_compile | MenuBar | self: Any | <complex_annotation> | 1114-1116 |
| method | _on_run | MenuBar | self: Any | <complex_annotation> | 1118-1120 |
| method | _on_test | MenuBar | self: Any | <complex_annotation> | 1122-1142 |
| method | _on_settings | MenuBar | self: Any | <complex_annotation> | 1144-1146 |
| method | _on_documentation | MenuBar | self: Any | <complex_annotation> | 1148-1152 |
| method | _on_keyboard_shortcuts | MenuBar | self: Any | <complex_annotation> | 1154-1158 |
| method | _on_about | MenuBar | self: Any | <complex_annotation> | 1160-1164 |
| method | _on_operation_status_changed | MenuBar | self: Any, data: Any | <complex_annotation> | 1168-1176 |
| method | _on_plate_selected | MenuBar | self: Any, data: Any | <complex_annotation> | 1178-1186 |
| method | _on_is_compiled_changed | MenuBar | self: Any, data: Any | <complex_annotation> | 1188-1196 |
| method | shutdown | MenuBar | self: Any |  | 1198-1209 |
| method | __pt_container__ | MenuBar | self: Any | Container | 1211-1213 |
| method | get_children | MenuBar | self: Any |  | 1216-1217 |
| method | preferred_width | MenuBar | self: Any, max_available_width: Any |  | 1219-1220 |
| method | preferred_height | MenuBar | self: Any, max_available_height: Any, width: Any |  | 1222-1223 |
| method | reset | MenuBar | self: Any |  | 1225-1226 |
| method | write_to_screen | MenuBar | self: Any, screen: Any, mouse_handlers: Any, write_position: Any, parent_style: Any, erase_bg: Any, z_index: Any |  | 1228-1231 |
| method | mouse_handler | MenuBar | self: Any, mouse_event: Any |  | 1233-1249 |

