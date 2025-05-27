# Detailed Code Definition Matrix
Generated on: 2025-05-22 18:08:36

### Detailed Matrix for `openhcs/tui/action_menu_pane.py`

| Definition Type | Name | Parent Class | Parameters | Return Type | Lines |
| --- | --- | --- | --- | --- | --- |
| class | ActionMenuPane |  |  |  | 41-839 |
| method | __init__ | ActionMenuPane | self: Any, state: <complex_annotation>, initial_tui_context: <complex_annotation> |  | 49-87 |
| method | _show_error | ActionMenuPane | self: Any, message: str |  | 89-95 |
| method | _clear_error | ActionMenuPane | self: Any |  | 97-99 |
| method | _set_status | ActionMenuPane | self: Any, message: str |  | 101-103 |
| method | _clear_status | ActionMenuPane | self: Any |  | 105-107 |
| method | _create_buttons | ActionMenuPane | self: Any | List[Container] | 109-483 |
| method | _add_handler | ActionMenuPane |  |  | 114-116 |
| method | _pre_compile_handler | ActionMenuPane |  |  | 120-189 |
| method | _compile_handler | ActionMenuPane |  |  | 191-282 |
| method | _run_handler | ActionMenuPane |  |  | 284-366 |
| method | _save_handler | ActionMenuPane |  |  | 368-430 |
| method | _test_handler | ActionMenuPane |  |  | 432-466 |
| method | _settings_handler | ActionMenuPane | self: Any |  | 485-553 |
| method | _create_main_settings_dialog | ActionMenuPane | self: Any | Dialog | 556-695 |
| method | save_tui_settings_handler | ActionMenuPane |  |  | 627-671 |
| method | close_dialog_handler | ActionMenuPane |  |  | 673-676 |
| method | _apply_and_save_global_config_settings | ActionMenuPane | self: Any |  | 697-785 |
| method | _persist_global_config_to_file | ActionMenuPane | self: Any, config_to_save: GlobalPipelineConfig |  | 787-836 |
| method | __pt_container__ | ActionMenuPane | self: Any |  | 838-839 |

