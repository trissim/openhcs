# Async Pattern Analysis for /home/ts/code/projects/openhcs/openhcs/tui

## Summary

| File | Async Functions | Without Return Type | Unawaited Coroutines | Awaited Coroutines |
|------|----------------|---------------------|---------------------|-------------------|
| components.py | 0 | 0 | 0 | 0 |
| __init__.py | 0 | 0 | 0 | 0 |
| pipeline_editor.py | 18 | 16 | 5 | 0 |
| file_browser.py | 9 | 9 | 1 | 1 |
| tui_launcher.py | 6 | 6 | 0 | 2 |
| dual_step_func_editor.py | 6 | 6 | 0 | 0 |
| __main__.py | 1 | 1 | 1 | 0 |
| menu_bar.py | 29 | 3 | 0 | 0 |
| utils.py | 3 | 2 | 1 | 0 |
| function_pattern_editor.py | 15 | 15 | 0 | 0 |
| plate_manager_core.py | 36 | 29 | 6 | 0 |
| tui_architecture.py | 12 | 9 | 2 | 0 |
| commands.py | 14 | 0 | 0 | 24 |
| status_bar.py | 6 | 3 | 0 | 0 |
| dialogs/global_settings_editor.py | 3 | 2 | 0 | 0 |
| dialogs/__init__.py | 0 | 0 | 0 | 0 |
| dialogs/manager.py | 1 | 0 | 0 | 1 |
| dialogs/plate_config_editor.py | 5 | 5 | 0 | 0 |
| dialogs/plate_dialog_manager.py | 10 | 6 | 5 | 0 |
| dialogs/help_dialog.py | 1 | 0 | 0 | 0 |
| dialogs/base.py | 2 | 0 | 0 | 2 |
| components/interactive_list_item.py | 0 | 0 | 0 | 0 |
| components/__init__.py | 0 | 0 | 0 | 0 |
| components/spinner.py | 1 | 1 | 0 | 0 |
| components/loading_screen.py | 0 | 0 | 0 | 0 |
| components/grouped_dropdown.py | 0 | 0 | 0 | 0 |
| components/parameter_editor.py | 0 | 0 | 0 | 0 |
| commands/__init__.py | 0 | 0 | 0 | 0 |
| commands/pipeline_commands.py | 5 | 0 | 0 | 5 |
| commands/registry.py | 2 | 0 | 0 | 2 |
| commands/dialog_commands.py | 2 | 0 | 0 | 3 |
| commands/plate_commands.py | 3 | 0 | 0 | 2 |
| services/external_editor_service.py | 2 | 1 | 0 | 0 |
| services/plate_validation.py | 7 | 2 | 0 | 0 |
| utils/dialog_helpers.py | 4 | 2 | 1 | 0 |
| utils/__init__.py | 0 | 0 | 0 | 0 |
| utils/error_handling.py | 3 | 0 | 0 | 2 |

## components.py

### Async Functions Without Return Type Annotations

No async functions without return type annotations found.

### Unawaited Coroutines

No unawaited coroutines found.

## __init__.py

### Async Functions Without Return Type Annotations

No async functions without return type annotations found.

### Unawaited Coroutines

No unawaited coroutines found.

## pipeline_editor.py

### Async Functions Without Return Type Annotations

| Name | Line | Class |
|------|------|-------|
| create | 65 | PipelineEditorPane |
| setup | 87 | PipelineEditorPane |
| _handle_item_select | 236 | PipelineEditorPane |
| _handle_item_move_up | 245 | PipelineEditorPane |
| _handle_item_move_down | 251 | PipelineEditorPane |
| _update_selection | 406 | PipelineEditorPane |
| _select_step | 423 | PipelineEditorPane |
| _save_pipeline | 437 | PipelineEditorPane |
| _handle_step_pattern_saved | 483 | PipelineEditorPane |
| _on_plate_selected | 502 | PipelineEditorPane |
| _load_steps_for_plate | 515 | PipelineEditorPane |
| _edit_step | 584 | PipelineEditorPane |
| _refresh_steps | 603 | PipelineEditorPane |
| _move_step_up | 618 | PipelineEditorPane |
| _move_step_down | 672 | PipelineEditorPane |
| shutdown | 726 | PipelineEditorPane |

### Unawaited Coroutines

| Name | Line | Function | Class |
|------|------|----------|-------|
| execute | 111 | setup | PipelineEditorPane |
| execute | 114 | setup | PipelineEditorPane |
| execute | 117 | setup | PipelineEditorPane |
| execute | 120 | setup | PipelineEditorPane |
| execute | 123 | setup | PipelineEditorPane |

## file_browser.py

### Async Functions Without Return Type Annotations

| Name | Line | Class |
|------|------|-------|
| _load_directory_listing | 242 | FileManagerBrowser |
| _handle_item_activated | 304 | FileManagerBrowser |
| _handle_ok | 347 | FileManagerBrowser |
| _handle_cancel | 400 | FileManagerBrowser |
| _handle_up_directory | 404 | FileManagerBrowser |
| _handle_refresh | 412 | FileManagerBrowser |
| main_async | 489 | None |
| on_path_selected_cb | 491 | None |
| on_cancel_cb | 495 | None |

### Unawaited Coroutines

| Name | Line | Function | Class |
|------|------|----------|-------|
| main_async | 528 | None | None |

## tui_launcher.py

### Async Functions Without Return Type Annotations

| Name | Line | Class |
|------|------|-------|
| _handle_global_config_update | 103 | OpenHCSTUILauncher |
| _on_plate_added | 152 | OpenHCSTUILauncher |
| _on_plate_removed | 196 | OpenHCSTUILauncher |
| _on_plate_selected | 235 | OpenHCSTUILauncher |
| run | 305 | OpenHCSTUILauncher |
| _cleanup | 339 | OpenHCSTUILauncher |

### Unawaited Coroutines

No unawaited coroutines found.

## dual_step_func_editor.py

### Async Functions Without Return Type Annotations

| Name | Line | Class |
|------|------|-------|
| _save_changes | 407 | DualStepFuncEditorPane |
| _close_editor | 462 | DualStepFuncEditorPane |
| shutdown | 493 | DualStepFuncEditorPane |
| _load_step_object | 498 | DualStepFuncEditorPane |
| _save_step_object_as | 539 | DualStepFuncEditorPane |
| _reset_step_parameter_field | 578 | DualStepFuncEditorPane |

### Unawaited Coroutines

No unawaited coroutines found.

## __main__.py

### Async Functions Without Return Type Annotations

| Name | Line | Class |
|------|------|-------|
| main | 26 | None |

### Unawaited Coroutines

| Name | Line | Function | Class |
|------|------|----------|-------|
| main | 149 | None | None |

## menu_bar.py

### Async Functions Without Return Type Annotations

| Name | Line | Class |
|------|------|-------|
| __aenter__ | 65 | ReentrantLock |
| __aexit__ | 81 | ReentrantLock |
| shutdown | 1184 | MenuBar |

### Unawaited Coroutines

No unawaited coroutines found.

## utils.py

### Async Functions Without Return Type Annotations

| Name | Line | Class |
|------|------|-------|
| show_error_dialog | 8 | None |
| focus_text_area | 100 | None |

### Unawaited Coroutines

| Name | Line | Function | Class |
|------|------|----------|-------|
| focus_text_area | 104 | prompt_for_path_dialog | None |

## function_pattern_editor.py

### Async Functions Without Return Type Annotations

| Name | Line | Class |
|------|------|-------|
| _handle_parameter_change | 183 | FunctionPatternEditor |
| _handle_reset_parameter | 214 | FunctionPatternEditor |
| _handle_reset_all_parameters | 237 | FunctionPatternEditor |
| _switch_key | 539 | FunctionPatternEditor |
| _convert_list_to_dict_pattern | 564 | FunctionPatternEditor |
| _add_key | 574 | FunctionPatternEditor |
| _remove_key | 596 | FunctionPatternEditor |
| _edit_in_vim | 628 | FunctionPatternEditor |
| _update_function | 659 | FunctionPatternEditor |
| _move_function_up | 762 | FunctionPatternEditor |
| _move_function_down | 772 | FunctionPatternEditor |
| _delete_function | 782 | FunctionPatternEditor |
| _add_function | 792 | FunctionPatternEditor |
| _load_func_pattern_from_file_handler | 838 | FunctionPatternEditor |
| _save_func_pattern_as_file_handler | 883 | FunctionPatternEditor |

### Unawaited Coroutines

No unawaited coroutines found.

## plate_manager_core.py

### Async Functions Without Return Type Annotations

| Name | Line | Class |
|------|------|-------|
| _on_filemanager_available | 135 | PlateManagerPane |
| _initialize_ui | 155 | PlateManagerPane |
| _handle_request_show_add_plate_dialog | 296 | PlateManagerPane |
| _handle_add_predefined_plate | 300 | PlateManagerPane |
| _initialize_and_refresh | 325 | PlateManagerPane |
| _update_selection | 343 | PlateManagerPane |
| _handle_plate_item_select | 431 | PlateManagerPane |
| _handle_plate_item_move_up | 437 | PlateManagerPane |
| _handle_plate_item_move_down | 443 | PlateManagerPane |
| _update_selection_and_notify_order | 449 | PlateManagerPane |
| _move_plate_up | 488 | PlateManagerPane |
| _move_plate_down | 498 | PlateManagerPane |
| _handle_add_dialog_result | 522 | PlateManagerPane |
| _handle_delete_plates_request | 578 | PlateManagerPane |
| _handle_error | 611 | PlateManagerPane |
| _handle_validation_result | 620 | PlateManagerPane |
| _handle_add_dialog_result | 669 | PlateManagerPane |
| _handle_remove_dialog_result | 693 | PlateManagerPane |
| _handle_remove_dialog_result | 759 | PlateManagerPane |
| _handle_validation_result | 775 | PlateManagerPane |
| _handle_error | 796 | PlateManagerPane |
| _on_edit_plate_clicked | 804 | PlateManagerPane |
| _on_init_plate_clicked | 818 | PlateManagerPane |
| _on_compile_plate_clicked | 821 | PlateManagerPane |
| _on_run_plate_clicked | 824 | PlateManagerPane |
| _update_plate_status | 865 | PlateManagerPane |
| _update_plate_status_locally_and_notify | 874 | PlateManagerPane |
| _refresh_plates | 895 | PlateManagerPane |
| shutdown | 993 | PlateManagerPane |

### Unawaited Coroutines

| Name | Line | Function | Class |
|------|------|----------|-------|
| execute | 188 | _initialize_ui | PlateManagerPane |
| execute | 191 | _initialize_ui | PlateManagerPane |
| execute | 194 | _initialize_ui | PlateManagerPane |
| execute | 197 | _initialize_ui | PlateManagerPane |
| execute | 200 | _initialize_ui | PlateManagerPane |
| execute | 203 | _initialize_ui | PlateManagerPane |

## tui_architecture.py

### Async Functions Without Return Type Annotations

| Name | Line | Class |
|------|------|-------|
| show_dialog | 197 | TUIState |
| _handle_show_edit_plate_config_request | 820 | OpenHCSTUI |
| _handle_plate_config_editing_cancelled | 832 | OpenHCSTUI |
| _handle_plate_config_saved | 840 | OpenHCSTUI |
| _handle_step_editing_cancelled | 850 | OpenHCSTUI |
| _handle_edit_step_dialog_requested | 858 | OpenHCSTUI |
| shutdown_components | 875 | OpenHCSTUI |
| _check_components_initialized | 911 | OpenHCSTUI |
| _async_initialize_pipeline_editor | 944 | OpenHCSTUI |

### Unawaited Coroutines

| Name | Line | Function | Class |
|------|------|----------|-------|
| execute | 398 | None | OpenHCSTUI |
| execute | 399 | None | OpenHCSTUI |

## commands.py

### Async Functions Without Return Type Annotations

No async functions without return type annotations found.

### Unawaited Coroutines

No unawaited coroutines found.

## status_bar.py

### Async Functions Without Return Type Annotations

| Name | Line | Class |
|------|------|-------|
| _on_operation_status_changed | 349 | StatusBar |
| _on_error_event | 375 | StatusBar |
| shutdown | 433 | StatusBar |

### Unawaited Coroutines

No unawaited coroutines found.

## dialogs/global_settings_editor.py

### Async Functions Without Return Type Annotations

| Name | Line | Class |
|------|------|-------|
| _save_settings | 285 | GlobalSettingsEditorDialog |
| _cancel | 327 | GlobalSettingsEditorDialog |

### Unawaited Coroutines

No unawaited coroutines found.

## dialogs/__init__.py

### Async Functions Without Return Type Annotations

No async functions without return type annotations found.

### Unawaited Coroutines

No unawaited coroutines found.

## dialogs/manager.py

### Async Functions Without Return Type Annotations

No async functions without return type annotations found.

### Unawaited Coroutines

No unawaited coroutines found.

## dialogs/plate_config_editor.py

### Async Functions Without Return Type Annotations

| Name | Line | Class |
|------|------|-------|
| notify_error_no_config_attr | 269 | PlateConfigEditorPane |
| notify_save_success | 278 | PlateConfigEditorPane |
| notify_apply_error | 284 | PlateConfigEditorPane |
| notify_cancel | 291 | PlateConfigEditorPane |
| shutdown | 295 | PlateConfigEditorPane |

### Unawaited Coroutines

No unawaited coroutines found.

## dialogs/plate_dialog_manager.py

### Async Functions Without Return Type Annotations

| Name | Line | Class |
|------|------|-------|
| browser_on_path_selected | 208 | PlateDialogManager |
| browser_on_cancel | 225 | PlateDialogManager |
| _dialog_ok | 329 | PlateDialogManager |
| _dialog_cancel | 361 | PlateDialogManager |
| _show_error_dialog | 452 | PlateDialogManager |
| show_remove_plate_dialog | 509 | PlateDialogManager |

### Unawaited Coroutines

| Name | Line | Function | Class |
|------|------|----------|-------|
| show | 352 | _dialog_ok | PlateDialogManager |
| show | 414 | None | PlateDialogManager |
| show | 422 | None | PlateDialogManager |
| show | 430 | None | PlateDialogManager |
| show | 432 | None | PlateDialogManager |

## dialogs/help_dialog.py

### Async Functions Without Return Type Annotations

No async functions without return type annotations found.

### Unawaited Coroutines

No unawaited coroutines found.

## dialogs/base.py

### Async Functions Without Return Type Annotations

No async functions without return type annotations found.

### Unawaited Coroutines

No unawaited coroutines found.

## components/interactive_list_item.py

### Async Functions Without Return Type Annotations

No async functions without return type annotations found.

### Unawaited Coroutines

No unawaited coroutines found.

## components/__init__.py

### Async Functions Without Return Type Annotations

No async functions without return type annotations found.

### Unawaited Coroutines

No unawaited coroutines found.

## components/spinner.py

### Async Functions Without Return Type Annotations

| Name | Line | Class |
|------|------|-------|
| spin | 55 | Spinner |

### Unawaited Coroutines

No unawaited coroutines found.

## components/loading_screen.py

### Async Functions Without Return Type Annotations

No async functions without return type annotations found.

### Unawaited Coroutines

No unawaited coroutines found.

## components/grouped_dropdown.py

### Async Functions Without Return Type Annotations

No async functions without return type annotations found.

### Unawaited Coroutines

No unawaited coroutines found.

## components/parameter_editor.py

### Async Functions Without Return Type Annotations

No async functions without return type annotations found.

### Unawaited Coroutines

No unawaited coroutines found.

## commands/__init__.py

### Async Functions Without Return Type Annotations

No async functions without return type annotations found.

### Unawaited Coroutines

No unawaited coroutines found.

## commands/pipeline_commands.py

### Async Functions Without Return Type Annotations

No async functions without return type annotations found.

### Unawaited Coroutines

No unawaited coroutines found.

## commands/registry.py

### Async Functions Without Return Type Annotations

No async functions without return type annotations found.

### Unawaited Coroutines

No unawaited coroutines found.

## commands/dialog_commands.py

### Async Functions Without Return Type Annotations

No async functions without return type annotations found.

### Unawaited Coroutines

No unawaited coroutines found.

## commands/plate_commands.py

### Async Functions Without Return Type Annotations

No async functions without return type annotations found.

### Unawaited Coroutines

No unawaited coroutines found.

## services/external_editor_service.py

### Async Functions Without Return Type Annotations

| Name | Line | Class |
|------|------|-------|
| _show_error_dialog | 121 | ExternalEditorService |

### Unawaited Coroutines

No unawaited coroutines found.

## services/plate_validation.py

### Async Functions Without Return Type Annotations

| Name | Line | Class |
|------|------|-------|
| close | 65 | PlateValidationService |
| close | 232 | PlateValidationService |

### Unawaited Coroutines

No unawaited coroutines found.

## utils/dialog_helpers.py

### Async Functions Without Return Type Annotations

| Name | Line | Class |
|------|------|-------|
| show_error_dialog | 17 | None |
| focus_text_area | 108 | None |

### Unawaited Coroutines

| Name | Line | Function | Class |
|------|------|----------|-------|
| focus_text_area | 112 | prompt_for_path_dialog | None |

## utils/__init__.py

### Async Functions Without Return Type Annotations

No async functions without return type annotations found.

### Unawaited Coroutines

No unawaited coroutines found.

## utils/error_handling.py

### Async Functions Without Return Type Annotations

No async functions without return type annotations found.

### Unawaited Coroutines

No unawaited coroutines found.
