# Detailed Code Definition Matrix
Generated on: 2025-05-22 18:08:36

### Detailed Matrix for `openhcs/tui/dialogs/plate_dialog_manager.py`

| Definition Type | Name | Parent Class | Parameters | Return Type | Lines |
| --- | --- | --- | --- | --- | --- |
| class | DialogResultCallback |  |  |  | 37-39 |
| method | __call__ | DialogResultCallback | self: Any, result: Any | <complex_annotation> | 39-39 |
| class | ErrorCallback |  |  |  | 41-43 |
| method | __call__ | ErrorCallback | self: Any, message: str, details: Optional[str] | <complex_annotation> | 43-43 |
| class | ErrorBanner |  |  |  | 45-154 |
| method | __init__ | ErrorBanner | self: Any |  | 52-73 |
| method | show | ErrorBanner | self: Any, message: str |  | 75-85 |
| method | hide | ErrorBanner | self: Any |  | 87-91 |
| method | reset | ErrorBanner | self: Any |  | 93-96 |
| method | __pt_container__ | ErrorBanner | self: Any |  | 98-100 |
| method | get_children | ErrorBanner | self: Any |  | 103-104 |
| method | preferred_width | ErrorBanner | self: Any, max_available_width: Any |  | 106-107 |
| method | preferred_height | ErrorBanner | self: Any, max_available_height: Any, width: Any |  | 109-110 |
| method | reset | ErrorBanner | self: Any |  | 112-113 |
| method | write_to_screen | ErrorBanner | self: Any, screen: Any, mouse_handlers: Any, write_position: Any, parent_style: Any, erase_bg: Any, z_index: Any |  | 115-118 |
| method | mouse_handler | ErrorBanner | self: Any, mouse_event: Any |  | 120-122 |
| method | find_in_container | ErrorBanner | container: Any |  | 125-154 |
| class | PlateDialogManager |  |  |  | 157-593 |
| method | __init__ | PlateDialogManager | self: Any, on_add_dialog_result: DialogResultCallback, on_remove_dialog_result: DialogResultCallback, on_error: ErrorCallback, storage_registry: Any |  | 166-185 |
| method | show_add_plate_dialog | PlateDialogManager | self: Any | <complex_annotation> | 187-212 |
| method | _create_file_browser_dialog | PlateDialogManager | self: Any | Dialog | 214-303 |
| method | selection_tracking_handler | PlateDialogManager | mouse_event: Any |  | 253-257 |
| method | _show_dialog | PlateDialogManager | self: Any, dialog: Dialog | Optional[Any] | 305-357 |
| method | _dialog_ok | PlateDialogManager | self: Any, dlg: Any, result: Any |  | 359-389 |
| method | _dialog_cancel | PlateDialogManager | self: Any, dlg: Any |  | 391-426 |
| method | _handle_ok_button_press | PlateDialogManager | self: Any, dialog: Any, path_input: Any, backend_selector: Any, user_selected: Any |  | 428-498 |
| method | _show_error_dialog | PlateDialogManager | self: Any, message: str, details: str |  | 502-511 |
| method | _create_error_dialog | PlateDialogManager | self: Any, message: str, details: str |  | 513-557 |
| method | show_remove_plate_dialog | PlateDialogManager | self: Any, plate: Dict[str, Any] |  | 559-593 |

