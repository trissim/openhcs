# Detailed Code Definition Matrix
Generated on: 2025-05-22 18:08:35

### Detailed Matrix for `openhcs/tui/plate_manager_core.py`

| Definition Type | Name | Parent Class | Parameters | Return Type | Lines |
| --- | --- | --- | --- | --- | --- |
| class | PlateEventHandler |  |  |  | 52-57 |
| method | on_plate_added | PlateEventHandler | self: Any, plate: Dict[str, Any] | <complex_annotation> | 54-54 |
| method | on_plate_removed | PlateEventHandler | self: Any, plate: Dict[str, Any] | <complex_annotation> | 55-55 |
| method | on_plate_selected | PlateEventHandler | self: Any, plate: Dict[str, Any] | <complex_annotation> | 56-56 |
| method | on_plate_status_changed | PlateEventHandler | self: Any, plate_id: str, status: str | <complex_annotation> | 57-57 |
| class | PlateManagerPane |  |  |  | 60-592 |
| method | __init__ | PlateManagerPane | self: Any, state: Any, context: ProcessingContext, storage_registry: Any |  | 70-118 |
| method | _initialize_ui | PlateManagerPane | self: Any |  | 120-170 |
| method | container | PlateManagerPane | self: Any | Container | 173-185 |
| method | _handle_request_show_add_plate_dialog | PlateManagerPane | self: Any, data: Any |  | 187-190 |
| method | _handle_add_predefined_plate | PlateManagerPane | self: Any, data: Optional[Dict[str, Any]] |  | 192-210 |
| method | _on_filemanager_available | PlateManagerPane | self: Any, data: Any |  | 212-222 |
| method | _initialize_and_refresh | PlateManagerPane | self: Any |  | 224-243 |
| method | _create_plate_list | PlateManagerPane | self: Any | TextArea | 245-262 |
| method | _format_plate_list | PlateManagerPane | self: Any, lock_already_held: bool | str | 264-337 |
| method | _do_format | PlateManagerPane |  |  | 275-329 |
| method | _create_key_bindings | PlateManagerPane | self: Any | KeyBindings | 339-379 |
| method | _ | PlateManagerPane | event: Any |  | 345-348 |
| method | _ | PlateManagerPane | event: Any |  | 351-354 |
| method | _ | PlateManagerPane | event: Any |  | 361-364 |
| method | _ | PlateManagerPane | event: Any |  | 367-370 |
| method | _ | PlateManagerPane | event: Any |  | 374-377 |
| method | _show_add_plate_dialog | PlateManagerPane | self: Any |  | 382-384 |
| method | _show_remove_plate_dialog | PlateManagerPane | self: Any |  | 386-392 |
| method | _handle_add_dialog_result | PlateManagerPane | self: Any, result: Dict[str, Any] |  | 395-405 |
| method | _handle_remove_dialog_result | PlateManagerPane | self: Any, plate: Dict[str, Any] |  | 407-422 |
| method | _handle_validation_result | PlateManagerPane | self: Any, plate: Dict[str, Any] |  | 425-448 |
| method | _handle_error | PlateManagerPane | self: Any, message: str, details: str |  | 450-467 |
| method | _move_selection | PlateManagerPane | self: Any, delta: int | <complex_annotation> | 470-483 |
| method | _ensure_selection_visible | PlateManagerPane | self: Any | <complex_annotation> | 485-491 |
| method | _select_plate | PlateManagerPane | self: Any, index: int | <complex_annotation> | 493-512 |
| method | _update_plate_status | PlateManagerPane | self: Any, data: Any |  | 514-537 |
| method | _refresh_plates | PlateManagerPane | self: Any, _: Any |  | 539-565 |
| method | shutdown | PlateManagerPane | self: Any |  | 567-592 |

