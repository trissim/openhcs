# Detailed Code Definition Matrix
Generated on: 2025-05-22 18:08:36

### Detailed Matrix for `openhcs/tui/step_viewer.py`

| Definition Type | Name | Parent Class | Parameters | Return Type | Lines |
| --- | --- | --- | --- | --- | --- |
| class | StepViewerPane |  |  |  | 15-505 |
| method | __init__ | StepViewerPane | self: Any, state: Any, context: ProcessingContext |  | 23-52 |
| method | create | StepViewerPane | cls: Any, state: Any, context: ProcessingContext |  | 55-75 |
| method | setup | StepViewerPane | self: Any |  | 77-115 |
| method | _create_step_list | StepViewerPane | self: Any | TextArea | 117-130 |
| method | _format_step_list | StepViewerPane | self: Any | str | 132-175 |
| method | _get_status_icon | StepViewerPane | self: Any, status: str | str | 177-194 |
| method | _get_function_name | StepViewerPane | self: Any, step: Dict[str, Any] | str | 196-227 |
| method | _create_key_bindings | StepViewerPane | self: Any | KeyBindings | 229-268 |
| method | _ | StepViewerPane | event: Any |  | 239-243 |
| method | _ | StepViewerPane | event: Any |  | 246-250 |
| method | _ | StepViewerPane | event: Any |  | 253-256 |
| method | _ | StepViewerPane | event: Any |  | 259-261 |
| method | _ | StepViewerPane | event: Any |  | 264-266 |
| method | _update_selection | StepViewerPane | self: Any |  | 270-274 |
| method | _select_step | StepViewerPane | self: Any, index: int |  | 276-286 |
| method | _on_plate_selected | StepViewerPane | self: Any, plate: Any |  | 288-299 |
| method | _load_steps_for_plate | StepViewerPane | self: Any, plate_id: str |  | 301-354 |
| method | _edit_step | StepViewerPane | self: Any |  | 356-361 |
| method | _add_step | StepViewerPane | self: Any |  | 363-408 |
| method | _remove_step | StepViewerPane | self: Any |  | 410-420 |
| method | _refresh_steps | StepViewerPane | self: Any, _: Any |  | 422-424 |
| method | _move_step_up | StepViewerPane | self: Any |  | 426-459 |
| method | _move_step_down | StepViewerPane | self: Any |  | 461-494 |
| method | shutdown | StepViewerPane | self: Any |  | 496-505 |

