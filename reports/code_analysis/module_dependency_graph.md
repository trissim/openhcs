# Module Dependency Graph

## `__init__.py`
### Depends on:
- `openhcs.tui.function_pattern_editor`

## `__main__.py`
### Depends on:
- `openhcs.tui.tui_launcher`
- `openhcs.core.config`
- `openhcs.core.orchestrator.gpu_scheduler`

## `action_menu_pane.py`
### Depends on:
- `openhcs.tui.tui_architecture`
- `openhcs.core.steps.abstract`
- `openhcs.core.config`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.core.context.processing_context`

## `dialogs/plate_dialog_manager.py`
### Depends on:
- `SyntaxError: invalid syntax. Maybe you meant '==' or ':=' instead of '='? at line 255`

## `function_pattern_editor.py`
### Depends on:
- `openhcs.constants.constants`
- `openhcs.processing.func_registry`
- `openhcs.core.pipeline.funcstep_contract_validator`

## `menu_bar.py`
### Depends on:
- `openhcs.core.context.processing_context`
- `openhcs.core.steps.abstract`
- `openhcs.core.orchestrator.orchestrator`

## `plate_manager_core.py`
### Depends on:
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.core.context.processing_context`

## `services/plate_validation.py`
### Depends on:
- `SyntaxError: invalid syntax at line 252`

## `status_bar.py`
### No local dependencies found.

## `step_viewer.py`
### Depends on:
- `SyntaxError: invalid syntax at line 495`

## `tui_architecture.py`
### Depends on:
- `openhcs.tui.menu_bar`
- `openhcs.tui.action_menu_pane`
- `openhcs.processing.func_registry`
- `openhcs.tui.plate_manager_core`
- `openhcs.core.config`
- `openhcs.io.filemanager`
- `openhcs.tui.status_bar`
- `openhcs.core.context.processing_context`
- `openhcs.tui.step_viewer`
- `openhcs.tui.function_pattern_editor`

## `tui_launcher.py`
### Depends on:
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.io.filemanager`
- `openhcs.core.config`
- `openhcs.core.context.processing_context`
- `openhcs.tui.tui_architecture`
- `openhcs.io.base`

