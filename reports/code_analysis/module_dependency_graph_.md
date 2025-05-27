# Module Dependency Graph for `openhcs/tui/`
Generated on: 2025-05-22 16:36:24

## `openhcs/tui/__init__.py`
### Depends on:
- `openhcs.tui.function_pattern_editor`

## `openhcs/tui/__main__.py`
### Depends on:
- `openhcs.tui.tui_launcher`
- `openhcs.core.orchestrator.gpu_scheduler`
- `openhcs.core.config`

## `openhcs/tui/action_menu_pane.py`
### Depends on:
- `openhcs.core.steps.abstract`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.core.context.processing_context`
- `openhcs.tui.tui_architecture`
- `openhcs.core.config`

## `openhcs/tui/dialogs/plate_dialog_manager.py`
### Errors encountered:
- SyntaxError: invalid syntax. Maybe you meant '==' or ':=' instead of '='? at line 255

## `openhcs/tui/function_pattern_editor.py`
### Depends on:
- `openhcs.processing.func_registry`
- `openhcs.constants.constants`
- `openhcs.core.pipeline.funcstep_contract_validator`

## `openhcs/tui/menu_bar.py`
### Depends on:
- `openhcs.core.context.processing_context`
- `openhcs.core.steps.abstract`
- `openhcs.core.orchestrator.orchestrator`

## `openhcs/tui/plate_manager_core.py`
### Depends on:
- `openhcs.core.context.processing_context`
- `openhcs.core.orchestrator.orchestrator`

## `openhcs/tui/services/plate_validation.py`
### Errors encountered:
- SyntaxError: invalid syntax at line 252

## `openhcs/tui/status_bar.py`
### No local dependencies found or file has issues.

## `openhcs/tui/step_viewer.py`
### Errors encountered:
- SyntaxError: invalid syntax at line 495

## `openhcs/tui/tui_architecture.py`
### Depends on:
- `openhcs.tui.action_menu_pane`
- `openhcs.tui.step_viewer`
- `openhcs.processing.func_registry`
- `openhcs.tui.status_bar`
- `openhcs.core.context.processing_context`
- `openhcs.tui.function_pattern_editor`
- `openhcs.core.config`
- `openhcs.tui.menu_bar`
- `openhcs.tui.plate_manager_core`
- `openhcs.io.filemanager`

## `openhcs/tui/tui_launcher.py`
### Depends on:
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.core.context.processing_context`
- `openhcs.tui.tui_architecture`
- `openhcs.core.config`
- `openhcs.io.filemanager`
- `openhcs.io.base`

