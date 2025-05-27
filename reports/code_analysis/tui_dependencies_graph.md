# Module Dependency Graph for `openhcs/tui/`
Generated on: 2025-05-22 17:48:41

## `openhcs/tui/__init__.py`
### Depends on:
- `openhcs.tui.function_pattern_editor`

## `openhcs/tui/__main__.py`
### Depends on:
- `openhcs.core.orchestrator.gpu_scheduler`
- `openhcs.tui.tui_launcher`
- `openhcs.core.config`

## `openhcs/tui/action_menu_pane.py`
### Depends on:
- `openhcs.core.context.processing_context`
- `openhcs.core.steps.abstract`
- `openhcs.tui.tui_architecture`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.core.config`

## `openhcs/tui/dialogs/plate_dialog_manager.py`
### No local dependencies found or file has issues.

## `openhcs/tui/function_pattern_editor.py`
### Depends on:
- `openhcs.constants.constants`
- `openhcs.core.pipeline.funcstep_contract_validator`
- `openhcs.processing.func_registry`

## `openhcs/tui/menu_bar.py`
### Depends on:
- `openhcs.core.context.processing_context`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.core.steps.abstract`

## `openhcs/tui/plate_manager_core.py`
### Depends on:
- `openhcs.io.filemanager`
- `openhcs.core.context.processing_context`
- `openhcs.io.base`
- `openhcs.tui.dialogs.plate_dialog_manager`
- `openhcs.tui.status_bar`
- `openhcs.tui.services.plate_validation`

## `openhcs/tui/services/plate_validation.py`
### Depends on:
- `openhcs.io.filemanager`
- `openhcs.core.context.processing_context`

## `openhcs/tui/status_bar.py`
### No local dependencies found or file has issues.

## `openhcs/tui/step_viewer.py`
### Depends on:
- `openhcs.core.context.processing_context`
- `openhcs.core.steps.function_step`

## `openhcs/tui/tui_architecture.py`
### Depends on:
- `openhcs.io.filemanager`
- `openhcs.core.context.processing_context`
- `openhcs.processing.func_registry`
- `openhcs.tui.step_viewer`
- `openhcs.tui.menu_bar`
- `openhcs.tui.function_pattern_editor`
- `openhcs.tui.action_menu_pane`
- `openhcs.tui.status_bar`
- `openhcs.tui.plate_manager_core`
- `openhcs.core.config`

## `openhcs/tui/tui_launcher.py`
### Depends on:
- `openhcs.io.filemanager`
- `openhcs.core.context.processing_context`
- `openhcs.io.base`
- `openhcs.tui.tui_architecture`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.core.config`

