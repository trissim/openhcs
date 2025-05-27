# Module Dependency Graph for `openhcs/tui`
Generated on: 2025-05-23 12:39:12

## `openhcs/tui/__init__.py`
### Depends on:
- `openhcs.tui.function_pattern_editor`

## `openhcs/tui/__main__.py`
### Depends on:
- `openhcs.tui.tui_launcher`
- `openhcs.core.config`
- `openhcs.core.orchestrator.gpu_scheduler`

## `openhcs/tui/commands.py`
### Depends on:
- `openhcs.core.steps.abstract`
- `openhcs.core.steps.function_step`
- `openhcs.processing.func_registry`
- `openhcs.core.config`
- `openhcs.constants.constants`
- `openhcs.tui.utils`
- `openhcs.tui.dialogs.global_settings_editor`
- `openhcs.core.context.processing_context`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.tui.dialogs.plate_dialog_manager`
- `openhcs.tui.tui_architecture`

## `openhcs/tui/commands/__init__.py`
### Depends on:
- `openhcs.tui.commands.pipeline_commands`
- `openhcs.tui.commands.plate_commands`
- `openhcs.tui.commands.registry`
- `openhcs.tui.commands.dialog_commands`

## `openhcs/tui/commands/dialog_commands.py`
### Depends on:
- `openhcs.core.context.processing_context`
- `openhcs.core.config`
- `openhcs.tui.dialogs.global_settings_editor`
- `openhcs.tui.tui_architecture`

## `openhcs/tui/commands/pipeline_commands.py`
### Depends on:
- `openhcs.core.context.processing_context`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.tui.tui_architecture`

## `openhcs/tui/commands/plate_commands.py`
### Depends on:
- `openhcs.core.context.processing_context`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.tui.dialogs.plate_dialog_manager`
- `openhcs.tui.tui_architecture`

## `openhcs/tui/commands/registry.py`
### Depends on:
- `openhcs.tui.commands`

## `openhcs/tui/components.py`
### No local dependencies found or file has issues.

## `openhcs/tui/components/__init__.py`
### Depends on:
- `openhcs.tui.components.spinner`
- `openhcs.tui.components.parameter_editor`
- `openhcs.tui.components.interactive_list_item`
- `openhcs.tui.components.grouped_dropdown`
- `openhcs.tui.components.loading_screen`
- `openhcs.tui.components.framed_button`

## `openhcs/tui/components/framed_button.py`
### No local dependencies found or file has issues.

## `openhcs/tui/components/grouped_dropdown.py`
### No local dependencies found or file has issues.

## `openhcs/tui/components/interactive_list_item.py`
### No local dependencies found or file has issues.

## `openhcs/tui/components/loading_screen.py`
### Depends on:
- `openhcs.tui.components.spinner`

## `openhcs/tui/components/parameter_editor.py`
### No local dependencies found or file has issues.

## `openhcs/tui/components/spinner.py`
### No local dependencies found or file has issues.

## `openhcs/tui/dialogs/__init__.py`
### Depends on:
- `openhcs.tui.dialogs.base`
- `openhcs.tui.dialogs.manager`

## `openhcs/tui/dialogs/base.py`
### No local dependencies found or file has issues.

## `openhcs/tui/dialogs/global_settings_editor.py`
### Depends on:
- `openhcs.tui.utils.__init__`
- `openhcs.core.config`
- `openhcs.constants.constants`

## `openhcs/tui/dialogs/help_dialog.py`
### No local dependencies found or file has issues.

## `openhcs/tui/dialogs/manager.py`
### Depends on:
- `openhcs.tui.dialogs.base`

## `openhcs/tui/dialogs/plate_config_editor.py`
### Depends on:
- `openhcs.constants`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.core.config`
- `openhcs.tui.tui_architecture`

## `openhcs/tui/dialogs/plate_dialog_manager.py`
### Depends on:
- `openhcs.tui.file_browser`
- `openhcs.constants.constants`
- `openhcs.io.filemanager`

## `openhcs/tui/dual_step_func_editor.py`
### Depends on:
- `openhcs.constants.constants`
- `openhcs.core.steps.function_step`
- `openhcs.core.steps.abstract`
- `openhcs.tui.utils.__init__`
- `openhcs.tui.function_pattern_editor`

## `openhcs/tui/file_browser.py`
### Depends on:
- `openhcs.constants.constants`
- `openhcs.io.filemanager`

## `openhcs/tui/function_pattern_editor.py`
### Depends on:
- `openhcs.tui.components`
- `openhcs.constants.constants`
- `openhcs.processing.func_registry`
- `openhcs.core.pipeline.funcstep_contract_validator`
- `openhcs.tui.utils`
- `openhcs.tui.services.external_editor_service`

## `openhcs/tui/menu_bar.py`
### Depends on:
- `openhcs.core.steps.abstract`
- `openhcs.core.config`
- `openhcs.core.context.processing_context`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.tui.commands`
- `openhcs.tui.tui_architecture`

## `openhcs/tui/pipeline_editor.py`
### Depends on:
- `openhcs.core.steps.abstract`
- `openhcs.core.steps.function_step`
- `openhcs.tui.components.__init__`
- `openhcs.tui.utils.__init__`
- `openhcs.core.context.processing_context`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.tui.commands.__init__`

## `openhcs/tui/plate_manager_core.py`
### Depends on:
- `openhcs.tui.status_bar`
- `openhcs.constants.constants`
- `openhcs.io.base`
- `openhcs.tui.components.__init__`
- `openhcs.tui.commands`
- `openhcs.tui.services.plate_validation`
- `openhcs.core.context.processing_context`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.tui.dialogs.plate_dialog_manager`
- `openhcs.io.filemanager`

## `openhcs/tui/services/external_editor_service.py`
### Depends on:
- `openhcs.core.pipeline.funcstep_contract_validator`

## `openhcs/tui/services/plate_validation.py`
### Depends on:
- `openhcs.core.context.processing_context`
- `openhcs.io.filemanager`

## `openhcs/tui/status_bar.py`
### No local dependencies found or file has issues.

## `openhcs/tui/tui_architecture.py`
### Depends on:
- `openhcs.tui.status_bar`
- `openhcs.core.config`
- `openhcs.core.steps.function_step`
- `openhcs.core.steps.abstract`
- `openhcs.io.base`
- `openhcs.tui.plate_manager_core`
- `openhcs.processing.func_registry`
- `openhcs.tui.dialogs`
- `openhcs.tui.function_pattern_editor`
- `openhcs.tui.dialogs.plate_config_editor`
- `openhcs.tui.pipeline_editor`
- `openhcs.tui.dual_step_func_editor`
- `openhcs.tui.components.loading_screen`
- `openhcs.core.context.processing_context`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.tui.commands`
- `openhcs.io.filemanager`

## `openhcs/tui/tui_launcher.py`
### Depends on:
- `openhcs.core.config`
- `openhcs.core.steps.abstract`
- `openhcs.io.base`
- `openhcs.tui.tui_architecture`
- `openhcs.core.context.processing_context`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.io.filemanager`

## `openhcs/tui/utils.py`
### No local dependencies found or file has issues.

## `openhcs/tui/utils/__init__.py`
### Depends on:
- `openhcs.tui.utils.error_handling`
- `openhcs.tui.utils.dialog_helpers`

## `openhcs/tui/utils/dialog_helpers.py`
### No local dependencies found or file has issues.

## `openhcs/tui/utils/error_handling.py`
### No local dependencies found or file has issues.

