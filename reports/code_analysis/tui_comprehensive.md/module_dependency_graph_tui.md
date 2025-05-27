# Module Dependency Graph for `openhcs/tui`
Generated on: 2025-05-23 12:47:46

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
- `openhcs.tui.dialogs.global_settings_editor`
- `openhcs.tui.dialogs.plate_dialog_manager`
- `openhcs.tui.tui_architecture`
- `openhcs.constants.constants`
- `openhcs.processing.func_registry`
- `openhcs.core.context.processing_context`
- `openhcs.core.config`
- `openhcs.tui.utils`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.core.steps.function_step`
- `openhcs.core.steps.abstract`

## `openhcs/tui/commands/__init__.py`
### Depends on:
- `openhcs.tui.commands.plate_commands`
- `openhcs.tui.commands.pipeline_commands`
- `openhcs.tui.commands.registry`
- `openhcs.tui.commands.dialog_commands`

## `openhcs/tui/commands/dialog_commands.py`
### Depends on:
- `openhcs.tui.dialogs.global_settings_editor`
- `openhcs.core.context.processing_context`
- `openhcs.tui.tui_architecture`
- `openhcs.core.config`

## `openhcs/tui/commands/pipeline_commands.py`
### Depends on:
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.core.context.processing_context`
- `openhcs.tui.tui_architecture`

## `openhcs/tui/commands/plate_commands.py`
### Depends on:
- `openhcs.tui.dialogs.plate_dialog_manager`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.core.context.processing_context`
- `openhcs.tui.tui_architecture`

## `openhcs/tui/commands/registry.py`
### Depends on:
- `openhcs.tui.commands`

## `openhcs/tui/components.py`
### No local dependencies found or file has issues.

## `openhcs/tui/components/__init__.py`
### Depends on:
- `openhcs.tui.components.spinner`
- `openhcs.tui.components.interactive_list_item`
- `openhcs.tui.components.framed_button`
- `openhcs.tui.components.parameter_editor`
- `openhcs.tui.components.loading_screen`
- `openhcs.tui.components.grouped_dropdown`

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
- `openhcs.tui.tui_architecture`
- `openhcs.core.config`

## `openhcs/tui/dialogs/plate_dialog_manager.py`
### Depends on:
- `openhcs.tui.file_browser`
- `openhcs.constants.constants`
- `openhcs.io.filemanager`

## `openhcs/tui/dual_step_func_editor.py`
### Depends on:
- `openhcs.constants.constants`
- `openhcs.tui.utils.__init__`
- `openhcs.tui.function_pattern_editor`
- `openhcs.core.steps.function_step`
- `openhcs.core.steps.abstract`

## `openhcs/tui/file_browser.py`
### Depends on:
- `openhcs.constants.constants`
- `openhcs.io.filemanager`

## `openhcs/tui/function_pattern_editor.py`
### Depends on:
- `openhcs.constants.constants`
- `openhcs.processing.func_registry`
- `openhcs.tui.components`
- `openhcs.tui.utils`
- `openhcs.core.pipeline.funcstep_contract_validator`
- `openhcs.tui.services.external_editor_service`

## `openhcs/tui/menu_bar.py`
### Depends on:
- `openhcs.tui.tui_architecture`
- `openhcs.core.context.processing_context`
- `openhcs.core.config`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.tui.commands`
- `openhcs.core.steps.abstract`

## `openhcs/tui/pipeline_editor.py`
### Depends on:
- `openhcs.tui.utils.__init__`
- `openhcs.core.context.processing_context`
- `openhcs.tui.components.__init__`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.tui.commands.__init__`
- `openhcs.core.steps.function_step`
- `openhcs.core.steps.abstract`

## `openhcs/tui/plate_manager_core.py`
### Depends on:
- `openhcs.tui.dialogs.plate_dialog_manager`
- `openhcs.constants.constants`
- `openhcs.core.context.processing_context`
- `openhcs.io.base`
- `openhcs.io.filemanager`
- `openhcs.tui.components.__init__`
- `openhcs.tui.services.plate_validation`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.tui.commands`
- `openhcs.tui.status_bar`

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
- `openhcs.core.steps.abstract`
- `openhcs.tui.dialogs.plate_config_editor`
- `openhcs.processing.func_registry`
- `openhcs.tui.function_pattern_editor`
- `openhcs.core.config`
- `openhcs.io.base`
- `openhcs.io.filemanager`
- `openhcs.core.context.processing_context`
- `openhcs.tui.dual_step_func_editor`
- `openhcs.tui.commands`
- `openhcs.tui.plate_manager_core`
- `openhcs.tui.dialogs`
- `openhcs.tui.components.loading_screen`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.core.steps.function_step`
- `openhcs.tui.status_bar`
- `openhcs.tui.pipeline_editor`

## `openhcs/tui/tui_launcher.py`
### Depends on:
- `openhcs.tui.tui_architecture`
- `openhcs.core.context.processing_context`
- `openhcs.core.config`
- `openhcs.io.base`
- `openhcs.io.filemanager`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.core.steps.abstract`

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

