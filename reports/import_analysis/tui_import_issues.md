# Import Analysis

Found 19 files with import issues:

## openhcs/tui/components.py

### Missing Imports
These symbols are used but not imported:
```
- buffer
- buttons_container
- can_move_down
- can_move_up
- current_kwargs
- current_value
- default
- default_value
- display_text_func
- func
- func_index
- input_field
- is_selected
- is_special
- item_button
- item_data
- item_index
- label
- label_text
- move_buttons_children
- name
- new_kwargs
- on_move_down
- on_move_up
- on_parameter_change
- on_reset_all_parameters
- on_reset_parameter
- on_select
- on_text_changed_handler
- options
- p_info
- param
- param_fields
- param_type
- params_info
- params_list
- required
- reset_all_button
- reset_button
- result
- self
- sig
- style
- text
- text_area
- value
- value_str
```

### Unused Imports
These symbols are imported but not used:
```
- Dialog
- HTML
- KeyBindings
- ScrollablePane
- Tuple
- Union
- ast
```

### Module Structure Issues
| Line | Import | Issue |
| ---- | ------ | ----- |
| 7 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

## openhcs/tui/__init__.py

### Unused Imports
These symbols are imported but not used:
```
- FunctionPatternEditor
```

## openhcs/tui/file_browser.py

### Missing Imports
These symbols are used but not imported:
```
- Dialog
- FileManagerBrowser
- action_buttons
- app
- backend
- bottom_bar
- display_formatted_text
- e
- error_container
- event
- ext
- file_browser
- file_manager
- filter_extensions
- fm
- i
- icon
- idx
- index
- initial_path
- is_dir
- is_focused
- is_selected_for_multi
- item
- item_button
- item_info
- item_name
- item_path
- items_ui
- kb
- layout
- logger
- loop
- main_async
- max_name_width
- mtime
- mtime_part
- name_part
- nav_buttons
- on_cancel
- on_cancel_cb
- on_path_selected
- on_path_selected_cb
- parent_path
- path
- path_to_check
- path_window
- prefix
- processed_listing
- raw_items
- sample_size
- select_files
- select_multiple
- selected_item_info
- selected_path
- selected_paths
- self
- show_hidden_files
- size
- size_bytes
- size_part
- stat_exc
- stats
- t
- text_fragments
- timestamp
- unit
- x
```

### Unused Imports
These symbols are imported but not used:
```
- TextArea
```

### Module Structure Issues
| Line | Import | Issue |
| ---- | ------ | ----- |
| 14 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

## openhcs/tui/tui_launcher.py

### Missing Imports
These symbols are used but not imported:
```
- common_output_directory
- core_global_config
- e
- logger
- new_config
- orchestrator
- plate_id
- plate_info
- plate_path_str
- removed_orchestrator
- safe_plate_id_for_path
- self
- tui_app
- tui_config_path
- workspace_path_for_plate
```

## openhcs/tui/dual_step_func_editor.py

### Missing Imports
These symbols are used but not imported:
```
- actual_type
- associated_widget
- buff
- converted_value
- current_name
- current_text
- current_value
- dynamic_content_area
- e
- editing_val
- enum_class
- f
- field_label
- file_path
- file_path_str
- func_step
- get_current_view_container
- has_changed
- initial_selection
- is_optional
- load_step_button
- loaded_object
- logger
- member
- menu_bar
- n
- name_to_check
- options
- original_name
- original_val
- original_value
- p_name
- param_name
- param_name_to_reset
- param_obj
- param_type_hint
- parameter_fields_container
- reset_button
- rows
- save_step_as_button
- selected_enum_member
- self
- sig
- sig_abstract
- state
- step_settings_toolbar
- step_to_save
- t
- val
- view_content
- view_name
- w
- widget
- widget_value
```

### Unused Imports
These symbols are imported but not used:
```
- Callable
- Dialog
- Enum
- FormattedTextControl
- KeyBindings
- List
- asyncio
```

### Module Structure Issues
| Line | Import | Issue |
| ---- | ------ | ----- |
| 21 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

## openhcs/tui/__main__.py

### Missing Imports
These symbols are used but not imported:
```
- Optional
- USER_CONFIG_DIR
- USER_GLOBAL_CONFIG_FILE
- args
- core_global_config
- current_pp
- current_vfs
- default_global_conf_for_top_level
- default_pp
- default_vfs
- e
- f
- final_pp_args
- final_vfs_args
- launcher
- loaded_data
- log_level
- logger
- main
- parser
- pp_data
- vfs_data
```

### Unused Imports
These symbols are imported but not used:
```
- os
```

## openhcs/tui/menu_bar.py

### Missing Imports
These symbols are used but not imported:
```
- LayoutContract
- MenuItem
- MenuItemSchema
- MenuItemType
- MenuStructureSchema
- MissingStateError
- ReentrantLock
- StateConditionRegistry
- StateConditionType
- _DEFAULT_MENU_STRUCTURE
- active_orchestrator
- app
- attribute_name
- checked
- child
- children
- cls
- condition_name
- container
- context
- create_handler
- current_index
- current_pipeline
- current_task
- current_valid_index
- delta
- e
- enabled
- erase_bg
- err_msg
- f
- filename
- handler
- handler_map
- handler_name
- i
- item
- item_dict
- item_mouse_handler
- item_type
- item_type_str
- items
- json_content
- kb
- key
- label
- label_text
- labels
- logger
- max_available_height
- max_available_width
- menu
- menu_active
- menu_data
- menu_item
- menu_items
- menu_mouse_handler
- menu_name
- menu_names
- menu_structure
- mnemonic
- mouse_event
- mouse_handlers
- msg
- new_index
- new_valid_index
- original_length
- original_mouse_handler
- padding
- parent_style
- pipeline_definition
- pipeline_dicts
- plate_dir_path
- plate_dir_path_str
- raw_structure
- save_fail_msg
- save_path
- save_success_msg
- screen
- selected_plate
- selected_step
- selected_step_dict
- self
- shortcut
- state
- step
- step_uid_to_remove
- structure
- submenu_active
- submenu_container
- test_plate_backend
- test_plate_path_str
- test_plate_relative_path
- theme
- valid_indices
- width
- write_position
- z_index
```

### Unused Imports
These symbols are imported but not used:
```
- AnyContainer
- ConditionalContainer
- FormattedTextControl
- GlobalPipelineConfig
- HTML
- PipelineOrchestrator
- ProcessingContext
- TUIState
- Tuple
- Window
- field
- has_focus
```

### Module Structure Issues
| Line | Import | Issue |
| ---- | ------ | ----- |
| 31 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

## openhcs/tui/utils.py

### Missing Imports
These symbols are used but not imported:
```
- accept_path
- app
- app_state
- buff
- cancel_dialog
- dialog
- error_dialog
- future
- initial_value
- logger
- message
- path_text
- path_text_area
- prompt_message
- title
```

## openhcs/tui/function_pattern_editor.py

### Missing Imports
These symbols are used but not imported:
```
- add_func_button
- add_function_button
- add_key_button
- backend
- backend_funcs
- change_callback
- convert_to_dict_button
- current_func
- current_kwargs
- current_value
- default
- default_func
- default_val
- default_value
- delete_button
- display_keys
- display_name
- dropdown
- dropdown_options
- e
- edit_in_vim_button
- f
- file_path
- file_path_str
- func
- func_dropdown
- func_index
- func_info
- func_item
- funcs
- function_items
- functions
- functions_by_backend
- get_function_info
- i
- idx
- index
- info
- initial_content
- initial_func_for_editor
- initial_kwargs_for_editor
- initial_pattern
- input_field
- is_special
- item
- item_param_editor
- k
- key
- key_dropdown
- key_management_buttons
- kwargs
- label
- label_text
- load_func_button
- loaded_pattern
- logger
- message
- move_down
- move_up
- name
- new_key
- new_kwargs
- new_pattern
- new_value_str
- on_key_change
- on_selection_change
- p
- p_name
- p_val_str
- param
- param_editor_container
- param_fields
- param_name
- params
- params_with_defaults_info
- parsed_value
- pattern
- remove_key_button
- required
- reset_all_button
- reset_button
- save_as_func_button
- self
- sig
- state
- step
- success
- title
- v
```

### Unused Imports
These symbols are imported but not used:
```
- Dialog
- TextArea
- asyncio
```

### Module Structure Issues
| Line | Import | Issue |
| ---- | ------ | ----- |
| 26 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

## openhcs/tui/plate_manager_core.py

### Missing Imports
These symbols are used but not imported:
```
- CompilePlatesCommand
- DeleteSelectedPlatesCommand
- InitializePlatesCommand
- RunPlatesCommand
- ShowAddPlateDialogCommand
- ShowEditPlateConfigDialogCommand
- added_plate_details
- app
- backend
- can_move_down
- can_move_up
- context
- current_selection_valid
- data
- delta
- detail
- details
- e
- error_message
- existing_plate_index
- fm
- i
- ids_to_remove
- idx
- index
- is_selected
- is_valid_selection
- item_widget
- item_widgets
- kb
- list_container_focused
- logger
- loop
- message
- name
- new_index
- new_plate_entry
- new_status
- notify_state
- num_removed
- orchestrator
- orchestrators
- original_length
- p
- path
- path_str
- paths_to_process
- plate
- plate_data
- plate_dict
- plate_entry
- plate_id
- plate_id_to_remove
- plate_in_list
- plate_status
- plate_to_move
- plate_to_remove
- plate_to_select
- plate_tui_id
- plates_to_delete_data
- prefix
- raw_path
- removed_plate_detail
- removed_plate_details
- result
- selected_data
- self
- state
- status_symbol
- updated
- validated_plate
- vim_mode_condition
```

### Unused Imports
These symbols are imported but not used:
```
- Callable
- Tuple
- Union
- os
- shutil
- signal
```

### Module Structure Issues
| Line | Import | Issue |
| ---- | ------ | ----- |
| 38 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

## openhcs/tui/tui_architecture.py

### Missing Imports
These symbols are used but not imported:
```
- Clause5Violation
- TUIState
- attr_name
- callback
- component
- component_attributes
- data
- e
- erase_bg
- event
- event_type
- global_config
- initial_context
- is_editing_plate_config
- is_editing_step_config
- kb
- logger
- max_available_height
- max_available_width
- mouse_event
- mouse_handlers
- new_core_config
- orchestrator
- orchestrator_to_edit_config
- parent_style
- plate
- screen
- self
- state
- step
- step_to_edit_config
- width
- write_position
- z_index
```

### Unused Imports
These symbols are imported but not used:
```
- Button
- FUNC_REGISTRY
- FileManager
- Float
- FloatContainer
- PipelineOrchestrator
- TextArea
- Union
- has_focus
```

### Module Structure Issues
| Line | Import | Issue |
| ---- | ------ | ----- |
| 61 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

## openhcs/tui/commands.py

### Missing Imports
These symbols are used but not imported:
```
- Command
- active_orchestrator
- compiled_pipeline_data
- confirm_dialog
- context
- default_filename
- dialog
- e
- f
- file_path
- file_path_str
- fm
- help_text
- ids_to_delete
- item
- kwargs
- loaded_pipeline
- logger
- loop
- new_pipeline
- new_step_instance
- orchestrator
- original_pipeline
- p
- pipeline_data_to_save
- plate_id
- plate_names
- plate_path
- result
- result_config
- s
- save_path
- selected_orchestrators
- selected_plates_data
- selected_step_data
- selected_steps_data
- self
- single_selected_step
- state
- step
- step_data
- step_names
- step_obj
- valid_pipeline
```

### Unused Imports
These symbols are imported but not used:
```
- List
- PipelineOrchestrator
- PlateDialogManager
- ProcessingContext
- TUIState
- get_app
- uuid
```

## openhcs/tui/step_viewer.py

### Missing Imports
These symbols are used but not imported:
```
- AddStepCommand
- DeleteSelectedStepsCommand
- LoadPipelineCommand
- SavePipelineCommand
- ShowEditStepDialogCommand
- active_orchestrator
- actual_step_instance
- can_move_down
- can_move_up
- cls
- context
- current_orchestrator_idx
- current_step_dict
- current_step_id
- current_step_pipeline_id
- data
- e
- f
- field
- file_path
- file_path_str
- func
- func_display_name
- func_name
- i
- icons
- index
- instance
- invalid_steps_from_context
- is_selected
- item_widget
- item_widgets
- kb
- list_container_focused
- logger
- memory_type
- message_dialog
- missing_fields
- name
- next_orchestrator_idx
- next_step_dict
- next_step_id
- next_step_pipeline_id
- p
- pipeline
- pipeline_to_save
- plate
- plate_id
- prefix
- prev_orchestrator_idx
- prev_step_dict
- prev_step_id
- prev_step_pipeline_id
- raw_step_objects
- required_fields
- saved_step
- selected_step_data
- self
- state
- status
- status_icon
- step
- step_data
- step_dict
- step_id_to_edit
- step_in_pipeline
- step_obj
- temp_func_dict
- transformed_steps
```

### Unused Imports
These symbols are imported but not used:
```
- TextArea
- uuid
```

### Module Structure Issues
| Line | Import | Issue |
| ---- | ------ | ----- |
| 10 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

## openhcs/tui/status_bar.py

### Missing Imports
These symbols are used but not imported:
```
- LogFormatter
- LogLevel
- Priority
- STATUS_ICONS
- StatusBar
- StatusBarSchema
- StatusBarState
- TUIStatusBarLogHandler
- app
- cls
- data
- dataclasses
- details
- entries
- entry
- erase_bg
- error_data
- expanded
- field_name
- formatted_entry_tuples
- get_display_text
- handler
- l
- label
- level
- level_enum
- level_no
- level_str
- log_level
- log_level_enum
- log_level_str
- log_message
- log_msg
- logger
- logger_to_adjust
- logging_level
- max_available_height
- max_available_width
- max_log_entries
- message
- mouse_event
- mouse_handlers
- msg
- new_buffer
- new_level_str
- new_state
- op_status_icon
- operation_status
- parent_style
- priority
- record
- required_fields
- result_fragments
- root_logger
- screen
- segments
- self
- source
- src
- status_bar_instance
- style_class
- ts
- tui_state
- width
- write_position
- z_index
```

### Unused Imports
These symbols are imported but not used:
```
- Box
```

### Module Structure Issues
| Line | Import | Issue |
| ---- | ------ | ----- |
| 39 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

## openhcs/tui/dialogs/global_settings_editor.py

### Missing Imports
These symbols are used but not imported:
```
- Float
- app
- b
- body_content
- cancel_button
- e
- float_
- future
- initial_config
- logger
- m
- ok_button
- previous_focus
- self
- state
```

### Unused Imports
These symbols are imported but not used:
```
- Checkbox
- PathPlanningConfig
- VFSConfig
- VSplit
```

## openhcs/tui/dialogs/plate_config_editor.py

### Missing Imports
These symbols are used but not imported:
```
- List
- Union
- arg
- args
- base_config
- buttons
- config_obj
- config_widgets_list
- converted_value
- current_value
- current_value_type
- current_widget_value
- dc_field
- e
- enum_values
- f
- field_meta
- field_name
- field_path
- field_type
- full_path
- handler_key
- input_widget
- int_accept_handler
- is_optional
- key
- label
- label_text
- logger
- member
- new_value
- notify_apply_error
- notify_cancel
- notify_error_no_config_attr
- notify_save_success
- obj_ptr
- orchestrator
- parent_path
- part
- parts
- path_capture
- self
- state
- str_accept_handler
- text_val
- title_text
- widget
- widget_capture
- widget_identifier
- widget_type
- widgets
```

### Unused Imports
These symbols are imported but not used:
```
- Dialog
- DynamicContainer
- FormattedTextControl
- GlobalPipelineConfig
- Microscope
- PathPlanningConfig
- PipelineOrchestrator
- TUIState
- VFSConfig
```

### Module Structure Issues
| Line | Import | Issue |
| ---- | ------ | ----- |
| 14 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

## openhcs/tui/dialogs/plate_dialog_manager.py

### Missing Imports
These symbols are used but not imported:
```
- DialogResultCallback
- ErrorBanner
- ErrorCallback
- app
- backend
- body
- browser_on_cancel
- browser_on_path_selected
- child
- container
- default_backend
- details
- dialog
- dlg
- e_focus
- e_focus_cancel
- erase_bg
- error_banner
- file_browser_component
- file_manager
- float_container
- future
- logger
- max_available_height
- max_available_width
- message
- message_label
- mouse_handlers
- on_add_dialog_result
- on_error
- on_remove_dialog_result
- p
- p_str
- parent_style
- path_input
- path_obj
- path_strs
- paths
- paths_text
- plate
- previous_focus
- result
- result_path
- screen
- selected_data
- selected_paths
- self
- width
- write_position
- z_index
```

### Unused Imports
These symbols are imported but not used:
```
- Coroutine
- Dropdown
- is_done
```

### Module Structure Issues
| Line | Import | Issue |
| ---- | ------ | ----- |
| 29 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

## openhcs/tui/services/external_editor_service.py

### Missing Imports
These symbols are used but not imported:
```
- content
- dialog
- e
- editor
- error_message
- f
- initial_content
- is_valid
- message
- modified_content
- pattern
- pattern_str
- self
- state
- stmt
- tmp_file
- tmp_file_path
- tree
```

### Unused Imports
These symbols are imported but not used:
```
- subprocess
```

## openhcs/tui/services/plate_validation.py

### Missing Imports
These symbols are used but not imported:
```
- ErrorCallback
- ValidationResultCallback
- backend
- combined
- context
- e
- error_details
- hash_obj
- io_executor
- is_valid
- loop
- on_error
- on_validation_result
- path
- plate
- plate_id
- plate_name
- resolved_path
- self
- standardized_path
- storage_registry
```

### Unused Imports
These symbols are imported but not used:
```
- Callable
```
