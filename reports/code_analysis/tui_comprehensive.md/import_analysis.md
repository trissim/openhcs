# Import Analysis

Found 38 files with import issues:

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
- display_text
- display_text_func
- full_text
- func
- func_index
- index_display
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
| 1 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |
| 8 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

## openhcs/tui/__init__.py

### Unused Imports
These symbols are imported but not used:
```
- FunctionPatternEditor
```

## openhcs/tui/pipeline_editor.py

### Missing Imports
These symbols are used but not imported:
```
- PlaceholderCommand
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
- missing_fields
- name
- next_orchestrator_idx
- next_step_dict
- next_step_id
- next_step_pipeline_id
- output_memory_type
- p
- pipeline
- pipeline_to_save
- plate
- plate_id
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
- Box
- Dialog
- PipelineOrchestrator
- TextArea
- uuid
```

### Module Structure Issues
| Line | Import | Issue |
| ---- | ------ | ----- |
| 9 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

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
- f
- logger
- new_config
- orchestrator
- pipeline_file_path
- pipeline_loaded
- plate_id
- plate_info
- plate_path_str
- removed_orchestrator_instance
- safe_plate_id_for_path
- self
- step
- tui_app
- tui_config_path
- workspace_path_for_plate
```

### Unused Imports
These symbols are imported but not used:
```
- get_app
```

## openhcs/tui/dual_step_func_editor.py

### Missing Imports
These symbols are used but not imported:
```
- actual_type
- associated_widget
- buff
- changed_in_step_settings
- converted_value
- current_name
- current_text
- current_value
- e
- editing_val
- enum_class
- f
- field_label
- file_path
- file_path_str
- func_changed
- func_step
- get_current_view_container
- has_changed
- initial_selection
- is_optional
- load_step_button
- loaded_object
- logger
- member
- mouse_event
- n
- name_to_check
- new_mouse_handler
- options
- original_handler
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
- result
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
- top_menu_bar
- val
- view_content
- view_name
- w
- widget
- widget_instance
- widget_value
```

### Unused Imports
These symbols are imported but not used:
```
- Callable
- CheckboxList
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
| 1 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |
| 22 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

## openhcs/tui/__main__.py

### Missing Imports
These symbols are used but not imported:
```
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
- file_handler
- final_pp_args
- final_vfs_args
- launcher
- loaded_data
- log_dir
- log_file
- log_level
- logger
- main
- parser
- pp_data
- root_logger
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
- focus_text_area
- future
- initial_value
- logger
- message
- ok_handler
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
| 1 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |
| 27 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

## openhcs/tui/plate_manager_core.py

### Missing Imports
These symbols are used but not imported:
```
- PlaceholderCommand
- added_plate_details
- app
- backend
- can_move_down
- can_move_up
- common_output_directory
- context
- current_selection_valid
- data
- delta
- detail
- details
- e
- error_details
- error_message
- existing_plate_index
- fm
- i
- ids_to_remove
- idx
- index
- is_selected
- is_valid
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
- paths
- paths_to_process
- plate
- plate_data
- plate_detail
- plate_dict
- plate_entry
- plate_id
- plate_id_to_remove
- plate_in_list
- plate_name
- plate_paths
- plate_status
- plate_to_move
- plate_to_remove
- plate_to_select
- plate_tui_id
- plates_to_delete_data
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
- Box
- Callable
- FileManager
- STATUS_ICONS
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
- app
- attr_name
- buttons_container
- callback
- component
- component_attributes
- content
- content_container
- data
- dialog
- e
- elapsed_time
- erase_bg
- event
- event_type
- float_container
- global_config
- initial_context
- is_editing_plate_config
- is_editing_step_config
- kb
- logger
- main_content
- main_layout
- max_available_height
- max_available_width
- max_wait_time
- mouse_event
- mouse_handlers
- new_core_config
- orchestrator
- orchestrator_to_edit_config
- parent_style
- plate
- previous_layout
- result
- result_future
- screen
- self
- state
- status_bar
- step
- step_data
- step_to_edit_config
- title
- top_bar
- wait_interval
- width
- write_position
- z_index
```

### Unused Imports
These symbols are imported but not used:
```
- AbstractStep
- FUNC_REGISTRY
- FileManager
- PipelineOrchestrator
- TextArea
- Union
- has_focus
```

### Module Structure Issues
| Line | Import | Issue |
| ---- | ------ | ----- |
| 68 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

## openhcs/tui/commands.py

### Missing Imports
These symbols are used but not imported:
```
- Command
- SHARED_EXECUTOR
- active_orchestrator
- backend_name
- compiled_pipeline_data
- confirm_dialog
- context
- default_filename
- default_func_name
- default_func_pattern
- dialog
- e
- f
- file_path
- file_path_str
- first_func_name
- fm
- funcs_in_backend
- help_text
- ids_to_delete
- item
- kwargs
- loaded_pipeline
- logger
- loop
- new_pipeline
- new_step
- orchestrator
- original_pipeline
- p
- pipeline_data_to_save
- plate_id
- plate_name
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
| 24 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |
| 41 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

## openhcs/tui/dialogs/global_settings_editor.py

### Missing Imports
These symbols are used but not imported:
```
- actual_type
- all_rows
- app
- config_obj
- create_checkbox_mouse_handler
- current_fields
- current_radio_value
- current_value
- current_widget_value
- dialog_height
- dialog_width
- e
- elem
- enum_current_value
- f
- field_annotation
- field_def
- field_definitions
- field_label_text
- field_name
- field_obj
- field_path
- field_path_parts
- field_type
- field_type_hint
- float_
- future
- i
- initial_config
- is_changed
- is_last_part
- is_optional
- is_still_in_layout
- item
- last_part
- logger
- member
- model_instance
- model_type
- mouse_event
- nested_prefix
- new_mouse_handler
- non_none_args
- obj
- obj_type
- options
- origin_type
- original_mouse_handler
- original_obj_segment
- original_value
- part
- part_name
- parts
- path
- path_parts
- prefix
- previous_focus
- res
- rows
- scrollable_body
- self
- state
- t
- temp_validation_config
- text
- title_label
- val
- value
- widget
```

### Unused Imports
These symbols are imported but not used:
```
- Backend
- Microscope
- MicroscopeConfig
- PathPlanningConfig
- VFSConfig
- show_error_dialog
```

## openhcs/tui/dialogs/__init__.py

### Unused Imports
These symbols are imported but not used:
```
- BaseDialog
- DialogManager
- dialog_manager
- initialize_dialog_manager
```

## openhcs/tui/dialogs/manager.py

### Missing Imports
These symbols are used but not imported:
```
- DialogManager
- dialog
- dialog_id
- e
- logger
- self
- state
```

### Unused Imports
These symbols are imported but not used:
```
- Generic
```

## openhcs/tui/dialogs/plate_config_editor.py

### Missing Imports
These symbols are used but not imported:
```
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
| 1 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |
| 15 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

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

## openhcs/tui/dialogs/help_dialog.py

### Missing Imports
These symbols are used but not imported:
```
- DEFAULT_HELP_TEXT
- app
- e
- elem
- float_
- help_text
- is_still_in_layout
- logger
- ok_button
- previous_focus
- self
```

## openhcs/tui/dialogs/base.py

### Missing Imports
These symbols are used but not imported:
```
- T
- e
- logger
- result
- self
- title
```

### Unused Imports
These symbols are imported but not used:
```
- Any
- Awaitable
- Callable
- Dict
- KeyBindings
- Layout
- List
- Type
- Union
```

## openhcs/tui/components/interactive_list_item.py

### Missing Imports
These symbols are used but not imported:
```
- buttons_container
- can_move_down
- can_move_up
- container_height
- container_width
- display_text_func
- down_box
- down_style
- erase_bg
- is_selected
- item_data
- item_index
- main_content
- max_available_height
- max_available_width
- mouse_event
- mouse_handlers
- on_move_down
- on_move_up
- on_select
- parent_style
- screen
- self
- style
- up_box
- up_style
- width
- write_position
- z_index
```

### Unused Imports
These symbols are imported but not used:
```
- Condition
- KeyBindings
- KeyBindingsBase
- List
- MouseHandlers
- Screen
- WritePosition
```

### Module Structure Issues
| Line | Import | Issue |
| ---- | ------ | ----- |
| 15 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

## openhcs/tui/components/__init__.py

### Unused Imports
These symbols are imported but not used:
```
- FramedButton
- GroupedDropdown
- InteractiveListItem
- LoadingScreen
- ParameterEditor
- Spinner
```

## openhcs/tui/components/framed_button.py

### Missing Imports
These symbols are used but not imported:
```
- button_window
- control
- handler
- mouse_event
- mouse_handler
- self
- style
- text
- width
```

### Unused Imports
These symbols are imported but not used:
```
- Button
- HSplit
- VSplit
```

### Module Structure Issues
| Line | Import | Issue |
| ---- | ------ | ----- |
| 8 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

## openhcs/tui/components/spinner.py

### Missing Imports
These symbols are used but not imported:
```
- app
- control
- done_callback
- erase_bg
- interval
- mouse_event
- mouse_handlers
- parent_style
- screen
- self
- spinner_chars
- spinner_text
- style
- window
- write_position
- z_index
```

### Unused Imports
These symbols are imported but not used:
```
- KeyBindingsBase
- MouseHandlers
- Screen
- WritePosition
- time
```

### Module Structure Issues
| Line | Import | Issue |
| ---- | ------ | ----- |
| 12 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

## openhcs/tui/components/loading_screen.py

### Missing Imports
These symbols are used but not imported:
```
- container_height
- container_width
- erase_bg
- logger
- max_available_height
- max_available_width
- message
- mouse_event
- mouse_handlers
- on_complete
- parent_style
- screen
- self
- width
- write_position
- z_index
```

### Unused Imports
These symbols are imported but not used:
```
- FormattedTextControl
- KeyBindingsBase
- List
- MouseHandlers
- Screen
- WritePosition
- asyncio
```

### Module Structure Issues
| Line | Import | Issue |
| ---- | ------ | ----- |
| 12 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

## openhcs/tui/components/grouped_dropdown.py

### Missing Imports
These symbols are used but not imported:
```
- container_height
- container_width
- default
- erase_bg
- group_name
- handler
- label
- max_available_height
- max_available_width
- min_height
- mouse_event
- mouse_handlers
- on_change
- options
- options_by_group
- parent_style
- screen
- self
- value
- width
- write_position
- z_index
```

### Unused Imports
These symbols are imported but not used:
```
- Button
- HSplit
- KeyBindingsBase
- Label
- MouseHandlers
- Screen
- Union
- VSplit
- Window
- WritePosition
- get_app
```

### Module Structure Issues
| Line | Import | Issue |
| ---- | ------ | ----- |
| 12 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

## openhcs/tui/components/parameter_editor.py

### Missing Imports
These symbols are used but not imported:
```
- accept_handler
- buffer
- container_height
- container_width
- current_kwargs
- current_value
- current_value_str
- default_value
- erase_bg
- func
- func_index
- kwargs
- max_available_height
- max_available_width
- min_height
- mouse_event
- mouse_handlers
- name
- on_parameter_change
- on_reset_all_parameters
- on_reset_parameter
- param
- param_container
- param_count
- parent_style
- reset_all_button
- reset_button
- screen
- self
- sig
- text_area
- width
- write_position
- z_index
```

### Unused Imports
These symbols are imported but not used:
```
- HTML
- KeyBindingsBase
- List
- MouseHandlers
- Screen
- Tuple
- Union
- Window
- WritePosition
```

### Module Structure Issues
| Line | Import | Issue |
| ---- | ------ | ----- |
| 13 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

## openhcs/tui/commands/__init__.py

### Unused Imports
These symbols are imported but not used:
```
- CommandRegistry
- CompilePlatesCommand
- DeletePlateCommand
- DeleteSelectedPlatesCommand
- InitializePlatesCommand
- RunPlatesCommand
- ShowAddPlateDialogCommand
- ShowEditPlateConfigDialogCommand
- ShowEditPlateDialogCommand
- ShowGlobalSettingsDialogCommand
- ShowHelpCommand
- command_registry
```

## openhcs/tui/commands/pipeline_commands.py

### Missing Imports
These symbols are used but not imported:
```
- compiled_pipeline_data
- context
- e
- kwargs
- logger
- orchestrator
- orchestrators_to_compile
- orchestrators_to_init
- orchestrators_to_run
- selected_plates_data
- self
- state
```

### Unused Imports
These symbols are imported but not used:
```
- Dict
- List
- Optional
- PipelineOrchestrator
- ProcessingContext
- TUIState
- get_app
```

## openhcs/tui/commands/registry.py

### Missing Imports
These symbols are used but not imported:
```
- CommandRegistry
- command
- command_id
- command_type
- context
- e
- handler
- logger
- self
- state
```

### Unused Imports
These symbols are imported but not used:
```
- Command
- Generic
- List
- Union
```

## openhcs/tui/commands/dialog_commands.py

### Missing Imports
These symbols are used but not imported:
```
- context
- dialog
- help_text
- result_config
- self
- state
```

### Unused Imports
These symbols are imported but not used:
```
- Optional
- ProcessingContext
- TUIState
- get_app
```

## openhcs/tui/commands/plate_commands.py

### Missing Imports
These symbols are used but not imported:
```
- context
- plate_dialog_manager
- plate_name
- result
- self
- state
```

### Unused Imports
These symbols are imported but not used:
```
- Dict
- List
- Optional
- Path
- PipelineOrchestrator
- ProcessingContext
- TUIState
- get_app
- os
```

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

## openhcs/tui/utils/dialog_helpers.py

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
- focus_text_area
- future
- handler
- initial_value
- logger
- message
- ok_handler
- path_text
- path_text_area
- prompt_message
- text_area
- title
```

## openhcs/tui/utils/__init__.py

### Unused Imports
These symbols are imported but not used:
```
- cancel_dialog
- focus_text_area
- handle_async_errors
- handle_async_errors_decorator
- ok_handler
- prompt_for_path_dialog
- show_error_dialog
```

## openhcs/tui/utils/error_handling.py

### Missing Imports
These symbols are used but not imported:
```
- T
- app_state
- args
- coro
- decorator
- e
- error_message
- error_types
- func
- handle_async_errors
- kwargs
- logger
- message
- notify_state
- operation_name
- source
- state
- title
- wrapper
```

### Unused Imports
These symbols are imported but not used:
```
- Dict
- traceback
```
