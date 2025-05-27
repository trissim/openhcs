# Import Analysis

Found 118 files with import issues:

## openhcs/microscopes/__init__.py

### Unused Imports
These symbols are imported but not used:
```
- ImageXpressFilenameParser
- ImageXpressMetadataHandler
- OperaPhenixFilenameParser
- OperaPhenixMetadataHandler
```

## openhcs/microscopes/opera_phenix.py

### Missing Imports
These symbols are used but not imported:
```
- OperaPhenixFilenameParser
- OperaPhenixMetadataHandler
- OperaPhenixXmlParser
- basename
- channel
- channel_str
- cls
- col
- common_dir
- dest_path
- e
- entries
- entry
- entry_lower
- ext
- extension
- f
- field
- field_id
- field_mapping
- fields
- file_name
- file_path
- filemanager
- filename
- grid_size
- i
- image_dir
- image_files
- index_xml
- logger
- mapping
- match
- measurement_dir
- metadata
- new_field_id
- new_name
- new_path
- original_field_id
- parse_comp
- pattern_format
- pixel_size
- plate_path
- result
- root
- row
- s
- self
- site
- site_padding
- site_part
- site_str
- temp_dir
- temp_file
- temp_file_name
- temp_files
- tree
- well
- workspace_path
- x
- xml_parser
- xml_path
- y
- z_index
- z_padding
- z_part
- z_str
```

### Unused Imports
These symbols are imported but not used:
```
- Tuple
```

## openhcs/microscopes/opera_phenix_xml_parser.py

### Missing Imports
These symbols are used but not imported:
```
- OperaPhenixXmlContentError
- OperaPhenixXmlError
- OperaPhenixXmlParseError
- attr_name
- channel_elem
- channel_id_text
- col
- col_elem
- col_text
- e
- elem
- epsilon
- field_elem
- field_id
- field_id_elem
- field_id_text
- field_positions
- grid
- grid_size
- group_key
- i
- image
- image_data
- image_elements
- image_elems
- image_groups
- image_id
- image_info
- images
- img
- j
- logger
- mapping
- match
- max_field_id
- num_x_positions
- num_y_positions
- parent_elem
- pixel_size
- plane_elem
- plane_id_text
- plate_columns_text
- plate_elem
- plate_info
- plate_rows_text
- pos
- pos_x
- pos_x_elem
- pos_y
- pos_y_elem
- positions
- resolution_x
- resolution_y
- row
- row_elem
- row_fields
- row_str
- row_text
- self
- sorted_field_ids
- sorted_images
- tag_name
- unique_x
- unique_y
- well
- well_elems
- well_id
- well_positions
- x
- x_coords
- x_idx
- x_positions
- x_range
- x_value
- xml_path
- y
- y_coords
- y_idx
- y_positions
- y_value
```

## openhcs/microscopes/microscope_base.py

### Missing Imports
These symbols are used but not imported:
```
- MICROSCOPE_HANDLERS
- MicroscopeHandler
- _auto_detect_microscope_type
- backend
- channel
- dir_name
- directory
- directory_path
- e
- entries
- entry
- entry_path
- extension
- extensions
- file_path
- filemanager
- filename
- folder_path
- group_by
- handler
- handler_class
- image_dir
- image_files
- item
- item_name
- logger
- metadata
- metadata_handler
- microscope_type
- msg
- new_name
- new_path
- original_name
- original_path
- parent_dir
- parser
- pattern
- pattern_engine
- pattern_format
- patterns_by_well
- plate_folder
- plate_path
- prepared_dir
- rename_map
- self
- site
- site_padding
- subdirs
- supported_types
- variable_components
- well
- well_filter
- width
- workspace_path
- z_index
- z_padding
```

## openhcs/microscopes/imagexpress.py

### Missing Imports
These symbols are used but not imported:
```
- ImageXpressFilenameParser
- ImageXpressMetadataHandler
- base_name
- basename
- channel
- channel_str
- cls
- cols_match
- common_dir_found
- components
- context
- d
- desc
- dir_name
- directory
- e
- entries
- entry
- entry_path
- ext
- extension
- f
- filemanager
- filename
- first_file
- first_image_path
- fm
- grid_size_x
- grid_size_y
- htd_content
- htd_file
- htd_files
- image_files
- img_file
- img_file_name
- img_files
- logger
- match
- new_name
- new_path
- parse_comp
- parts
- pattern_format
- plate_path
- potential_z_folders
- result
- rows_match
- s
- self
- site
- site_padding
- site_str
- subdir
- subdirs
- tif
- well
- workspace_path
- x
- z_dir
- z_folders
- z_index
- z_padding
- z_str
- zstep_pattern
```

## openhcs/microscopes/microscope_interfaces.py

### Missing Imports
These symbols are used but not imported:
```
- _auto_detect_microscope_type
- e
- filemanager
- handler
- handler_class
- logger
- microscope_type
- msg
- pattern_format
- plate_folder
- supported_types
```

## openhcs/runtime/napari_stream_visualizer.py

### Missing Imports
These symbols are used but not imported:
```
- SHUTDOWN_SENTINEL
- backend
- cpu_tensor
- data
- display_data
- display_slice
- e
- e_conv
- e_load
- filemanager
- item
- item_to_queue
- layer_name
- loaded_data
- logger
- metadata
- path
- self
- slicer
- step_id
- step_id_for_log
- viewer_title
- well_id
```

### Unused Imports
These symbols are imported but not used:
```
- List
```

## openhcs/constants/__init__.py

### Unused Imports
These symbols are imported but not used:
```
- Backend
- CPU_MEMORY_TYPES
- Clause
- DEFAULT_ASSEMBLER_LOG_LEVEL
- DEFAULT_BACKEND
- DEFAULT_CPU_THREAD_COUNT
- DEFAULT_GROUP_BY
- DEFAULT_IMAGE_EXTENSION
- DEFAULT_IMAGE_EXTENSIONS
- DEFAULT_INTERPOLATION_MODE
- DEFAULT_INTERPOLATION_ORDER
- DEFAULT_MARGIN_RATIO
- DEFAULT_MAX_SHIFT
- DEFAULT_MICROSCOPE
- DEFAULT_NUM_WORKERS
- DEFAULT_OUT_DIR_SUFFIX
- DEFAULT_PIXEL_SIZE
- DEFAULT_POSITIONS_DIR_SUFFIX
- DEFAULT_RECURSIVE_PATTERN_SEARCH
- DEFAULT_SITE_PADDING
- DEFAULT_STITCHED_DIR_SUFFIX
- DEFAULT_TILE_OVERLAP
- DEFAULT_VARIABLE_COMPONENTS
- FORCE_DISK_WRITE
- GPU_MEMORY_TYPES
- GroupBy
- MEMORY_TYPE_CUPY
- MEMORY_TYPE_JAX
- MEMORY_TYPE_NUMPY
- MEMORY_TYPE_TENSORFLOW
- MEMORY_TYPE_TORCH
- MemoryType
- Microscope
- READ_BACKEND
- REQUIRES_DISK_READ
- REQUIRES_DISK_WRITE
- SUPPORTED_MEMORY_TYPES
- VALID_GPU_MEMORY_TYPES
- VALID_MEMORY_TYPES
- VariableComponents
- WRITE_BACKEND
```

## openhcs/constants/clauses.py

### Missing Imports
These symbols are used but not imported:
```
- Clause
- clause_numbers
- self
```

## openhcs/constants/constants.py

### Missing Imports
These symbols are used but not imported:
```
- Backend
- CPU_MEMORY_TYPES
- DEFAULT_IMAGE_EXTENSIONS
- GPU_MEMORY_TYPES
- GroupBy
- MemoryType
- Microscope
- VariableComponents
- mt
```

### Unused Imports
These symbols are imported but not used:
```
- Dict
- List
```

## openhcs/core/__init__.py

### Unused Imports
These symbols are imported but not used:
```
- ProcessingContext
- Step
```

## openhcs/core/config.py

### Missing Imports
These symbols are used but not imported:
```
- GlobalPipelineConfig
- PathPlanningConfig
- VFSConfig
- _DEFAULT_PATH_PLANNING_CONFIG
- _DEFAULT_VFS_CONFIG
- logger
```

### Unused Imports
These symbols are imported but not used:
```
- Any
- Dict
- Path
- Union
```

## openhcs/core/utils.py

### Missing Imports
These symbols are used but not imported:
```
- active_threads
- activities
- activity
- analysis
- analyze_thread_activity
- args
- context
- decorator
- duration
- end
- end1
- end2
- end_time
- f
- func
- func1
- func2
- func_name
- i
- j
- kwargs
- log_func
- log_level
- logger
- max_concurrent
- module_name
- overlap
- overlap_duration
- overlap_end
- overlap_start
- overlaps
- result
- start1
- start2
- start_time
- t
- thread1
- thread2
- thread_activity
- thread_ends
- thread_id
- thread_lock
- thread_name
- thread_starts
- time_val
- w
- well
- well1
- well2
- wrapper
- x
```

### Unused Imports
These symbols are imported but not used:
```
- Union
```

## openhcs/core/exceptions.py

### Missing Imports
These symbols are used but not imported:
```
- OpenHCSError
```

## openhcs/core/pipeline/funcstep_contract_validator.py

### Missing Imports
These symbols are used but not imported:
```
- ERROR_COMPLEX_PATTERN_WITH_SPECIAL_CONTRACTS
- ERROR_INCONSISTENT_MEMORY_TYPES
- ERROR_INVALID_FUNCTION
- ERROR_INVALID_MEMORY_TYPE
- ERROR_INVALID_PATTERN
- ERROR_MISSING_MEMORY_TYPE
- ERROR_MISSING_REQUIRED_ARGS
- FuncStepContractValidator
- all_callables
- arg
- e
- exc
- f
- f_callable
- first_fn
- fn
- fn_input_type
- fn_output_type
- func
- func_pattern
- functions
- i
- input_type
- is_structurally_simple
- item
- key
- kwargs
- logger
- memory_types
- missing_args
- name
- nested_functions
- output_type
- param
- pipeline_context
- required_args
- sig
- step
- step_memory_types
- step_name
- steps
- uses_special_contracts
```

## openhcs/core/pipeline/__init__.py

### Missing Imports
These symbols are used but not imported:
```
- metadata
- name
- self
- step
- steps
```

### Unused Imports
These symbols are imported but not used:
```
- Backend
- DEFAULT_BACKEND
- Dict
- FORCE_DISK_WRITE
- FuncStepContractValidator
- List
- MaterializationFlagPlanner
- MemoryType
- PipelineCompiler
- PipelineExecutor
- PipelinePathPlanner
- READ_BACKEND
- REQUIRES_DISK_READ
- REQUIRES_DISK_WRITE
- StepAttributeStripper
- VALID_GPU_MEMORY_TYPES
- VALID_MEMORY_TYPES
- WRITE_BACKEND
```

## openhcs/core/pipeline/compiler.py

### Missing Imports
These symbols are used but not imported:
```
- context
- current_plan
- global_enable_visualizer
- gpu_assignment
- gpu_assignments
- input_type
- is_gpu_step
- k
- logger
- memory_types
- output_type
- plan
- step
- step_gpu_assignment
- step_id
- step_memory_types
- step_name
- step_plan_val
- steps_definition
```

### Unused Imports
These symbols are imported but not used:
```
- Any
- Dict
- Optional
- Path
- Union
```

## openhcs/core/pipeline/step_attribute_stripper.py

### Missing Imports
These symbols are used but not imported:
```
- ERROR_ATTRIBUTE_DELETION_FAILED
- ERROR_RESERVED_ATTRIBUTE
- attr
- attributes
- e
- logger
- remaining_attrs
- step
- step_id
- step_name
- steps
```

## openhcs/core/pipeline/gpu_memory_validator.py

### Missing Imports
These symbols are used but not imported:
```
- e
- gpu_assignments
- gpu_id
- gpu_registry
- input_memory_type
- least_loaded_gpu
- logger
- output_memory_type
- requires_gpu
- step_id
- step_plan
- step_plans
- x
```

### Unused Imports
These symbols are imported but not used:
```
- optional_import
```

## openhcs/core/pipeline/function_contracts.py

### Missing Imports
These symbols are used but not imported:
```
- F
- decorator
- func
- input_names
- name
- output_names
```

### Unused Imports
These symbols are imported but not used:
```
- Dict
- Set
```

## openhcs/core/pipeline/pipeline_factories.py

### Missing Imports
These symbols are used but not imported:
```
- assembly_pipeline
- channel_weights
- flatten_z
- input_dir
- input_dir_vp
- normalization_params
- normalize
- output_dir
- pos_pipeline
- positions_dir
- self
- well_filter
- z_method
```

## openhcs/core/pipeline/pipeline.py

### Unused Imports
These symbols are imported but not used:
```
- PipelineCompiler
- PipelineExecutor
```

## openhcs/core/pipeline/executor.py

### Missing Imports
These symbols are used but not imported:
```
- context
- logger
```

## openhcs/core/pipeline/pipeline_utils.py

### Missing Imports
These symbols are used but not imported:
```
- func_pattern
- get_core_callable
- name
- s_final
- s_final_recheck
- s_intermediate
```

### Unused Imports
These symbols are imported but not used:
```
- List
- Tuple
```

## openhcs/core/pipeline/materialization_flag_planner.py

### Missing Imports
These symbols are used but not imported:
```
- context
- current_step_plan
- force_disk_output
- i
- is_function_step
- logger
- pipeline_definition
- read_backend
- requires_disk_input
- requires_disk_output
- step
- step_id
- step_name
- step_plans
- vfs_config
- well_id
- write_backend
```

### Unused Imports
These symbols are imported but not used:
```
- Any
- Dict
- logging
```

## openhcs/core/pipeline/path_planner.py

### Missing Imports
These symbols are used but not imported:
```
- PlanError
- context
- core_callable
- curr_step_id
- curr_step_input_dir
- curr_step_name
- current_suffix
- declared_outputs
- first_step_input
- first_step_instance
- has_special_connection
- i
- initial_pipeline_input_dir
- input_info
- is_cb
- is_chain_breaker_flag_from_plan
- k
- key
- logger
- next_step
- next_step_id
- output_path
- path_config
- pipeline_definition
- prev_step
- prev_step_id
- prev_step_name
- prev_step_output_dir
- producer
- producer_step_name
- raw_s_inputs
- raw_s_outputs
- s_inputs_info
- s_outputs_keys
- snake_case_key
- special_inputs
- special_outputs
- step
- step_id
- step_input_dir
- step_name
- step_name_lower
- step_output_dir
- step_paths
- step_plans
- steps
```

### Unused Imports
These symbols are imported but not used:
```
- Union
```

## openhcs/core/context/processing_context.py

### Missing Imports
These symbols are used but not imported:
```
- global_config
- key
- kwargs
- name
- plan
- self
- step_id
- step_plans
- value
- well_id
```

### Unused Imports
These symbols are imported but not used:
```
- Path
- Union
```

## openhcs/core/context/__init__.py

### Unused Imports
These symbols are imported but not used:
```
- ProcessingContext
```

## openhcs/core/memory/converters.py

### Missing Imports
These symbols are used but not imported:
```
- allow_cpu_roundtrip
- data
- gpu_id
- m
- memory_type
- source_type
- target_type
```

## openhcs/core/memory/decorators.py

### Missing Imports
These symbols are used but not imported:
```
- F
- decorator
- func
- input_type
- memory_types
- output_type
```

## openhcs/core/memory/__init__.py

### Unused Imports
These symbols are imported but not used:
```
- MemoryWrapper
- cupy
- jax
- memory_types
- numpy
- tensorflow
- torch
```

## openhcs/core/memory/wrapper.py

### Missing Imports
These symbols are used but not imported:
```
- MemoryWrapper
- allow_cpu_roundtrip
- cupy_data
- data
- device_str
- gpu_id
- is_gpu_array
- is_gpu_tensor
- jax
- jax_data
- memory_type
- numpy_data
- result_gpu_id
- self
- tf
- tf_data
- torch_data
```

## openhcs/core/memory/gpu_utils.py

### Missing Imports
These symbols are used but not imported:
```
- cp
- d
- device_id
- device_str
- devices
- e
- gpu_devices
- gpus
- jax
- logger
- tf
- torch
```

## openhcs/core/memory/stack_utils.py

### Missing Imports
These symbols are used but not imported:
```
- _detect_memory_type
- _enforce_gpu_device_requirements
- _is_2d
- _is_3d
- allow_single_slice
- array
- converted_slice
- converted_slices
- cp
- data
- gpu_id
- i
- jax
- jnp
- mem_type
- memory_type
- slice_data
- slices
- tf
- torch
- validate_slices
- wrapped
```

## openhcs/core/memory/utils.py

### Missing Imports
These symbols are used but not imported:
```
- _ensure_module
- cupy
- data
- device_id
- device_str
- e
- jax
- logger
- major
- memory_type
- min_version
- minor
- module
- module_name
- obj
- tf_version
```

## openhcs/core/memory/conversion_functions.py

### Missing Imports
These symbols are used but not imported:
```
- _jax_to_numpy
- allow_cpu_roundtrip
- cupy
- current_device
- data
- device_id
- device_str
- dlpack
- e
- gpu_id
- is_on_gpu
- jax
- major
- minor
- numpy_data
- result
- tensor
- tf
- tf_version
- torch
```

## openhcs/core/memory/exceptions.py

### Missing Imports
These symbols are used but not imported:
```
- message
- method
- reason
- self
- source_type
- target_type
```

## openhcs/core/memory/trackers/tf_tracker.py

### Missing Imports
These symbols are used but not imported:
```
- TF_GPU_AVAILABLE
- current_usage_bytes
- current_usage_mb
- details
- device_id
- e
- e_details
- free_mb
- logger
- memory_info
- physical_gpus
- target_gpu
- total_memory_mb
```

### Unused Imports
These symbols are imported but not used:
```
- Dict
```

## openhcs/core/memory/trackers/cupy_tracker.py

### Missing Imports
These symbols are used but not imported:
```
- CUPY_AVAILABLE
- device_id
- e
- free_bytes
- free_mb
- logger
- total_bytes
```

## openhcs/core/memory/trackers/memory_tracker.py

### Missing Imports
These symbols are used but not imported:
```
- MemoryTracker
```

## openhcs/core/memory/trackers/numpy_tracker.py

### Missing Imports
These symbols are used but not imported:
```
- NUMPY_AVAILABLE
- after_memory_mb
- array_size
- before_memory_mb
- cls
- e
- free_bytes
- free_mb
- logger
- mem_info
- memory_per_element_mb
- process
- process_memory_mb
```

## openhcs/core/memory/trackers/memory_tracker_registry.py

### Missing Imports
These symbols are used but not imported:
```
- MemoryTrackerRegistry
- MemoryTrackerSpec
- accurate
- available
- e
- factory
- get_tracker
- include_sync_true_only
- list_trackers
- logger
- memory_tracker_registry
- name
- schema_path
- self
- spec
- specs
- synchronous
- tracker
- tracker_cls
```

## openhcs/core/memory/trackers/torch_tracker.py

### Missing Imports
These symbols are used but not imported:
```
- TORCH_CUDA_AVAILABLE
- allocated_bytes
- device_id
- e
- free_in_pytorch_pool_bytes
- free_mb
- logger
- reserved_bytes
```

### Unused Imports
These symbols are imported but not used:
```
- Dict
```

## openhcs/core/steps/__init__.py

### Unused Imports
These symbols are imported but not used:
```
- AbstractStep
- FunctionStep
```

## openhcs/core/steps/abstract.py

### Missing Imports
These symbols are used but not imported:
```
- force_disk_output
- group_by
- input_dir
- logger_instance
- name
- output_dir
- self
- variable_components
```

### Unused Imports
These symbols are imported but not used:
```
- ProcessingContext
```

## openhcs/core/steps/function_step.py

### Missing Imports
These symbols are used but not imported:
```
- _execute_chain_core
- _execute_function_core
- _is_3d
- _process_single_pattern_group
- actual_callable
- actual_func_for_name
- arg_name
- array
- base_func_args
- base_kwargs
- base_kwargs_for_item
- comp_to_base_args
- comp_to_funcs
- comp_val
- component_value
- context
- current_pattern_list
- current_stack
- device_id
- e
- exec_func_or_chain
- executable_func_or_chain
- file_path_suffix
- final_base_kwargs
- final_kwargs
- first_item
- force_disk_output
- force_disk_output_flag
- full_file_path
- func
- func_callable
- func_chain
- func_item
- group_by
- grouped_patterns
- i
- image
- img_slice
- initial_data_stack
- input_mem_type
- input_memory_type_from_plan
- is_last_in_chain
- logger
- main_data_arg
- main_data_stack
- main_output_data
- matching_files
- name
- num_special_outputs
- output_filename
- output_key
- output_mem_type
- output_memory_type_from_plan
- output_path
- output_slices
- outputs_plan_for_this_call
- pattern_group_info
- pattern_item
- pattern_repr
- patterns_by_well
- processed_stack
- raw_function_output
- raw_slices
- read_backend
- returned_special_values_tuple
- same_dir
- same_directory
- self
- site
- special_inputs
- special_inputs_map
- special_inputs_plan
- special_outputs
- special_outputs_map
- special_outputs_plan
- special_path_value
- start_time
- step_input_dir
- step_output_dir
- step_plan
- step_special_inputs_plan
- step_special_outputs_plan
- value_to_save
- variable_components
- vfs_path
- well_id
- write_backend
```

### Unused Imports
These symbols are imported but not used:
```
- ProcessingContext
```

## openhcs/core/steps/specialized/focus_step.py

### Missing Imports
These symbols are used but not imported:
```
- self
```

## openhcs/core/steps/specialized/__init__.py

### Unused Imports
These symbols are imported but not used:
```
- CompositeStep
- FocusStep
- NormStep
- ZFlatStep
```

## openhcs/core/steps/specialized/composite_step.py

### Missing Imports
These symbols are used but not imported:
```
- self
- weights
```

## openhcs/core/steps/specialized/zflat_step.py

### Missing Imports
These symbols are used but not imported:
```
- method
- projection_method
- self
- valid_methods
```

## openhcs/core/steps/specialized/norm_step.py

### Missing Imports
These symbols are used but not imported:
```
- high_percentile
- low_percentile
- msg
- self
```

## openhcs/core/orchestrator/__init__.py

### Unused Imports
These symbols are imported but not used:
```
- PipelineOrchestrator
- acquire_gpu_slot
- get_gpu_registry_status
- initialize_gpu_registry
- is_gpu_registry_initialized
- release_gpu_slot
```

## openhcs/core/orchestrator/gpu_scheduler.py

### Missing Imports
These symbols are used but not imported:
```
- GPU_REGISTRY
- _detect_available_gpus
- _registry_initialized
- _registry_lock
- available_gpus
- config_to_use
- configured_num_workers
- cupy_gpu
- e
- global_config
- gpu_id
- info
- initialize_gpu_registry
- is_gpu_registry_initialized
- jax_gpu
- logger
- max_cpu_threads
- max_pipelines_per_gpu
- tf_gpu
- torch_gpu
```

## openhcs/core/orchestrator/orchestrator.py

### Missing Imports
These symbols are used but not imported:
```
- actual_max_workers
- all_wells
- all_wells_set
- compiled_contexts
- concurrent
- context
- e
- enable_visualizer_override
- error_msg
- exc
- execution_results
- executor
- filename
- filenames
- frozen_context
- future
- future_to_well_id
- global_config
- logger
- max_workers
- new_config
- num_links
- output_dir
- parsed_info
- pipeline_definition
- plate_path
- result
- selected_wells
- self
- step
- step_plan
- str_well_filter
- visualizer
- w
- well
- well_filter
- well_id
- wells_to_process
- workspace_path
- write_backend
```

### Unused Imports
These symbols are imported but not used:
```
- futures
```

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
| 1 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |
| 8 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

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
| 1 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |
| 27 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

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
| 1 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |
| 11 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

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
| 1 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |
| 40 | `prompt_toolkit.layout.Container` | Container is imported from prompt_toolkit.layout |

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
- List
- PathPlanningConfig
- VFSConfig
- VSplit
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

## openhcs/formats/position_format.py

### Missing Imports
These symbols are used but not imported:
```
- CSVPositionFormat
- POSITION_CSV_FORMATS
- ParserFunction
- PatternFormatSchema
- PositionCSVFormatSpec
- PositionRecordData
- SerializerFunction
- _parse_kv_semicolon_csv
- _parse_standard_csv
- _serialize_kv_semicolon_csv
- _serialize_standard_csv
- cls
- content
- data
- data_dict
- df
- dict_records
- expected_type
- field
- format_enum
- format_spec
- line_num
- line_pattern
- line_text
- lines
- match
- missing
- record
- records
- required_cols
- row
- self
- value
```

## openhcs/formats/func_arg_prep.py

### Missing Imports
These symbols are used but not imported:
```
- args
- comp_value
- component
- component_to_args
- component_to_funcs
- extract_func_and_args
- func
- func_item
- grouped_patterns
- patterns
- processing_funcs
```

### Unused Imports
These symbols are imported but not used:
```
- Dict
- List
```

## openhcs/formats/pattern/pattern_discovery.py

### Missing Imports
These symbols are used but not imported:
```
- PatternPath
- all_images
- backend
- cls
- comp
- component
- component_combinations
- directory
- directory_path
- e
- extensions
- file_exists
- file_list
- file_metadata
- file_path
- filemanager
- filename
- files
- files_by_well
- files_metadata
- folder_path
- group_by
- grouped_patterns
- image_paths
- img_path
- is_match
- key
- key_parts
- logger
- matching_files
- metadata
- natural_sort_key
- p
- parser
- pattern
- pattern_args
- pattern_metadata
- pattern_str
- pattern_string
- pattern_template
- patterns
- recursive
- result
- s
- self
- template_metadata
- test_instance
- text
- value
- variable_components
- well
- well_filter
```

## openhcs/formats/pattern/pattern_resolver.py

### Missing Imports
These symbols are used but not imported:
```
- DirectoryLister
- InvalidPatternError
- PathListProvider
- PatternDetector
- _extract_patterns_from_data
- _process_pattern_list
- _validate_filename_pattern
- all_patterns
- backend
- backend_instance
- convert_pattern_string
- detector
- directory
- filemanager
- filename_pattern
- pattern
- pattern_data
- patterns
- patterns_by_well
- patterns_list
- recursive
- result
- variable_components
- well
- well_patterns_data
```

## openhcs/io/filemanager.py

### Missing Imports
These symbols are used but not imported:
```
- all_files
- backend
- backend_class
- backend_instance
- backend_name
- base_dir
- current_path
- dest_parent
- dest_path
- directory
- dirs
- e
- entries
- entry
- extensions
- file_path
- filename
- files
- full_path
- logger
- path
- path_str
- paths
- pattern
- recursive
- registry
- rel_path
- result
- self
- source_dir
- source_path
- stack
- symlink_path
- target_dir
```

### Unused Imports
These symbols are imported but not used:
```
- Any
- Optional
- PathMismatchError
- os
- traceback
- validate_backend_parameter
- validate_path_types
```

## openhcs/io/__init__.py

### Unused Imports
These symbols are imported but not used:
```
- DiskStorageBackend
- FileManager
- MemoryStorageBackend
- StorageBackend
- ZarrStorageBackend
```

## openhcs/io/memory.py

### Missing Imports
These symbols are used but not imported:
```
- MemorySymlink
- StorageResolutionError
- _resolve_parent
- base_dict
- base_path
- comp
- components
- current
- data
- directory
- dst
- dst_dict
- dst_key
- dst_name
- dst_parts
- e
- extensions
- file_path
- final_key
- full_path
- k
- key
- link_name
- name
- obj
- output_path
- parent
- parent_stack
- part
- parts
- path
- pattern
- py_copy
- recurse
- recursive
- resolved
- result
- seen
- self
- source
- src
- src_dict
- src_key
- src_name
- src_parts
- target
- v
- visited
```

### Unused Imports
These symbols are imported but not used:
```
- Literal
- PathLike
- pycopy
```

## openhcs/io/zarr.py

### Missing Imports
These symbols are used but not imported:
```
- StorageResolutionError
- _matches_filters
- attrs
- chunk_divisor
- chunks
- data
- directory
- dst
- dst_group
- dst_key
- dst_store
- e
- entries
- ext
- extensions
- file_path
- full_key
- group
- group_prefix
- i
- is_array
- is_link
- k
- key
- kwargs
- link_group
- link_name
- name
- obj
- output_path
- parts
- path
- pattern
- prefix
- recursive
- relative_key
- result
- root_group
- s
- seen_keys
- self
- shape
- source
- src
- src_group
- src_key
- src_store
- store
- store2
- store_path
- target
- visit_group
- visited
- zarr_obj
```

### Unused Imports
These symbols are imported but not used:
```
- Literal
- storage
```

## openhcs/io/base.py

### Missing Imports
These symbols are used but not imported:
```
- StorageBackend
- StorageResolutionError
- path
- self
```

### Unused Imports
These symbols are imported but not used:
```
- Callable
- wraps
```

## openhcs/io/disk.py

### Missing Imports
These symbols are used but not imported:
```
- FileFormatRegistry
- StorageResolutionError
- cupy
- d
- data
- directory
- disk_directory
- disk_output_path
- disk_path
- dst
- e
- entry
- ext
- extensions
- f
- file_path
- files
- formats
- glob_pattern
- jax
- jnp
- kwargs
- link_name
- lowercase_extensions
- module_name
- optional_import
- output_path
- p
- path
- path_str
- pattern
- reader
- recursive
- resolved
- self
- source
- src
- target_exists
- tf
- tifffile
- torch
- writer
```

### Unused Imports
These symbols are imported but not used:
```
- PathLike
- fnmatch
```

## openhcs/validation/__init__.py

### Unused Imports
These symbols are imported but not used:
```
- ValidationViolation
- validate_backend_parameter
- validate_file
- validate_path_types
```

## openhcs/validation/validate.py

### Missing Imports
These symbols are used but not imported:
```
- all_violations
- args
- d
- dir_violations
- directory
- dirs
- exclude_dirs
- file
- file_path
- file_violations
- files
- find_python_files
- main
- output_file
- p
- parser
- path
- paths
- python_files
- root
- v
- validate_directory
- violation
- violations
- violations_by_file
```

## openhcs/validation/ast_validator.py

### Missing Imports
These symbols are used but not imported:
```
- ASTValidator
- BACKEND_PARAM
- BackendParameterValidator
- MEMORY_TYPE
- MemoryTypeValidator
- PATH_TYPE
- PathTypeValidator
- VFSBoundaryValidator
- VFS_BOUNDARY
- ValidationViolation
- alias
- arg
- args
- backend_arg
- decorator
- e
- elt
- f
- file_path
- filemanager_methods
- func
- has_memory_decorator
- has_validator
- kwargs
- line_number
- memory_decorators
- message
- node
- old_function
- self
- slice_value
- source
- tree
- type_annotations
- types
- validator
- validators
- violation_type
- violations
- wrapper
```

### Unused Imports
These symbols are imported but not used:
```
- Any
- Callable
- Dict
- Path
- Set
- Tuple
- Type
- Union
- inspect
- sys
```

## openhcs/tests/generators/generate_synthetic_data.py

### Missing Imports
These symbols are used but not imported:
```
- a
- abs_pos_z
- auto_image_size
- b
- background_intensity
- base_cells
- base_x
- base_y
- blur_sigma
- cc
- cell
- cell_eccentricity_range
- cell_intensity_range
- cell_size_range
- cells
- channel
- channel_idx
- channel_names
- col
- current_time
- eccentricity
- emission_wavelengths
- excitation_wavelengths
- exposure_times
- f
- filename
- filepath
- full_image
- grid_size
- height
- htd_content
- htd_filename
- htd_path
- image
- image_id
- image_size
- index_xml_path
- intensity
- key
- margin
- max_num_cells
- min_height
- min_width
- noise
- noise_level
- num_cells
- output_dir
- overlap_percent
- overlap_x
- overlap_y
- pixel_size_meters
- plate_name
- pos_x
- pos_y
- pos_z
- random_seed
- rotation
- row
- row_wells
- rr
- self
- shared_cell_fraction
- site
- site_col
- site_index
- site_positions
- site_row
- size
- stage_error_px
- step_x
- step_y
- target_dir
- tile
- tile_size
- timepoint_htd_path
- unique_id
- url
- w
- w_background
- w_cell_intensity_range
- w_cell_size_range
- w_noise_level
- w_num_cells
- w_params
- wavelength
- wavelength_backgrounds
- wavelength_idx
- wavelength_intensities
- wavelength_params
- wavelengths
- well
- well_index
- well_indices
- well_seed
- wells
- width
- x
- x_error
- x_pos
- xml_content
- y
- y_error
- y_pos
- z
- z_center
- z_distance
- z_factor
- z_level
- z_stack_levels
- z_step_size
- z_step_size_meters
- zstep_dir
```

### Unused Imports
These symbols are imported but not used:
```
- Dict
- List
- Optional
- Tuple
```

## openhcs/tests/helpers/unsafe_registry.py

### Missing Imports
These symbols are used but not imported:
```
- BackendClass
- allowed
- available_backends
- backend_class
- filename
- frame
- i
- name
- normalized_filename
- self
```

## openhcs/processing/func_registry.py

### Missing Imports
These symbols are used but not imported:
```
- FUNC_REGISTRY
- VALID_MEMORY_TYPES
- _register_function
- _registry_initialized
- _registry_lock
- _scan_and_register_functions
- e
- func
- funcs
- input_type
- is_pkg
- logger
- memory_type
- module
- module_name
- obj
- output_type
- processing_package
- processing_path
```

### Unused Imports
These symbols are imported but not used:
```
- Optional
- Tuple
- sys
```

## openhcs/processing/__init__.py

### Unused Imports
These symbols are imported but not used:
```
- FUNC_REGISTRY
- analysis
- assemblers
- cupy
- enhance
- get_function_info
- get_functions_by_memory_type
- get_valid_memory_types
- is_registry_initialized
- jax
- numpy
- pos_gen
- processors
- tensorflow
- torch
```

## openhcs/processing/registry_base.py

### Missing Imports
These symbols are used but not imported:
```
- FUNC_REGISTRY
- backend
- func
- logger
- memory_type
- memory_types
```

## openhcs/processing/function_registry.py

### Missing Imports
These symbols are used but not imported:
```
- F
- MEMORY_TYPE_CUPY
- MEMORY_TYPE_JAX
- MEMORY_TYPE_NUMPY
- MEMORY_TYPE_TENSORFLOW
- MEMORY_TYPE_TORCH
- core_decorator
- decorated_func
- decorator
- func
- info
- input_type
- memory_types
- output_type
- registry_decorator
- test_func
- torch
- x
```

### Unused Imports
These symbols are imported but not used:
```
- get_functions_by_memory_type
```

## openhcs/processing/backends/assemblers/assemble_stack_cupy.py

### Missing Imports
These symbols are used but not imported:
```
- _create_gaussian_blend_mask
- blend_mask
- blend_radius
- blended_tile
- canvas_height
- canvas_max_x
- canvas_max_y
- canvas_min_x
- canvas_min_y
- canvas_width
- canvas_x_start_int
- canvas_y_start_int
- composite_accum
- coords_x
- coords_y
- cupyx_scipy
- epsilon
- first_tile_shape
- i
- image_tiles
- image_tiles_float
- logger
- mask
- max_x_pos
- max_y_pos
- min_x_pos
- min_y_pos
- num_tiles
- pos_x
- pos_y
- positions
- shift_x_subpixel
- shift_y_subpixel
- shifted_tile
- stitched_image_float
- stitched_image_uint16
- target_canvas_x_float
- target_canvas_y_float
- tile_float
- tile_h
- tile_shape
- tile_w
- tile_x_end_src
- tile_x_start_src
- tile_y_end_src
- tile_y_start_src
- weight_accum
- x_end_on_canvas
- x_start_on_canvas
- xx
- y_end_on_canvas
- y_start_on_canvas
- yy
```

### Unused Imports
These symbols are imported but not used:
```
- gaussian_filter
```

## openhcs/processing/backends/assemblers/__init__.py

### Unused Imports
These symbols are imported but not used:
```
- assemble_stack_cpu
- assemble_stack_cupy
```

## openhcs/processing/backends/assemblers/self_supervised_stitcher.py

### Missing Imports
These symbols are used but not imported:
```
- D
- F
- FeatureEncoder
- H
- H_12
- H_21
- H_ab
- H_ba
- H_global_i
- H_inv
- H_out
- H_p1_p2
- H_p2_p1
- H_tile
- HomographyPredictionNet
- N
- W_out
- W_tile
- X
- Y
- Z
- Z_tiles
- adjacency_pairs
- adjacent_tile_pairs
- all_corners_global_frame
- all_corners_stacked
- all_graph_pairs
- all_tile_features_final
- all_x
- all_y
- ax
- axes
- barlow_twins_loss
- base_image
- base_img_H
- base_img_W
- batch_idx1_list
- batch_idx2_list
- batch_size
- bottom_idx
- c
- canvas_dims
- col_idx
- col_idx_init
- combined_features
- corners_global_frame_i
- corners_transformed_homog_i
- current_batch_pairs_indices
- current_idx
- current_start_x
- current_start_y
- device
- dummy_features
- dx
- dx_init
- dy
- dy_init
- e
- empty_homographies
- empty_positions
- feat1_g
- feat2_g
- feature_dim
- feature_dim_encoder
- feature_encoder
- features1
- features2
- final_H_i
- final_global_transforms_list
- flat_features1
- flat_features2
- geometry_consistency_loss
- get_adjacency_from_layout
- global_transforms
- grid
- grid_transformed
- grid_transformed_normalized
- grid_x
- grid_y
- homographies_learn
- homography
- homography_net
- i
- i_init
- i_pair
- i_rand_pair
- ideal_starts
- identity
- idx1
- idx2
- initial_global_transforms
- initial_transforms_for_opt
- iter_idx
- jitter_x
- jitter_y
- lambda_coeff
- layout_cols
- layout_rows
- layout_shape
- layout_shape_override
- learn
- loss
- loss_bt
- loss_geom
- loss_photo
- loss_total_batch
- m
- min_global_coords
- models
- nn
- normalization_offset_matrix
- num_layout_slots
- num_pairs_in_batch
- num_tiles
- num_train_iterations
- off_diag
- offset_x_to_origin
- offset_y_to_origin
- on_diag
- optimize_pose_graph
- optimizer
- output_homographies
- output_positions
- output_shape
- overlap
- overlap_percent
- p1
- p1_g
- p2
- p2_g
- p_idx1
- p_idx2
- padded_tile_stack
- padding_count
- padding_tensor
- pairwise_H_matrices
- params_8
- perm_idx
- photometric_loss
- plot_layout
- positions_cpu
- pretrained
- product
- r
- rect
- resnet
- return_homographies
- rgb_weights
- right_idx
- row_idx
- row_idx_init
- safe_w_coords
- safe_w_coords_i
- sampling_grid
- self
- self_supervised_stitcher
- start_x
- start_y
- synthetic_tile_stack
- synthetic_tiles_list
- tile
- tile_H
- tile_W
- tile_local_corners_homog
- tile_positions_infer
- tile_positions_learn
- tile_positions_xy
- tile_shape
- tile_shape_override
- tile_stack
- tile_stack_cpu
- tile_target
- tile_warped
- tiles1_batch
- tiles2_batch
- tiles2_warped_to_1
- tiles_for_cnn
- title
- torch
- total_loss_iter
- translate_matrix
- translate_matrix_init
- w_coords
- w_coords_i
- warp_tile_homography
- warped_tile
- x
- y
- z1
- z1_norm
- z2
- z2_norm
```

### Unused Imports
These symbols are imported but not used:
```
- Any
```

## openhcs/processing/backends/assemblers/assemble_stack_cpu.py

### Missing Imports
These symbols are used but not imported:
```
- _create_gaussian_blend_mask
- blend_mask
- blend_radius
- blended_tile
- canvas_height
- canvas_max_x
- canvas_max_y
- canvas_min_x
- canvas_min_y
- canvas_width
- canvas_x_start_int
- canvas_y_start_int
- composite_accum
- coords_x
- coords_y
- epsilon
- first_tile_shape
- i
- image_tiles
- image_tiles_float
- logger
- mask
- max_x_pos
- max_y_pos
- min_x_pos
- min_y_pos
- num_tiles
- pos_x
- pos_y
- positions
- shift_x_subpixel
- shift_y_subpixel
- shifted_tile
- stitched_image_float
- stitched_image_uint16
- target_canvas_x_float
- target_canvas_y_float
- tile_float
- tile_h
- tile_shape
- tile_w
- tile_x_end_src
- tile_x_start_src
- tile_y_end_src
- tile_y_start_src
- weight_accum
- x_end_on_canvas
- x_start_on_canvas
- xx
- y_end_on_canvas
- y_start_on_canvas
- yy
```

## openhcs/processing/backends/analysis/rrs_vectorized_tracer.py

### Missing Imports
These symbols are used but not imported:
```
- D
- N
- active_indices
- active_mask
- actual_points_for_trace
- angle_continuity_scores
- angle_tolerance
- angular_offsets
- base_angles
- best_score_indices
- cand_dx
- cand_dy
- candidate_angles
- candidate_direction_vectors
- candidate_next_positions
- chosen_directions
- chosen_next_positions
- continuing_global_indices
- continuing_mask_local
- coord
- cos_sim_step
- current_dirs_active
- current_path_lengths
- current_pos_active
- current_positions
- device
- direction_buffer
- dot_products_continuity
- i
- i_global
- image
- img_dims_tensor
- img_for_sample
- intensities_at_chosen
- intensity_scores
- intensity_threshold
- len_of_terminated_path
- mask_buffer
- max_angle_change_cosine
- max_path_length
- new_reaction_dir
- new_reaction_dir_norm
- new_seed_pos_reaction
- normalized_cand_pos
- num_active
- num_candidate_dirs_K
- num_pixels
- output_traces
- path_idx_to_write
- perturbs
- point_coords
- reaction_counts
- reaction_retries
- restart_idx_in_trace
- sampled_intensities
- seed_density
- seeds
- term_angle
- term_intensity
- term_length
- terminate_this_step_mask_local
- terminating_global_indices
- terminating_mask_local
- torch
- total_scores
- trace_buffer
- trace_len
- trace_list
- trace_radius
- valid_points_in_segment
```

### Unused Imports
These symbols are imported but not used:
```
- Any
```

## openhcs/processing/backends/analysis/__init__.py

### Unused Imports
These symbols are imported but not used:
```
- FocusAnalyzer
- dxf_mask_pipeline
```

## openhcs/processing/backends/analysis/self_supervised_segmentation_3d.py

### Missing Imports
These symbols are used but not imported:
```
- D
- D_feat
- Decoder3D
- Encoder3D
- F
- H
- H_orig
- K
- N
- W
- W_orig
- X
- Z_orig
- _affine_augment_patch
- _extract_random_patches
- _kmeans_torch
- _nt_xent_loss
- apply_segmentation
- assigned_points
- batch_size
- centroids
- cluster_k
- contrastive_weight
- current_patch_orig
- d_start
- decoder
- denominator
- dense_features_full_vol
- dense_features_upsampled
- device
- dists_sq
- emb_affine
- emb_mvm
- embedding
- embedding_dim
- encoder
- epoch
- exp_sim_no_self
- features
- features_conv1
- features_conv2
- features_for_kmeans
- flattened_features
- h_start
- i
- identity_mask
- image_volume
- img_max_orig
- img_min_orig
- img_vol_norm
- img_vol_proc
- in_channels
- k
- k_idx
- kwargs
- labels
- lambda_bound
- learning_rate
- log_probs
- logger
- loss
- loss_bound
- loss_contrastive
- loss_rec
- mask_fraction
- mask_mvm
- masked_original
- masked_reconstruction
- masks_batch
- masks_for_loss_list
- max_val_norm
- min_val_norm
- n_epochs
- n_iters
- new_centroids
- nn
- noise
- normalized_embedding
- num_patches
- numerator
- optimizer
- original_dtype
- original_input_shape_len
- out_channels
- p
- pD
- pH
- pW
- patch
- patch_affine
- patch_mvm
- patch_size_dhw
- patch_size_dhw_default
- patches
- patches_affine_batch
- patches_affine_list
- patches_mvm_batch
- patches_mvm_list
- patches_orig_batch
- pooled_features
- pos_mask
- reconstructed_patches
- reconstruction_weight
- return_features_before_pool
- segmentation_mask
- self
- sigma_noise
- sim_matrix
- temperature
- torch
- total_loss
- volume
- voxel_labels_flat
- w_start
- x
- z
- z_i
- z_j
```

## openhcs/processing/backends/analysis/focus_analyzer.py

### Missing Imports
These symbols are used but not imported:
```
- FocusAnalyzer
- best_focus_idx
- best_idx
- best_image_slice
- fft
- fft_shift
- fm
- focus_func
- focus_scores
- gx
- gy
- high_freq_count
- i
- image_stack
- img
- img_slice
- img_std
- ksize
- lap
- magnitude
- mean_val
- metric
- nvar
- score
- scores
- ten
- threshold
- threshold_factor
- weights
- x
```

## openhcs/processing/backends/analysis/dxf_mask_pipeline.py

### Missing Imports
These symbols are used but not imported:
```
- C
- C_img
- H
- HAS_TORCH
- H_data
- ModulePlaceholder
- W
- W_data
- Z
- _RegistrationCNN_torch
- _apply_displacement_field_torch
- _rasterize_polygons_slice_torch
- _smooth_field_z_torch
- abs_x
- abs_y
- aligned_mask_slices_list
- aligned_mask_stack_bool
- aligned_slice
- apply_mask
- bb_H
- bb_W
- c_idx
- cnn_input
- coords_z
- current_poly_mask_bb
- data_slice
- data_slice_unsqueezed
- device
- displaced_grid
- displacement_field
- displacement_field_slice
- displacement_field_slices
- displacement_field_stack
- dxf_polygons
- field_permuted
- field_reshaped
- grid_x
- grid_y
- i
- identity_grid
- image_slice_gray
- image_slice_norm
- image_stack
- image_stack_reg
- img_max
- img_min
- initial_rasterized_masks_float
- intersections
- j_indices
- k_indices
- kernel_1d_z
- kernel_1d_z_reshaped
- kernel_size_z
- kwargs
- logger
- mask_slice
- mask_to_apply
- masked_img
- masked_img_float
- masking_mode
- max_x
- max_xy
- max_y
- min_x
- min_xy
- min_y
- nans
- num_poly_pts
- original_dtype
- p
- p1x
- p1y
- p2x
- p2y
- padding_z
- poly_tensor
- poly_y
- polygons_gpu
- r_idx
- raster_slice
- registration_cnn
- self
- sigma_z
- smoothed_permuted
- smoothed_reshaped
- smoothed_stack
- smoothing_sigma_z
- test_points
- test_y
- warped_slice
- x
- xinters
- xx_bb
- yy_bb
- z_idx
```

### Unused Imports
These symbols are imported but not used:
```
- jnp
```

## openhcs/processing/backends/analysis/straighten_object_3d.py

### Missing Imports
These symbols are used but not imported:
```
- F
- H_orig
- L
- V_pca_transpose
- W_orig
- Z_orig
- _moving_average_1d_torch
- aligned_volume
- aligned_volume_slices
- alpha
- arbitrary_vec
- binary_mask
- centered_coords
- centerline_points_smooth
- centerline_points_smooth_x
- centerline_points_smooth_y
- centerline_points_smooth_z
- cum_lengths
- current_point
- curve_length_est
- data
- data_reshaped
- device
- e
- empty_grid
- empty_vol
- final_grid
- final_sampling_grid
- final_volume
- grid_for_sampling
- grid_u_local
- grid_v_local
- h_s
- i
- image_volume
- img_vol_for_sampling
- img_vol_proc
- indices
- kwargs
- len_next_cum
- len_prev_cum
- logger
- max_components_val
- max_z
- mean_coord
- min_voxel_threshold
- min_z
- norm_coords_x
- norm_coords_y
- norm_coords_z
- num_fallback_points
- num_samples_L
- num_sorted_pts
- object_coords
- object_coords_idx
- original_dtype
- padded_slice
- padding
- patch_dim
- patch_radius_val
- plane_points_world_x
- plane_points_world_y
- plane_points_world_z
- principal_axis
- projected_scalar
- pt_next
- pt_prev
- resampled_centerline
- return_grid_val
- sampling_grid_slices
- sampling_spacing_val
- segment_len_for_alpha
- segment_lengths
- segment_lengths_est
- single_slice
- smoothed_data
- sorted_coords_on_axis
- sorted_indices
- spline_smoothness_val
- tangent
- tangent_vec
- target_cum_lengths
- torch
- total_curve_length
- u_coords
- v_coords
- vec_u_prime
- vec_u_prime_unnorm
- vec_v_prime
- w_s
- weights
- window_size
- x_start
- y_start
```

## openhcs/processing/backends/enhance/__init__.py

### Unused Imports
These symbols are imported but not used:
```
- basic_flatfield_correction_batch_cupy
- basic_flatfield_correction_batch_numpy
- basic_flatfield_correction_cupy
- basic_flatfield_correction_numpy
- n2v2_denoise_torch
```

## openhcs/processing/backends/enhance/self_supervised_3d_deconvolution.py

### Missing Imports
These symbols are used but not imported:
```
- D
- F
- H
- W
- _LearnedBlur3D_torch
- _Simple3DCNN_torch
- _blur_fft_torch
- _extract_random_patches_torch
- _gaussian_kernel_3d_torch
- apply_deconvolution
- blur_kernel_size
- blur_mode
- blur_sigma_depth
- blur_sigma_spatial
- blurred_fft
- blurred_vol
- conv_kernel
- coords_d
- coords_h
- coords_w
- current_patch_masked
- current_patch_orig
- d_start
- deconvolved_final
- deconvolved_norm
- device
- epoch
- f_model
- f_x_masked
- f_x_orig
- features
- fixed_blur_kernel
- g_f_x_masked
- g_f_x_orig
- g_model_blur
- h_start
- i
- image_volume
- img_max_orig
- img_min_orig
- img_vol_norm
- in_channels
- kD
- kH
- kW
- ker_fft
- kernel
- kernel_d
- kernel_d_1d
- kernel_h
- kernel_h_1d
- kernel_padded
- kernel_size
- kernel_w
- kernel_w_1d
- kwargs
- lambda_bound
- lambda_inv
- lambda_rec
- learning_rate
- logger
- loss_bound
- loss_bound_f_masked
- loss_bound_f_orig
- loss_inv
- loss_rec
- mask
- mask_fraction
- max_val
- min_val
- n_epochs
- nn
- noise
- num_patches
- optimizer
- out_channels
- pD
- pH
- pW
- pad_size
- padding
- patch
- patch_size_dhw
- patches
- self
- shape
- sigma
- sigma_depth
- sigma_noise
- sigma_spatial
- torch
- total_loss
- vol_fft
- volume
- volume_single_batch_channel
- w_start
- weights
- x
```

## openhcs/processing/backends/enhance/focus_torch.py

### Missing Imports
These symbols are used but not imported:
```
- F
- H
- W
- Z
- composite
- gx
- gy
- h_start
- i
- image
- image_stack
- j
- kernel
- laplacian
- laplacian_img
- max_indices
- method
- normalize_sharpness
- original_ndim
- patch_size
- patches
- sharpness
- stride
- torch
- w_start
- weight_slice
- weights
- z_idx
```

## openhcs/processing/backends/enhance/dl_edof_unsupervised.py

### Missing Imports
These symbols are used but not imported:
```
- H_orig
- UNetLite
- W_orig
- Z
- Z_patch
- blend_patches_to_2d_image
- blurred_stack
- ch1
- ch2
- consistency_loss_fn
- count_map
- current_model_depth_config
- current_patch_size
- current_stride
- denoise
- device
- diff_sq
- extract_patches_2d_from_3d_stack
- final_fused_patch
- fused_2d_normalized
- fused_image
- fused_output_patch
- fused_patch
- fused_patch_outputs
- fused_uint16
- h_end
- h_start
- i
- image_batch
- image_stack
- in_channels_z
- input_patch_stack
- j
- kernel
- laplacian_filter_torch
- laplacian_response
- loss_c
- loss_s
- min_diff_sq_over_z
- model
- model_config_depth
- model_depth
- model_input
- multiplier
- normalize
- num_blocks_h
- num_blocks_w
- num_epochs_per_patch
- optimizer
- original_dtype
- out
- patch_content
- patch_idx
- patch_outputs
- patch_size
- patch_stack_z
- patches
- self
- sharpness_loss_fn
- stack_3d
- stack_f32
- stack_to_blur
- stride
- target_h
- target_w
- total_loss
- w_end
- w_start
- x
- x1
- x2
- x3
```

## openhcs/processing/backends/enhance/basic_processor_cupy.py

### Missing Imports
These symbols are used but not imported:
```
- D
- L
- L_stack
- S
- U
- Vh
- _low_rank_approximation
- _soft_threshold
- _validate_cupy_array
- array
- b
- basic_flatfield_correction_cupy
- batch_dim
- corrected
- correction_mode
- cupyx_scipy
- eps
- image
- image_batch
- image_float
- iteration
- kwargs
- lambda_lowrank
- lambda_sparse
- logger
- low_rank
- matrix
- max_iters
- max_val
- name
- norm_D
- normalize_output
- orig_dtype
- rank
- residual
- result_list
- s
- threshold
- tol
- verbose
- x
- y
- z
```

## openhcs/processing/backends/enhance/basic_processor_numpy.py

### Missing Imports
These symbols are used but not imported:
```
- D
- L
- L_stack
- S
- U
- Vh
- _low_rank_approximation
- _soft_threshold
- _validate_numpy_array
- array
- b
- basic_flatfield_correction_numpy
- batch_dim
- corrected
- correction_mode
- eps
- image
- image_batch
- image_float
- iteration
- kwargs
- lambda_lowrank
- lambda_sparse
- logger
- low_rank
- matrix
- max_iters
- max_val
- name
- norm_D
- normalize_output
- orig_dtype
- rank
- residual
- result_list
- s
- threshold
- tol
- verbose
- x
- y
- z
```

## openhcs/processing/backends/enhance/n2v2_processor_torch.py

### Missing Imports
These symbols are used but not imported:
```
- BlurPool3d
- DoubleConv3d
- Down3d
- F
- Module
- N2V2UNet
- Up3d
- batch_size
- blindspot_prob
- channels
- count
- denoised
- device
- diff_x
- diff_y
- diff_z
- epoch
- epoch_loss
- extract_random_patches
- features
- generate_blindspot_mask
- i
- image
- in_channels
- kernel
- kernel_range
- kernel_size
- kernel_x
- kernel_y
- kernel_z
- kwargs
- learning_rate
- logger
- loss
- loss_fn
- masked_input
- masked_loss
- masks
- max_epochs
- max_val
- model
- model_path
- nn
- num_batches
- num_patches
- optimizer
- out_channels
- padded
- patch
- patch_size
- patches
- pred_patch
- prediction
- prob
- random_seed
- save_model_path
- self
- shape
- sigma
- skip
- skip_connection
- stride
- torch
- verbose
- x
- x1
- x2
- x3
- x4
- x5
- x_end
- x_start
- y
- y_end
- y_start
- z
- z_end
- z_start
```

## openhcs/processing/backends/pos_gen/__init__.py

### Unused Imports
These symbols are imported but not used:
```
- gpu_ashlar_align_cupy
- mist_compute_tile_positions
```

## openhcs/processing/backends/pos_gen/mist_processor_cupy.py

### Missing Imports
These symbols are used but not imported:
```
- H
- W
- Z
- _validate_cupy_array
- array
- bottom_idx
- bottom_patch
- bottom_tile
- c
- center_x
- center_y
- correlation
- cross_power
- cross_power_norm
- cupyx_scipy
- current_patch
- current_region
- current_tile
- dx
- dy
- expected_x
- expected_y
- extract_patch
- fft1
- fft2
- fft_backend
- global_optimization
- h
- half_size
- idx
- image
- image1
- image2
- image_stack
- iteration
- left_idx
- left_patch
- left_region
- left_tile
- logger
- name
- normalize
- num_cols
- num_rows
- overlap_center_x
- overlap_center_y
- overlap_ratio
- overlap_x
- overlap_y
- pad_x_after
- pad_x_before
- pad_y_after
- pad_y_before
- patch
- patch_size
- peak_idx
- phase_correlation
- position_updates
- positions
- prev_positions
- r
- refinement_iterations
- region
- return_full
- right_idx
- right_patch
- right_tile
- subpixel
- subpixel_radius
- tile
- tile_grid
- tile_max
- tile_min
- top_idx
- top_patch
- top_region
- top_tile
- total_mass
- verbose
- w
- weights
- window
- window_2d
- window_x
- window_y
- x_com
- x_indices
- x_max
- x_min
- x_peak
- x_shift
- y_com
- y_indices
- y_max
- y_min
- y_peak
- y_shift
- z
```

### Unused Imports
These symbols are imported but not used:
```
- ndimage
```

## openhcs/processing/backends/pos_gen/ashlar_processor_cupy.py

### Missing Imports
These symbols are used but not imported:
```
- H
- W
- Z
- _validate_cupy_array
- affine_mats
- array
- c
- correlation
- cross_power
- cross_power_norm
- cupyx_scipy
- current_tile
- dx
- dy
- fft1
- fft2
- h
- idx
- image1
- image2
- left_idx
- left_tile
- mean_offset
- method
- name
- normalize
- num_cols
- num_rows
- offsets
- peak_idx
- phase_correlation
- positions
- r
- region
- return_affine
- subpixel
- subpixel_radius
- tile
- tile_max
- tile_min
- tiles
- tiles_grid
- top_idx
- top_tile
- total_mass
- w
- window
- window_2d
- window_x
- window_y
- x_com
- x_indices
- x_max
- x_min
- x_peak
- x_shift
- y_com
- y_indices
- y_max
- y_min
- y_peak
- y_shift
- z
```

### Unused Imports
These symbols are imported but not used:
```
- ndimage
```

## openhcs/processing/backends/processors/jax_processor.py

### Missing Imports
These symbols are used but not imported:
```
- amount
- args
- array
- background_large
- background_small
- bin_width
- bins
- blocks
- blurred
- cdf
- cdf_max
- cdf_normalized
- clipped
- cls
- composite
- create_linear_weight_mask
- downsample_factor
- dtype
- equal_percentiles
- equalized_flat
- equalized_stack
- eroded
- first_image
- flat_stack
- grid_x
- grid_y
- height
- high_percentile
- hist
- i
- image
- image_small
- images
- img
- indices
- input_dtype
- jax
- jnp
- kernel
- kernel_1d
- kernel_2d
- kernel_reshaped
- kernel_size
- lax
- logger
- low_percentile
- margin_ratio
- margin_x
- margin_y
- mask
- masked
- masked_values
- max_val
- method
- min_val
- name
- neighborhood
- new_h
- new_w
- normalize_single_slice
- normalize_slice
- normalize_stack_fn
- normalized
- opened
- p_high
- p_low
- pad_size
- padded
- padded_eroded
- padded_reshaped
- process_slice
- radius
- ramp_bottom
- ramp_left
- ramp_right
- ramp_top
- range_max
- range_min
- result
- result_list
- return_constant
- return_constant_stack
- scale
- selem_radius
- shape
- sharpened
- sigma
- slice_data
- slice_data_float
- slice_float
- slice_idx
- slice_result
- small_mask
- small_selem
- small_selem_radius
- stack
- stack_data
- target_max
- target_min
- tophat_small
- total_weight
- weight
- weight_mask
- weight_x
- weight_y
- weights
- width
- x
- x_range
- y
- y_range
- z
```

### Unused Imports
These symbols are imported but not used:
```
- ImageProcessorInterface
```

## openhcs/processing/backends/processors/torch_processor.py

### Missing Imports
These symbols are used but not imported:
```
- F
- amount
- array
- background_4d
- background_large
- background_small
- bins
- blurred
- cdf
- clipped
- cls
- composite
- coords
- create_linear_weight_mask
- device
- downsample_factor
- dtype
- equalized_flat
- equalized_stack
- eroded
- first_image
- flat_stack
- gauss
- height
- high_percentile
- hist
- i
- image
- image_small
- images
- img
- img_4d
- indices
- input_dtype
- kernel
- kernel_size
- kernel_x
- kernel_y
- logger
- low_percentile
- margin_ratio
- margin_x
- margin_y
- mask
- masked
- masked_patches
- masked_patches_eroded
- max_val
- method
- min_val
- name
- new_h
- new_w
- normalized
- opened
- p_high
- p_low
- pad_size
- padded
- padded_eroded
- patch_size
- patches
- patches_eroded
- projection_2d
- radius
- ramp_bottom
- ramp_left
- ramp_right
- ramp_top
- range_max
- range_min
- result
- scale
- selem_radius
- shape
- sharpened
- sigma
- slice_float
- slice_result
- small_grid_size
- small_grid_x
- small_grid_y
- small_mask
- small_selem
- small_selem_radius
- stack
- target_max
- target_min
- tophat_small
- torch
- total_weight
- weight
- weight_mask
- weight_x
- weight_y
- weights
- width
- z
```

### Unused Imports
These symbols are imported but not used:
```
- ImageProcessorInterface
```

## openhcs/processing/backends/processors/numpy_processor.py

### Missing Imports
These symbols are used but not imported:
```
- amount
- array
- background_large
- background_small
- bin_edges
- bins
- blurred
- cdf
- clipped
- cls
- composite
- create_linear_weight_mask
- downsample_factor
- dtype
- equalized_stack
- first_image
- flat_stack
- height
- high_percentile
- hist
- i
- image
- image_small
- images
- img
- input_dtype
- logger
- low_percentile
- margin_ratio
- margin_x
- margin_y
- mask
- masked
- max_val
- method
- name
- normalized
- p_high
- p_low
- projection_2d
- radius
- ramp_bottom
- ramp_left
- ramp_right
- ramp_top
- range_max
- range_min
- result
- selem_radius
- selem_small
- shape
- sharpened
- slice_float
- slice_result
- stack
- target_max
- target_min
- tophat_small
- total_weight
- weight
- weight_mask
- weight_x
- weight_y
- weights
- width
- z
```

### Unused Imports
These symbols are imported but not used:
```
- ImageProcessorInterface
- TYPE_CHECKING
```

## openhcs/processing/backends/processors/cupy_processor.py

### Missing Imports
These symbols are used but not imported:
```
- amount
- array
- background_large
- background_small
- bin_width
- bins
- blurred
- cdf
- clipped
- cls
- composite
- cp
- create_linear_weight_mask
- cupyx_scipy
- downsample_factor
- dtype
- equalized_flat
- equalized_stack
- eroded
- first_image
- flat_stack
- grid_x
- grid_y
- height
- high_percentile
- hist
- i
- image
- image_small
- images
- img
- indices
- input_dtype
- logger
- low_percentile
- margin_ratio
- margin_x
- margin_y
- mask
- masked
- max_val
- method
- min_val
- name
- ndimage
- new_h
- new_w
- normalized
- opened
- p_high
- p_low
- projection_2d
- radius
- ramp_bottom
- ramp_left
- ramp_right
- ramp_top
- range_max
- range_min
- result
- selem_radius
- shape
- sharpened
- slice_float
- slice_result
- small_mask
- small_selem
- small_selem_radius
- stack
- target_max
- target_min
- tophat_small
- total_weight
- weight
- weight_mask
- weight_x
- weight_y
- weights
- width
- x_range
- y_range
- z
```

### Unused Imports
These symbols are imported but not used:
```
- ImageProcessorInterface
```

## openhcs/processing/backends/processors/tensorflow_processor.py

### Missing Imports
These symbols are used but not imported:
```
- amount
- array
- background_4d
- background_large
- background_small
- bins
- blurred
- cdf
- clipped
- cls
- composite
- create_linear_weight_mask
- downsample_factor
- dtype
- e
- equalized_flat
- equalized_stack
- eroded
- eroded_neg
- first_image
- flat_slice
- flat_stack
- height
- high_idx
- high_percentile
- hist
- i
- image
- image_small
- images
- img
- img_4d
- indices
- input_dtype
- kernel_size
- large_neg
- logger
- low_idx
- low_percentile
- margin_ratio
- margin_x
- margin_y
- mask
- mask_complement
- mask_expanded
- masked
- max_val
- method
- min_val
- min_version
- name
- neg_padded
- new_h
- new_w
- normalized
- opened
- p_high
- p_low
- pad_size
- padded
- padded_eroded
- radius
- ramp_bottom
- ramp_left
- ramp_right
- ramp_top
- range_max
- range_min
- result
- result_list
- scale
- selem_radius
- shape
- sharpened
- sigma
- slice_float
- slice_result
- slice_size
- small_grid_size
- small_mask
- small_selem_radius
- small_x_grid
- small_y_grid
- sorted_slice
- sorted_stack
- stack
- stack_size
- target_max
- target_min
- tf
- tf_version
- tophat_small
- total_weight
- weight
- weight_mask
- weight_x
- weight_y
- weights
- width
- x_range
- y_range
- z
```

### Unused Imports
These symbols are imported but not used:
```
- ImageProcessorInterface
```

## openhcs/utils/import_utils.py

### Missing Imports
These symbols are used but not imported:
```
- Placeholder
- base_class
- item
- module_name
- name
- required_library
- self
```
