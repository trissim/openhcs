# Import Analysis

Found 44 files with import issues:

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
