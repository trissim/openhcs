# Code Reference Analysis

Found 77 files with references:

## openhcs/microscopes/opera_phenix.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 43 | OperaPhenixXmlParser.get_field_id_mapping | `Dict` |
| 94 | OperaPhenixHandler.common_dirs | `List` |
| 294 | OperaPhenixFilenameParser.parse_filename | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 12 | `from typing import Any, Dict, List, Optional, Union` |
| 43 | `def get_field_id_mapping(self) -> Dict[str, int]:` |
| 48 | `Dictionary mapping original field IDs to remapped field IDs` |
| 94 | `def common_dirs(self) -> List[str]:` |
| 294 | `def parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:` |
| 303 | `dict or None: Dictionary with extracted components or None if parsing fails.` |

## openhcs/microscopes/microscope_interfaces_base.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 38 | FilenameParser.parse_filename | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 11 | `from typing import Any, Dict, Optional, Tuple, Union` |
| 38 | `def parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:` |
| 46 | `dict or None: Dictionary with extracted components or None if parsing fails` |

## openhcs/microscopes/opera_phenix_xml_parser.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 102 | OperaPhenixXmlParser.get_plate_info | `Dict` |
| 314 | OperaPhenixXmlParser.get_image_info | `Dict` |
| 314 | OperaPhenixXmlParser.get_image_info | `Dict` |
| 373 | OperaPhenixXmlParser.get_well_positions | `Dict` |
| 414 | OperaPhenixXmlParser.get_field_positions | `Dict` |
| 458 | OperaPhenixXmlParser.sort_fields_by_position | `Dict` |
| 508 | OperaPhenixXmlParser.get_field_id_mapping | `Dict` |
| 524 | OperaPhenixXmlParser.remap_field_id | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 11 | `from typing import Any, Dict, Optional, Tuple, Union` |
| 102 | `def get_plate_info(self) -> Dict[str, Any]:` |
| 107 | `Dict containing plate information` |
| 314 | `def get_image_info(self) -> Dict[str, Dict[str, Any]]:` |
| 319 | `Dictionary mapping image IDs to dictionaries containing image information` |
| 373 | `def get_well_positions(self) -> Dict[str, Tuple[int, int]]:` |
| 378 | `Dictionary mapping well IDs to (row, column) tuples` |
| 414 | `def get_field_positions(self) -> Dict[int, Tuple[float, float]]:` |
| 458 | `def sort_fields_by_position(self, positions: Dict[int, Tuple[float, float]]) -> ...` |
| 464 | `positions: Dictionary mapping field IDs to (x, y) position tuples` |
| 508 | `def get_field_id_mapping(self) -> Dict[int, int]:` |
| 524 | `def remap_field_id(self, field_id: int, mapping: Optional[Dict[int, int]] = None...` |

## openhcs/microscopes/microscope_base.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 52 | MicroscopeHandler.common_dirs | `List` |
| 248 | MicroscopeHandler.parse_filename | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 12 | `from typing import Any, Dict, List, Optional, Tuple, Union` |
| 25 | `# Dictionary to store registered microscope handlers` |
| 52 | `def common_dirs(self) -> List[str]:` |
| 248 | `def parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:` |
| 274 | `variable_components: List of components to make variable (e.g., ['site', 'z_inde...` |
| 277 | `Dict[str, Any]: Dictionary mapping wells to patterns` |
| 321 | `List of matching filenames` |

## openhcs/microscopes/imagexpress.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 40 | ImageXpressHandler.common_dirs | `List` |
| 228 | ImageXpressFilenameParser.parse_filename | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 12 | `from typing import Any, Dict, List, Optional, Tuple, Union` |
| 40 | `def common_dirs(self) -> List[str]:` |
| 95 | `# List all subdirectories using the filemanager` |
| 131 | `# List all files in the Z folder` |
| 228 | `def parse_filename(self, filename: Union[str, Any]) -> Optional[Dict[str, Any]]:` |
| 236 | `dict or None: Dictionary with extracted components or None if parsing fails` |

## openhcs/runtime/napari_stream_visualizer.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 39 | NapariStreamVisualizer.__init__ | `Dict` |
| 114 | NapariStreamVisualizer._update_layer_in_thread | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 17 | `from typing import Any, Dict, List, Optional # Added List` |
| 39 | `self.layers: Dict[str, napari.layers.Image] = {} # Consider if layer type should...` |
| 114 | `def _update_layer_in_thread(self, layer_name: str, data: np.ndarray, metadata: O...` |

## openhcs/constants/constants.py
**Imports target:** True

### String References
| Line | Reference |
| ---- | --------- |
| 9 | `from typing import Any, Callable, Dict, List, Set, TypeVar` |

## openhcs/core/config.py
**Imports target:** False

### String References
| Line | Reference |
| ---- | --------- |
| 66 | `# logging_config: Optional[Dict[str, Any]] = None # For configuring logging leve...` |
| 67 | `# plugin_settings: Dict[str, Any] = field(default_factory=dict) # For plugin-spe...` |

## openhcs/core/utils.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 55 | get_thread_activity | `Dict` |
| 55 | get_thread_activity | `List` |
| 55 | get_thread_activity | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 10 | `from typing import Any, Callable, Dict, List, Optional, Union` |
| 55 | `def get_thread_activity() -> Dict[int, List[Dict[str, Any]]]:` |
| 60 | `Dict mapping thread IDs to lists of activity records` |
| 171 | `Dict containing analysis results` |

## openhcs/core/pipeline/funcstep_contract_validator.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 86 | FuncStepContractValidator.validate_pipeline | `List` |
| 86 | FuncStepContractValidator.validate_pipeline | `Dict` |
| 86 | FuncStepContractValidator.validate_pipeline | `Dict` |
| 86 | FuncStepContractValidator.validate_pipeline | `Dict` |
| 160 | FuncStepContractValidator.validate_funcstep | `Dict` |
| 310 | FuncStepContractValidator._validate_required_args | `Dict` |
| 351 | FuncStepContractValidator.validate_pattern_structure | `List` |
| 380 | FuncStepContractValidator._extract_functions_from_pattern | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 17 | `from typing import Any, Callable, Dict, List, Optional, Tuple` |
| 62 | `"containing exactly one such item. Dictionaries or multi-item lists are not perm...` |
| 86 | `def validate_pipeline(steps: List[Any], pipeline_context: Optional[Dict[str, Any...` |
| 100 | `Dictionary mapping step UIDs to memory type dictionaries` |
| 160 | `def validate_funcstep(step: FunctionStep) -> Dict[str, str]:` |
| 169 | `Dictionary of validated memory types` |
| 310 | `def _validate_required_args(func: Callable, kwargs: Dict[str, Any], step_name: s...` |
| 351 | `) -> List[Callable]:` |
| 361 | `- List of callables or patterns` |
| 362 | `- Dict of keyed callables or patterns` |
| 369 | `List of functions in the pattern` |
| 380 | `) -> List[Callable]:` |
| 387 | `- List of callables or patterns` |
| 388 | `- Dict of keyed callables or patterns` |
| 395 | `List of functions in the pattern` |
| 417 | `# Case 3: List of patterns` |
| 431 | `# Case 4: Dict of keyed patterns` |

## openhcs/core/pipeline/__init__.py
**Imports target:** False

### String References
| Line | Reference |
| ---- | --------- |
| 52 | `steps: List of steps in the pipeline` |
| 62 | `steps: List of steps in the pipeline` |
| 87 | `Dictionary representation of the pipeline` |

## openhcs/core/pipeline/compiler.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 57 | PipelineCompiler.initialize_step_plans_for_context | `List` |
| 133 | PipelineCompiler.plan_materialization_flags_for_context | `List` |
| 172 | PipelineCompiler.validate_memory_contracts_for_context | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 25 | `from typing import Any, Dict, List, Optional, Union # Callable removed` |
| 26 | `from collections import OrderedDict # For special_outputs and special_inputs ord...` |
| 57 | `steps_definition: List[AbstractStep]` |
| 104 | `# Ensure these keys exist as OrderedDicts if PathPlanner doesn't guarantee it` |
| 105 | `# (PathPlanner currently creates them as dicts, OrderedDict might not be strictl...` |
| 106 | `current_plan.setdefault("special_inputs", OrderedDict())` |
| 107 | `current_plan.setdefault("special_outputs", OrderedDict())` |
| 133 | `steps_definition: List[AbstractStep]` |
| 172 | `steps_definition: List[AbstractStep]` |

## openhcs/core/pipeline/step_attribute_stripper.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 49 | StepAttributeStripper.strip_step_attributes | `List` |
| 49 | StepAttributeStripper.strip_step_attributes | `Dict` |
| 49 | StepAttributeStripper.strip_step_attributes | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 17 | `from typing import Any, Dict, List` |
| 49 | `def strip_step_attributes(steps: List[Any], step_plans: Dict[str, Dict[str, Any]...` |
| 54 | `steps: List of Step instances` |
| 55 | `step_plans: Dictionary mapping step UIDs to step plans` |

## openhcs/core/pipeline/gpu_memory_validator.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 57 | GPUMemoryTypeValidator.validate_step_plans | `Dict` |
| 57 | GPUMemoryTypeValidator.validate_step_plans | `Dict` |
| 58 | GPUMemoryTypeValidator.validate_step_plans | `Dict` |
| 58 | GPUMemoryTypeValidator.validate_step_plans | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 15 | `from typing import Any, Dict` |
| 57 | `step_plans: Dict[str, Dict[str, Any]]` |
| 58 | `) -> Dict[str, Dict[str, Any]]:` |
| 67 | `step_plans: Dictionary mapping step IDs to step plans` |
| 70 | `Dictionary mapping step IDs to dictionaries containing GPU assignments` |

## openhcs/core/pipeline/function_contracts.py
**Imports target:** True

### String References
| Line | Reference |
| ---- | --------- |
| 20 | `from typing import Callable, Any, TypeVar, Set, Dict` |

## openhcs/core/pipeline/pipeline_factories.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 43 | AutoPipelineFactory.__init__ | `Dict` |
| 46 | AutoPipelineFactory.__init__ | `List` |
| 46 | AutoPipelineFactory.__init__ | `Dict` |
| 47 | AutoPipelineFactory.__init__ | `List` |
| 85 | AutoPipelineFactory.create_pipelines | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 12 | `from typing import Any, Dict, List, Optional, Union` |
| 43 | `normalization_params: Optional[Dict[str, Any]] = None,` |
| 46 | `channel_weights: Optional[Union[List[float], Dict[str, float]]] = None,` |
| 47 | `well_filter: Optional[List[str]] = None,` |
| 85 | `def create_pipelines(self) -> List[Pipeline]:` |

## openhcs/core/pipeline/executor.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 42 | PipelineExecutor.execute | `List` |
| 42 | PipelineExecutor.execute | `List` |
| 42 | PipelineExecutor.execute | `List` |
| 43 | PipelineExecutor.execute | `List` |
| 46 | PipelineExecutor.execute | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 18 | `from typing import Any, List, Optional, Union` |
| 42 | `steps: Union[List[AbstractStep], List[List[AbstractStep]]],` |
| 43 | `context: Union[ProcessingContext, List[ProcessingContext]],` |
| 46 | `) -> Union[ProcessingContext, List[ProcessingContext]]:` |

## openhcs/core/pipeline/pipeline_utils.py
**Imports target:** True

### String References
| Line | Reference |
| ---- | --------- |
| 4 | `from typing import Any, Callable, List, Optional, Tuple` |

## openhcs/core/pipeline/materialization_flag_planner.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 50 | MaterializationFlagPlanner.prepare_pipeline_flags | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 20 | `from typing import Any, Dict, List` |
| 50 | `pipeline_definition: List[AbstractStep] # Renamed 'steps' for clarity` |
| 63 | `pipeline_definition: List of AbstractStep instances defining the pipeline.` |

## openhcs/core/pipeline/path_planner.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 35 | PipelinePathPlanner.prepare_pipeline_paths | `List` |
| 81 | PipelinePathPlanner.prepare_pipeline_paths | `Dict` |
| 37 | PipelinePathPlanner.prepare_pipeline_paths | `Dict` |
| 37 | PipelinePathPlanner.prepare_pipeline_paths | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 10 | `from typing import Any, Dict, List, Optional, Set, Union` |
| 35 | `pipeline_definition: List[AbstractStep]` |
| 37 | `) -> Dict[str, Dict[str, Any]]: # Return type is still the modified step_plans f...` |
| 44 | `pipeline_definition: List of AbstractStep instances.` |
| 81 | `s_inputs_info: Dict[str, bool] = {}` |

## openhcs/core/context/processing_context.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 37 | ProcessingContext.__init__ | `Dict` |
| 37 | ProcessingContext.__init__ | `Dict` |
| 75 | ProcessingContext.inject_plan | `Dict` |
| 122 | ProcessingContext.get_step_plan | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 7 | `from typing import Any, Dict, Optional, Union` |
| 24 | `step_plans: Dictionary mapping step IDs to execution plans.` |
| 25 | `outputs: Dictionary for step outputs (usage may change with VFS-centric model).` |
| 26 | `intermediates: Dictionary for intermediate results (usage may change).` |
| 37 | `step_plans: Optional[Dict[str, Dict[str, Any]]] = None,` |
| 46 | `step_plans: Dictionary mapping step IDs to execution plans.` |
| 75 | `def inject_plan(self, step_id: str, plan: Dict[str, Any]) -> None:` |
| 122 | `def get_step_plan(self, step_id: str) -> Optional[Dict[str, Any]]:` |

## openhcs/core/memory/stack_utils.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 129 | stack_slices | `List` |
| 220 | unstack_slices | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 13 | `from typing import Any, List` |
| 129 | `def stack_slices(slices: List[Any], memory_type: str, gpu_id: int, allow_single_...` |
| 137 | `slices: List of 2D slices (numpy arrays, cupy arrays, torch tensors, etc.)` |
| 220 | `def unstack_slices(array: Any, memory_type: str, gpu_id: int, validate_slices: b...` |
| 233 | `List of 2D slices in the specified memory type` |

## openhcs/core/memory/trackers/tf_tracker.py
**Imports target:** True

### String References
| Line | Reference |
| ---- | --------- |
| 2 | `from typing import TYPE_CHECKING, Dict` |

## openhcs/core/memory/trackers/numpy_tracker.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 92 | NumPyMemoryTracker.get_numpy_memory_info | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 15 | `from typing import TYPE_CHECKING, Dict` |
| 92 | `def get_numpy_memory_info(self) -> Dict[str, float]:` |
| 97 | `Dictionary containing NumPy memory information in MB` |

## openhcs/core/memory/trackers/memory_tracker_registry.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 68 | MemoryTrackerRegistry.list_trackers | `List` |
| 101 | list_trackers | `List` |
| 141 | list_available_tracker_specs | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 16 | `from typing import Any, List, Type` |
| 68 | `def list_trackers(self) -> List[str]:` |
| 70 | `List all available memory tracker names.` |
| 73 | `List of memory tracker names` |
| 101 | `def list_trackers() -> List[str]:` |
| 103 | `List all available memory tracker names.` |
| 106 | `List of memory tracker names` |
| 141 | `def list_available_tracker_specs(include_sync_true_only: bool = False) -> List[M...` |
| 143 | `List specifications for all registered and available memory trackers.` |

## openhcs/core/memory/trackers/torch_tracker.py
**Imports target:** True

### String References
| Line | Reference |
| ---- | --------- |
| 2 | `from typing import TYPE_CHECKING, Dict` |

## openhcs/core/steps/abstract.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 82 | AbstractStep.__init__ | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 28 | `from typing import TYPE_CHECKING, List, Optional, Union` |
| 82 | `variable_components: Optional[List[str]] = None,` |
| 96 | `variable_components: List of variable components for this step.` |

## openhcs/core/steps/function_step.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 33 | _execute_function_core | `Dict` |
| 35 | _execute_function_core | `Dict` |
| 88 | _execute_chain_core | `List` |
| 88 | _execute_chain_core | `Dict` |
| 90 | _execute_chain_core | `Dict` |
| 96 | _execute_chain_core | `Dict` |
| 122 | _process_single_pattern_group | `Dict` |
| 134 | _process_single_pattern_group | `Dict` |
| 225 | FunctionStep.__init__ | `Dict` |
| 225 | FunctionStep.__init__ | `List` |
| 225 | FunctionStep.__init__ | `Dict` |
| 226 | FunctionStep.__init__ | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 13 | `from typing import Any, Callable, Dict, List, Optional, Tuple, Union, OrderedDic...` |
| 33 | `base_kwargs: Dict[str, Any],` |
| 35 | `special_inputs_plan: Dict[str, str],  # {'arg_name_for_func': 'special_path_valu...` |
| 36 | `special_outputs_plan: TypingOrderedDict[str, str] # {'output_key': 'special_path...` |
| 88 | `func_chain: List[Union[Callable, Tuple[Callable, Dict]]],` |
| 90 | `step_special_inputs_plan: Dict[str, str],` |
| 91 | `step_special_outputs_plan: TypingOrderedDict[str, str]` |
| 96 | `base_kwargs_for_item: Dict[str, Any] = {}` |
| 122 | `base_func_args: Dict[str, Any],` |
| 134 | `special_inputs_map: Dict[str, str],` |
| 135 | `special_outputs_map: TypingOrderedDict[str, str]` |
| 225 | `func: Union[Callable, Tuple[Callable, Dict], List[Union[Callable, Tuple[Callable...` |
| 226 | `*, name: Optional[str] = None, variable_components: Optional[List[str]] = ['site...` |
| 257 | `special_outputs = step_plan.get('special_outputs', {}) # Should be OrderedDict i...` |

## openhcs/core/steps/specialized/composite_step.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 44 | CompositeStep.__init__ | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 21 | `from typing import List, Optional` |
| 44 | `weights: Optional[List[float]] = None,` |

## openhcs/core/orchestrator/gpu_scheduler.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 45 | module level | `Dict` |
| 45 | module level | `Dict` |
| 117 | _detect_available_gpus | `List` |
| 241 | get_gpu_registry_status | `Dict` |
| 241 | get_gpu_registry_status | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 27 | `from typing import Dict, List, Optional` |
| 45 | `GPU_REGISTRY: Dict[int, Dict[str, int]] = {}` |
| 117 | `def _detect_available_gpus() -> List[int]:` |
| 122 | `List of available GPU IDs` |
| 241 | `def get_gpu_registry_status() -> Dict[int, Dict[str, int]]:` |

## openhcs/core/orchestrator/orchestrator.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 99 | PipelineOrchestrator.__init__ | `List` |
| 200 | PipelineOrchestrator.compile_pipelines | `List` |
| 201 | PipelineOrchestrator.compile_pipelines | `List` |
| 230 | PipelineOrchestrator.compile_pipelines | `Dict` |
| 203 | PipelineOrchestrator.compile_pipelines | `Dict` |
| 263 | PipelineOrchestrator._execute_single_well | `List` |
| 266 | PipelineOrchestrator._execute_single_well | `Dict` |
| 303 | PipelineOrchestrator.execute_compiled_plate | `List` |
| 304 | PipelineOrchestrator.execute_compiled_plate | `Dict` |
| 335 | PipelineOrchestrator.execute_compiled_plate | `Dict` |
| 335 | PipelineOrchestrator.execute_compiled_plate | `Dict` |
| 307 | PipelineOrchestrator.execute_compiled_plate | `Dict` |
| 307 | PipelineOrchestrator.execute_compiled_plate | `Dict` |
| 359 | PipelineOrchestrator.get_wells | `List` |
| 359 | PipelineOrchestrator.get_wells | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 19 | `from typing import Any, Dict, List, Optional, Union, Set` |
| 99 | `self.default_pipeline_definition: Optional[List[AbstractStep]] = None` |
| 200 | `pipeline_definition: List[AbstractStep],` |
| 201 | `well_filter: Optional[List[str]] = None,` |
| 203 | `) -> Dict[str, ProcessingContext]:` |
| 228 | `raise ValueError("A valid pipeline definition (List[AbstractStep]) must be provi...` |
| 230 | `compiled_contexts: Dict[str, ProcessingContext] = {}` |
| 263 | `pipeline_definition: List[AbstractStep],` |
| 266 | `) -> Dict[str, Any]:` |
| 303 | `pipeline_definition: List[AbstractStep],` |
| 304 | `compiled_contexts: Dict[str, ProcessingContext],` |
| 307 | `) -> Dict[str, Dict[str, Any]]:` |
| 313 | `compiled_contexts: Dict of well_id to its compiled, frozen ProcessingContext.` |
| 335 | `execution_results: Dict[str, Dict[str, Any]] = {}` |
| 359 | `def get_wells(self, well_filter: Optional[List[str]] = None) -> List[str]:` |

## openhcs/tui/components.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 17 | InteractiveListItem.__init__ | `Dict` |
| 18 | InteractiveListItem.__init__ | `Dict` |
| 140 | ParameterEditor.__init__ | `Dict` |
| 155 | ParameterEditor.update_function | `Dict` |
| 191 | ParameterEditor._get_function_parameters | `List` |
| 191 | ParameterEditor._get_function_parameters | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 3 | `from typing import Any, Callable, Dict, List, Optional, Tuple, Union` |
| 9 | `TextArea, RadioList as Dropdown)` |
| 12 | `class InteractiveListItem:` |
| 17 | `def __init__(self, item_data: Dict[str, Any], item_index: int, is_selected: bool...` |
| 18 | `display_text_func: Callable[[Dict[str, Any], bool], str],` |
| 140 | `current_kwargs: Dict[str, Any],` |
| 155 | `def update_function(self, func: Optional[Callable], new_kwargs: Dict[str, Any], ...` |
| 191 | `def _get_function_parameters(self, func: Callable) -> List[Dict]:` |

## openhcs/tui/file_browser.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 48 | FileManagerBrowser.__init__ | `List` |
| 55 | FileManagerBrowser.__init__ | `List` |
| 67 | FileManagerBrowser.__init__ | `List` |
| 67 | FileManagerBrowser.__init__ | `Dict` |
| 69 | FileManagerBrowser.__init__ | `List` |
| 348 | FileManagerBrowser._handle_ok | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 10 | `from typing import Any, Callable, Dict, List, Optional, Union, Coroutine # Added...` |
| 31 | `# from openhcs.tui.components import InteractiveListItem # Not used in current b...` |
| 48 | `on_path_selected: Callable[[List[Path]], Coroutine[Any, Any, None]], # Expects a...` |
| 55 | `filter_extensions: Optional[List[str]] = None # e.g., [".h5", ".zarr"]` |
| 67 | `self.current_listing: List[Dict[str, Any]] = [] # List of {'name': str, 'path': ...` |
| 69 | `self.selected_item_indices: List[int] = [] # For multi-selection` |
| 348 | `selected_paths: List[Path] = []` |

## openhcs/tui/tui_launcher.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 56 | OpenHCSTUILauncher.__init__ | `Dict` |
| 147 | OpenHCSTUILauncher._on_plate_added | `Dict` |
| 191 | OpenHCSTUILauncher._on_plate_removed | `Dict` |
| 213 | OpenHCSTUILauncher._on_plate_selected | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 11 | `from typing import Dict, Optional, Any` |
| 56 | `self.orchestrators: Dict[str, PipelineOrchestrator] = {}` |
| 147 | `async def _on_plate_added(self, plate_info: Dict[str, Any]):` |
| 191 | `async def _on_plate_removed(self, plate_info: Dict[str, Any]):` |
| 213 | `async def _on_plate_selected(self, plate_info: Dict[str, Any]):` |

## openhcs/tui/dual_step_func_editor.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 56 | DualStepFuncEditorPane.__init__ | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 15 | `from typing import Any, Callable, Dict, List, Optional, Union as TypingUnion, ge...` |
| 22 | `from prompt_toolkit.widgets import Box, Button, Frame, Label, TextArea, RadioLis...` |
| 56 | `self.step_param_inputs: Dict[str, Any] = {}` |
| 159 | `widget = CheckboxList(values=[(param_name, "")])` |
| 162 | `# For CheckboxList, changes are typically handled on save or via a dedicated cal...` |
| 167 | `# Determine initial selection for RadioList` |
| 176 | `widget = RadioList(values=options, default=initial_selection)` |
| 177 | `# The handler for RadioList is set directly on the widget instance` |
| 191 | `widget = RadioList(values=options, default=initial_selection)` |
| 302 | `# Handle variable_components specifically if it was changed by RadioList` |
| 349 | `if param_name == "variable_components" and isinstance(widget, RadioList):` |
| 355 | `elif param_name == "group_by" and isinstance(widget, RadioList):` |
| 361 | `elif isinstance(widget, CheckboxList) and actual_type is bool: # Existing bool h...` |
| 503 | `elif isinstance(associated_widget, CheckboxList):` |
| 504 | `# Assuming param_name_to_reset is the value for boolean CheckboxList` |
| 506 | `elif isinstance(associated_widget, RadioList):` |
| 524 | `logger.warning(f"Cannot reset RadioList for unknown enum param: {param_name_to_r...` |
| 530 | `self._something_changed(param_name=param_name_to_reset, widget_value=original_va...` |

## openhcs/tui/menu_bar.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 136 | MenuItemSchema.validate_menu_item | `Dict` |
| 177 | MenuStructureSchema.validate_menu_structure | `Dict` |
| 177 | MenuStructureSchema.validate_menu_structure | `List` |
| 177 | MenuStructureSchema.validate_menu_structure | `Dict` |
| 226 | MenuItem.__init__ | `List` |
| 266 | MenuItem.from_dict | `Dict` |
| 266 | MenuItem.from_dict | `Dict` |
| 399 | MenuBar.__init__ | `List` |
| 431 | MenuBar._create_handler_map | `Dict` |
| 471 | MenuBar._create_condition_map | `Dict` |
| 512 | MenuBar._load_menu_structure | `Dict` |
| 512 | MenuBar._load_menu_structure | `List` |
| 577 | MenuBar._create_menu_labels | `List` |
| 834 | MenuBar._create_submenu_container | `List` |
| 930 | MenuBar._on_save_pipeline | `Dict` |
| 939 | MenuBar._on_save_pipeline | `List` |
| 1043 | MenuBar._on_remove_step | `Dict` |
| 1065 | MenuBar._on_remove_step | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 6 | `from typing import (Any, Callable, ClassVar, Dict, FrozenSet, List, Optional,` |
| 136 | `def validate_menu_item(item: Dict[str, Any]) -> None:` |
| 177 | `def validate_menu_structure(structure: Dict[str, List[Dict[str, Any]]]) -> None:` |
| 226 | `children: Optional[List['MenuItem']] = None` |
| 238 | `children: List of child menu items (for submenu items)` |
| 266 | `def from_dict(cls, item_dict: Dict[str, Any], handler_map: Dict[str, Callable]) ...` |
| 271 | `item_dict: Dictionary with menu item data` |
| 399 | `self.active_submenu: Optional[List[MenuItem]] = None` |
| 431 | `def _create_handler_map(self) -> Dict[str, Union[Callable, Command]]: # Return t...` |
| 436 | `Dictionary mapping handler names to callables or Commands` |
| 471 | `def _create_condition_map(self) -> Dict[str, Condition]:` |
| 476 | `Dictionary mapping condition names to Condition objects` |
| 512 | `def _load_menu_structure(self) -> Dict[str, List[MenuItem]]:` |
| 517 | `Dictionary mapping menu names to lists of MenuItem objects` |
| 577 | `def _create_menu_labels(self) -> List[Label]:` |
| 582 | `List of Label widgets for menu categories` |
| 834 | `def _create_submenu_container(self, menu_items: List[MenuItem]) -> Container:` |
| 839 | `menu_items: List of menu items to display` |
| 930 | `selected_plate: Optional[Dict[str, Any]] = getattr(self.state, 'selected_plate',...` |
| 939 | `pipeline_definition: Optional[List[AbstractStep]] = getattr(active_orchestrator,...` |
| 1043 | `selected_step_dict: Optional[Dict[str, Any]] = getattr(self.state, 'selected_ste...` |
| 1065 | `current_pipeline: List[AbstractStep] = active_orchestrator.pipeline_definition` |
| 1160 | `data: Dictionary with operation and status` |

## openhcs/tui/function_pattern_editor.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 90 | FunctionPatternEditor.__init__ | `List` |
| 90 | FunctionPatternEditor.__init__ | `Dict` |
| 149 | FunctionPatternEditor._extract_pattern | `List` |
| 149 | FunctionPatternEditor._extract_pattern | `Dict` |
| 155 | FunctionPatternEditor._clone_pattern | `List` |
| 155 | FunctionPatternEditor._clone_pattern | `Dict` |
| 173 | FunctionPatternEditor._get_initial_func_for_param_editor | `Dict` |
| 298 | FunctionPatternEditor.get_pattern | `List` |
| 298 | FunctionPatternEditor.get_pattern | `Dict` |
| 450 | FunctionPatternEditor._get_current_functions | `List` |
| 473 | FunctionPatternEditor._extract_func_and_kwargs | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 9 | `When converting function patterns from List to Dict, use None as the key` |
| 22 | `from typing import Any, Callable, Dict, List, Optional, Tuple, Union` |
| 27 | `from prompt_toolkit.widgets import (Box, Button, Dialog, Label, TextArea, RadioL...` |
| 54 | `Dictionary of function metadata` |
| 90 | `def __init__(self, state: Any, initial_pattern: Union[List, Dict, None] = None, ...` |
| 149 | `def _extract_pattern(self, step) -> Union[List, Dict]:` |
| 155 | `def _clone_pattern(self, pattern) -> Union[List, Dict]:` |
| 173 | `def _get_initial_func_for_param_editor(self) -> Tuple[Optional[Callable], Dict[s...` |
| 298 | `def get_pattern(self) -> Union[List, Dict]:` |
| 366 | `"Convert to Dict Pattern",` |
| 450 | `def _get_current_functions(self) -> List:` |
| 473 | `def _extract_func_and_kwargs(self, func_item) -> Tuple[Optional[Callable], Dict]...` |

## openhcs/tui/plate_manager_core.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 56 | PlateEventHandler.on_plate_added | `Dict` |
| 57 | PlateEventHandler.on_plate_removed | `Dict` |
| 58 | PlateEventHandler.on_plate_selected | `Dict` |
| 72 | PlateManagerPane.__init__ | `List` |
| 72 | PlateManagerPane.__init__ | `Dict` |
| 154 | PlateManagerPane._get_selected_plate_data_for_action | `List` |
| 154 | PlateManagerPane._get_selected_plate_data_for_action | `Dict` |
| 169 | PlateManagerPane._get_selected_orchestrators_for_action | `List` |
| 163 | PlateManagerPane._get_selected_orchestrators_for_action | `List` |
| 222 | PlateManagerPane._handle_add_predefined_plate | `Dict` |
| 303 | PlateManagerPane._get_plate_display_text | `Dict` |
| 439 | PlateManagerPane._handle_add_dialog_result | `Dict` |
| 495 | PlateManagerPane._handle_delete_plates_request | `Dict` |
| 535 | PlateManagerPane._handle_remove_dialog_result | `Dict` |
| 551 | PlateManagerPane._handle_validation_result | `Dict` |
| 619 | PlateManagerPane._select_plate | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 33 | `from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union` |
| 42 | `from .components import InteractiveListItem # Import the new component` |
| 56 | `async def on_plate_added(self, plate: Dict[str, Any]) -> None: ...` |
| 57 | `async def on_plate_removed(self, plate: Dict[str, Any]) -> None: ...` |
| 58 | `async def on_plate_selected(self, plate: Dict[str, Any]) -> None: ...` |
| 72 | `self.plates: List[Dict[str, Any]] = []` |
| 154 | `def _get_selected_plate_data_for_action(self) -> Optional[List[Dict[str, Any]]]:` |
| 163 | `def _get_selected_orchestrators_for_action(self) -> List["PipelineOrchestrator"]...` |
| 166 | `# e.g. if multi-selection is supported by InteractiveListItem or a similar mecha...` |
| 169 | `orchestrators: List["PipelineOrchestrator"] = []` |
| 222 | `async def _handle_add_predefined_plate(self, data: Optional[Dict[str, Any]] = No...` |
| 279 | `"""Builds the HSplit container holding individual InteractiveListItem widgets fo...` |
| 291 | `item_widget = InteractiveListItem(` |
| 303 | `def _get_plate_display_text(self, plate_data: Dict[str, Any], is_selected: bool)...` |
| 340 | `# The ^/v symbols for reordering are best handled by InteractiveListItem itself` |
| 347 | `# --- New callback handlers for InteractiveListItem ---` |
| 439 | `async def _handle_add_dialog_result(self, result: Dict[str, Any]):` |
| 495 | `async def _handle_delete_plates_request(self, data: Dict[str, Any]):` |
| 535 | `async def _handle_remove_dialog_result(self, plate_to_remove: Dict[str, Any]):` |
| 551 | `async def _handle_validation_result(self, validated_plate: Dict[str, Any]):` |
| 619 | `plate_to_select: Optional[Dict[str, Any]] = None` |

## openhcs/tui/tui_architecture.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 83 | TUIState.__init__ | `Dict` |
| 84 | TUIState.__init__ | `Dict` |
| 91 | TUIState.__init__ | `Dict` |
| 96 | TUIState.__init__ | `Dict` |
| 114 | TUIState.__init__ | `Dict` |
| 114 | TUIState.__init__ | `List` |
| 143 | TUIState.set_selected_plate | `Dict` |
| 157 | TUIState.set_selected_step | `Dict` |
| 486 | OpenHCSTUI._handle_show_edit_plate_config_request | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 25 | `from typing import Any, Callable, Container, Dict, List, Optional, Union, TYPE_C...` |
| 83 | `self.selected_plate: Optional[Dict[str, Any]] = None` |
| 84 | `self.selected_step: Optional[Dict[str, Any]] = None` |
| 91 | `self.compiled_contexts: Optional[Dict[str, ProcessingContext]] = None` |
| 96 | `self.step_to_edit_config: Optional[Dict[str, Any]] = None # Renamed from selecte...` |
| 114 | `self.observers: Dict[str, List[Callable]] = {}` |
| 143 | `async def set_selected_plate(self, plate: Dict[str, Any]) -> None:` |
| 157 | `async def set_selected_step(self, step: Dict[str, Any]) -> None:` |
| 486 | `async def _handle_show_edit_plate_config_request(self, data: Dict[str, Any]):` |

## openhcs/tui/commands.py
**Imports target:** True

### String References
| Line | Reference |
| ---- | --------- |
| 8 | `from typing import Protocol, Any, TYPE_CHECKING, List # Added List` |

## openhcs/tui/step_viewer.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 42 | PipelineEditorPane.__init__ | `List` |
| 42 | PipelineEditorPane.__init__ | `Dict` |
| 43 | PipelineEditorPane.__init__ | `List` |
| 43 | PipelineEditorPane.__init__ | `Dict` |
| 139 | PipelineEditorPane._get_selected_steps_for_action | `List` |
| 139 | PipelineEditorPane._get_selected_steps_for_action | `Dict` |
| 205 | PipelineEditorPane._get_step_display_text | `Dict` |
| 264 | PipelineEditorPane._get_function_name | `Dict` |
| 282 | PipelineEditorPane._handle_edit_step_request | `Dict` |
| 466 | PipelineEditorPane._handle_step_pattern_saved | `Dict` |
| 607 | PipelineEditorPane._move_step_up | `List` |
| 661 | PipelineEditorPane._move_step_down | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 3 | `from typing import Any, Dict, List, Optional` |
| 14 | `from .components import InteractiveListItem` |
| 42 | `self.steps: List[Dict[str, Any]] = []` |
| 43 | `self.pipelines: List[Dict[str, Any]] = [] # This might be simplified if only one...` |
| 48 | `self.step_items_container_widget: Optional[HSplit] = None # Will hold HSplit of ...` |
| 139 | `def _get_selected_steps_for_action(self) -> List[Dict[str, Any]]:` |
| 191 | `item_widget = InteractiveListItem(` |
| 205 | `def _get_step_display_text(self, step_data: Dict[str, Any], is_selected: bool) -...` |
| 208 | `Reordering symbols (^/v) should be handled by InteractiveListItem.` |
| 215 | `# InteractiveListItem will handle its own selection highlighting.` |
| 216 | `# The ^/v symbols for reordering are also best handled by InteractiveListItem.` |
| 222 | `# --- New callback handlers for InteractiveListItem ---` |
| 264 | `def _get_function_name(self, step: Dict[str, Any]) -> str:` |
| 282 | `async def _handle_edit_step_request(self, data: Optional[Dict[str, Any]]) -> Non...` |
| 372 | `# If on_select is implemented in InteractiveListItem to call _edit_step,` |
| 394 | `# Rebuild the HSplit with new InteractiveListItem instances` |
| 466 | `async def _handle_step_pattern_saved(self, data: Dict[str, Any]):` |
| 607 | `pipeline: List[FunctionStep] = self.state.active_orchestrator.pipeline_definitio...` |
| 661 | `pipeline: List[FunctionStep] = self.state.active_orchestrator.pipeline_definitio...` |

## openhcs/tui/status_bar.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 107 | StatusBarSchema.validate_log_entry | `Dict` |
| 131 | StatusBarState.None | `Dict` |
| 157 | StatusBarState.with_log_entry | `Dict` |
| 167 | LogFormatter.None | `Dict` |
| 176 | LogFormatter.format_log_entry | `Dict` |
| 201 | LogFormatter.format_log_entries | `Dict` |
| 203 | LogFormatter.format_log_entries | `List` |
| 344 | StatusBar._on_operation_status_changed | `Dict` |
| 370 | StatusBar._on_error_event | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 31 | `from typing import List, Optional, Dict, Any, Deque, ClassVar, Tuple, Union` |
| 107 | `def validate_log_entry(entry: Dict[str, Any]) -> None:` |
| 131 | `log_buffer: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=S...` |
| 157 | `def with_log_entry(self, entry: Dict[str, Any]) -> 'StatusBarState':` |
| 167 | `LEVEL_STYLES: ClassVar[Dict[LogLevel, Tuple[str, str]]] = {` |
| 176 | `def format_log_entry(cls, entry: Dict[str, Any]) -> FormattedText:` |
| 201 | `def format_log_entries(cls, entries: Deque[Dict[str, Any]]) -> FormattedText:` |
| 203 | `result_fragments: List[Tuple[str, str]] = []` |
| 248 | `# Listen for requests to toggle the log drawer (e.g., from MenuBar)` |
| 344 | `async def _on_operation_status_changed(self, data: Dict[str, Any]):` |
| 370 | `async def _on_error_event(self, error_data: Dict[str, Any]):` |

## openhcs/tui/dialogs/global_settings_editor.py
**Imports target:** False

### String References
| Line | Reference |
| ---- | --------- |
| 13 | `from prompt_toolkit.widgets import Button, Dialog, Label, TextArea, RadioList, C...` |
| 55 | `# Example: VFS default_storage_backend (using RadioList as Dropdown)` |
| 57 | `self.vfs_backend_selector = RadioList(` |
| 65 | `self.microscope_selector = RadioList(` |

## openhcs/tui/dialogs/plate_config_editor.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 54 | PlateConfigEditorPane.__init__ | `Dict` |
| 125 | PlateConfigEditorPane._create_config_widgets | `List` |
| 123 | PlateConfigEditorPane._create_config_widgets | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 9 | `from typing import Any, Dict, Optional, TYPE_CHECKING` |
| 15 | `from prompt_toolkit.widgets import Box, Button, Frame, Label, TextArea, RadioLis...` |
| 54 | `self.config_param_inputs: Dict[str, Any] = {} # To store UI input widgets` |
| 97 | `elif current_value_type == bool: # For Checkbox or RadioList representing bool` |
| 102 | `elif issubclass(current_value_type, Enum): # For RadioList returning string memb...` |
| 123 | `def _create_config_widgets(self, config_obj: Any, parent_path: str = "") -> List...` |
| 125 | `widgets: List[Any] = []` |
| 170 | `input_widget = RadioList(values=enum_values, current_value=current_value.name)` |
| 171 | `# RadioList changes are typically polled. For live update:` |
| 173 | `#     # This is tricky as RadioList doesn't have a direct on_change.` |
| 174 | `#     # This would require a custom RadioList or polling.` |
| 235 | `# This loop explicitly processes RadioLists and Checkboxes.` |
| 247 | `if widget_identifier == "radiolist" and isinstance(widget, RadioList):` |

## openhcs/tui/dialogs/plate_dialog_manager.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 208 | PlateDialogManager.browser_on_path_selected | `List` |
| 509 | PlateDialogManager.show_remove_plate_dialog | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 24 | `from typing import Any, Dict, List, Optional, Protocol, Coroutine # Added List` |
| 32 | `from prompt_toolkit.widgets import RadioList as Dropdown` |
| 208 | `async def browser_on_path_selected(selected_paths: List[Path]): # Expects a List...` |
| 509 | `async def show_remove_plate_dialog(self, plate: Dict[str, Any]):` |

## openhcs/tui/services/external_editor_service.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 21 | ExternalEditorService.edit_pattern_in_external_editor | `List` |
| 21 | ExternalEditorService.edit_pattern_in_external_editor | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 5 | `from typing import Any, Dict, List, Optional, Tuple, Union` |
| 21 | `async def edit_pattern_in_external_editor(self, initial_content: str) -> Tuple[b...` |
| 30 | `A tuple: (success: bool, pattern: Optional[Union[List, Dict]], error_message: Op...` |

## openhcs/tui/services/plate_validation.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 15 | ValidationResultCallback.__call__ | `Dict` |
| 74 | PlateValidationService.validate_plate | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 6 | `from typing import Any, Callable, Dict, Optional, Protocol` |
| 15 | `async def __call__(self, result: Dict[str, Any]) -> None: ...` |
| 74 | `async def validate_plate(self, path: str, backend: str, /) -> Dict[str, Any]:` |

## openhcs/formats/position_format.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 35 | PatternFormatSchema.None | `List` |
| 36 | PatternFormatSchema.None | `List` |
| 37 | PatternFormatSchema.None | `Dict` |
| 39 | PatternFormatSchema.validate | `Dict` |
| 96 | PositionRecordData.from_dict | `Dict` |
| 120 | PositionRecordData.to_dict | `Dict` |
| 146 | module level | `List` |
| 147 | module level | `List` |
| 185 | _parse_standard_csv | `List` |
| 221 | _parse_kv_semicolon_csv | `List` |
| 267 | _serialize_standard_csv | `List` |
| 293 | _serialize_kv_semicolon_csv | `List` |
| 318 | module level | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 12 | `from typing import Any, Callable, Dict, List, Optional, Type` |
| 29 | `required_fields: List of field names that must be present` |
| 30 | `optional_fields: List of field names that may be present` |
| 31 | `field_types: Dictionary mapping field names to their expected types` |
| 35 | `required_fields: List[str]` |
| 36 | `optional_fields: List[str]` |
| 37 | `field_types: Dict[str, Type]` |
| 39 | `def validate(self, data: Dict[str, Any]) -> bool:` |
| 96 | `def from_dict(cls, data: Dict[str, Any]) -> 'PositionRecordData':` |
| 101 | `data: Dictionary containing position record data` |
| 120 | `def to_dict(self) -> Dict[str, Any]:` |
| 125 | `Dictionary representation of the position record` |
| 146 | `ParserFunction = Callable[[str], List[PositionRecordData]]` |
| 147 | `SerializerFunction = Callable[[List[PositionRecordData]], str]` |
| 185 | `def _parse_standard_csv(content: str) -> List[PositionRecordData]:` |
| 195 | `List of PositionRecordData objects` |
| 221 | `def _parse_kv_semicolon_csv(content: str) -> List[PositionRecordData]:` |
| 231 | `List of PositionRecordData objects` |
| 267 | `def _serialize_standard_csv(records: List[PositionRecordData]) -> str:` |
| 274 | `records: List of PositionRecordData objects` |
| 293 | `def _serialize_kv_semicolon_csv(records: List[PositionRecordData]) -> str:` |
| 300 | `records: List of PositionRecordData objects` |
| 318 | `POSITION_CSV_FORMATS: Dict[CSVPositionFormat, PositionCSVFormatSpec] = {` |

## openhcs/formats/func_arg_prep.py
**Imports target:** False

### String References
| Line | Reference |
| ---- | --------- |
| 19 | `- grouped_patterns: Dictionary mapping component values to patterns` |
| 20 | `- component_to_funcs: Dictionary mapping component values to processing function...` |
| 21 | `- component_to_args: Dictionary mapping component values to processing args` |
| 54 | `# List of functions or function tuples` |

## openhcs/formats/pattern/pattern_discovery.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 72 | PatternDiscoveryEngine.path_list_from_pattern | `List` |
| 169 | PatternDiscoveryEngine._natural_sort | `List` |
| 169 | PatternDiscoveryEngine._natural_sort | `List` |
| 180 | PatternDiscoveryEngine.group_patterns_by_component | `List` |
| 182 | PatternDiscoveryEngine.group_patterns_by_component | `Dict` |
| 182 | PatternDiscoveryEngine.group_patterns_by_component | `List` |
| 231 | PatternDiscoveryEngine.auto_detect_patterns | `List` |
| 232 | PatternDiscoveryEngine.auto_detect_patterns | `List` |
| 234 | PatternDiscoveryEngine.auto_detect_patterns | `List` |
| 236 | PatternDiscoveryEngine.auto_detect_patterns | `Dict` |
| 268 | PatternDiscoveryEngine._find_and_filter_images | `List` |
| 269 | PatternDiscoveryEngine._find_and_filter_images | `List` |
| 272 | PatternDiscoveryEngine._find_and_filter_images | `Dict` |
| 272 | PatternDiscoveryEngine._find_and_filter_images | `List` |
| 335 | PatternDiscoveryEngine._generate_patterns_for_files | `List` |
| 336 | PatternDiscoveryEngine._generate_patterns_for_files | `List` |
| 337 | PatternDiscoveryEngine._generate_patterns_for_files | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 14 | `from typing import Any, Dict, List, Optional, Union` |
| 72 | `def path_list_from_pattern(self, directory: Union[str, Path], pattern: Union[str...` |
| 82 | `List of matching filenames` |
| 169 | `def _natural_sort(self, file_list: List[str]) -> List[str]:` |
| 180 | `patterns: List[Union[str, 'PatternPath']],` |
| 182 | `) -> Dict[str, List[Union[str, 'PatternPath']]]:` |
| 187 | `patterns: List of patterns to group` |
| 191 | `Dictionary mapping component values to lists of patterns` |
| 231 | `well_filter: List[str],` |
| 232 | `extensions: List[str],` |
| 234 | `variable_components: List[str],` |
| 236 | `) -> Dict[str, Any]:` |
| 268 | `well_filter: List[str],` |
| 269 | `extensions: List[str],` |
| 272 | `) -> Dict[str, List[Any]]:` |
| 278 | `well_filter: List of wells to include` |
| 279 | `extensions: List of file extensions to include` |
| 284 | `Dictionary mapping wells to lists of image paths` |
| 335 | `files: List[Any],` |
| 336 | `variable_components: List[str]` |
| 337 | `) -> List['PatternPath']:` |
| 342 | `files: List of file path objects representing files` |
| 343 | `variable_components: List of components that can vary in the pattern` |
| 346 | `List of PatternPath objects` |

## openhcs/formats/pattern/pattern_resolver.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 32 | PatternDetector.auto_detect_patterns | `List` |
| 33 | PatternDetector.auto_detect_patterns | `List` |
| 37 | PatternDetector.auto_detect_patterns | `Dict` |
| 49 | PathListProvider.path_list_from_pattern | `List` |
| 63 | DirectoryLister.list_files | `List` |
| 86 | ManualRecursivePatternDetector.auto_detect_patterns | `List` |
| 87 | ManualRecursivePatternDetector.auto_detect_patterns | `List` |
| 91 | ManualRecursivePatternDetector.auto_detect_patterns | `Dict` |
| 138 | _extract_patterns_from_data | `List` |
| 123 | _extract_patterns_from_data | `List` |
| 158 | _process_pattern_list | `List` |
| 176 | _process_pattern_list | `List` |
| 161 | _process_pattern_list | `List` |
| 230 | get_patterns_for_well | `List` |
| 290 | get_patterns_for_well | `List` |
| 232 | get_patterns_for_well | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 18 | `from typing import Any, Dict, List, Optional, Protocol, Set, Union` |
| 32 | `well_filter: List[str],` |
| 33 | `variable_components: List[str],` |
| 37 | `) -> Dict[str, Any]:` |
| 42 | `class PathListProvider(Protocol):` |
| 49 | `) -> List[Union[str, Path]]:` |
| 50 | `"""List paths matching a pattern in a directory."""` |
| 54 | `class DirectoryLister(Protocol):` |
| 63 | `) -> List[Union[str, Path]]:` |
| 64 | `"""List files in a directory."""` |
| 80 | `parser: PathListProvider` |
| 81 | `filemanager: DirectoryLister` |
| 86 | `well_filter: List[str],` |
| 87 | `variable_components: List[str],` |
| 91 | `) -> Dict[str, Any]:` |
| 123 | `) -> List[str]:` |
| 133 | `List of standardized pattern strings` |
| 138 | `result: List[str] = []` |
| 158 | `patterns: List[Any],` |
| 161 | `) -> List[str]:` |
| 166 | `patterns: List of pattern objects (str or Path)` |
| 171 | `List of standardized pattern strings` |
| 176 | `result: List[str] = []` |
| 230 | `variable_components: List[str],` |
| 232 | `) -> List[str]:` |
| 246 | `List of path patterns for the well (as strings)` |
| 290 | `all_patterns: List[str] = []` |

## openhcs/io/filemanager.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 96 | FileManager.list_image_files | `List` |
| 136 | FileManager.list_files | `List` |
| 206 | FileManager.list_dir | `List` |
| 522 | FileManager.collect_dirs_and_files | `List` |
| 523 | FileManager.collect_dirs_and_files | `List` |
| 513 | FileManager.collect_dirs_and_files | `List` |
| 513 | FileManager.collect_dirs_and_files | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 11 | `from typing import List, Set, Union, Tuple` |
| 96 | `pattern: str = None, extensions: Set[str] = DEFAULT_IMAGE_EXTENSIONS, recursive:...` |
| 98 | `List all image files in a directory using the specified backend.` |
| 113 | `List of string paths for image files found` |
| 123 | `# List image files` |
| 136 | `pattern: str = None, extensions: Set[str] = None, recursive: bool = False) -> Li...` |
| 138 | `List all files in a directory using the specified backend.` |
| 153 | `List of string paths for files found` |
| 163 | `# List files` |
| 194 | `# List all files recursively` |
| 206 | `def list_dir(self, path: Union[str, Path], backend: str) -> List[str]:` |
| 513 | `) -> Tuple[List[str], List[str]]:` |
| 518 | `(dirs, files): Lists of string paths for directories and files` |
| 522 | `dirs: List[str] = []` |
| 523 | `files: List[str] = []` |

## openhcs/io/memory.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 73 | MemoryStorageBackend.list_files | `List` |
| 100 | MemoryStorageBackend.list_dir | `List` |
| 413 | MemoryStorageBackend.stat | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 17 | `from typing import Any, Dict, List, Literal, Optional, Set, Union` |
| 28 | `self._memory_store = {}  # Dict[str, Any]` |
| 73 | `) -> List[Path]:` |
| 100 | `def list_dir(self, path: Union[str, Path]) -> List[str]:` |
| 413 | `def stat(self, path: Union[str, Path]) -> Dict[str, Any]:` |

## openhcs/io/zarr.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 100 | ZarrStorageBackend.list_files | `List` |
| 92 | ZarrStorageBackend.list_files | `List` |
| 130 | ZarrStorageBackend.list_dir | `List` |
| 483 | ZarrStorageBackend.stat | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 13 | `from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union` |
| 92 | `recursive: bool = False) -> List[Path]:` |
| 100 | `result: List[Path] = []` |
| 130 | `def list_dir(self, path: Union[str, Path]) -> List[str]:` |
| 483 | `def stat(self, path: Union[str, Path]) -> Dict[str, Any]:` |

## openhcs/io/base.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 65 | StorageBackend.list_files | `List` |
| 86 | StorageBackend.list_dir | `List` |
| 235 | StorageBackend.stat | `Dict` |
| 269 | storage_registry | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 13 | `from typing import Any, Dict, List, Optional, Set, Type, Union, Callable` |
| 65 | `extensions: Optional[Set[str]] = None, recursive: bool = False) -> List[Path]:` |
| 67 | `List files in a directory, optionally filtering by pattern and extensions.` |
| 77 | `List of paths to matching files.` |
| 86 | `def list_dir(self, path: Union[str, Path]) -> List[str]:` |
| 88 | `List the names of immediate entries in a directory.` |
| 94 | `List of entry names (not full paths) in the directory.` |
| 235 | `def stat(self, path: Union[str, Path]) -> Dict[str, Any]:` |
| 269 | `def storage_registry() -> Dict[str, Type[StorageBackend]]:` |

## openhcs/io/disk.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 33 | FileFormatRegistry.__init__ | `Dict` |
| 34 | FileFormatRegistry.__init__ | `Dict` |
| 181 | DiskStorageBackend.list_files | `List` |
| 222 | DiskStorageBackend.list_dir | `List` |
| 412 | DiskStorageBackend.stat | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 14 | `from typing import Any, Callable, Dict, List, Optional, Set, Union` |
| 33 | `self._writers: Dict[str, Callable[[Path, Any], None]] = {}` |
| 34 | `self._readers: Dict[str, Callable[[Path], Any]] = {}` |
| 181 | `extensions: Optional[Set[str]] = None, recursive: bool = False) -> List[Union[st...` |
| 183 | `List files on disk, optionally filtering by pattern and extensions.` |
| 193 | `List of paths to matching files.` |
| 222 | `def list_dir(self, path: Union[str, Path]) -> List[str]:` |
| 412 | `def stat(self, path: Union[str, Path]) -> Dict[str, Any]:` |

## openhcs/validation/validate.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 18 | find_python_files | `List` |
| 45 | validate_directory | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 13 | `from typing import List, Optional, Set` |
| 18 | `def find_python_files(directory: Path, exclude_dirs: Optional[Set[str]] = None) ...` |
| 27 | `List of Python file paths.` |
| 45 | `def validate_directory(directory: Path, exclude_dirs: Optional[Set[str]] = None)...` |
| 54 | `List of validation violations.` |

## openhcs/validation/ast_validator.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 65 | ASTValidator.__init__ | `List` |
| 321 | validate_file | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 14 | `from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union` |
| 65 | `self.violations: List[ValidationViolation] = []` |
| 155 | `# List of FileManager methods that require a backend parameter` |
| 321 | `def validate_file(file_path: str) -> List[ValidationViolation]:` |
| 329 | `List of validation violations.` |

## openhcs/tests/generators/generate_synthetic_data.py
**Imports target:** False

### String References
| Line | Reference |
| ---- | --------- |
| 59 | `wells=['A01'],  # List of wells to generate` |
| 103 | `wavelength_intensities: Dictionary mapping wavelength indices to fixed intensiti...` |
| 105 | `wavelength_backgrounds: Dictionary mapping wavelength indices to background inte...` |
| 107 | `wells: List of well IDs to generate (e.g., ['A01', 'A02'])` |

## openhcs/tests/helpers/unsafe_registry.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 40 | UnsafeRegistry.None | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 12 | `from typing import Any, Dict, Type` |
| 40 | `_backends: Dict[str, BackendClass]` |

## openhcs/processing/func_registry.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 40 | module level | `Dict` |
| 40 | module level | `List` |
| 162 | get_functions_by_memory_type | `List` |
| 197 | get_function_info | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 31 | `from typing import Any, Callable, Dict, List, Optional, Set, Tuple` |
| 40 | `FUNC_REGISTRY: Dict[str, List[Callable]] = {}` |
| 162 | `def get_functions_by_memory_type(memory_type: str) -> List[Callable]:` |
| 197 | `def get_function_info(func: Callable) -> Dict[str, Any]:` |

## openhcs/processing/registry_base.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 14 | module level | `Dict` |
| 14 | module level | `List` |
| 16 | initialize_registry | `List` |
| 58 | get_functions_by_backend | `List` |
| 78 | get_function_info | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 9 | `from typing import Any, Callable, Dict, List` |
| 14 | `FUNC_REGISTRY: Dict[str, List[Callable]] = {}` |
| 16 | `def initialize_registry(memory_types: List[str]) -> None:` |
| 21 | `memory_types: List of memory type strings` |
| 58 | `def get_functions_by_backend(backend: str) -> List[Callable]:` |
| 78 | `def get_function_info(func: Callable) -> Dict[str, Any]:` |

## openhcs/processing/backends/assemblers/self_supervised_stitcher.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 165 | get_adjacency_from_layout | `List` |
| 193 | optimize_pose_graph | `Dict` |
| 196 | optimize_pose_graph | `List` |
| 197 | optimize_pose_graph | `List` |
| 312 | self_supervised_stitcher_func | `Dict` |
| 314 | self_supervised_stitcher_func | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 2 | `from typing import Any, Dict, List, Optional, Tuple` |
| 165 | `def get_adjacency_from_layout(layout_rows: int, layout_cols: int, num_tiles: int...` |
| 193 | `pairwise_homographies: Dict[Tuple[int, int], torch.Tensor],` |
| 196 | `initial_global_transforms: Optional[List[torch.Tensor]] = None` |
| 197 | `) -> List[torch.Tensor]:` |
| 312 | `pairwise_H_matrices: Dict[Tuple[int, int], torch.Tensor] = {}` |
| 314 | `global_transforms: List[torch.Tensor] = [torch.eye(3, device=device) for _ in ra...` |

## openhcs/processing/backends/analysis/rrs_vectorized_tracer.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 186 | trace_neurites_rrs_vectorized | `Dict` |
| 186 | trace_neurites_rrs_vectorized | `List` |
| 18 | trace_neurites_rrs_vectorized | `Dict` |
| 18 | trace_neurites_rrs_vectorized | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 1 | `from typing import Any, Dict, List, Tuple` |
| 18 | `) -> Dict[str, List[Tuple[float, ...]]]:` |
| 186 | `output_traces: Dict[str, List[Tuple[float, ...]]] = {}` |

## openhcs/processing/backends/analysis/focus_analyzer.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 156 | FocusAnalyzer.combined_focus_measure | `Dict` |
| 210 | FocusAnalyzer._get_focus_function | `Dict` |
| 247 | FocusAnalyzer.find_best_focus | `Dict` |
| 248 | FocusAnalyzer.find_best_focus | `List` |
| 281 | FocusAnalyzer.select_best_focus | `Dict` |
| 282 | FocusAnalyzer.select_best_focus | `List` |
| 302 | FocusAnalyzer.compute_focus_metrics | `Dict` |
| 302 | FocusAnalyzer.compute_focus_metrics | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 2 | `from typing import Dict, List, Optional, Tuple, Union` |
| 156 | `weights: Optional[Dict[str, float]] = None` |
| 210 | `def _get_focus_function(metric: Union[str, Dict[str, float]]):` |
| 246 | `image_stack: np.ndarray, # Changed from List[np.ndarray] to np.ndarray (Z, H, W)` |
| 247 | `metric: Union[str, Dict[str, float]] = "combined"` |
| 248 | `) -> Tuple[int, List[Tuple[int, float]]]:` |
| 280 | `image_stack: np.ndarray, # Changed from List[np.ndarray] to np.ndarray (Z, H, W)` |
| 281 | `metric: Union[str, Dict[str, float]] = "combined"` |
| 282 | `) -> Tuple[np.ndarray, int, List[Tuple[int, float]]]: # Return best image as (1,...` |
| 301 | `def compute_focus_metrics(image_stack: np.ndarray, # Changed from List[np.ndarra...` |
| 302 | `metric: Union[str, Dict[str, float]] = "combined") -> List[float]:` |
| 313 | `List of focus scores for each image slice` |

## openhcs/processing/backends/analysis/dxf_mask_pipeline.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 71 | _rasterize_polygons_slice_torch | `List` |
| 216 | dxf_mask_pipeline | `List` |
| 216 | dxf_mask_pipeline | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 2 | `from typing import TYPE_CHECKING, List, Tuple, Union` |
| 71 | `polygons_gpu: List["torch.Tensor"], H: int, W: int, device: "torch.device"` |
| 216 | `dxf_polygons: List[List[Tuple[float, float]]],` |

## openhcs/processing/backends/enhance/dl_edof_unsupervised.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 45 | blend_patches_to_2d_image | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 1 | `from typing import TYPE_CHECKING, List, Optional` |
| 45 | `patch_outputs: List["torch.Tensor"],  # List of [1, patch_size, patch_size]` |
| 54 | `Input patch_outputs: List of [1, patch_size, patch_size] tensors.` |

## openhcs/processing/backends/enhance/n2v2_processor_torch.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 218 | N2V2UNet.__init__ | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 16 | `from typing import List, Optional, Tuple` |
| 218 | `def __init__(self, in_channels: int = 1, out_channels: int = 1, features: List[i...` |
| 225 | `features: List of feature dimensions for each level` |

## openhcs/processing/backends/processors/jax_processor.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 352 | create_composite | `List` |
| 352 | create_composite | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 14 | `from typing import Any, List, Optional, Tuple` |
| 352 | `cls, images: List["jnp.ndarray"], weights: Optional[List[float]] = None` |
| 358 | `images: List of 3D JAX arrays, each of shape (Z, Y, X)` |
| 359 | `weights: List of weights for each image. If None, equal weights are used.` |

## openhcs/processing/backends/processors/torch_processor.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 273 | create_composite | `List` |
| 273 | create_composite | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 14 | `from typing import Any, List, Optional, Tuple` |
| 273 | `cls, images: List["torch.Tensor"], weights: Optional[List[float]] = None` |
| 279 | `images: List of 3D PyTorch tensors, each of shape (Z, Y, X)` |
| 280 | `weights: List of weights for each image. If None, equal weights are used.` |

## openhcs/processing/backends/processors/numpy_processor.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 209 | create_composite | `List` |
| 209 | create_composite | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 14 | `from typing import TYPE_CHECKING, Any, List, Optional, Tuple` |
| 209 | `cls, images: List[np.ndarray], weights: Optional[List[float]] = None` |
| 215 | `images: List of 3D NumPy arrays, each of shape (Z, Y, X)` |
| 216 | `weights: List of weights for each image. If None, equal weights are used.` |

## openhcs/processing/backends/processors/cupy_processor.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 231 | create_composite | `List` |
| 231 | create_composite | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 14 | `from typing import Any, List, Optional, Tuple` |
| 231 | `cls, images: List["cp.ndarray"], weights: Optional[List[float]] = None` |
| 237 | `images: List of 3D CuPy arrays, each of shape (Z, Y, X)` |
| 238 | `weights: List of weights for each image. If None, equal weights are used.` |

## openhcs/processing/backends/processors/tensorflow_processor.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 332 | create_composite | `List` |
| 332 | create_composite | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 14 | `from typing import Any, List, Optional, Tuple` |
| 332 | `cls, images: List["tf.Tensor"], weights: Optional[List[float]] = None` |
| 338 | `images: List of 3D TensorFlow tensors, each of shape (Z, Y, X)` |
| 339 | `weights: List of weights for each image. If None, equal weights are used.` |

## openhcs/ez/functions.py
**Imports target:** False

### String References
| Line | Reference |
| ---- | --------- |
| 26 | `channel_weights (List[float]): Weights for channel compositing` |
| 27 | `well_filter (List[str]): List of wells to process` |

## openhcs/ez/core.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 29 | OpenHCS.__init__ | `List` |
| 30 | OpenHCS.__init__ | `List` |

### String References
| Line | Reference |
| ---- | --------- |
| 9 | `from typing import List, Literal, Optional, Union` |
| 29 | `channel_weights: Optional[List[float]] = None,` |
| 30 | `well_filter: Optional[List[str]] = None,` |
| 43 | `well_filter: List of wells to process (processes all if None)` |

## openhcs/ez/utils.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 11 | detect_wells | `List` |
| 26 | suggest_channel_weights | `List` |
| 41 | create_config | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 8 | `from typing import Any, Dict, List, Optional, Union` |
| 11 | `def detect_wells(plate_path: Union[str, Path]) -> List[str]:` |
| 19 | `List[str]: List of well identifiers` |
| 26 | `def suggest_channel_weights(plate_path: Union[str, Path]) -> Optional[List[float...` |
| 34 | `List[float] or None: Suggested channel weights` |
| 41 | `def create_config(input_path: Union[str, Path, 'VirtualPath'], **kwargs) -> Dict...` |

## openhcs/ez/api.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 47 | create_config | `List` |
| 103 | run_pipeline | `Dict` |
| 106 | run_pipeline | `Dict` |
| 154 | stitch_images | `List` |
| 156 | stitch_images | `Dict` |

### String References
| Line | Reference |
| ---- | --------- |
| 10 | `from typing import Any, Dict, List, Optional, Union` |
| 47 | `well_filter: Optional[List[str]] = None,` |
| 59 | `well_filter: List of wells to process (default: None, process all wells)` |
| 103 | `config: Union[PipelineConfig, Dict[str, Any]],` |
| 106 | `) -> Dict[str, Any]:` |
| 116 | `Dictionary of results` |
| 154 | `well_filter: Optional[List[str]] = None,` |
| 156 | `) -> Dict[str, Any]:` |
| 168 | `well_filter: List of wells to process (default: None, process all wells)` |
| 172 | `Dictionary of results` |
