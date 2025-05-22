"""
FunctionStep implementation for pattern-based processing.

This module contains the FunctionStep class. During execution, FunctionStep instances
are stateless regarding their configuration. All operational parameters, including
the function(s) to execute, special input/output keys, their VFS paths, and memory types,
are retrieved from this step's entry in `context.step_plans`.
"""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, OrderedDict as TypingOrderedDict

from openhcs.constants.constants import (DEFAULT_IMAGE_EXTENSION,
                                             DEFAULT_IMAGE_EXTENSIONS,
                                             DEFAULT_SITE_PADDING, Backend,
                                             MemoryType)
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.steps.abstract import AbstractStep
from openhcs.formats.func_arg_prep import prepare_patterns_and_functions
from openhcs.core.memory.stack_utils import stack_slices, unstack_slices

logger = logging.getLogger(__name__)

def _is_3d(array: Any) -> bool:
    """Check if an array is 3D."""
    return hasattr(array, 'ndim') and array.ndim == 3

def _execute_function_core(
    func_callable: Callable,
    main_data_arg: Any,
    base_kwargs: Dict[str, Any], 
    context: 'ProcessingContext',
    special_inputs_plan: Dict[str, str],  # {'arg_name_for_func': 'special_path_value'}
    special_outputs_plan: TypingOrderedDict[str, str] # {'output_key': 'special_path_value'}, order matters
) -> Any: # Returns the main processed data stack
    """
    Executes a single callable, handling its special I/O.
    - Loads special inputs from VFS paths in `special_inputs_plan`.
    - Calls `func_callable(main_data_arg, **all_kwargs)`.
    - If `special_outputs_plan` is non-empty, expects func to return (main_out, sp_val1, sp_val2,...).
    - Saves special outputs positionally to VFS paths in `special_outputs_plan`.
    - Returns the main processed data stack.
    """
    final_kwargs = base_kwargs.copy()
    
    if special_inputs_plan:
        for arg_name, special_path_value in special_inputs_plan.items():
            logger.debug(f"Loading special input '{arg_name}' from path '{special_path_value}' (memory backend)")
            try:
                final_kwargs[arg_name] = context.filemanager.load(special_path_value, MemoryType.MEMORY.value)
            except Exception as e:
                logger.error(f"Failed to load special input '{arg_name}' from '{special_path_value}': {e}", exc_info=True)
                raise 
    
    raw_function_output = func_callable(main_data_arg, **final_kwargs)
    main_output_data = raw_function_output
    
    if special_outputs_plan: 
        num_special_outputs = len(special_outputs_plan)
        if not isinstance(raw_function_output, tuple) or len(raw_function_output) != (1 + num_special_outputs):
            raise ValueError(
                f"Function '{getattr(func_callable, '__name__', 'unknown')}' was expected to return a tuple of "
                f"{1 + num_special_outputs} values (main_output + {num_special_outputs} special) "
                f"based on 'special_outputs' in step plan, but returned {len(raw_function_output) if isinstance(raw_function_output, tuple) else type(raw_function_output)} values."
            )
        main_output_data = raw_function_output[0]
        returned_special_values_tuple = raw_function_output[1:]

        # Iterate through special_outputs_plan (which must be ordered by compiler)
        # and match with positionally returned special values.
        for i, (output_key, vfs_path) in enumerate(special_outputs_plan.items()):
            if i < len(returned_special_values_tuple):
                value_to_save = returned_special_values_tuple[i]
                logger.debug(f"Saving special output '{output_key}' to VFS path '{vfs_path}' (memory backend)")
                context.filemanager.save(value_to_save, vfs_path, MemoryType.MEMORY.value)
            else:
                # This indicates a mismatch that should ideally be caught by schema/validation
                logger.error(f"Mismatch: {num_special_outputs} special outputs planned, but fewer values returned by function for key '{output_key}'.")
                # Or, if partial returns are allowed, this might be a warning. For now, error.
                raise ValueError(f"Function did not return enough values for all planned special outputs. Missing value for '{output_key}'.")
    
    return main_output_data

def _execute_chain_core(
    initial_data_stack: Any,
    func_chain: List[Union[Callable, Tuple[Callable, Dict]]], 
    context: 'ProcessingContext',
    step_special_inputs_plan: Dict[str, str], 
    step_special_outputs_plan: TypingOrderedDict[str, str] 
) -> Any: 
    current_stack = initial_data_stack
    for i, func_item in enumerate(func_chain):
        actual_callable: Callable
        base_kwargs_for_item: Dict[str, Any] = {}
        is_last_in_chain = (i == len(func_chain) - 1)

        if isinstance(func_item, tuple) and len(func_item) == 2 and callable(func_item[0]):
            actual_callable, base_kwargs_for_item = func_item
        elif callable(func_item):
            actual_callable = func_item
        else:
            raise TypeError(f"Invalid item in function chain: {func_item}.")
        
        outputs_plan_for_this_call = step_special_outputs_plan if is_last_in_chain else {}
        
        current_stack = _execute_function_core(
            func_callable=actual_callable,
            main_data_arg=current_stack,
            base_kwargs=base_kwargs_for_item,
            context=context,
            special_inputs_plan=step_special_inputs_plan, 
            special_outputs_plan=outputs_plan_for_this_call
        )
    return current_stack

def _process_single_pattern_group(
    context: 'ProcessingContext',
    pattern_group_info: Any, 
    executable_func_or_chain: Any, 
    base_func_args: Dict[str, Any], 
    step_input_dir: Path,
    step_output_dir: Path,
    well_id: str,
    component_value: str, 
    read_backend: str,
    write_backend: str,
    input_memory_type_from_plan: str, # Explicitly from plan
    output_memory_type_from_plan: str, # Explicitly from plan
    device_id: Optional[int],
    same_directory: bool,
    force_disk_output_flag: bool,
    special_inputs_map: Dict[str, str], 
    special_outputs_map: TypingOrderedDict[str, str]
) -> None:
    start_time = time.time()
    pattern_repr = str(pattern_group_info)[:100]
    logger.debug(f"Processing pattern group {pattern_repr} for well {well_id}, component {component_value}")

    try:
        if not context.microscope_handler:
             raise RuntimeError("MicroscopeHandler not available in context.")

        matching_files = context.microscope_handler.path_list_from_pattern(
            str(step_input_dir), pattern_group_info, read_backend
        )

        if not matching_files:
            logger.warning(f"No matching files for pattern group {pattern_repr} in {step_input_dir}")
            return

        raw_slices = []
        for file_path_suffix in matching_files:
            full_file_path = step_input_dir / file_path_suffix
            try:
                image = context.filemanager.load_image(str(full_file_path), read_backend)
                if image is not None: raw_slices.append(image)
            except Exception as e:
                logger.error(f"Error loading image {full_file_path}: {e}", exc_info=True)
        
        if not raw_slices:
            logger.warning(f"No valid images loaded for pattern group {pattern_repr} in {step_input_dir}")
            return

        main_data_stack = stack_slices(
            slices=raw_slices, memory_type=input_memory_type_from_plan, gpu_id=device_id, allow_single_slice=False
        )
        
        final_base_kwargs = base_func_args.copy()
        
        if isinstance(executable_func_or_chain, list): 
            processed_stack = _execute_chain_core(
                main_data_stack, executable_func_or_chain, context,
                special_inputs_map, special_outputs_map
            )
        elif callable(executable_func_or_chain):
            processed_stack = _execute_function_core(
                executable_func_or_chain, main_data_stack, final_base_kwargs, context,
                special_inputs_map, special_outputs_map
            )
        else:
            raise TypeError(f"Invalid executable_func_or_chain: {type(executable_func_or_chain)}")

        if not _is_3d(processed_stack):
             raise ValueError(f"Main processing must result in a 3D array, got {getattr(processed_stack, 'shape', 'unknown')}")

        output_slices = unstack_slices(
            array=processed_stack, memory_type=output_memory_type_from_plan, gpu_id=device_id, validate_slices=True
        )

        for i, img_slice in enumerate(output_slices):
            try:
                site = pattern_group_info.get('site', i + 1) if isinstance(pattern_group_info, dict) else i + 1
                output_filename = context.microscope_handler.parser.construct_filename(
                    well=well_id, site=site, channel=component_value, z_step=i + 1, 
                    extension=DEFAULT_IMAGE_EXTENSION, site_padding=DEFAULT_SITE_PADDING
                )
                output_path = Path(step_output_dir) / output_filename

                if same_directory and context.filemanager.exists(str(output_path), write_backend):
                    context.filemanager.delete_file(str(output_path), write_backend)
                
                context.filemanager.save_image(img_slice, str(output_path), write_backend)

                if force_disk_output_flag and write_backend != Backend.DISK.value:
                    logger.info(f"Force disk output: saving additional copy to disk: {output_path}")
                    context.filemanager.save_image(img_slice, str(output_path), Backend.DISK.value)
            except Exception as e:
                logger.error(f"Error saving output slice {i} for pattern {pattern_repr}: {e}", exc_info=True)
        
        logger.debug(f"Finished pattern group {pattern_repr} in {(time.time() - start_time):.2f}s.")
    except Exception as e:
        logger.error(f"Error processing pattern group {pattern_repr}: {e}", exc_info=True)
        raise ValueError(f"Failed to process pattern group {pattern_repr}: {e}") from e

class FunctionStep(AbstractStep):
    @property
    def requires_disk_input(self) -> bool: return False 
    @property
    def requires_disk_output(self) -> bool: return False

    def __init__(
        self,
        func: Union[Callable, Tuple[Callable, Dict], List[Union[Callable, Tuple[Callable, Dict]]]], 
        *, name: Optional[str] = None, variable_components: Optional[List[str]] = ['site'], 
        group_by: str = "channel", force_disk_output: bool = False
    ):
        actual_func_for_name = func
        if isinstance(func, tuple): actual_func_for_name = func[0]
        elif isinstance(func, list) and func:
             first_item = func[0]
             if isinstance(first_item, tuple): actual_func_for_name = first_item[0]
             elif callable(first_item): actual_func_for_name = first_item
        
        super().__init__(
            name=name or getattr(actual_func_for_name, '__name__', 'FunctionStep'),
            variable_components=variable_components, group_by=group_by,
            force_disk_output=force_disk_output
        )
        self.func = func # This is used by prepare_patterns_and_functions at runtime

    def process(self, context: 'ProcessingContext') -> None:
        step_plan = context.get_step_plan(self.step_id)
        if not step_plan:
            raise ValueError(f"Step plan not found for step: {self.name} (ID: {self.step_id})")

        try:
            well_id = step_plan.get('well_id')
            step_input_dir = Path(step_plan.get('input_dir'))
            step_output_dir = Path(step_plan.get('output_dir'))
            variable_components = step_plan.get('variable_components', ['site'])
            group_by = step_plan.get('group_by', 'channel')
            
            # special_inputs/outputs are dicts: {'key': 'vfs_path_value'}
            special_inputs = step_plan.get('special_inputs', {}) 
            special_outputs = step_plan.get('special_outputs', {}) # Should be OrderedDict if order matters

            force_disk_output = step_plan.get('force_disk_output', False)
            read_backend = step_plan.get('read_backend', Backend.DISK.value)
            write_backend = step_plan.get('write_backend', Backend.DISK.value)
            input_mem_type = step_plan.get('input_memory_type', MemoryType.NUMPY.value)
            output_mem_type = step_plan.get('output_memory_type', MemoryType.NUMPY.value)
            device_id = step_plan.get('gpu_id')

            if not all([well_id, step_input_dir, step_output_dir]):
                raise ValueError(f"Plan missing essential keys for step {self.step_id}")

            same_dir = str(step_input_dir) == str(step_output_dir)
            logger.info(f"Step {self.step_id} ({step_plan.get('step_name', self.name)}) I/O: read='{read_backend}', write='{write_backend}'.")

            if not context.microscope_handler:
                raise RuntimeError(f"MicroscopeHandler not in context for step {self.step_id}")

            patterns_by_well = context.microscope_handler.auto_detect_patterns(
                str(step_input_dir), [well_id], DEFAULT_IMAGE_EXTENSIONS,
                group_by, variable_components, read_backend # backend is positional
            )

            if well_id not in patterns_by_well or not patterns_by_well[well_id]:
                raise ValueError(f"No patterns for well {well_id} in {step_input_dir} for step {self.step_id}.")
            
            # self.func (the callable/tuple/list from __init__) is used here
            grouped_patterns, comp_to_funcs, comp_to_base_args = prepare_patterns_and_functions(
                patterns_by_well[well_id], self.func, component=group_by
            )

            for comp_val, current_pattern_list in grouped_patterns.items():
                exec_func_or_chain = comp_to_funcs[comp_val]
                base_kwargs = comp_to_base_args[comp_val]
                for pattern_item in current_pattern_list:
                    _process_single_pattern_group(
                        context, pattern_item, exec_func_or_chain, base_kwargs,
                        step_input_dir, step_output_dir, well_id, comp_val,
                        read_backend, write_backend, input_mem_type, output_mem_type,
                        device_id, same_dir, force_disk_output,
                        special_inputs, special_outputs # Pass the maps from step_plan
                    )
            logger.info(f"FunctionStep {self.step_id} ({step_plan.get('step_name', self.name)}) completed for well {well_id}.")
        except Exception as e:
            logger.error(f"Error in FunctionStep {self.step_id} ({step_plan.get('step_name', self.name)}): {e}", exc_info=True)
            raise