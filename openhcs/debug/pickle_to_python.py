#!/usr/bin/env python3
"""
Pickle to Python Converter - Convert OpenHCS debug pickle files to runnable Python scripts
"""

import sys
import dill as pickle
import inspect
import dataclasses
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from enum import Enum

# It's better to have these imports at the top level
from openhcs.core.config import GlobalPipelineConfig, PathPlanningConfig, VFSConfig, ZarrConfig
from openhcs.core.steps.function_step import FunctionStep

def _value_to_repr(value):
    """Converts a value to its Python representation string."""
    if isinstance(value, Enum):
        return f"{value.__class__.__name__}.{value.name}"
    elif isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, Path):
        return f'Path("{value}")'
    return repr(value)

def generate_clean_dataclass_repr(instance, indent_level=0, clean_mode=False):
    """
    Generates a clean, readable Python representation of a dataclass instance,
    omitting fields that are set to their default values if clean_mode is True.
    This function is recursive and handles nested dataclasses.
    """
    if not dataclasses.is_dataclass(instance):
        return _value_to_repr(instance)

    lines = []
    indent_str = "    " * indent_level
    child_indent_str = "    " * (indent_level + 1)
    
    # Get a default instance of the same class for comparison
    default_instance = instance.__class__()

    for field in dataclasses.fields(instance):
        field_name = field.name
        current_value = getattr(instance, field_name)
        default_value = getattr(default_instance, field_name)

        if clean_mode and current_value == default_value:
            continue

        if dataclasses.is_dataclass(current_value):
            # Recursively generate representation for nested dataclasses
            nested_repr = generate_clean_dataclass_repr(current_value, indent_level + 1, clean_mode)
            lines.append(f"{child_indent_str}{field_name}={current_value.__class__.__name__}(\n{nested_repr}\n{child_indent_str})")
        else:
            value_repr = _value_to_repr(current_value)
            lines.append(f"{child_indent_str}{field_name}={value_repr}")

    if not lines:
        return "" # Return empty string if all fields were default in clean_mode

    return ",\n".join(lines)


def convert_pickle_to_python(pickle_path, output_path=None, clean_mode=False):
    """Convert an OpenHCS debug pickle file to a runnable Python script."""
    
    pickle_file = Path(pickle_path)
    if not pickle_file.exists():
        print(f"Error: Pickle file not found: {pickle_path}")
        return
    
    if output_path is None:
        output_path = pickle_file.with_suffix('.py')
    
    print(f"Converting {pickle_file} to {output_path} (Clean Mode: {clean_mode})")
    
    try:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        
        # Generate Python script
        with open(output_path, 'w') as f:
            f.write('#!/usr/bin/env python3\n')
            f.write('"""\n')
            f.write(f'OpenHCS Pipeline Script - Generated from {pickle_file.name}\n')
            f.write(f'Generated: {datetime.now()}\n')
            f.write('"""\n\n')
            
            # Imports
            f.write('import sys\n')
            f.write('import os\n')
            f.write('from pathlib import Path\n\n')
            f.write('# Add OpenHCS to path\n')
            f.write('sys.path.insert(0, "/home/ts/code/projects/openhcs")\n\n')
            
            f.write('from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator\n')
            f.write('from openhcs.core.steps.function_step import FunctionStep\n')
            f.write('from openhcs.core.config import (GlobalPipelineConfig, PathPlanningConfig, VFSConfig, ZarrConfig, \n'
                    '                         MaterializationBackend, ZarrCompressor, ZarrChunkStrategy)\n')
            f.write('from openhcs.constants.constants import VariableComponents, Backend, Microscope\n\n')
            
            # Import all the functions used in the pipeline
            function_imports = defaultdict(set)
            enum_imports = defaultdict(set)

            def find_and_register_imports(obj):
                """Recursively find and register all function and enum imports."""
                if isinstance(obj, Enum):
                    module = obj.__class__.__module__
                    name = obj.__class__.__name__
                    if module and name and module.startswith('openhcs'):
                        enum_imports[module].add(name)
                elif callable(obj):
                    module = getattr(obj, '__module__', None)
                    name = getattr(obj, '__name__', None)
                    if module and name and module.startswith('openhcs'):
                        function_imports[module].add(name)
                elif isinstance(obj, (list, tuple)):
                    for item in obj:
                        find_and_register_imports(item)
                elif isinstance(obj, dict):
                    for value in obj.values():
                        find_and_register_imports(value)

            # Extract all imports by traversing the data structure
            if 'pipeline_data' in data:
                for steps in data['pipeline_data'].values():
                    for step in steps:
                        if hasattr(step, 'func'):
                            find_and_register_imports(step.func)
                        if hasattr(step, 'variable_components'):
                            find_and_register_imports(step.variable_components)
            # Traverse global_config too
            find_and_register_imports(data.get('global_config'))


            # Write function and enum imports
            f.write('# Function and Enum imports\n')
            all_imports = function_imports
            for module, names in enum_imports.items():
                all_imports[module].update(names)

            for module, names in sorted(all_imports.items()):
                f.write(f"from {module} import {', '.join(sorted(names))}\n")
            f.write('\n')

            # Write the main function
            f.write('def create_pipeline():\n')
            f.write('    """Create and return the pipeline configuration."""\n\n')
            
            # Write plate paths
            f.write('    # Plate paths\n')
            f.write(f'    plate_paths = {repr(data["plate_paths"])}\n\n')
            
            # Write global config using the new generic generator
            f.write('    # Global configuration\n')
            config_repr = generate_clean_dataclass_repr(data['global_config'], indent_level=1, clean_mode=clean_mode)
            f.write(f'    global_config = GlobalPipelineConfig(\n{config_repr}\n    )\n\n')
            
            # Write pipeline steps
            f.write('    # Pipeline steps\n')
            f.write('    pipeline_data = {}\n\n')
            
            default_step = FunctionStep(func=lambda: None) # For comparing default values
            
            for plate_path, steps in data['pipeline_data'].items():
                f.write(f'    # Steps for plate: {Path(plate_path).name}\n')
                f.write(f'    steps = []\n\n')
                
                for i, step in enumerate(steps):
                    f.write(f'    # Step {i+1}: {step.name}\n')
                    func_repr = generate_readable_function_repr(step.func, indent=2, clean_mode=clean_mode)
                    
                    # Generate arguments for FunctionStep, respecting clean_mode
                    step_args = [f'func={func_repr}']
                    
                    params_to_check = {
                        "name": (f'name="{step.name}"', step.name, default_step.name),
                        "variable_components": (
                            f'variable_components=[VariableComponents.{step.variable_components[0].name}]',
                            step.variable_components,
                            default_step.variable_components
                        ),
                        "force_disk_output": (f'force_disk_output={step.force_disk_output}', step.force_disk_output, default_step.force_disk_output)
                    }

                    for name, (repr_str, current_val, default_val) in params_to_check.items():
                        if not clean_mode or current_val != default_val:
                            step_args.append(repr_str)
                    
                    args_str = ",\n        ".join(step_args)
                    f.write(f'    step_{i+1} = FunctionStep(\n        {args_str}\n    )\n')
                    f.write(f'    steps.append(step_{i+1})\n\n')
                
                f.write(f'    pipeline_data["{plate_path}"] = steps\n\n')
            
            f.write('    return plate_paths, pipeline_data, global_config\n\n')
            
            # ... (rest of the file remains the same for now) ...
            f.write('def setup_signal_handlers():\n')
            f.write('    """Setup signal handlers to kill all child processes and threads on Ctrl+C."""\n')
            f.write('    import signal\n')
            f.write('    import os\n')
            f.write('    import sys\n\n')
            f.write('    def cleanup_and_exit(signum, frame):\n')
            f.write('        print(f"\\n🔥 Signal {signum} received! Cleaning up all processes and threads...")\n\n')
            f.write('        os._exit(1)\n\n')
            f.write('    signal.signal(signal.SIGINT, cleanup_and_exit)\n')
            f.write('    signal.signal(signal.SIGTERM, cleanup_and_exit)\n\n')

            f.write('def run_pipeline():\n')
            f.write('    os.environ["OPENHCS_SUBPROCESS_MODE"] = "1"\n')
            f.write('    plate_paths, pipeline_data, global_config = create_pipeline()\n')
            f.write('    from openhcs.core.orchestrator.gpu_scheduler import setup_global_gpu_registry\n')
            f.write('    setup_global_gpu_registry(global_config=global_config)\n')
            f.write('    for plate_path in plate_paths:\n')
            f.write('        orchestrator = PipelineOrchestrator(plate_path, global_config=global_config)\n')
            f.write('        orchestrator.initialize()\n')
            f.write('        compiled_contexts = orchestrator.compile_pipelines(pipeline_data[plate_path])\n')
            f.write('        orchestrator.execute_compiled_plate(\n')
            f.write('            pipeline_definition=pipeline_data[plate_path],\n')
            f.write('            compiled_contexts=compiled_contexts,\n')
            f.write('            max_workers=global_config.num_workers\n')
            f.write('        )\n\n')

            f.write('if __name__ == "__main__":\n')
            f.write('    setup_signal_handlers()\n')
            f.write('    run_pipeline()\n')

        
        print(f"✅ Successfully converted to {output_path}")
        print(f"You can now run: python {output_path}")
        
    except Exception as e:
        print(f"Error converting pickle file: {e}")
        import traceback
        traceback.print_exc()


def generate_readable_function_repr(func_obj, indent=0, clean_mode=False):
    """
    Generate a readable and optionally clean Python representation of a function pattern.
    - Strips default kwargs from function tuples.
    - Simplifies `(func, {})` to `func`.
    - Simplifies `[func]` to `func`.
    """
    indent_str = "    " * indent
    next_indent_str = "    " * (indent + 1)

    if callable(func_obj):
        return f"{func_obj.__name__}"
    
    elif isinstance(func_obj, tuple) and len(func_obj) == 2 and callable(func_obj[0]):
        func, args = func_obj
        
        if not args and clean_mode:
            return f"{func.__name__}"

        # Get function signature to find default values
        try:
            sig = inspect.signature(func)
            default_params = {
                k: v.default for k, v in sig.parameters.items()
                if v.default is not inspect.Parameter.empty
            }
        except (ValueError, TypeError): # Handle built-ins or other un-inspectables
            default_params = {}

        # Filter out default values in clean_mode
        final_args = {}
        for k, v in args.items():
            if not clean_mode or k not in default_params or v != default_params[k]:
                final_args[k] = v
        
        if not final_args:
             return f"{func.__name__}" if clean_mode else f"({func.__name__}, {{}})"
        
        args_items = []
        for k, v in final_args.items():
            v_repr = generate_readable_function_repr(v, indent + 2, clean_mode)
            args_items.append(f"{next_indent_str}    '{k}': {v_repr}")
        args_str = "{\n" + ",\n".join(args_items) + f"\n{next_indent_str}}}"
        return f"({func.__name__}, {args_str})"

    elif isinstance(func_obj, list):
        if clean_mode and len(func_obj) == 1:
            return generate_readable_function_repr(func_obj[0], indent, clean_mode)
        if not func_obj:
            return "[]"
        items = [generate_readable_function_repr(item, indent, clean_mode) for item in func_obj]
        return f"[\n{next_indent_str}{f',\n{next_indent_str}'.join(items)}\n{indent_str}]"
    
    elif isinstance(func_obj, dict):
        if not func_obj:
            return "{}"
        items = []
        for key, value in func_obj.items():
            value_repr = generate_readable_function_repr(value, indent, clean_mode)
            items.append(f"{next_indent_str}'{key}': {value_repr}")
        return f"{{{',\n'.join(items)}\n{indent_str}}}"
        
    else:
        return _value_to_repr(func_obj)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert OpenHCS debug pickle files to runnable Python scripts.")
    parser.add_argument("pickle_file", help="Path to the input pickle file.")
    parser.add_argument("output_file", nargs='?', default=None, help="Path to the output Python script file (optional).")
    parser.add_argument("--clean", action="store_true", help="Generate a clean script with only non-default parameters.")
    
    args = parser.parse_args()
    
    convert_pickle_to_python(args.pickle_file, args.output_file, clean_mode=args.clean)

if __name__ == "__main__":
    main()
