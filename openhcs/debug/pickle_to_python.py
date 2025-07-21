#!/usr/bin/env python3
"""
Pickle to Python Converter - Convert OpenHCS debug pickle files to runnable Python scripts
"""

import sys
import dill as pickle
import inspect
from pathlib import Path
from datetime import datetime

def convert_pickle_to_python(pickle_path, output_path=None):
    """Convert an OpenHCS debug pickle file to a runnable Python script."""
    
    pickle_file = Path(pickle_path)
    if not pickle_file.exists():
        print(f"Error: Pickle file not found: {pickle_path}")
        return
    
    if output_path is None:
        output_path = pickle_file.with_suffix('.py')
    
    print(f"Converting {pickle_file} to {output_path}")
    
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
            f.write('from openhcs.core.config import GlobalPipelineConfig\n')
            f.write('from openhcs.constants.constants import VariableComponents\n\n')
            
            # Import all the functions used in the pipeline
            from collections import defaultdict
            function_imports = defaultdict(set)
            
            def extract_function_imports(func_obj):
                """Extract import statements for functions."""
                if callable(func_obj):
                    module = getattr(func_obj, '__module__', None)
                    name = getattr(func_obj, '__name__', None)
                    if module and name and module.startswith('openhcs'):
                        function_imports[module].add(name)
                elif isinstance(func_obj, (list, tuple)):
                    for item in func_obj:
                        if isinstance(item, tuple) and len(item) > 0:
                            extract_function_imports(item[0])
                        else:
                            extract_function_imports(item)
                elif isinstance(func_obj, dict):
                    for value in func_obj.values():
                        extract_function_imports(value)
            
            # Extract imports from pipeline
            if 'pipeline_data' in data:
                for plate_path, steps in data['pipeline_data'].items():
                    for step in steps:
                        if hasattr(step, 'func'):
                            extract_function_imports(step.func)
            
            # Write function imports
            f.write('# Function imports\n')
            for module, names in sorted(function_imports.items()):
                f.write(f"from {module} import {', '.join(sorted(names))}\n")
            f.write('\n')
            
            # Write enum imports
            f.write('# Enum imports\n')
            f.write('from openhcs.processing.backends.analysis.cell_counting_pyclesperanto import DetectionMethod\n')
            f.write('from openhcs.processing.backends.analysis.skan_axon_analysis import AnalysisDimension\n')
            f.write('from openhcs.core.config import PathPlanningConfig, VFSConfig, ZarrConfig, MaterializationBackend, ZarrCompressor, ZarrChunkStrategy\n')
            f.write('from openhcs.constants.constants import Backend, Microscope\n\n')
            
            # Write the main function
            f.write('def create_pipeline():\n')
            f.write('    """Create and return the pipeline configuration."""\n\n')
            
            # Write plate paths
            f.write('    # Plate paths\n')
            f.write(f'    plate_paths = {repr(data["plate_paths"])}\n\n')
            
            # Write global config
            f.write('    # Global configuration\n')
            config = data['global_config']
            f.write(f'    global_config = GlobalPipelineConfig(\n')
            f.write(f'        num_workers={config.num_workers},\n')
            f.write(f'        path_planning=PathPlanningConfig(\n')
            f.write(f'            output_dir_suffix="{config.path_planning.output_dir_suffix}",\n')
            f.write(f'            global_output_folder="{config.path_planning.global_output_folder}",\n')
            f.write(f'            materialization_results_path="{config.path_planning.materialization_results_path}"\n')
            f.write(f'        ),\n')
            f.write(f'        vfs=VFSConfig(\n')
            f.write(f'            intermediate_backend=Backend.{config.vfs.intermediate_backend.name},\n')
            f.write(f'            materialization_backend=MaterializationBackend.{config.vfs.materialization_backend.name}\n')
            f.write(f'        ),\n')
            f.write(f'        zarr=ZarrConfig(\n')
            f.write(f'            store_name="{config.zarr.store_name}",\n')
            f.write(f'            compressor=ZarrCompressor.{config.zarr.compressor.name},\n')
            f.write(f'            compression_level={config.zarr.compression_level},\n')
            f.write(f'            shuffle={config.zarr.shuffle},\n')
            f.write(f'            chunk_strategy=ZarrChunkStrategy.{config.zarr.chunk_strategy.name},\n')
            f.write(f'            ome_zarr_metadata={config.zarr.ome_zarr_metadata},\n')
            f.write(f'            write_plate_metadata={config.zarr.write_plate_metadata}\n')
            f.write(f'        ),\n')
            f.write(f'        microscope=Microscope.{config.microscope.name},\n')
            f.write(f'        use_threading={config.use_threading}\n')
            f.write(f'    )\n\n')
            
            # Write pipeline steps
            f.write('    # Pipeline steps\n')
            f.write('    pipeline_data = {}\n\n')
            
            for plate_path, steps in data['pipeline_data'].items():
                f.write(f'    # Steps for plate: {Path(plate_path).name}\n')
                f.write(f'    steps = []\n\n')
                
                for i, step in enumerate(steps):
                    f.write(f'    # Step {i+1}: {step.name}\n')
                    
                    # Handle different function types
                    func_repr = generate_function_repr(step.func)
                    
                    f.write(f'    step_{i+1} = FunctionStep(\n')
                    f.write(f'        func={func_repr},\n')
                    f.write(f'        name="{step.name}",\n')
                    f.write(f'        variable_components=[VariableComponents.{step.variable_components[0].name}],\n')
                    f.write(f'        force_disk_output=False\n')
                    f.write(f'    )\n')
                    f.write(f'    steps.append(step_{i+1})\n\n')
                
                f.write(f'    pipeline_data["{plate_path}"] = steps\n\n')
            
            f.write('    return plate_paths, pipeline_data, global_config\n\n')
            
            # Write signal handling and cleanup functions
            f.write('def setup_signal_handlers():\n')
            f.write('    """Setup signal handlers to kill all child processes and threads on Ctrl+C."""\n')
            f.write('    import signal\n')
            f.write('    import os\n')
            f.write('    import threading\n')
            f.write('    import sys\n\n')
            f.write('    def cleanup_and_exit(signum, frame):\n')
            f.write('        print(f"\\n🔥 Signal {signum} received! Cleaning up all processes and threads...")\n\n')
            f.write('        # Kill all child processes in our process group\n')
            f.write('        try:\n')
            f.write('            import psutil\n')
            f.write('            current_process = psutil.Process(os.getpid())\n')
            f.write('            children = current_process.children(recursive=True)\n')
            f.write('            print(f"🔥 Found {len(children)} child processes to terminate")\n\n')
            f.write('            for child in children:\n')
            f.write('                try:\n')
            f.write('                    print(f"🔥 Terminating child process {child.pid}")\n')
            f.write('                    child.terminate()\n')
            f.write('                except Exception as e:\n')
            f.write('                    print(f"🔥 Error terminating child {child.pid}: {e}")\n\n')
            f.write('            # Wait for children to terminate gracefully\n')
            f.write('            psutil.wait_procs(children, timeout=3)\n\n')
            f.write('            # Force kill any remaining children\n')
            f.write('            for child in children:\n')
            f.write('                try:\n')
            f.write('                    if child.is_running():\n')
            f.write('                        print(f"🔥 Force killing child process {child.pid}")\n')
            f.write('                        child.kill()\n')
            f.write('                except Exception as e:\n')
            f.write('                    print(f"🔥 Error force killing child {child.pid}: {e}")\n\n')
            f.write('        except ImportError:\n')
            f.write('            print("🔥 psutil not available, using process group kill")\n')
            f.write('            try:\n')
            f.write('                # Kill entire process group\n')
            f.write('                os.killpg(os.getpgrp(), signal.SIGTERM)\n')
            f.write('            except Exception as e:\n')
            f.write('                print(f"🔥 Error killing process group: {e}")\n\n')
            f.write('        # Force exit\n')
            f.write('        print("🔥 Forcing exit...")\n')
            f.write('        os._exit(1)\n\n')
            f.write('    # Register signal handlers\n')
            f.write('    signal.signal(signal.SIGINT, cleanup_and_exit)   # Ctrl+C\n')
            f.write('    signal.signal(signal.SIGTERM, cleanup_and_exit)  # Termination\n')
            f.write('    print("🔥 Signal handlers registered for cleanup")\n\n')
            f.write('def run_pipeline():\n')
            f.write('    """Run the pipeline - replicating subprocess runner logic."""\n')
            f.write('    # Setup signal handlers first\n')
            f.write('    setup_signal_handlers()\n\n')
            f.write('    # Create new process group for proper cleanup\n')
            f.write('    import os\n')
            f.write('    try:\n')
            f.write('        os.setpgrp()  # Create new process group\n')
            f.write('        print(f"🔥 Created process group {os.getpgrp()}")\n')
            f.write('    except Exception as e:\n')
            f.write('        print(f"🔥 Warning: Could not create process group: {e}")\n\n')
            f.write('    # Set subprocess mode like the subprocess runner does\n')
            f.write('    os.environ["OPENHCS_SUBPROCESS_MODE"] = "1"\n\n')
            f.write('    plate_paths, pipeline_data, global_config = create_pipeline()\n\n')
            f.write('    # Initialize GPU registry (like subprocess runner does)\n')
            f.write('    from openhcs.core.orchestrator.gpu_scheduler import setup_global_gpu_registry\n')
            f.write('    print("🔥 Setting up GPU registry...")\n')
            f.write('    setup_global_gpu_registry(global_config=global_config)\n')
            f.write('    print("🔥 GPU registry setup completed!")\n\n')
            f.write('    # Execute pipeline for each plate (replicating subprocess runner)\n')
            f.write('    for plate_path in plate_paths:\n')
            f.write('        print(f"🔥 Processing plate: {plate_path}")\n')
            f.write('        steps = pipeline_data[plate_path]\n\n')
            f.write('        # Create orchestrator for this plate\n')
            f.write('        orchestrator = PipelineOrchestrator(plate_path, global_config=global_config)\n\n')
            f.write('        # Initialize orchestrator (like subprocess runner does)\n')
            f.write('        print("🔥 Initializing orchestrator...")\n')
            f.write('        orchestrator.initialize()\n')
            f.write('        print("🔥 Orchestrator initialized!")\n\n')
            f.write('        # Step 1: Compile the plate (like subprocess runner does)\n')
            f.write('        print("🔥 Compiling plate...")\n')
            f.write('        compiled_contexts = orchestrator.compile_pipelines(steps)\n')
            f.write('        print(f"🔥 Compiled {len(compiled_contexts)} contexts")\n\n')
            f.write('        # Step 2: Execute the compiled plate (like subprocess runner does)\n')
            f.write('        print("🔥 Executing compiled plate...")\n')
            f.write('        results = orchestrator.execute_compiled_plate(\n')
            f.write('            pipeline_definition=steps,\n')
            f.write('            compiled_contexts=compiled_contexts,\n')
            f.write('            max_workers=global_config.num_workers\n')
            f.write('        )\n\n')
            f.write('        print(f"🔥 Execution completed! Results: {type(results)}")\n')
            f.write('        if isinstance(results, dict):\n')
            f.write('            print(f"🔥 Results keys: {list(results.keys())}")\n\n')
            f.write('        return results\n\n')
            f.write('if __name__ == "__main__":\n')
            f.write('    run_pipeline()\n')
        
        print(f"✅ Successfully converted to {output_path}")
        print(f"You can now run: python {output_path}")
        
    except Exception as e:
        print(f"Error converting pickle file: {e}")
        import traceback
        traceback.print_exc()

def convert_enum_value(value):
    """Convert enum values to proper Python representation."""
    if hasattr(value, '__class__') and hasattr(value.__class__, '__name__'):
        class_name = value.__class__.__name__
        if hasattr(value, 'name'):
            return f"{class_name}.{value.name}"
    return repr(value)

def convert_args_dict(args_dict):
    """Convert arguments dictionary, handling enum values properly."""
    converted = {}
    for key, value in args_dict.items():
        if hasattr(value, '__class__') and hasattr(value, 'name'):
            # This is likely an enum
            converted[key] = convert_enum_value(value)
        else:
            converted[key] = repr(value)
    return converted

def generate_function_repr(func_obj):
    """Generate a Python representation of a function object."""
    if callable(func_obj):
        return f"{func_obj.__name__}"
    elif isinstance(func_obj, tuple) and len(func_obj) == 2:
        func, args = func_obj
        # Convert args dict to handle enums properly
        converted_args = convert_args_dict(args)
        args_str = "{" + ", ".join(f"'{k}': {v}" for k, v in converted_args.items()) + "}"
        return f"({func.__name__}, {args_str})"
    elif isinstance(func_obj, list):
        items = []
        for item in func_obj:
            if isinstance(item, tuple) and len(item) == 2:
                func, args = item
                converted_args = convert_args_dict(args)
                args_str = "{" + ", ".join(f"'{k}': {v}" for k, v in converted_args.items()) + "}"
                items.append(f"({func.__name__}, {args_str})")
            elif callable(item):
                items.append(f"{item.__name__}")
            else:
                items.append(repr(item))
        return f"[{', '.join(items)}]"
    elif isinstance(func_obj, dict):
        items = []
        for key, value in func_obj.items():
            value_repr = generate_function_repr(value)
            items.append(f"'{key}': {value_repr}")
        return f"{{{', '.join(items)}}}"
    else:
        return repr(func_obj)

def generate_readable_function_repr(func_obj, indent=0):
    """Generate readable Python representation with newlines for better readability."""
    indent_str = "    " * indent
    next_indent_str = "    " * (indent + 1)

    if callable(func_obj):
        return f"{func_obj.__name__}"
    elif isinstance(func_obj, tuple) and len(func_obj) == 2:
        func, args = func_obj
        converted_args = convert_args_dict(args)
        if not converted_args:
            args_str = "{}"
        else:
            # Always format kwargs with newlines for readability
            args_items = []
            for k, v in converted_args.items():
                # Recursively format complex values like nested lists/dicts
                v_repr = generate_readable_function_repr(v, indent + 2)
                args_items.append(f"{next_indent_str}    '{k}': {v_repr}")
            args_str = "{\n" + ",\n".join(args_items) + f"\n{next_indent_str}}}"
        return f"({func.__name__}, {args_str})"
    elif isinstance(func_obj, list):
        if not func_obj:
            return "[]"
        items = []
        for item in func_obj:
            # Pass the *same* indent level for items in the list
            item_repr = generate_readable_function_repr(item, indent)
            items.append(f"{next_indent_str}{item_repr}")
        return f"[\n{',\n'.join(items)}\n{indent_str}]"
    elif isinstance(func_obj, dict):
        if not func_obj:
            return "{}"
        items = []
        for key, value in func_obj.items():
            # Pass the *same* indent level for items in the dict
            value_repr = generate_readable_function_repr(value, indent)
            items.append(f"{next_indent_str}'{key}': {value_repr}")
        return f"{{{',\n'.join(items)}\n{indent_str}}}"
    else:
        return repr(func_obj)

def generate_orchestrator_repr(data):
    """Generate complete orchestrator script like special_io_pipeline.py but without execution methods."""
    lines = []

    # Header comment
    lines.append('# Edit this orchestrator configuration and save to apply changes')
    lines.append('# Generated from plate manager - self-contained orchestrator definitions')
    lines.append('')

    # Core imports (like special_io_pipeline.py)
    lines.append('import sys')
    lines.append('import os')
    lines.append('from pathlib import Path')
    lines.append('')
    lines.append('# Add OpenHCS to path')
    lines.append('sys.path.insert(0, "/home/ts/code/projects/openhcs")')
    lines.append('')
    lines.append('from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator')
    lines.append('from openhcs.core.steps.function_step import FunctionStep')
    lines.append('from openhcs.core.config import GlobalPipelineConfig')
    lines.append('from openhcs.constants.constants import VariableComponents')
    lines.append('')

    # Extract function imports from pipeline data
    function_imports = set()
    if 'pipeline_data' in data:
        for plate_path, steps in data['pipeline_data'].items():
            for step in steps:
                if hasattr(step, 'func'):
                    _extract_function_imports_from_func(step.func, function_imports)

    # Write function imports
    if function_imports:
        lines.append('# Function imports')
        for import_stmt in sorted(function_imports):
            lines.append(import_stmt)
        lines.append('')

    # Write enum imports
    lines.append('# Enum imports')
    lines.append('from openhcs.processing.backends.analysis.cell_counting_pyclesperanto import DetectionMethod')
    lines.append('from openhcs.processing.backends.analysis.skan_axon_analysis import AnalysisDimension')
    lines.append('from openhcs.core.config import PathPlanningConfig, VFSConfig, ZarrConfig, MaterializationBackend, ZarrCompressor, ZarrChunkStrategy')
    lines.append('from openhcs.constants.constants import Backend, Microscope')
    lines.append('')

    # Write global config (like special_io_pipeline.py)
    lines.append('# Global configuration')
    lines.append('global_config = GlobalPipelineConfig(')
    if 'global_config' in data and data['global_config']:
        config = data['global_config']
        lines.append(f'    num_workers={config.num_workers},')
        # Add other config fields as needed
    else:
        lines.append('    num_workers=4,')
    lines.append(')')
    lines.append('')

    # Write pipeline data (like special_io_pipeline.py)
    lines.append('# Pipeline data')
    lines.append('pipeline_data = {}')
    lines.append('')

    if 'pipeline_data' in data:
        for plate_path, steps in data['pipeline_data'].items():
            lines.append(f'# Steps for plate: {Path(plate_path).name}')
            lines.append('steps = []')
            lines.append('')

            for i, step in enumerate(steps):
                lines.append(f'# Step {i+1}: {step.name}')
                func_repr = generate_readable_function_repr(step.func, indent=1)
                lines.append(f'step_{i+1} = FunctionStep(')
                lines.append(f'    func={func_repr},')
                lines.append(f'    name="{step.name}",')
                lines.append(f'    variable_components=[VariableComponents.{step.variable_components[0].name}],')
                lines.append(f'    force_disk_output=False')
                lines.append(f')')
                lines.append(f'steps.append(step_{i+1})')
                lines.append('')

            lines.append(f'pipeline_data["{plate_path}"] = steps')
            lines.append('')

    # Generate orchestrators list (the main assignment)
    lines.append('# Orchestrators for selected plates')
    lines.append('orchestrators = [')

    if 'plate_paths' in data:
        for plate_path in data['plate_paths']:
            lines.append(f'    PipelineOrchestrator(')
            lines.append(f'        plate_path=Path("{plate_path}"),')
            lines.append(f'        global_config=global_config')
            lines.append(f'    ),')

    lines.append(']')

    return '\n'.join(lines)

def _extract_function_imports_from_func(func_obj, function_imports):
    """Extract import statements for functions."""
    if callable(func_obj):
        module = getattr(func_obj, '__module__', None)
        name = getattr(func_obj, '__name__', None)
        if module and name and module.startswith('openhcs'):
            function_imports.add(f"from {module} import {name}")
    elif isinstance(func_obj, (list, tuple)):
        for item in func_obj:
            if isinstance(item, tuple) and len(item) > 0:
                _extract_function_imports_from_func(item[0], function_imports)
            else:
                _extract_function_imports_from_func(item, function_imports)
    elif isinstance(func_obj, dict):
        for value in func_obj.values():
            _extract_function_imports_from_func(value, function_imports)

def generate_pipeline_repr(pipeline_steps=None, plate_paths=None, global_config=None):
    """Generate Python code for editing pipeline steps (following function pattern editor approach)."""
    lines = []

    # Header comment
    lines.append('# Edit this pipeline and save to apply changes')
    lines.append('')

    # Core OpenHCS imports
    lines.append('from openhcs.core.steps.function_step import FunctionStep')
    lines.append('from openhcs.constants.constants import VariableComponents')
    lines.append('')

    # Collect function imports from pipeline steps
    function_imports = set()
    if isinstance(pipeline_steps, list):
        for step in pipeline_steps:
            _extract_function_imports_from_step(step, function_imports)

    # Add function imports
    if function_imports:
        lines.append('# Function imports')
        lines.extend(sorted(function_imports))
        lines.append('')

    # Add enum imports (same as function editor)
    lines.append('# Enum imports')
    lines.append('from openhcs.processing.backends.analysis.cell_counting_pyclesperanto import DetectionMethod')
    lines.append('from openhcs.processing.backends.analysis.skan_axon_analysis import AnalysisDimension')
    lines.append('')

    # Generate pipeline steps list
    if isinstance(pipeline_steps, list) and pipeline_steps:
        lines.append('# Pipeline steps')
        lines.append('pipeline_steps = [')

        for i, step in enumerate(pipeline_steps):
            step_name = getattr(step, 'name', f'step_{i+1}')
            func_repr = generate_readable_function_repr(getattr(step, 'func', None), indent=1)
            variable_components = getattr(step, 'variable_components', [])
            force_disk_output = getattr(step, 'force_disk_output', False)

            lines.append(f'    # Step {i+1}: {step_name}')
            lines.append('    FunctionStep(')
            lines.append(f'        func={func_repr},')
            lines.append(f'        name="{step_name}",')
            if variable_components:
                components_repr = f'[VariableComponents.{variable_components[0].name}]'
            else:
                components_repr = '[VariableComponents.SITE]'
            lines.append(f'        variable_components={components_repr},')
            lines.append(f'        force_disk_output={force_disk_output}')
            if i < len(pipeline_steps) - 1:
                lines.append('    ),')
            else:
                lines.append('    )')
            lines.append('')

        lines.append(']')
    else:
        lines.append('# Pipeline steps')
        lines.append('pipeline_steps = []')

    return '\n'.join(lines)

def _extract_function_imports_from_step(step, imports):
    """Extract function imports from a pipeline step."""
    if hasattr(step, 'func'):
        _extract_function_imports(step.func, imports)

def _extract_function_imports(func_obj, imports):
    """Extract import statements from function objects (same logic as function editor)."""
    if callable(func_obj):
        if hasattr(func_obj, '__module__') and hasattr(func_obj, '__name__'):
            imports.add(f"from {func_obj.__module__} import {func_obj.__name__}")
    elif isinstance(func_obj, tuple) and len(func_obj) > 0:
        _extract_function_imports(func_obj[0], imports)
    elif isinstance(func_obj, list):
        for item in func_obj:
            if isinstance(item, tuple) and len(item) > 0:
                _extract_function_imports(item[0], imports)
            else:
                _extract_function_imports(item, imports)
    elif isinstance(func_obj, dict):
        for value in func_obj.values():
            _extract_function_imports(value, imports)

def main():
    if len(sys.argv) < 2:
        print("Usage: python pickle_to_python.py <pickle_file> [output_file]")
        print("Example: python pickle_to_python.py special_io.pkl special_io_pipeline.py")
        sys.exit(1)
    
    pickle_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_pickle_to_python(pickle_path, output_path)

if __name__ == "__main__":
    main()
