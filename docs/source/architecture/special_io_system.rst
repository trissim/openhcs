Special I/O System: Cross-Step Communication
============================================

Overview
--------

The Special I/O system enables data exchange between pipeline steps
outside the primary input/output directories. It uses a declarative
decorator system combined with VFS path resolution to create directed
data flow connections between steps.

Architecture Components
-----------------------

Decorator System
~~~~~~~~~~~~~~~~

Functions declare their special I/O requirements using decorators:

.. code:: python

   from openhcs.core.pipeline.function_contracts import special_outputs, special_inputs

   @special_outputs("positions", "metadata")
   def generate_positions(image_stack):
       """Function that produces special outputs."""
       positions = calculate_positions(image_stack)
       metadata = extract_metadata(image_stack)
       
       # Return: (main_output, special_output_1, special_output_2, ...)
       return processed_image, positions, metadata

   @special_inputs("positions", "metadata")
   def stitch_images(image_stack, positions, metadata):
       """Function that consumes special inputs."""
       # positions and metadata are automatically loaded from VFS
       return stitch(image_stack, positions, metadata)

Decorator Implementation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   def special_outputs(*output_specs) -> Callable[[F], F]:
       """Mark function as producing special outputs with optional materialization."""
       def decorator(func: F) -> F:
           special_outputs_info = {}
           output_keys = set()

           for spec in output_specs:
               if isinstance(spec, str):
                   # String only - no materialization function
                   output_keys.add(spec)
                   special_outputs_info[spec] = None
               elif isinstance(spec, tuple) and len(spec) == 2:
                   # (key, materialization_function) tuple
                   key, mat_func = spec
                   output_keys.add(key)
                   special_outputs_info[key] = mat_func

           # Set both attributes for backward compatibility and new functionality
           func.__special_outputs__ = output_keys  # For path planner
           func.__materialization_functions__ = special_outputs_info  # For materialization
           return func
       return decorator

   def special_inputs(*input_names: str) -> Callable[[F], F]:
       """Mark function as consuming special inputs."""
       def decorator(func: F) -> F:
           # Store as dict with True values for compatibility
           func.__special_inputs__ = {name: True for name in input_names}
           return func
       return decorator

Compilation-Time Path Resolution
--------------------------------

Phase 1: Special Output Registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During path planning, the compiler extracts special outputs and creates
VFS paths:

.. code:: python

   # In PipelinePathPlanner.prepare_pipeline_paths()
   def process_special_outputs(step, step_output_dir, declared_outputs):
       """Process special outputs for a step."""
       
       # Extract special outputs from function decorators
       s_outputs_keys = getattr(step.func, '__special_outputs__', set())
       
       special_outputs = {}
       for key in sorted(list(s_outputs_keys)):
           # Use key directly - no unnecessary sanitization!
           output_path = Path(step_output_dir) / f"{key}.pkl"
           special_outputs[key] = {"path": str(output_path)}
           
           # Register this output globally for linking
           declared_outputs[key] = {
               "step_id": step.uid,
               "position": step_position,
               "path": str(output_path)
           }
       
       return special_outputs

Phase 2: Special Input Linking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The compiler links special inputs to previously declared outputs:

.. code:: python

   def process_special_inputs(step, step_position, declared_outputs):
       """Link special inputs to their source outputs."""
       
       # Extract special inputs from function decorators
       s_inputs_dict = getattr(step.func, '__special_inputs__', {})
       
       special_inputs = {}
       for key in s_inputs_dict.keys():
           # Find the source step that produces this output
           if key not in declared_outputs:
               raise ValueError(f"Special input '{key}' not found in any previous step")
           
           source_info = declared_outputs[key]
           source_step_position = source_info["position"]
           
           # Validate dependency order (inputs must come from earlier steps)
           if source_step_position >= step_position:
               raise ValueError(
                   f"Special input '{key}' in step {step_position} "
                   f"depends on output from step {source_step_position}. "
                   "Dependencies must be from earlier steps."
               )
           
           # Link to source path
           special_inputs[key] = {"path": source_info["path"]}
       
       return special_inputs

Path Generation Strategy
~~~~~~~~~~~~~~~~~~~~~~~~

Special I/O paths follow a standardized pattern:

.. code:: python

   def generate_special_io_path(step_output_dir, key):
       """Generate standardized VFS path for special I/O."""

       # Use key directly - predictable and simple!
       return str(Path(step_output_dir) / f"{key}.pkl")

   # Examples:
   # Key "positions" → "positions.pkl"
   # Key "cellMetadata" → "cellMetadata.pkl"
   # Key "stitchingParams" → "stitchingParams.pkl"

Runtime Execution
-----------------

Special Output Handling
~~~~~~~~~~~~~~~~~~~~~~~

During function execution, special outputs are saved to VFS:

.. code:: python

   def _execute_function_core(func_callable, main_data_arg, base_kwargs, 
                             context, special_inputs_plan, special_outputs_plan):
       """Execute function with special I/O handling."""
       
       # 1. Load special inputs from VFS
       final_kwargs = base_kwargs.copy()
       for arg_name, special_path in special_inputs_plan.items():
           logger.debug(f"Loading special input '{arg_name}' from '{special_path}'")
           special_data = context.filemanager.load(special_path, "memory")
           final_kwargs[arg_name] = special_data
       
       # 2. Execute function
       raw_function_output = func_callable(main_data_arg, **final_kwargs)
       
       # 3. Handle special outputs
       if special_outputs_plan:
           # Function returns (main_output, special_output_1, special_output_2, ...)
           if isinstance(raw_function_output, tuple):
               main_output = raw_function_output[0]
               special_values = raw_function_output[1:]
           else:
               raise ValueError("Function with special outputs must return tuple")
           
           # Save special outputs positionally
           for i, (output_key, vfs_path) in enumerate(special_outputs_plan.items()):
               if i < len(special_values):
                   value_to_save = special_values[i]
                   logger.debug(f"Saving special output '{output_key}' to '{vfs_path}'")
                   context.filemanager.save(value_to_save, vfs_path, "memory")
               else:
                   raise ValueError(f"Missing special output value for key '{output_key}'")
           
           return main_output
       else:
           return raw_function_output

Step Plan Integration
~~~~~~~~~~~~~~~~~~~~~

Special I/O information is stored in step plans:

.. code:: python

   # Example step plan with special I/O
   step_plan = {
       "step_name": "Position Generation",
       "step_id": "step_001",
       "input_dir": "/workspace/A01/input",
       "output_dir": "/workspace/A01/step1_out",
       
       # Special outputs produced by this step
       "special_outputs": {
           "positions": {"path": "/workspace/A01/step1_out/positions.pkl"},
           "metadata": {"path": "/workspace/A01/step1_out/metadata.pkl"}
       },
       
       # Special inputs consumed by this step (empty for first step)
       "special_inputs": {},
       
       # Other configuration...
   }

   # Later step that consumes the outputs
   step_plan_2 = {
       "step_name": "Image Stitching",
       "step_id": "step_002",
       "input_dir": "/workspace/A01/step1_out",
       "output_dir": "/workspace/A01/step2_out",
       
       # Special inputs linked to previous step's outputs
       "special_inputs": {
           "positions": {"path": "/workspace/A01/step1_out/positions.pkl"},
           "metadata": {"path": "/workspace/A01/step1_out/metadata.pkl"}
       },
       
       # No special outputs
       "special_outputs": {},
   }

Data Flow Validation
--------------------

Dependency Graph Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The compiler builds a dependency graph to validate special I/O
connections:

.. code:: python

   def validate_special_io_dependencies(steps):
       """Validate special I/O dependency graph."""
       
       # Build dependency graph
       dependency_graph = {}
       declared_outputs = {}
       
       for i, step in enumerate(steps):
           step_id = step.uid
           dependency_graph[step_id] = {"depends_on": [], "provides": []}
           
           # Register outputs
           special_outputs = getattr(step.func, '__special_outputs__', set())
           for output_key in special_outputs:
               if output_key in declared_outputs:
                   raise ValueError(f"Duplicate special output key: {output_key}")
               declared_outputs[output_key] = {"step_id": step_id, "position": i}
               dependency_graph[step_id]["provides"].append(output_key)
           
           # Register dependencies
           special_inputs = getattr(step.func, '__special_inputs__', {})
           for input_key in special_inputs.keys():
               if input_key not in declared_outputs:
                   raise ValueError(f"Unresolved special input: {input_key}")
               
               source_step = declared_outputs[input_key]["step_id"]
               dependency_graph[step_id]["depends_on"].append(source_step)
       
       # Check for cycles
       if has_cycles(dependency_graph):
           raise ValueError("Circular dependencies detected in special I/O")
       
       return dependency_graph

   def has_cycles(graph):
       """Check for cycles in dependency graph using DFS."""
       visited = set()
       rec_stack = set()
       
       def dfs(node):
           visited.add(node)
           rec_stack.add(node)
           
           for neighbor in graph[node]["depends_on"]:
               if neighbor not in visited:
                   if dfs(neighbor):
                       return True
               elif neighbor in rec_stack:
                   return True
           
           rec_stack.remove(node)
           return False
       
       for node in graph:
           if node not in visited:
               if dfs(node):
                   return True
       
       return False

Order Validation
~~~~~~~~~~~~~~~~

.. code:: python

   def validate_execution_order(steps):
       """Ensure special inputs come from earlier steps."""
       
       declared_outputs = {}
       
       for i, step in enumerate(steps):
           # Check inputs reference earlier steps
           special_inputs = getattr(step.func, '__special_inputs__', {})
           for input_key in special_inputs.keys():
               if input_key not in declared_outputs:
                   raise ValueError(f"Special input '{input_key}' not declared by any previous step")
               
               source_position = declared_outputs[input_key]["position"]
               if source_position >= i:
                   raise ValueError(
                       f"Special input '{input_key}' in step {i} "
                       f"references output from step {source_position}. "
                       "Dependencies must be from earlier steps."
                   )
           
           # Register outputs for future steps
           special_outputs = getattr(step.func, '__special_outputs__', set())
           for output_key in special_outputs:
               declared_outputs[output_key] = {"position": i, "step_id": step.uid}

VFS Integration
---------------

Backend Selection
~~~~~~~~~~~~~~~~~

Special I/O typically uses memory backend for performance:

.. code:: python

   def plan_special_io_backends(step_plans):
       """Plan backends for special I/O data."""
       
       for step_id, step_plan in step_plans.items():
           # Special I/O usually uses memory backend
           for output_key, output_info in step_plan.get("special_outputs", {}).items():
               output_info["backend"] = "memory"
           
           for input_key, input_info in step_plan.get("special_inputs", {}).items():
               input_info["backend"] = "memory"

Serialization Handling
~~~~~~~~~~~~~~~~~~~~~~

The VFS automatically handles serialization for special I/O data:

.. code:: python

   # Memory backend stores Python objects directly
   context.filemanager.save(positions_array, "/vfs/positions.pkl", "memory")
   # → Stored as Python object in memory

   # Disk backend would serialize to pickle format
   context.filemanager.save(positions_array, "/workspace/positions.pkl", "disk")
   # → Serialized to .pkl file on disk

Error Handling
--------------

Runtime Validation
~~~~~~~~~~~~~~~~~~

The system performs runtime validation during function execution:

.. code:: python

   # Validation occurs in _execute_function_core
   # - Special inputs are loaded from VFS memory backend
   # - Function output tuple length is validated against declared special outputs
   # - Missing special output values raise ValueError
   # - Failed special input loading propagates exceptions

Current Implementation Status
-----------------------------

Implemented Features
~~~~~~~~~~~~~~~~~~~~

-  ✅ Declarative decorator system (@special_inputs, @special_outputs)
-  ✅ Materialization function support for special outputs
-  ✅ Compilation-time path resolution and dependency validation
-  ✅ Runtime VFS integration with memory backend
-  ✅ Function execution with automatic special I/O handling
-  ✅ Order validation and dependency graph construction

Future Enhancements
~~~~~~~~~~~~~~~~~~~

1. **Optional Special Inputs**: Support for optional special inputs with
   default values
2. **Typed Special I/O**: Type hints and validation for special I/O data
3. **Performance Optimization**: Caching and memory management for
   special I/O
4. **Custom Error Classes**: Specialized exception types for special I/O
   errors
5. **Cross-Pipeline Special I/O**: Share special I/O data between
   different pipeline runs
