Special I/O System: Cross-Step Communication and Dict Pattern Integration
==========================================================================

Overview
--------

The Special I/O system enables sophisticated data exchange between pipeline steps outside the primary input/output directories. It uses a declarative decorator system combined with VFS path resolution to create directed data flow connections between steps. The system has evolved to support complex dict patterns through compiler-inspired namespacing techniques, enabling component-specific processing while maintaining cross-step communication capabilities.

**System Evolution**: Originally designed for simple single-function steps, the system was extended to handle dict patterns (multiple functions per step) through sophisticated namespacing and scope promotion rules, similar to symbol resolution in programming language compilers.

Architectural Evolution
-----------------------

Original Design: Single Function Steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The original Special I/O system was designed around a simple assumption: one function per step with direct key matching for cross-step communication.

**Original Architecture**:
- **Purpose**: Cross-step communication (positions generation → assembly) and analysis materialization
- **Assumption**: Single function per step with simple key matching
- **Limitation**: Could not handle component-specific processing patterns

**The Challenge**: OpenHCS needed to support dict patterns for component-specific processing (``{'DAPI': analyze_nuclei, 'GFP': analyze_proteins}``), but this created fundamental architectural tension between multiple functions per step and step-to-step communication.

Dict Pattern Integration: Compiler-Inspired Solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The system was extended using compiler design principles to handle complex patterns while maintaining architectural integrity.

**Key Innovation**: Namespacing system similar to symbol resolution in programming language compilers:

1. **Full Namespacing**: ``dict_key_chain_position_original_key`` pattern
2. **Scope Promotion**: Single-key dict patterns auto-promote to global scope
3. **Collision Detection**: Compiler validates unique output keys across patterns
4. **Execution Mapping**: Compilation-time generation of explicit execution plans

**Architectural Benefits**:
- Maintains single source of truth (compiled plans)
- Enables component-specific processing without breaking cross-step communication
- Provides fail-loud behavior with clear error messages
- Preserves compilation model integrity

Architecture Components
-----------------------

Decorator System
~~~~~~~~~~~~~~~~

The Special I/O system uses a declarative approach where functions simply declare what additional data they produce or consume beyond their main image processing. This creates a clean separation between the function's core logic and its communication requirements.

**Special Outputs**: Functions that generate useful side data (like position coordinates, analysis results, or metadata) declare these outputs using the ``@special_outputs`` decorator. The function returns its main processed image plus the additional data as a tuple.

**Special Inputs**: Functions that need data from previous steps declare their requirements using ``@special_inputs``. The system automatically loads this data from the VFS and provides it as function parameters.

**Materialization Support**: Special outputs can optionally include materialization functions that convert Python objects to persistent file formats (CSV, JSON, etc.) for analysis tools.

.. code:: python

   # Example: Position generation with materialization
   @special_outputs(("positions", materialize_positions_csv))
   def generate_positions(image_stack):
       positions = calculate_positions(image_stack)
       return processed_image, positions

   # Example: Assembly using positions
   @special_inputs("positions")
   def stitch_images(image_stack, positions):
       return stitch(image_stack, positions)

Decorator Implementation
~~~~~~~~~~~~~~~~~~~~~~~~

The decorators work by attaching metadata to function objects that the compilation system can discover and use for path planning. This approach keeps the function declarations clean while providing all the information needed for automatic data flow management.

**Metadata Attachment**: The decorators add attributes to functions (``__special_outputs__``, ``__special_inputs__``) that the compiler reads during pipeline analysis. This metadata-driven approach means functions remain normal Python functions that can be tested independently.

**Materialization Integration**: When special outputs include materialization functions, the decorator stores both the output keys (for path planning) and the materialization functions (for file conversion) as separate attributes.

**Backward Compatibility**: The implementation maintains compatibility with existing code while supporting new features like materialization functions.

Compilation-Time Path Resolution
--------------------------------

The compilation system transforms the declarative special I/O requirements into concrete execution plans with specific file paths and dependency relationships. This happens during pipeline compilation, before any actual processing begins.

Phase 1: Special Output Registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The compiler scans each step's function to discover what special outputs it produces, then creates VFS paths for storing this data. Each special output gets a unique path within the step's output directory, typically using pickle format for Python object serialization.

**Path Generation**: The system creates predictable paths based on the step's output directory and the special output key. This ensures that consuming steps can reliably find the data they need.

**Global Registration**: As outputs are discovered, they're registered in a global catalog that tracks which step produces each piece of special data. This catalog enables dependency validation and cross-step linking.

Phase 2: Special Input Linking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After discovering all special outputs, the compiler validates that every special input requirement can be satisfied by a previous step's output. This creates explicit dependency relationships and ensures the pipeline has a valid data flow.

**Dependency Validation**: The system checks that each special input has a corresponding special output from an earlier step. If any dependencies are missing, compilation fails with a clear error message indicating which data is unavailable.

**Order Validation**: The compiler also enforces that dependencies flow forward in time - a step cannot depend on outputs from later steps. This prevents circular dependencies and ensures the pipeline has a valid execution order.

**Path Linking**: When dependencies are satisfied, the compiler creates explicit links between consuming steps and the paths where their required data will be stored. This eliminates runtime path resolution and makes data flow explicit in the compiled plan.

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

Dict Pattern Integration: Compiler-Inspired Namespacing
-------------------------------------------------------

The Architectural Challenge
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The original Special I/O system was designed around a simple assumption: one function per step with direct key matching for cross-step communication. However, OpenHCS dict patterns enable component-specific processing where a single step can contain multiple functions, each processing different image channels or components.

**Cross-Step Communication Problem**: When a dict pattern produces special outputs, the keys become namespaced (like ``DAPI_positions``), but consuming steps expect the original key names (like ``positions``). This breaks the linking between steps because the namespaced output key doesn't match the expected input key.

**Execution Filtering Problem**: During execution, the system needs to determine which special outputs a specific function should produce. The compiled step plan contains namespaced keys, but the function's metadata contains original keys. Simple key matching fails because the namespaces don't align.

**Architectural Tension**: The system needed to support both component-specific processing (requiring namespacing) and cross-step communication (requiring consistent key names) without breaking existing functionality or creating complex workarounds.

Compiler-Inspired Solution Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The solution draws from compiler design principles, particularly symbol resolution and scoping mechanisms used in programming languages. The system implements a namespacing strategy that resolves the architectural tension while maintaining clean semantics.

**Full Namespacing System**: Every special output from a dict pattern gets a unique name that includes the dict key, chain position, and original output name. This ensures no conflicts while preserving traceability back to the source function.

**Scope Promotion Rules**: The system includes intelligent scope promotion that automatically handles common patterns. When a dict pattern has only one key, its outputs are promoted to global scope, removing the namespace prefix. This allows seamless integration with consuming steps that expect simple key names.

**Collision Detection**: The compiler validates that scope promotion doesn't create naming conflicts. If multiple dict patterns would produce the same promoted key name, compilation fails with a clear error message.

**Execution Mapping**: Rather than complex runtime filtering, the system generates explicit execution mappings during compilation. These mappings directly connect function execution contexts to their required special outputs, eliminating the need for key matching logic.

Funcplan System: Explicit Execution Mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The funcplan system eliminates runtime complexity by pre-computing all execution mappings during compilation. Instead of trying to match namespaced keys with original keys at runtime, the system creates explicit mappings that directly specify which special outputs each function execution should produce.

**Compilation-Time Generation**: During pipeline compilation, the system analyzes each dict pattern and generates a mapping from execution contexts (function name + dict key + chain position) to the list of special outputs that execution should produce. This mapping captures all the namespacing and scope promotion logic in a simple lookup table.

**Runtime Simplicity**: During execution, the system constructs an execution key and performs a simple dictionary lookup to determine which special outputs to save. This replaces complex filtering logic with a straightforward table lookup, improving both performance and reliability.

**Deterministic Behavior**: The funcplan approach ensures that special output handling is completely deterministic and debuggable. The mapping is generated once during compilation and used consistently throughout execution.

Materialization Function Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dict patterns require careful handling of materialization functions since multiple functions within a pattern may produce special outputs that need materialization. The system must extract and organize these materialization functions according to the same namespacing rules used for the outputs themselves.

**Pattern Analysis**: The system analyzes each dict pattern to discover which functions have materialization requirements. For function chains, each position is checked independently. For single functions, the analysis is straightforward.

**Namespace Coordination**: Materialization functions are organized using the same namespacing scheme as the special outputs they handle. This ensures that the correct materialization function is applied to each namespaced output.

**Directory Management**: Materialization functions are responsible for ensuring their target directories exist before writing files. The execution system provides the data and target paths, but doesn't pre-create directory structures for special outputs.

Architectural Benefits
~~~~~~~~~~~~~~~~~~~~~~

The dict pattern integration provides several key benefits while maintaining system integrity:

**Clear Separation of Concerns**: The solution distinguishes between pattern structure (which determines function identity and namespacing) and execution mechanics (which determines how functions are called). This separation makes the system easier to understand and maintain.

**Compilation Model Preservation**: The compiled step plans remain the authoritative source of execution information. All namespacing and scope promotion logic is resolved during compilation, not at runtime.

**Predictable Error Handling**: The system provides clear error messages for common problems like naming collisions, missing dependencies, and invalid pattern structures. Errors occur during compilation rather than during execution.

**Runtime Simplicity**: Complex filtering and matching logic is replaced with simple dictionary lookups, improving both performance and debuggability.

**Backward Compatibility**: The solution extends the existing special I/O system without breaking existing functionality or requiring changes to existing code.

VFS Integration
---------------

The Special I/O system integrates seamlessly with OpenHCS's Virtual File System (VFS) to provide transparent data storage and retrieval across different backends.

**Backend Selection**: Special I/O data typically uses the memory backend for optimal performance, since this data is usually consumed within the same pipeline run. The memory backend stores Python objects directly without serialization overhead, making data transfer between steps very efficient.

**Automatic Serialization**: When special I/O data needs to be persisted (for debugging or analysis), the VFS automatically handles serialization to appropriate formats. The system uses pickle format by default for Python objects, but materialization functions can convert data to other formats like CSV or JSON.

**Path Abstraction**: Functions work with abstract VFS paths rather than concrete file system paths. This abstraction allows the same function to work with different storage backends without modification.

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

**Core Special I/O System**:
-  ✅ Declarative decorator system (@special_inputs, @special_outputs)
-  ✅ Materialization function support for special outputs
-  ✅ Compilation-time path resolution and dependency validation
-  ✅ Runtime VFS integration with memory backend
-  ✅ Function execution with automatic special I/O handling
-  ✅ Order validation and dependency graph construction

**Dict Pattern Integration**:
-  ✅ Full namespacing system (dict_key_chain_position_original_key)
-  ✅ Scope promotion rules for single-key dict patterns
-  ✅ Collision detection and validation
-  ✅ Funcplan system with explicit execution mapping
-  ✅ Materialization function extraction from dict patterns
-  ✅ Directory creation responsibility in materialization functions

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

See Also
--------

- :doc:`storage_and_memory_system` - VFS integration and materialization system
- :doc:`pipeline_compilation_system` - How special I/O integrates with compilation phases
- :doc:`function_pattern_system` - Function decorators and pattern system
- :doc:`../api/processing_backends` - Special I/O usage in processing functions
- :doc:`../guides/pipeline_compilation_workflow` - Practical special I/O usage examples

Consolidated Documentation
--------------------------

This document consolidates and extends the following architectural components:

- **Original Special I/O System**: Cross-step communication and materialization
- **Dict Pattern Case Study**: Compiler-inspired namespacing and scope promotion
- **Funcplan System**: Explicit execution mapping for complex patterns
- **Materialization Integration**: Special output materialization with storage backends

The dict pattern integration represents a significant architectural evolution that maintains system integrity while enabling sophisticated component-specific processing patterns.
