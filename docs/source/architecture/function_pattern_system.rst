The Function Pattern System
============================

Overview
--------

The function pattern system evolved from EZStitcher. This system addresses
the problem of composing heterogeneous functions into flexible, type-safe
processing pipelines.

OpenHCS implements four fundamental patterns that provide a unified
interface for different execution strategies, allowing the same
``FunctionStep`` class to handle various processing scenarios through
elegant pattern matching.

**Note**: This document describes the actual function pattern
implementation in OpenHCS, including enhancements and evolution from the
original EZStitcher design.

The Problem It Solves
---------------------

Scientific image processing involves combining functions with different
interfaces, argument patterns, and execution models:

-  **Single-image functions** (most image processing libraries)
-  **Stack-aware functions** (microscopy-specific operations)
-  **Functions with parameters** (configurable processing)
-  **Channel-specific functions** (different processing per channel)
-  **Sequential processing chains** (multi-step operations)

Traditional approaches require manual adaptation, wrapper functions, or
complex orchestration code. The function pattern system automates this
process.

The Four Function Patterns
--------------------------

1. Single Function Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Syntax**: ``FunctionStep(func=my_function)``

**Use Case**: Apply the same function to all data groups

**Real-World Example** (from TUI-generated scripts):

.. code:: python

   from openhcs.core.steps.function_step import FunctionStep
   from openhcs.constants.constants import VariableComponents
   from openhcs.processing.backends.processors.cupy_processor import create_composite

   # Single function - clean and simple
   step = FunctionStep(
       func=create_composite,
       name="composite",
       variable_components=[VariableComponents.CHANNEL]
   )

**Execution Flow**: - Function called once per pattern group - Same
function applied to all channels/sites/etc. - Parameters come from
function defaults or global configuration

2. Parameterized Function Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Syntax**: ``FunctionStep(func=(my_function, {'param': value}))``

**Use Case**: Apply function with specific parameters

**Example**:

.. code:: python

   from openhcs.processing.backends.processors.torch_processor import stack_percentile_normalize

   # Function with parameters
   step = FunctionStep(
       func=(stack_percentile_normalize, {
           'low_percentile': 1.0,
           'high_percentile': 99.0,
           'target_max': 65535.0
       }),
       name="normalize",
       variable_components=[VariableComponents.SITE]
   )

**Execution Flow**: - Tuple unpacked: ``(function, kwargs)`` - Function
called with merged parameters (kwargs + defaults) - Same parameters
applied to all pattern groups

3. Sequential Processing Pattern (List)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Syntax**: ``FunctionStep(func=[func1, func2, func3])``

**Use Case**: Apply multiple functions in sequence to each data group

**Example**:

.. code:: python

   from openhcs.processing.backends.processors.torch_processor import stack_percentile_normalize
   from openhcs.processing.backends.processors.cupy_processor import tophat

   # Sequential processing pipeline
   step = FunctionStep(
       func=[
           (stack_percentile_normalize, {
               'low_percentile': 1.0,
               'high_percentile': 99.0,
               'target_max': 65535.0
           }),
           (tophat, {
               'selem_radius': 50,
               'downsample_factor': 4
           })
       ],
       name="preprocess",
       variable_components=[VariableComponents.SITE]
   )

**Execution Flow**: - Functions applied in order:
``output = func3(func2(func1(input)))`` - Each function can be single or
parameterized pattern - Pipeline applied to each pattern group
independently

4. Component-Specific Processing Pattern (Dict)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Syntax**: ``FunctionStep(func={'key1': func1, 'key2': func2})``

**Use Case**: Different processing for different components (channels,
sites, etc.)

**Example**:

.. code:: python

   from openhcs.processing.backends.analysis.cell_counting_cpu import count_cells_single_channel
   from openhcs.processing.backends.analysis.skan_axon_analysis import skan_axon_skeletonize_and_analyze
   from openhcs.processing.backends.analysis.cell_counting_pyclesperanto import DetectionMethod
   from openhcs.processing.backends.analysis.skan_axon_analysis import AnalysisDimension

   # Channel-specific processing
   step = FunctionStep(
       func={
           '1': (count_cells_single_channel, {
               'min_sigma': 1.0,
               'max_sigma': 10.0,
               'detection_method': DetectionMethod.WATERSHED
           }),
           '2': (skan_axon_skeletonize_and_analyze, {
               'voxel_spacing': (1.0, 1.0, 1.0),
               'min_branch_length': 10.0,
               'analysis_dimension': AnalysisDimension.TWO_D
           })
       },
       name="channel_specific_analysis",
       variable_components=[VariableComponents.SITE]
   )

**Execution Flow**: - Pattern groups routed by component value - Each
component gets its specific function - Used with ``group_by`` parameter
for automatic routing

Advanced Pattern Combinations
-----------------------------

Nested Patterns (Semantically Valid)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Lists within dictionaries: sequential processing per component
   func = {
       "1": [                           # Channel 1: sequential processing
           (sharpen, {'amount': 1.5}),
           normalize,
           denoise_dapi
       ],
       "2": [                           # Channel 2: different sequence
           (enhance, {'strength': 0.8}),
           process_calcein
       ]
   }

   # Functions with arguments in sequential lists
   func = [
       (sharpen, {'amount': 1.5}),      # First: sharpen with parameters
       normalize,                       # Then: normalize (no parameters)
       (denoise, {'strength': 0.8})     # Finally: denoise with parameters
   ]

**Note**: Nested dictionaries are NOT semantically valid (what would
nested routing keys mean in microscopy?).

Pattern Resolution and Execution
--------------------------------

Pattern Validation
~~~~~~~~~~~~~~~~~~

The system validates patterns during compilation using
``FuncStepContractValidator``:

.. code:: python

   def _extract_functions_from_pattern(func, step_name):
       """Extract all functions from a pattern with validation."""
       
       # Case 1: Direct callable
       if callable(func) and not isinstance(func, type):
           return [func]
       
       # Case 2: Tuple (function, kwargs)
       if isinstance(func, tuple) and len(func) == 2:
           return [func[0]]
       
       # Case 3: List of patterns (recursive)
       if isinstance(func, list):
           functions = []
           for f in func:
               functions.extend(_extract_functions_from_pattern(f, step_name))
           return functions
       
       # Case 4: Dict of keyed patterns (recursive)
       if isinstance(func, dict):
           functions = []
           for key, f in func.items():
               functions.extend(_extract_functions_from_pattern(f, step_name))
           return functions
       
       raise ValueError(f"Invalid function pattern: {func}")

Execution Coordination
~~~~~~~~~~~~~~~~~~~~~~

Pattern execution is coordinated by ``prepare_patterns_and_functions``:

.. code:: python

   def prepare_patterns_and_functions(patterns, processing_funcs, component='default'):
       """Prepare patterns and functions for execution."""
       
       # 1. Ensure patterns are component-keyed
       grouped_patterns = _group_patterns_by_component(patterns, component)
       
       # 2. Route functions to components
       component_to_funcs = _route_functions_to_components(processing_funcs, grouped_patterns)
       
       # 3. Extract arguments for each component
       component_to_args = _extract_component_arguments(component_to_funcs)
       
       return grouped_patterns, component_to_funcs, component_to_args

Memory Type Integration
-----------------------

Function patterns integrate seamlessly with the memory type system:

.. code:: python

   @cupy_func  # GPU processing
   def gpu_gaussian(image_stack, sigma=1.0):
       return cucim.skimage.filters.gaussian(image_stack, sigma)

   @numpy_func  # CPU processing  
   def cpu_gaussian(image_stack, sigma=1.0):
       return scipy.ndimage.gaussian_filter(image_stack, sigma)

   # Pattern can mix memory types - automatic conversion handled
   step = FunctionStep(
       func=[
           gpu_gaussian,     # GPU processing
           cpu_gaussian      # Automatic GPU→CPU conversion
       ]
   )

Historical Context: EZStitcher Evolution
----------------------------------------

EZStitcher Foundation
~~~~~~~~~~~~~~~~~~~~~

The function pattern system originated in EZStitcher as a solution to
the “function interface chaos” problem in scientific computing.
EZStitcher established the core patterns that remain central to OpenHCS.

OpenHCS Enhancements
~~~~~~~~~~~~~~~~~~~~

OpenHCS evolved the pattern system with:

-  **Memory type integration**: Automatic conversion between NumPy,
   CuPy, PyTorch, etc.
-  **GPU coordination**: Device-aware execution with resource management
-  **Validation system**: Compile-time pattern validation and contract
   checking
-  **Performance optimization**: Zero-copy conversions and intelligent
   materialization

Why This System Is Genius
-------------------------

Composability Without Complexity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pattern system enables complex workflows through simple composition:

.. code:: python

   # This simple pattern definition...
   func = {
       "dapi": [gaussian_blur, threshold_otsu, binary_opening],
       "calcein": [enhance_contrast, detect_cells],
       "brightfield": [normalize_illumination]
   }

   # ...automatically handles:
   # - Channel routing
   # - Sequential processing  
   # - Memory type conversions
   # - GPU resource management
   # - Error isolation
   # - Performance optimization

Type Safety and Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Unlike traditional scientific computing tools that fail at runtime,
OpenHCS validates patterns at compilation time, preventing entire
classes of errors before execution begins.

Universal Interface
~~~~~~~~~~~~~~~~~~~

The same ``FunctionStep`` interface handles everything from simple
single-function processing to complex multi-channel, multi-step
workflows with automatic optimization.

Performance Characteristics
---------------------------

-  **Pattern Resolution**: O(1) lookup after compilation
-  **Memory Conversions**: Zero-copy when possible, optimized otherwise
-  **GPU Coordination**: Automatic device placement and resource
   management
-  **Error Isolation**: Pattern failures don’t affect other components

Future Enhancements
-------------------

-  **Dynamic Pattern Generation**: Runtime pattern creation based on
   data characteristics
-  **Pattern Optimization**: Automatic reordering for performance
-  **Distributed Patterns**: Multi-node pattern execution
-  **Pattern Caching**: Compiled pattern reuse across executions

Stack Processing Evolution
--------------------------

The Bridge Between Single-Image and Stack-Based Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from openhcs.core.memory.stack_utils import stack_slices, unstack_slices
   from skimage.filters import gaussian

   # Problem: gaussian() works on single images, but we have image stacks
   # Solution: stack/unstack pattern in OpenHCS
   # (Note: stack() utility evolved into stack_slices/unstack_slices system)
   func = gaussian  # Applied per-slice automatically by OpenHCS

Automatic Stack Handling
~~~~~~~~~~~~~~~~~~~~~~~~

OpenHCS automatically handles the stack/unstack operations:

.. code:: python

   # Function operates on 2D slices
   @numpy_func
   def process_single_slice(image_2d, param=1.0):
       return skimage.filters.gaussian(image_2d, sigma=param)

   # OpenHCS automatically:
   # 1. Unstacks 3D array into 2D slices
   # 2. Applies function to each slice
   # 3. Restacks results into 3D array
   # 4. Maintains memory type consistency

Real-World Usage Examples
-------------------------

Neurite Tracing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Actual research pipeline for axon regeneration studies
   neurite_pipeline = Pipeline([
       FunctionStep(
           func=[
               (gaussian_filter, {'sigma': 1.0}),
               (top_hat_filter, {'footprint': disk(3)}),
               (contrast_enhancement, {'percentile_range': (1, 99)})
           ],
           name="Preprocessing"
       ),
       FunctionStep(
           func=trace_neurites_rrs_alva,
           name="HMM Neurite Tracing"
       ),
       FunctionStep(
           func={
               "measurements": [
                   measure_neurite_length,
                   count_branch_points,
                   calculate_regeneration_index
               ],
               "visualization": [
                   create_trace_overlay,
                   generate_summary_plot
               ]
           },
           group_by=GroupBy.ANALYSIS_TYPE,
           name="Analysis and Visualization"
       )
   ])

High-Content Screening Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Multi-channel cell analysis
   hcs_pipeline = Pipeline([
       FunctionStep(
           func={
               "1": [gaussian_blur, threshold_otsu],      # DAPI: nuclei
               "2": [enhance_contrast, detect_cells],     # Calcein: live cells
               "3": [normalize_illumination, segment]     # Brightfield: morphology
           },
           group_by=GroupBy.CHANNEL,
           name="Channel-Specific Processing"
       ),
       FunctionStep(
           func=[
               combine_channels,
               count_cells_multi_channel,
               calculate_viability_metrics
           ],
           name="Multi-Channel Analysis"
       )
   ])

Error Handling and Debugging
----------------------------

Pattern Validation Errors
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Common pattern validation errors:

   # Invalid nested dictionaries
   func = {
       "1": {
           "sub1": process_func  # ❌ Nested dicts not semantically valid
       }
   }

   # Invalid function types
   func = [
       "string_function_name",  # ❌ Must be callable
       42,                      # ❌ Must be callable
       SomeClass               # ❌ Must be function, not class
   ]

   # Valid corrections
   func = {
       "1": [process_func1, process_func2]  # ✅ List within dict
   }
   func = [actual_function, another_function]  # ✅ List of callables

Runtime Debugging
~~~~~~~~~~~~~~~~~

.. code:: python

   # Enable pattern debugging
   import logging
   logging.getLogger('openhcs.core.steps.function_step').setLevel(logging.DEBUG)

   # Logs show pattern resolution:
   # DEBUG: Pattern type: dict with keys ['1', '2', '3']
   # DEBUG: Component '1' executing: [gaussian_blur, threshold_otsu]
   # DEBUG: Component '2' executing: enhance_contrast

Integration with Special I/O
----------------------------

Function patterns work seamlessly with special I/O for cross-step
communication:

.. code:: python

   @special_outputs(("cell_counts", materialize_cell_counts))
   def count_cells_with_output(image_stack):
       """Function that produces both main output and special output."""
       processed = process_image(image_stack)
       cell_count = len(find_cells(processed))
       return processed, cell_count  # Main output, special output

   # Use in pattern
   step = FunctionStep(
       func={
           "dapi": count_cells_with_output,
           "calcein": simple_processing
       },
       group_by=GroupBy.CHANNEL
   )

Performance Optimization
------------------------

Pattern Compilation
~~~~~~~~~~~~~~~~~~~

Patterns are compiled once and reused:

.. code:: python

   # Compilation phase (once per pipeline)
   compiled_pattern = compile_function_pattern(func, step_name)

   # Execution phase (once per well/component)
   result = execute_compiled_pattern(compiled_pattern, data, context)

Memory Type Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Automatic memory type planning
   func = [
       gpu_function,    # Stays on GPU
       cpu_function,    # Converts GPU→CPU
       gpu_function2    # Converts CPU→GPU
   ]

   # Optimizer may reorder for efficiency:
   # gpu_function → gpu_function2 → cpu_function (minimize conversions)

Comparison with Other Systems
-----------------------------

ImageJ/FIJI Macros
~~~~~~~~~~~~~~~~~~

.. code:: java

   // ImageJ: Manual orchestration
   run("Gaussian Blur...", "sigma=2");
   run("Threshold...", "method=Otsu");
   run("Watershed");
   // No type safety, no composability, no GPU support

CellProfiler Modules
~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # CellProfiler: Fixed module pipeline
   # No dynamic routing, limited composability

OpenHCS Function Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # OpenHCS: Unified, composable, type-safe
   func = {
       "dapi": [gaussian_blur, threshold_otsu, watershed],
       "calcein": [enhance_contrast, detect_cells]
   }
   # Automatic GPU support, memory management, validation

See Also
--------

**Core Integration**:

- :doc:`memory_type_system` - Memory type decorators and automatic conversion
- :doc:`function_registry_system` - Function discovery and registration
- :doc:`pipeline_compilation_system` - How patterns are compiled and executed

**Practical Usage**:

- :doc:`../api/function_step` - FunctionStep API reference
- :doc:`../guides/memory_type_integration` - Memory type integration guide
- :doc:`../concepts/function_handling` - Basic function handling concepts

**Advanced Topics**:

- :doc:`dict_pattern_case_study` - Advanced dict pattern examples
- :doc:`special_io_system` - Cross-step communication patterns
- :doc:`compilation_system_detailed` - Deep dive into pattern compilation

--------------

**The function pattern system represents a fundamental breakthrough in
scientific computing architecture - providing the composability of
functional programming with the performance and type safety required for
production research workflows.**
