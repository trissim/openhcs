Function Patterns
================

OpenHCS provides four distinct function patterns that allow you to organize processing logic in different ways. Understanding these patterns is crucial for building effective analysis workflows that handle complex multi-channel, multi-condition experiments.

The Four Function Patterns
--------------------------

1. Single Function Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest pattern applies one function to all data.

.. code-block:: python

   from openhcs.core.steps.function_step import FunctionStep
   from openhcs.processing.backends.processors.cupy_processor import gaussian_filter

   # Apply Gaussian filter to all images
   step = FunctionStep(
       func=(gaussian_filter, {'sigma': 2.0}),
       name="blur"
   )

**When to Use**: Simple operations that apply the same processing to all data (normalization, basic filtering, format conversion).

2. Function Chain Pattern
~~~~~~~~~~~~~~~~~~~~~~~~

Applies multiple functions in sequence, creating a processing pipeline within a single step.

.. code-block:: python

   from openhcs.processing.backends.processors.cupy_processor import tophat, threshold_otsu

   # Sequential processing: tophat filter followed by thresholding
   step = FunctionStep(
       func=[
           (tophat, {'selem_radius': 25}),
           (threshold_otsu, {'binary': True})
       ],
       name="preprocess_and_threshold"
   )

**When to Use**: Sequential operations that should be grouped together (preprocessing chains, multi-step analysis).

3. Dictionary Pattern
~~~~~~~~~~~~~~~~~~~~

Routes different data to different functions based on data characteristics.

.. code-block:: python

   from openhcs.processing.backends.analysis.cell_counting_cpu import count_cells_single_channel
   from openhcs.processing.backends.analysis.skan_axon_analysis import skan_axon_skeletonize_and_analyze
   from openhcs.constants.constants import GroupBy

   # Different analysis for different channels
   step = FunctionStep(
       func={
           '1': (count_cells_single_channel, {}),      # DAPI channel - count nuclei
           '2': (skan_axon_skeletonize_and_analyze, {}) # GFP channel - trace neurites
       },
       group_by=GroupBy.CHANNEL,
       name="channel_specific_analysis"
   )

**When to Use**: Channel-specific processing, condition-specific analysis, or any scenario where different data needs different processing.

4. Nested Patterns
~~~~~~~~~~~~~~~~~~

Combines patterns for complex multi-dimensional routing.

.. code-block:: python

   # Dictionary of function chains - different preprocessing for each channel
   step = FunctionStep(
       func={
           '1': [  # DAPI channel preprocessing
               (gaussian_filter, {'sigma': 1.0}),
               (tophat, {'selem_radius': 15}),
               (threshold_otsu, {})
           ],
           '2': [  # GFP channel preprocessing  
               (gaussian_filter, {'sigma': 2.0}),
               (enhance_contrast, {'percentile_range': (1, 99)}),
               (binary_opening, {'footprint_radius': 3})
           ]
       },
       group_by=GroupBy.CHANNEL,
       name="channel_specific_preprocessing"
   )

**When to Use**: Complex workflows where different data types need different multi-step processing.

Understanding Group By
----------------------

The ``group_by`` parameter is essential for dictionary patterns. It tells OpenHCS how to interpret the dictionary keys.

.. code-block:: python

   from openhcs.constants.constants import GroupBy

   # group_by=GroupBy.CHANNEL means keys correspond to channel numbers
   step = FunctionStep(
       func={
           '1': (process_dapi, {}),     # Processes channel 1 data
           '2': (process_gfp, {})       # Processes channel 2 data
       },
       group_by=GroupBy.CHANNEL
   )

   # group_by=GroupBy.CHANNEL means keys correspond to channel numbers
   step = FunctionStep(
       func={
           '1': (process_dapi, {}),     # Processes channel 1 data
           '2': (process_gfp, {}),      # Processes channel 2 data
           '3': (process_rfp, {})       # Processes channel 3 data
       },
       group_by=GroupBy.CHANNEL
   )

**Available Group By Options**:

- ``GroupBy.CHANNEL``: Route by fluorescence channel (most common)

Pattern Selection Guide
----------------------

Choosing the Right Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~

**Single Function**: Use when all data gets the same processing

.. code-block:: python

   # All images need the same normalization
   FunctionStep(func=(stack_percentile_normalize, {
       'low_percentile': 1.0,
       'high_percentile': 99.0
   }))

**Function Chain**: Use for sequential operations that belong together

.. code-block:: python

   # Preprocessing pipeline that should be grouped
   FunctionStep(func=[
       (normalize, {}),
       (filter_func, {}),
       (enhance, {})
   ])

**Dictionary Pattern**: Use when different data needs different processing

.. code-block:: python

   # Different channels need different analysis
   FunctionStep(
       func={
           '1': (count_nuclei, {}),
           '2': (trace_neurites, {})
       },
       group_by=GroupBy.CHANNEL
   )

**Nested Patterns**: Use for complex multi-dimensional workflows

.. code-block:: python

   # Different channels need different preprocessing chains
   FunctionStep(
       func={
           '1': [(normalize_dapi, {}), (threshold_dapi, {})],
           '2': [(normalize_gfp, {}), (enhance_gfp, {}), (trace_gfp, {})]
       },
       group_by=GroupBy.CHANNEL
   )

Real-World Examples
------------------

Cell Viability Assay
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Multi-channel cell viability analysis
   pipeline = Pipeline([
       # Preprocessing - same for all channels
       FunctionStep(
           func=(stack_percentile_normalize, {}),
           name="normalize"
       ),

       # Channel-specific analysis
       FunctionStep(
           func={
               '1': (count_cells_single_channel, {}),     # DAPI - total cells
               '2': (measure_calcein_intensity, {})       # Calcein - live cells
           },
           group_by=GroupBy.CHANNEL,
           name="analyze_viability"
       ),

       # Combine results
       FunctionStep(
           func=(calculate_viability_ratio, {}),
           name="calculate_ratio"
       )
   ])

Neurite Outgrowth Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Complex neurite analysis with condition-specific processing
   pipeline = Pipeline([
       # Different preprocessing for different experimental conditions
       FunctionStep(
           func={
               'control': [
                   (gaussian_filter, {'sigma': 1.0}),
                   (tophat, {'selem_radius': 25})
               ],
               'treatment': [
                   (gaussian_filter, {'sigma': 1.5}),
                   (enhance_contrast, {'percentile_range': (2, 98)}),
                   (tophat, {'selem_radius': 30})
               ]
           },
           group_by=GroupBy.WELL,  # Route by experimental condition
           name="condition_specific_preprocessing"
       ),
       
       # Same analysis for all conditions
       FunctionStep(
           func=(skan_axon_skeletonize_and_analyze, {}),
           name="trace_neurites"
       )
   ])

Pattern Advantages
-----------------

**Composability**: Patterns can be combined to create complex workflows from simple building blocks.

**Readability**: The pattern structure makes it clear what processing applies to what data.

**Maintainability**: Changes to specific processing paths don't affect other parts of the workflow.

**Performance**: OpenHCS optimizes execution based on the pattern structure, minimizing data movement and memory usage.

**Flexibility**: The same pattern framework handles everything from simple single-function steps to complex multi-dimensional routing.

Common Patterns in Practice
---------------------------

**Preprocessing + Analysis**: Function chain for preprocessing, followed by dictionary pattern for channel-specific analysis.

**Condition-Specific Workflows**: Dictionary pattern routing by well for different experimental conditions.

**Multi-Scale Processing**: Nested patterns for different processing at different image scales or regions.

**Quality Control + Processing**: Function chains that include quality checks followed by conditional processing.

Memory Type Integration
----------------------

OpenHCS automatically handles memory type conversion between different computational backends within function patterns:

.. code-block:: python

   # Chain functions from different backends - automatic conversion
   step = FunctionStep(
       func=[
           (stack_percentile_normalize, {}),  # PyTorch function
           (tophat, {}),                      # CuPy function
           (count_cells_single_channel, {})   # NumPy function
       ],
       name="mixed_backend_processing",
       variable_components=[VariableComponents.SITE]
   )

**How it works**: OpenHCS detects memory type requirements and automatically converts data between NumPy arrays, CuPy arrays, PyTorch tensors, and pyclesperanto arrays as needed.

**Performance optimization**: Conversions are minimized by grouping operations by memory type when possible.

Advanced Pattern Examples
-------------------------

Complex Multi-Channel Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Sophisticated multi-channel analysis with preprocessing chains
   step = FunctionStep(
       func={
           '1': [  # DAPI channel - nuclear analysis
               (gaussian_filter, {'sigma': 1.0}),
               (tophat, {'selem_radius': 15}),
               (threshold_otsu, {}),
               (count_cells_single_channel, {
                   'detection_method': DetectionMethod.WATERSHED,
                   'min_sigma': 1.0,
                   'max_sigma': 10.0
               })
           ],
           '2': [  # GFP channel - neurite analysis
               (gaussian_filter, {'sigma': 2.0}),
               (enhance_contrast, {'percentile_range': (1, 99)}),
               (skan_axon_skeletonize_and_analyze, {
                   'analysis_dimension': AnalysisDimension.TWO_D,
                   'min_branch_length': 10.0
               })
           ]
       },
       group_by=GroupBy.CHANNEL,
       variable_components=[VariableComponents.SITE],
       name="comprehensive_analysis"
   )

The function pattern system provides a systematic way to organize complex analysis workflows while maintaining clarity and performance. By understanding these patterns, you can build sophisticated analysis pipelines that handle the complexity of modern high-content screening experiments.
