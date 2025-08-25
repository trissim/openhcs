Data Dimensions
===============

High-content screening data has multiple dimensions - wells, sites, channels, Z-planes, and timepoints. OpenHCS provides systematic ways to organize processing across these dimensions through variable components and group by parameters.

Understanding Microscopy Data Dimensions
----------------------------------------

Typical HCS data structure:

.. code-block:: text

   Plate/
   ├── A01_s1_w1.tif    # Well A01, Site 1, Channel 1
   ├── A01_s1_w2.tif    # Well A01, Site 1, Channel 2  
   ├── A01_s2_w1.tif    # Well A01, Site 2, Channel 1
   ├── A01_s2_w2.tif    # Well A01, Site 2, Channel 2
   ├── A02_s1_w1.tif    # Well A02, Site 1, Channel 1
   └── ...

**Dimensions**:

- **Well**: Sample position (A01, A02, B01, etc.) - represents experimental conditions
- **Site**: Imaging position within a well (1, 2, 3, etc.) - multiple fields of view
- **Channel**: Fluorescence channel (1, 2, 3, etc.) - different markers or wavelengths
- **Z-Index**: Z-plane depth (1, 2, 3, etc.) - for 3D imaging
- **Timepoint**: Time series point (1, 2, 3, etc.) - for live imaging

Variable Components
------------------

Variable components tell OpenHCS how to group files for processing. They define which dimensions vary within each processing group.

.. code-block:: python

   from openhcs.constants.constants import VariableComponents

   # Available variable components
   VariableComponents.SITE      # Process each site separately
   VariableComponents.CHANNEL   # Process each channel separately  
   VariableComponents.Z_INDEX   # Process each Z-plane separately
   VariableComponents.WELL      # Process each well separately

Most Common: Process Each Site Separately
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from openhcs.core.steps.function_step import FunctionStep

   # Process each site independently (most common pattern)
   step = FunctionStep(
       func=(normalize_images, {}),
       variable_components=[VariableComponents.SITE],
       name="normalize"
   )

**What this does**: Groups files by (well, channel, z_index) and processes each site separately.

**Example grouping**:
- Group 1: [A01_s1_w1.tif, A01_s1_w2.tif] → Process site 1 of well A01
- Group 2: [A01_s2_w1.tif, A01_s2_w2.tif] → Process site 2 of well A01  
- Group 3: [A02_s1_w1.tif, A02_s1_w2.tif] → Process site 1 of well A02

**When to use**: Most image processing operations (filtering, segmentation, analysis) that work on complete images from one imaging position.

Process Each Channel Separately
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Process each channel independently
   step = FunctionStep(
       func=(create_composite, {}),
       variable_components=[VariableComponents.CHANNEL],
       name="composite"
   )

**What this does**: Groups files by (well, site, z_index) and processes each channel separately.

**Example grouping**:
- Group 1: [A01_s1_w1.tif, A01_s2_w1.tif] → Process channel 1 across all sites
- Group 2: [A01_s1_w2.tif, A01_s2_w2.tif] → Process channel 2 across all sites

**When to use**: Operations that combine data across sites for each channel (creating channel composites, channel-specific normalization).

Process Each Z-Plane Separately
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Process each Z-plane independently for 3D analysis
   step = FunctionStep(
       func=(max_projection_across_sites, {}),
       variable_components=[VariableComponents.Z_INDEX],
       name="z_projection"
   )

**What this does**: Groups files by (well, site, channel) and processes each Z-plane separately.

**When to use**: 3D imaging where you need to combine or analyze across Z-planes, such as creating maximum projections.

Multiple Variable Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Process each site and channel combination separately
   step = FunctionStep(
       func=(single_image_analysis, {}),
       variable_components=[VariableComponents.SITE, VariableComponents.CHANNEL],
       name="single_image"
   )

**What this does**: Creates separate groups for each unique combination of site and channel.

**When to use**: Operations that work on individual images rather than image stacks.

Group By Parameter
-----------------

The ``group_by`` parameter works with dictionary function patterns to route different data to different functions.

.. code-block:: python

   from openhcs.constants.constants import GroupBy

   # Route different channels to different functions
   step = FunctionStep(
       func={
           '1': (analyze_nuclei, {}),    # Channel 1 → nuclei analysis
           '2': (analyze_neurites, {})   # Channel 2 → neurite analysis
       },
       group_by=GroupBy.CHANNEL,
       variable_components=[VariableComponents.SITE]
   )

How Group By Works
~~~~~~~~~~~~~~~~~

1. **Data Grouping**: Files are first grouped by ``variable_components``
2. **Function Routing**: Within each group, data is routed to functions based on ``group_by``
3. **Execution**: Each function processes its assigned data

**Example with channel routing**:

.. code-block:: text

   Files: A01_s1_w1.tif, A01_s1_w2.tif
   
   Step 1 - Group by variable_components=[SITE]:
   Group: [A01_s1_w1.tif, A01_s1_w2.tif]  # Same site
   
   Step 2 - Route by group_by=CHANNEL:
   Channel 1: A01_s1_w1.tif → analyze_nuclei()
   Channel 2: A01_s1_w2.tif → analyze_neurites()

Available Group By Options
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   GroupBy.CHANNEL   # Route by channel number (primary use case)

Common Data Organization Patterns
---------------------------------

Site-by-Site Processing
~~~~~~~~~~~~~~~~~~~~~~

Most common pattern for standard image processing:

.. code-block:: python

   # Process each imaging site independently
   step = FunctionStep(
       func=(segment_cells, {}),
       variable_components=[VariableComponents.SITE],
       name="segmentation"
   )

**Use cases**: Filtering, segmentation, feature extraction, most analysis operations.

Channel-Specific Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

Different analysis for different fluorescent markers:

.. code-block:: python

   # Different analysis for each channel
   step = FunctionStep(
       func={
           '1': (count_nuclei, {}),        # DAPI channel
           '2': (measure_intensity, {}),   # GFP channel
           '3': (detect_structures, {})    # RFP channel
       },
       group_by=GroupBy.CHANNEL,
       variable_components=[VariableComponents.SITE],
       name="channel_analysis"
   )

**Use cases**: Multi-marker experiments where each channel represents different biological features.

Multi-Channel Processing
~~~~~~~~~~~~~~~~~~~~~~~

Different processing for different fluorescent markers:

.. code-block:: python

   # Different preprocessing for different channels
   step = FunctionStep(
       func={
           '1': [(gaussian_filter, {'sigma': 1.0}), (tophat, {'selem_radius': 15})],     # DAPI preprocessing
           '2': [(gaussian_filter, {'sigma': 2.0}), (enhance_contrast, {'percentile_range': (1, 99)})]    # GFP preprocessing
       },
       group_by=GroupBy.CHANNEL,
       variable_components=[VariableComponents.SITE],
       name="channel_preprocessing"
   )

**Use cases**: Multi-marker experiments where each channel requires different preprocessing approaches.

Z-Stack Processing
~~~~~~~~~~~~~~~~~

Processing 3D image stacks:

.. code-block:: python

   # Combine Z-planes into maximum projection
   step = FunctionStep(
       func=(max_projection, {}),
       variable_components=[VariableComponents.Z_INDEX],
       name="z_projection"
   )

**Use cases**: 3D imaging where you need to combine or analyze across Z-planes.

Choosing the Right Organization
------------------------------

**Consider Your Analysis Goal**:

- **Single image operations**: Use ``[SITE, CHANNEL]`` to process individual images
- **Multi-channel analysis**: Use ``[SITE]`` with channel-specific functions
- **Cross-site analysis**: Use ``[CHANNEL]`` to combine data across sites
- **Well-level summaries**: Use ``[WELL]`` to analyze entire wells

**Consider Your Data Structure**:

- **2D images**: Typically use ``[SITE]`` 
- **3D stacks**: May need ``[Z_INDEX]`` for projection operations
- **Time series**: May need ``[TIME]`` for temporal analysis
- **Multi-condition**: Use ``group_by=WELL`` for condition-specific processing

**Performance Considerations**:

- **Parallel processing**: More variable components = more parallel groups
- **Memory usage**: Fewer variable components = larger data groups = more memory per group
- **I/O efficiency**: Group organization affects how data is loaded and cached

The data dimensions system provides systematic control over how your analysis processes the multiple dimensions of HCS data, enabling both simple single-image operations and complex multi-dimensional workflows.
